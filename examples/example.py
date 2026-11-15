"""Run the SQmdb workflow on local queried-data Parquet files."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from itertools import zip_longest
from pathlib import Path

import polars as pl

from sqmdb import (
    PassHorizonModel,
    PitchValueModel,
    SQConfig,
    fit_player_movement_model,
    iter_defensive_influence_samples,
    iter_pitch_value,
    iter_player_influence,
    space_quality,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate SQmdb surfaces from local canonical Parquet inputs. "
            "No database connection is made."
        )
    )
    parser.add_argument("--queried-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracking-file", default="tracking-live.parquet")
    parser.add_argument("--passes-file", default="passes.parquet")
    parser.add_argument("--pitch-length-m", type=float, default=105.0)
    parser.add_argument("--pitch-width-m", type=float, default=68.0)
    parser.add_argument("--frame-rate", type=float, default=25.0)
    parser.add_argument("--grid-resolution-m", type=float, default=2.0)
    parser.add_argument("--frame-stride", type=int, default=25)
    parser.add_argument("--batch-frames", type=int, default=5)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--max-training-samples", type=int, default=1_000_000)
    return parser.parse_args()


def write_batches(
    batches: Iterable[pl.DataFrame],
    directory: Path,
) -> tuple[int, int]:
    directory.mkdir(parents=True, exist_ok=True)
    row_count = 0
    batch_count = 0
    for batch_count, batch in enumerate(batches, start=1):
        if batch.height:
            batch.write_parquet(directory / f"part-{batch_count:06d}.parquet")
            row_count += batch.height
    return batch_count, row_count


def main() -> None:
    args = parse_args()
    queried_data_dir = args.queried_data_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    tracking_path = queried_data_dir / args.tracking_file
    passes_path = queried_data_dir / args.passes_file

    for path in (tracking_path, passes_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required local Parquet input not found: {path}")

    input_dir = queried_data_dir.resolve()
    output_path = output_dir.resolve()
    if input_dir == output_path or input_dir in output_path.parents:
        raise ValueError("Choose an output directory outside queried-data.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be empty to avoid mixing results: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SQConfig(
        pitch_length_m=args.pitch_length_m,
        pitch_width_m=args.pitch_width_m,
        frame_rate=args.frame_rate,
        grid_resolution_m=args.grid_resolution_m,
        frame_stride=args.frame_stride,
        batch_frames=args.batch_frames,
    )
    tracking = pl.scan_parquet(tracking_path)
    passes = pl.scan_parquet(passes_path)

    movement_model = fit_player_movement_model(
        tracking,
        config,
        sample_stride=args.sample_stride,
    )
    horizon_model = PassHorizonModel.fit(passes)

    labels_dir = output_dir / "defensive-influence-labels"
    label_batches, label_rows = write_batches(
        iter_defensive_influence_samples(
            iter_player_influence(
                tracking,
                movement_model,
                config,
                horizon_model=horizon_model,
            ),
            tracking,
        ),
        labels_dir,
    )
    if label_rows == 0:
        raise ValueError(
            "No defensive-influence training labels were produced; "
            "check possession metadata and the selected tracking frames."
        )

    pitch_value_model = PitchValueModel.fit(
        pl.scan_parquet(str(labels_dir / "*.parquet")),
        max_training_samples=args.max_training_samples,
    )

    influence_batches = iter_player_influence(
        tracking,
        movement_model,
        config,
        horizon_model=horizon_model,
    )
    value_batches = iter_pitch_value(tracking, pitch_value_model, config)
    influence_dir = output_dir / "player-influence"
    value_dir = output_dir / "pitch-value"
    quality_dir = output_dir / "space-quality"
    for directory in (influence_dir, value_dir, quality_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sentinel = object()
    influence_rows = 0
    value_rows = 0
    quality_rows = 0
    part_count = 0
    for part, pair in enumerate(
        zip_longest(influence_batches, value_batches, fillvalue=sentinel),
        start=1,
    ):
        part_count = part
        influence_batch, value_batch = pair
        if influence_batch is sentinel or value_batch is sentinel:
            raise RuntimeError(
                "Player-influence and pitch-value iterators produced different "
                "numbers of frame batches."
            )

        frame_keys = ["match_id", "frame_id", "grid_i", "grid_j"]
        influence_frames = (
            influence_batch.select(*frame_keys).unique().sort(frame_keys)
        )
        value_frames = value_batch.select(*frame_keys).unique().sort(frame_keys)
        if not influence_frames.equals(value_frames):
            raise RuntimeError(
                "Player-influence and pitch-value batches do not contain the "
                "same frames and grid cells."
            )

        quality_batch = space_quality(influence_batch, value_batch)
        influence_batch.write_parquet(
            influence_dir / f"part-{part:06d}.parquet"
        )
        value_batch.write_parquet(
            value_dir / f"part-{part:06d}.parquet"
        )
        quality_batch.write_parquet(
            quality_dir / f"part-{part:06d}.parquet"
        )
        influence_rows += influence_batch.height
        value_rows += value_batch.height
        quality_rows += quality_batch.height

    if part_count == 0:
        raise ValueError("No output frames were calculated from the tracking input.")

    print(
        f"Wrote {label_rows:,} label rows in {label_batches} batches; "
        f"{influence_rows:,} player-influence rows; "
        f"{value_rows:,} pitch-value rows; and "
        f"{quality_rows:,} team control/space-quality rows to {output_dir}."
    )


if __name__ == "__main__":
    main()
