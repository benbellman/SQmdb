from __future__ import annotations

from collections.abc import Iterator
from typing import Any
import polars as pl

from .config import SQConfig

TRACKING_COLUMNS = {"frame_id", "player_id", "team_id", "x", "y"}


def as_lazy(
    data: pl.DataFrame | pl.LazyFrame,
    required: set[str] | None = None,
) -> pl.LazyFrame:
    if not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError("Expected a Polars DataFrame or LazyFrame.")
    lazy = data.lazy() if isinstance(data, pl.DataFrame) else data
    names = set(lazy.collect_schema().names())
    missing = (required or set()) - names
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}.")
    if "match_id" not in names:
        lazy = lazy.with_columns(pl.lit("__single_match__").alias("match_id"))
    if "frame_id" in names:
        lazy = lazy.with_columns(pl.col("frame_id").cast(pl.Int64, strict=True))
    numeric_coordinates = [
        column for column in ("x", "y", "ball_x", "ball_y") if column in names
    ]
    if numeric_coordinates:
        lazy = lazy.with_columns(
            *[
                pl.col(column).cast(pl.Float64, strict=True)
                for column in numeric_coordinates
            ]
        )
    return lazy


def tracking_batches(
    data: pl.DataFrame | pl.LazyFrame,
    config: SQConfig,
    required: set[str] | None = None,
    frame_stride: int | None = None,
) -> Iterator[pl.DataFrame]:
    lazy = as_lazy(data, TRACKING_COLUMNS | (required or set()))
    stride = config.frame_stride if frame_stride is None else frame_stride
    if stride < 1:
        raise ValueError("frame_stride must be a positive integer.")

    chunks = lazy.collect_batches(
        chunk_size=max(1024, config.batch_frames * 256),
        maintain_order=True,
        engine="streaming",
    )
    pending: pl.DataFrame | None = None
    frame_buffer: list[pl.DataFrame] = []
    first_frame: int | None = None
    active_match: Any = None
    completed_matches: set[Any] = set()
    last_frame: int | None = None

    def consume(frame: pl.DataFrame) -> pl.DataFrame | None:
        nonlocal first_frame, active_match, last_frame
        match_id = frame.get_column("match_id")[0]
        raw_frame_id = frame.get_column("frame_id")[0]
        if raw_frame_id is None:
            raise ValueError("frame_id cannot contain null values.")
        frame_id = int(raw_frame_id)
        if match_id is None:
            raise ValueError("match_id cannot contain null values.")
        if match_id != active_match:
            if match_id in completed_matches:
                raise ValueError("Input rows must be grouped by match_id and ordered by frame_id.")
            if active_match is not None:
                completed_matches.add(active_match)
            active_match = match_id
            first_frame = frame_id
            last_frame = None
        if last_frame is not None and frame_id <= last_frame:
            raise ValueError("Input rows must be grouped by match_id and ordered by frame_id.")
        last_frame = frame_id
        if first_frame is not None and (frame_id - first_frame) % stride == 0:
            frame_buffer.append(frame)
        if len(frame_buffer) >= config.batch_frames:
            ready = pl.concat(frame_buffer, how="vertical")
            frame_buffer.clear()
            return ready
        return None

    for chunk in chunks:
        combined = pl.concat([pending, chunk], how="vertical") if pending is not None else chunk
        run_keys = (
            combined.select("match_id", "frame_id")
            .with_columns(
                (
                    (pl.col("match_id") != pl.col("match_id").shift(1))
                    | (pl.col("frame_id") != pl.col("frame_id").shift(1))
                )
                .fill_null(True)
                .alias("_new_frame")
            )
            .filter("_new_frame")
            .select("match_id", "frame_id")
        )
        repeated_frames = (
            run_keys.group_by("match_id", "frame_id")
            .len()
            .filter(pl.col("len") > 1)
            .limit(1)
            .height
        )
        if repeated_frames:
            raise ValueError(
                "Input rows must be grouped by match_id and ordered by frame_id."
            )
        last_key = combined.select("match_id", "frame_id").row(-1)
        is_pending = (pl.col("match_id") == last_key[0]) & (
            pl.col("frame_id") == last_key[1]
        )
        complete = combined.filter(~is_pending)
        pending = combined.filter(is_pending)
        for frame in complete.partition_by(
            ["match_id", "frame_id"], maintain_order=True
        ):
            batch = consume(frame)
            if batch is not None:
                yield batch
    if pending is not None and pending.height:
        for frame in pending.partition_by(
            ["match_id", "frame_id"], maintain_order=True
        ):
            batch = consume(frame)
            if batch is not None:
                yield batch
    if frame_buffer:
        yield pl.concat(frame_buffer, how="vertical")


def require_finite(data: pl.DataFrame, columns: list[str]) -> None:
    for column in columns:
        values = data.get_column(column)
        if values.null_count() or values.is_nan().any() or values.is_infinite().any():
            raise ValueError(f"Column {column!r} must contain only finite, non-null values.")


def validate_tracking_batch(batch: pl.DataFrame, config: SQConfig) -> None:
    require_finite(batch, ["x", "y"])
    if batch.select(
        pl.any_horizontal(
            pl.col("player_id").is_null(),
            pl.col("team_id").is_null(),
            pl.col("frame_id").is_null(),
        ).any()
    ).item():
        raise ValueError("frame_id, player_id, and team_id cannot contain null values.")
    keys = ["match_id", "frame_id", "player_id"]
    duplicate = (
        batch.group_by(keys)
        .len()
        .filter(pl.col("len") > 1)
        .limit(1)
        .height
    )
    if duplicate:
        raise ValueError("Tracking rows must be unique by match_id, frame_id, and player_id.")
    if config.frame_rate <= 0:
        raise ValueError("frame_rate must be positive.")


def ensure_frame_metadata_consistent(
    data: pl.DataFrame | pl.LazyFrame,
    columns: list[str],
) -> pl.DataFrame:
    lazy = as_lazy(data, {"frame_id", *columns})
    aggregates: list[pl.Expr] = []
    for column in columns:
        aggregates.extend(
            [
                pl.col(column).n_unique().alias(f"_{column}_count"),
                pl.col(column).first().alias(column),
            ]
        )
    frames = lazy.group_by("match_id", "frame_id").agg(aggregates).collect()
    for column in columns:
        if frames.filter(pl.col(f"_{column}_count") > 1).height:
            raise ValueError(f"{column!r} must be constant within each match/frame.")
    return frames.select("match_id", "frame_id", *columns)
