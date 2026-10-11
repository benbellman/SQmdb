from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import polars as pl
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ._data import (
    as_lazy,
    ensure_frame_metadata_consistent,
    tracking_batches,
)
from .config import SQConfig

PITCH_VALUE_SAMPLE_COLUMNS = {
    "ball_x",
    "ball_y",
    "grid_x",
    "grid_y",
    "defensive_influence",
}


class PitchValueModel:
    """Feed-forward predictor of defensive influence from ball and grid location."""

    def __init__(self, estimator: Any) -> None:
        self._estimator = estimator

    @classmethod
    def fit(
        cls,
        samples: pl.DataFrame | pl.LazyFrame,
        *,
        hidden_layer_sizes: tuple[int, ...] = (64, 64),
        random_state: int = 0,
        max_iter: int = 300,
        max_training_samples: int = 1_000_000,
    ) -> "PitchValueModel":
        if max_training_samples < 10:
            raise ValueError("max_training_samples must be at least 10.")
        source = as_lazy(samples, PITCH_VALUE_SAMPLE_COLUMNS)
        training_columns = [
            "ball_x",
            "ball_y",
            "grid_x",
            "grid_y",
            "defensive_influence",
        ]
        sampling_keys = [
            key
            for key in ("match_id", "frame_id", "grid_i", "grid_j")
            if key in source.collect_schema().names()
        ]
        lazy = source.select(
            *[
                pl.col(column).cast(pl.Float64, strict=True)
                for column in training_columns
            ],
            *sampling_keys,
        )
        total_samples = int(lazy.select(pl.len()).collect().item())
        if total_samples > max_training_samples:
            divisor = int(np.ceil(total_samples / max_training_samples))
            lazy = lazy.filter(
                (
                    pl.struct(*(sampling_keys or training_columns))
                    .hash(seed=random_state)
                    % divisor
                )
                == 0
            )
        data = lazy.select(training_columns).collect(engine="streaming")
        if data.height < 10:
            raise ValueError("At least 10 pitch-value training examples are required.")
        if not hidden_layer_sizes or any(size < 1 for size in hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must contain positive integers.")
        if max_iter < 1:
            raise ValueError("max_iter must be positive.")
        for column in data.columns:
            series = data.get_column(column)
            if series.null_count() or series.is_nan().any() or series.is_infinite().any():
                raise ValueError(f"Training column {column!r} must be finite and non-null.")
        target = data.get_column("defensive_influence").to_numpy()
        if np.any((target < 0) | (target > 1)):
            raise ValueError("defensive_influence targets must be in [0, 1].")
        features = data.select("ball_x", "ball_y", "grid_x", "grid_y").to_numpy()
        estimator = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation="relu",
                solver="adam",
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=data.height >= 100,
                n_iter_no_change=15,
            ),
        )
        estimator.fit(features, target)
        return cls(estimator)

    def predict_defensive_influence(
        self,
        ball_x: np.ndarray,
        ball_y: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
    ) -> np.ndarray:
        features = np.column_stack((ball_x, ball_y, grid_x, grid_y))
        return np.clip(self._estimator.predict(features), 0.0, 1.0)


def defensive_influence_samples(
    player_surfaces: pl.DataFrame | pl.LazyFrame | Iterable[pl.DataFrame],
    tracking: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame:
    """Build observed defensive-influence labels for pitch-value model fitting.

    The tracking data must include ball_x, ball_y, and possession_team_id,
    repeated consistently on the player rows for each frame.
    """
    if isinstance(player_surfaces, pl.LazyFrame):
        surfaces = player_surfaces.collect(engine="streaming")
        batches = [surfaces]
    elif isinstance(player_surfaces, pl.DataFrame):
        batches = [player_surfaces]
    else:
        batches = player_surfaces
    results = list(iter_defensive_influence_samples(batches, tracking))
    return pl.concat(results, how="vertical") if results else pl.DataFrame()


def iter_defensive_influence_samples(
    player_surface_batches: Iterable[pl.DataFrame],
    tracking: pl.DataFrame | pl.LazyFrame,
) -> Iterator[pl.DataFrame]:
    """Yield bounded defensive-influence training labels from player surfaces."""
    tracking_lazy = as_lazy(
        tracking,
        {"frame_id", "player_id", "team_id", "possession_team_id", "ball_x", "ball_y"},
    )
    tracking_names = set(tracking_lazy.collect_schema().names())
    has_goalkeeper_flag = "is_goalkeeper" in tracking_names
    for player_surfaces in player_surface_batches:
        surfaces = as_lazy(
            player_surfaces,
            {
                "frame_id",
                "player_id",
                "team_id",
                "grid_i",
                "grid_j",
                "x",
                "y",
                "player_influence",
            },
        )
        frame_keys = surfaces.select("match_id", "frame_id").unique()
        frame_data = ensure_frame_metadata_consistent(
            tracking_lazy.join(frame_keys, on=["match_id", "frame_id"], how="semi"),
            ["possession_team_id", "ball_x", "ball_y"],
        )
        joined = surfaces.join(
            frame_data.lazy(),
            on=["match_id", "frame_id"],
            how="inner",
        ).filter(pl.col("team_id") != pl.col("possession_team_id"))
        if has_goalkeeper_flag:
            player_keys = surfaces.select("match_id", "frame_id", "player_id").unique()
            goalkeepers = tracking_lazy.select(
                "match_id", "frame_id", "player_id", "is_goalkeeper"
            ).join(player_keys, on=["match_id", "frame_id", "player_id"], how="semi")
            joined = joined.join(
                goalkeepers,
                on=["match_id", "frame_id", "player_id"],
                how="left",
            ).filter(~pl.col("is_goalkeeper").fill_null(False))
        result = (
            joined.group_by("match_id", "frame_id", "grid_i", "grid_j", "x", "y")
            .agg(
                pl.col("player_influence")
                .sum()
                .clip(upper_bound=1.0)
                .alias("defensive_influence"),
                pl.col("ball_x").first(),
                pl.col("ball_y").first(),
            )
            .rename({"x": "grid_x", "y": "grid_y"})
            .collect(engine="streaming")
        )
        if result.height:
            yield result


def _pitch_value_rows(
    batch: pl.DataFrame,
    model: PitchValueModel,
    config: SQConfig,
) -> pl.DataFrame:
    frames = ensure_frame_metadata_consistent(
        batch,
        ["ball_x", "ball_y"],
    )
    grid = config.grid()
    grid_x = grid.get_column("x").to_numpy()
    grid_y = grid.get_column("y").to_numpy()
    grid_i = grid.get_column("grid_i").to_numpy()
    grid_j = grid.get_column("grid_j").to_numpy()
    rows: list[pl.DataFrame] = []
    goal_x = config.pitch_length_m / 2
    max_distance = np.hypot(config.pitch_length_m, config.pitch_width_m / 2)
    distance_value = np.clip(
        1.0 - np.hypot(grid_x - goal_x, grid_y) / max_distance,
        0.0,
        1.0,
    )
    for record in frames.iter_rows(named=True):
        ball_x = np.full(grid.height, float(record["ball_x"]))
        ball_y = np.full(grid.height, float(record["ball_y"]))
        defensive = model.predict_defensive_influence(ball_x, ball_y, grid_x, grid_y)
        rows.append(
            pl.DataFrame(
                {
                    "match_id": [record["match_id"]] * grid.height,
                    "frame_id": [record["frame_id"]] * grid.height,
                    "grid_i": grid_i,
                    "grid_j": grid_j,
                    "x": grid_x,
                    "y": grid_y,
                    "pitch_value": (defensive * distance_value).astype(np.float32),
                }
            )
        )
    return pl.concat(rows, how="vertical") if rows else pl.DataFrame()


def iter_pitch_value(
    tracking: pl.DataFrame | pl.LazyFrame,
    model: PitchValueModel,
    config: SQConfig = SQConfig(),
    *,
    frame_stride: int | None = None,
) -> Iterator[pl.DataFrame]:
    """Yield bounded long-form pitch-value surfaces."""
    for batch in tracking_batches(
        tracking,
        config,
        required={"ball_x", "ball_y"},
        frame_stride=frame_stride,
    ):
        yield _pitch_value_rows(batch, model, config)


def pitch_value(
    tracking: pl.DataFrame | pl.LazyFrame,
    model: PitchValueModel,
    config: SQConfig = SQConfig(),
    *,
    frame_stride: int | None = None,
) -> pl.DataFrame:
    """Eager convenience wrapper around :func:`iter_pitch_value`."""
    batches = list(iter_pitch_value(tracking, model, config, frame_stride=frame_stride))
    return pl.concat(batches, how="vertical") if batches else pl.DataFrame()
