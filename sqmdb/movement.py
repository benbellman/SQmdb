from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter, map_coordinates

from ._data import (
    TRACKING_COLUMNS,
    as_lazy,
    require_finite,
    tracking_batches,
    validate_tracking_batch,
)
from .config import SQConfig, velocity_bin
from .horizon import PassHorizonModel


@dataclass(frozen=True, slots=True)
class _MovementDensity:
    values: np.ndarray
    extent: float
    resolution: float

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix = (x + self.extent) / self.resolution - 0.5
        iy = (y + self.extent) / self.resolution - 0.5
        sampled = map_coordinates(
            self.values,
            np.vstack((iy, ix)),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        peak = float(self.values.max())
        if peak <= 0:
            return np.zeros_like(sampled)
        return np.clip(sampled / peak, 0.0, 1.0)


class PlayerMovementModel:
    """Velocity-conditioned, per-player movement KDEs.

    Density maps use a binned Gaussian KDE approximation to make repeated
    pitch-grid queries inexpensive. A pooled model provides a fallback for
    players or velocity bins that were not observed during training.
    """

    def __init__(
        self,
        distributions: dict[tuple[Any, str, float], _MovementDensity],
        horizons: tuple[float, ...],
        extent: float,
        resolution: float,
    ) -> None:
        self._distributions = distributions
        self.horizons = horizons
        self.extent = extent
        self.resolution = resolution

    def _select(self, player_id: Any, speed_bin: str, horizon_seconds: float) -> _MovementDensity:
        player_horizons = [
            horizon
            for horizon in self.horizons
            if (player_id, speed_bin, horizon) in self._distributions
        ]
        if not player_horizons:
            player_horizons = [
                horizon
                for horizon in self.horizons
                if (None, speed_bin, horizon) in self._distributions
            ]
            key_player = None
        else:
            key_player = player_id
        if not player_horizons:
            raise ValueError(
                f"No movement observations for velocity bin {speed_bin!r} "
                f"at or near horizon {horizon_seconds:g}s."
            )
        horizon = min(player_horizons, key=lambda item: abs(item - horizon_seconds))
        return self._distributions[(key_player, speed_bin, horizon)]


def _triplets_for_horizon(
    data: pl.LazyFrame,
    lag_frames: int,
    horizon_frames: int,
    sample_stride: int,
) -> pl.DataFrame:
    keys = ["match_id", "player_id", "frame_id"]
    positions = data.select(*keys, "x", "y")
    first_frames = positions.group_by("match_id").agg(
        pl.col("frame_id").min().alias("_first_frame")
    )
    current = positions.join(first_frames, on="match_id").filter(
        ((pl.col("frame_id") - pl.col("_first_frame")) % sample_stride) == 0
    ).drop("_first_frame")
    past = positions.select(
        "match_id",
        pl.col("player_id"),
        (pl.col("frame_id") + lag_frames).alias("frame_id"),
        pl.col("x").alias("past_x"),
        pl.col("y").alias("past_y"),
    )
    future = positions.select(
        "match_id",
        pl.col("player_id"),
        (pl.col("frame_id") - horizon_frames).alias("frame_id"),
        pl.col("x").alias("future_x"),
        pl.col("y").alias("future_y"),
    )
    joined = (
        current.join(past, on=keys, how="inner")
        .join(future, on=keys, how="inner")
        .filter(
            pl.col("x").is_not_null()
            & pl.col("y").is_not_null()
            & pl.col("past_x").is_not_null()
            & pl.col("future_x").is_not_null()
        )
    )
    return joined.select(
        "player_id",
        ((pl.col("x") - pl.col("past_x"))).alias("_vx_delta"),
        ((pl.col("y") - pl.col("past_y"))).alias("_vy_delta"),
        (pl.col("future_x") - pl.col("x")).alias("_dx"),
        (pl.col("future_y") - pl.col("y")).alias("_dy"),
    ).collect()


def fit_player_movement_model(
    tracking: pl.DataFrame | pl.LazyFrame,
    config: SQConfig = SQConfig(),
    *,
    sample_stride: int = 1,
) -> PlayerMovementModel:
    """Fit movement densities from historical tracking trajectories.

    Uses 0.2-second velocity estimates, the paper's five speed bins, and the
    configured future horizons. Endpoint displacements are translated to the
    origin and rotated so the estimated initial heading aligns with +x.
    """
    if sample_stride < 1:
        raise ValueError("sample_stride must be a positive integer.")
    data = as_lazy(tracking, TRACKING_COLUMNS)
    positions = data.select(
        "match_id",
        pl.col("frame_id").cast(pl.Int64),
        "player_id",
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
    )
    invalid_rows = positions.filter(
        pl.any_horizontal(
            pl.col("match_id").is_null(),
            pl.col("frame_id").is_null(),
            pl.col("player_id").is_null(),
            pl.col("x").is_null(),
            pl.col("y").is_null(),
            pl.col("x").is_nan(),
            pl.col("y").is_nan(),
            pl.col("x").is_infinite(),
            pl.col("y").is_infinite(),
        )
    ).limit(1).collect().height
    if invalid_rows:
        raise ValueError("frame_id, player_id, x, and y must be finite and non-null.")
    # Validate row identity without materializing tracking positions.
    duplicates = (
        positions.group_by("match_id", "frame_id", "player_id")
        .len()
        .filter(pl.col("len") > 1)
        .limit(1)
        .collect()
        .height
    )
    if duplicates:
        raise ValueError("Tracking rows must be unique by match_id, frame_id, and player_id.")

    lag = config.movement_lag_frames
    lag_seconds = lag / config.frame_rate
    extent = config.movement_density_extent_m
    resolution = config.movement_density_resolution_m
    shape = (int(np.ceil(2 * extent / resolution)),) * 2
    histogram_counts: dict[tuple[Any, str, float], np.ndarray] = {}
    histogram_totals: dict[tuple[Any, str, float], int] = {}
    for horizon in config.movement_horizons_seconds:
        horizon_frames = max(1, int(round(horizon * config.frame_rate)))
        effective_horizon = horizon_frames / config.frame_rate
        # Exact frame IDs are required for the configured fixed-rate triplets.
        triplets = _triplets_for_horizon(positions, lag, horizon_frames, sample_stride)
        if not triplets.height:
            continue
        require_finite(triplets, ["_vx_delta", "_vy_delta", "_dx", "_dy"])
        vx = triplets.get_column("_vx_delta").to_numpy() / lag_seconds
        vy = triplets.get_column("_vy_delta").to_numpy() / lag_seconds
        speed_kmh = np.hypot(vx, vy) * 3.6
        angle = np.arctan2(vy, vx)
        dx = triplets.get_column("_dx").to_numpy()
        dy = triplets.get_column("_dy").to_numpy()
        forward = dx * np.cos(angle) + dy * np.sin(angle)
        lateral = -dx * np.sin(angle) + dy * np.cos(angle)
        player_ids = triplets.get_column("player_id").to_list()
        bins = np.asarray([velocity_bin(float(speed)) for speed in speed_kmh], dtype=object)

        frame = pl.DataFrame(
            {
                "player_id": player_ids,
                "speed_bin": bins,
                "forward": forward,
                "lateral": lateral,
            }
        )
        for group_key, part in frame.partition_by(
            ["player_id", "speed_bin"], as_dict=True
        ).items():
            player_id, speed_name = group_key
            points = np.column_stack(
                (part.get_column("forward").to_numpy(), part.get_column("lateral").to_numpy())
            )
            player_key = (player_id, str(speed_name), effective_horizon)
            pooled_key = (None, str(speed_name), effective_horizon)
            for key in (player_key, pooled_key):
                counts = histogram_counts.setdefault(key, np.zeros(shape, dtype=np.float32))
                counts += np.histogram2d(
                    points[:, 1],
                    points[:, 0],
                    bins=shape,
                    range=((-extent, extent), (-extent, extent)),
                )[0].astype(np.float32)
                histogram_totals[key] = histogram_totals.get(key, 0) + len(points)

    distributions: dict[tuple[Any, str, float], _MovementDensity] = {}
    sigma = config.movement_kde_bandwidth_m / resolution
    for key, counts in histogram_counts.items():
        if histogram_totals[key] < config.movement_min_samples:
            continue
        smooth = gaussian_filter(counts, sigma=sigma, mode="constant").astype(np.float32)
        total = float(smooth.sum())
        if total > 0:
            smooth /= total
        distributions[key] = _MovementDensity(smooth, extent, resolution)
    if not distributions:
        raise ValueError(
            "Not enough valid player trajectories to fit movement densities. "
            "Check frame continuity, sample volume, and movement_min_samples."
        )
    fitted_horizons = tuple(sorted({key[2] for key in distributions}))
    return PlayerMovementModel(distributions, fitted_horizons, extent, resolution)


def iter_player_influence(
    tracking: pl.DataFrame | pl.LazyFrame,
    model: PlayerMovementModel,
    config: SQConfig = SQConfig(),
    *,
    horizon_seconds: float | None = None,
    horizon_model: PassHorizonModel | None = None,
    frame_stride: int | None = None,
) -> Iterator[pl.DataFrame]:
    """Yield bounded long-form player influence surfaces, one frame batch at a time."""
    if horizon_model is not None and horizon_seconds is not None:
        raise ValueError("Pass either horizon_seconds or horizon_model, not both.")
    if horizon_seconds is not None and (
        horizon_seconds <= 0 or not np.isfinite(horizon_seconds)
    ):
        raise ValueError("horizon_seconds must be a positive finite number.")
    selected_horizon = 1.0 if horizon_seconds is None and horizon_model is None else horizon_seconds
    required = TRACKING_COLUMNS | (
        {"ball_x", "ball_y"} if horizon_model is not None else set()
    )
    source = as_lazy(tracking, required)
    grid = config.grid()
    grid_x = grid.get_column("x").to_numpy()
    grid_y = grid.get_column("y").to_numpy()
    grid_i = grid.get_column("grid_i").to_numpy()
    grid_j = grid.get_column("grid_j").to_numpy()
    lag_frames = config.movement_lag_frames
    lag_seconds = lag_frames / config.frame_rate
    output_stride = config.frame_stride if frame_stride is None else frame_stride
    if output_stride < 1:
        raise ValueError("frame_stride must be a positive integer.")
    history: dict[tuple[Any, Any, int], tuple[float, float]] = {}
    first_frame_by_match: dict[Any, int] = {}
    active_match_id: Any = None

    for batch in tracking_batches(source, config, frame_stride=1):
        validate_tracking_batch(batch, config)
        output_frames: list[pl.DataFrame] = []
        for frame in batch.partition_by(
            ["match_id", "frame_id"], maintain_order=True
        ):
            match_id = frame.get_column("match_id")[0]
            frame_id = int(frame.get_column("frame_id")[0])
            if active_match_id != match_id:
                history.clear()
                active_match_id = match_id
                first_frame_by_match[match_id] = frame_id
            first_frame = first_frame_by_match[match_id]
            frame_is_selected = (frame_id - first_frame) % output_stride == 0
            if horizon_model is not None:
                for column in ("ball_x", "ball_y"):
                    if frame.get_column(column).null_count() or frame.get_column(
                        column
                    ).n_unique() != 1:
                        raise ValueError(
                            f"{column!r} must be non-null and constant within each frame."
                        )
            player_records = frame.iter_rows(named=True)
            frame_players = list(player_records)
            player_ids: list[Any] = []
            team_ids: list[Any] = []
            values_by_player: list[np.ndarray] = []
            for player in frame_players:
                vx = 0.0
                vy = 0.0
                previous = history.get(
                    (match_id, player["player_id"], frame_id - lag_frames)
                )
                if previous is not None:
                    vx = (float(player["x"]) - previous[0]) / lag_seconds
                    vy = (float(player["y"]) - previous[1]) / lag_seconds
                if frame_is_selected:
                    speed_bin = velocity_bin(float(np.hypot(vx, vy) * 3.6))
                    requested_horizon = selected_horizon
                    if horizon_model is not None:
                        distance = np.hypot(
                            float(player["x"]) - float(player["ball_x"]),
                            float(player["y"]) - float(player["ball_y"]),
                        )
                        requested_horizon = float(horizon_model.predict(distance)[0])
                    if requested_horizon is None:
                        raise RuntimeError("No movement horizon was selected.")
                    density = model._select(
                        player["player_id"], speed_bin, requested_horizon
                    )
                    theta = float(np.arctan2(vy, vx))
                    dx = grid_x - float(player["x"])
                    dy = grid_y - float(player["y"])
                    forward = dx * np.cos(theta) + dy * np.sin(theta)
                    lateral = -dx * np.sin(theta) + dy * np.cos(theta)
                    values_by_player.append(density.evaluate(forward, lateral))
                    player_ids.append(player["player_id"])
                    team_ids.append(player["team_id"])

            if frame_is_selected:
                output_frames.append(
                    pl.DataFrame(
                        {
                            "match_id": [match_id] * (len(player_ids) * grid.height),
                            "frame_id": [frame_id] * (len(player_ids) * grid.height),
                            "player_id": [
                                player_id for player_id in player_ids for _ in range(grid.height)
                            ],
                            "team_id": [
                                team_id for team_id in team_ids for _ in range(grid.height)
                            ],
                            "grid_i": np.tile(grid_i, len(player_ids)),
                            "grid_j": np.tile(grid_j, len(player_ids)),
                            "x": np.tile(grid_x, len(player_ids)),
                            "y": np.tile(grid_y, len(player_ids)),
                            "player_influence": np.concatenate(values_by_player).astype(
                                np.float32
                            ),
                        }
                    )
                )

            for player in frame_players:
                history[(match_id, player["player_id"], frame_id)] = (
                    float(player["x"]),
                    float(player["y"]),
                )
            cutoff = frame_id - lag_frames + 1
            for history_key in list(history):
                if history_key[0] == match_id and history_key[2] < cutoff:
                    del history[history_key]
        if output_frames:
            yield pl.concat(output_frames, how="vertical")


def player_influence(
    tracking: pl.DataFrame | pl.LazyFrame,
    model: PlayerMovementModel,
    config: SQConfig = SQConfig(),
    *,
    horizon_seconds: float | None = None,
    horizon_model: PassHorizonModel | None = None,
    frame_stride: int | None = None,
) -> pl.DataFrame:
    """Eager convenience wrapper around :func:`iter_player_influence`."""
    batches = list(
        iter_player_influence(
            tracking,
            model,
            config,
            horizon_seconds=horizon_seconds,
            horizon_model=horizon_model,
            frame_stride=frame_stride,
        )
    )
    if not batches:
        return pl.DataFrame(
            schema={
                "match_id": pl.String,
                "frame_id": pl.Int64,
                "player_id": pl.String,
                "team_id": pl.String,
                "grid_i": pl.Int32,
                "grid_j": pl.Int32,
                "x": pl.Float64,
                "y": pl.Float64,
                "player_influence": pl.Float32,
            }
        )
    return pl.concat(batches, how="vertical")
