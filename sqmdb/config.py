from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class SQConfig:
    """Pitch, grid, and bounded-execution settings.

    Input coordinates are expected in metres, centered at (0, 0), with the
    attacking team oriented toward positive x. Normalize other coordinate
    conventions before calling SQmdb.
    """

    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    grid_resolution_m: float = 1.0
    frame_rate: float = 25.0
    velocity_lag_seconds: float = 0.2
    frame_stride: int = 1
    batch_frames: int = 1
    movement_density_resolution_m: float = 0.5
    movement_density_extent_m: float = 30.0
    movement_kde_bandwidth_m: float = 1.0
    movement_horizons_seconds: tuple[float, ...] = tuple(
        float(value) for value in np.arange(0.2, 5.01, 0.2)
    )
    movement_min_samples: int = 20

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, float | None],
        **overrides: object,
    ) -> "SQConfig":
        """Use canonical pitch dimensions in metadata, falling back to defaults."""
        dimensions: dict[str, object] = {
            name: metadata[name]
            for name in ("pitch_length_m", "pitch_width_m")
            if metadata.get(name) is not None
        }
        dimensions.update(overrides)
        return cls(**dimensions)

    def __post_init__(self) -> None:
        positive = {
            "pitch_length_m": self.pitch_length_m,
            "pitch_width_m": self.pitch_width_m,
            "grid_resolution_m": self.grid_resolution_m,
            "frame_rate": self.frame_rate,
            "velocity_lag_seconds": self.velocity_lag_seconds,
            "movement_density_resolution_m": self.movement_density_resolution_m,
            "movement_density_extent_m": self.movement_density_extent_m,
            "movement_kde_bandwidth_m": self.movement_kde_bandwidth_m,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
        if self.frame_stride < 1 or self.batch_frames < 1:
            raise ValueError("frame_stride and batch_frames must be positive integers.")
        if self.movement_min_samples < 1:
            raise ValueError("movement_min_samples must be positive.")
        if not self.movement_horizons_seconds or any(
            not np.isfinite(value) or value <= 0 for value in self.movement_horizons_seconds
        ):
            raise ValueError("movement_horizons_seconds must contain positive finite values.")
        if tuple(sorted(set(self.movement_horizons_seconds))) != self.movement_horizons_seconds:
            raise ValueError("movement_horizons_seconds must be unique and sorted.")

    def grid(self) -> pl.DataFrame:
        """Return the configured cell-center grid and its stable integer indices."""
        nx = max(1, int(round(self.pitch_length_m / self.grid_resolution_m)))
        ny = max(1, int(round(self.pitch_width_m / self.grid_resolution_m)))
        xs = np.linspace(
            -self.pitch_length_m / 2 + self.pitch_length_m / (2 * nx),
            self.pitch_length_m / 2 - self.pitch_length_m / (2 * nx),
            nx,
        )
        ys = np.linspace(
            -self.pitch_width_m / 2 + self.pitch_width_m / (2 * ny),
            self.pitch_width_m / 2 - self.pitch_width_m / (2 * ny),
            ny,
        )
        return pl.DataFrame(
            {
                "grid_i": np.repeat(np.arange(nx, dtype=np.int32), ny),
                "grid_j": np.tile(np.arange(ny, dtype=np.int32), nx),
                "x": np.repeat(xs, ny),
                "y": np.tile(ys, nx),
            }
        )

    @property
    def movement_lag_frames(self) -> int:
        return max(1, int(round(self.velocity_lag_seconds * self.frame_rate)))


def velocity_bin(speed_kmh: float) -> str:
    if speed_kmh < 1.0:
        return "standing"
    if speed_kmh < 7.0:
        return "walking"
    if speed_kmh < 14.0:
        return "jogging"
    if speed_kmh < 20.0:
        return "running"
    return "sprinting"
