from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer

from ._data import as_lazy


class PassHorizonModel:
    """Power-transformed regression from pass distance to pass travel time."""

    def __init__(self, estimator: object, min_seconds: float, max_seconds: float) -> None:
        self._estimator = estimator
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

    @classmethod
    def fit(
        cls,
        passes: pl.DataFrame | pl.LazyFrame,
    ) -> "PassHorizonModel":
        """Fit using canonical ``pass_distance_m`` and ``pass_duration_seconds`` columns."""
        data = as_lazy(
            passes,
            {"pass_distance_m", "pass_duration_seconds"},
        ).select(
            pl.col("pass_distance_m").cast(pl.Float64, strict=True),
            pl.col("pass_duration_seconds").cast(pl.Float64, strict=True),
        ).collect(engine="streaming")
        if data.height < 10:
            raise ValueError("At least 10 completed passes are required to fit a horizon model.")
        if any(
            data.get_column(column).null_count()
            or data.get_column(column).is_nan().any()
            or data.get_column(column).is_infinite().any()
            for column in data.columns
        ):
            raise ValueError("Pass distances and durations must be finite and non-null.")
        distances = data.get_column("pass_distance_m").to_numpy()
        durations = data.get_column("pass_duration_seconds").to_numpy()
        if np.any(distances < 0) or np.any(durations <= 0):
            raise ValueError("Pass distances must be non-negative and durations positive.")
        if np.unique(distances).size < 2:
            raise ValueError("Pass distances must contain at least two distinct values.")
        estimator = make_pipeline(
            PowerTransformer(method="yeo-johnson"),
            LinearRegression(),
        )
        estimator.fit(distances.reshape(-1, 1), durations)
        return cls(estimator, float(durations.min()), float(durations.max()))

    def predict(self, pass_distance_m: np.ndarray | float) -> np.ndarray:
        distances = np.asarray(pass_distance_m, dtype=float)
        if np.any(~np.isfinite(distances)) or np.any(distances < 0):
            raise ValueError("pass_distance_m must contain finite, non-negative values.")
        predictions = self._estimator.predict(distances.reshape(-1, 1))
        return np.clip(predictions, self.min_seconds, self.max_seconds)
