from __future__ import annotations

import pytest

from examples.local_validation import cursor_to_polars
from sqmdb import (
    PassHorizonModel,
    SQConfig,
    fit_player_movement_model,
    iter_player_influence,
)


class FakeCursor:
    def __init__(self, columns: list[str], rows: list[tuple[object, ...]]) -> None:
        self.description = [(name, None, None, None, None, None, None) for name in columns]
        self._rows = rows

    def fetchmany(self, size: int) -> list[tuple[object, ...]]:
        rows, self._rows = self._rows[:size], self._rows[size:]
        return rows


def test_cursor_to_polars_reads_batches_and_empty_results() -> None:
    cursor = FakeCursor(["frame_id", "x"], [(1, 2.5), (2, 3.5), (3, 4.5)])

    result = cursor_to_polars(cursor, batch_size=2)

    assert result.to_dict(as_series=False) == {
        "frame_id": [1, 2, 3],
        "x": [2.5, 3.5, 4.5],
    }
    empty = cursor_to_polars(FakeCursor(["frame_id"], []), batch_size=2)
    assert empty.columns == ["frame_id"]


def test_cursor_to_polars_rejects_invalid_result_shapes() -> None:
    with pytest.raises(ValueError, match="column names"):
        cursor_to_polars(FakeCursor(["x", "x"], [(1, 2)]))
    with pytest.raises(ValueError, match="batch_size"):
        cursor_to_polars(FakeCursor(["x"], []), batch_size=0)

    class NoResult:
        description = None

        def fetchmany(self, size: int) -> list[tuple[object, ...]]:
            return []

    with pytest.raises(ValueError, match="query result"):
        cursor_to_polars(NoResult())


def test_dbapi_polars_data_runs_through_sqmdb() -> None:
    tracking_rows = []
    for frame_id in range(31):
        for player_id, team_id, x, y in (
            ("runner", "attack", frame_id * 0.1, 0.0),
            ("static", "defend", 2.0, 1.0),
        ):
            tracking_rows.append(
                (
                    "match-1",
                    frame_id,
                    player_id,
                    team_id,
                    x,
                    y,
                    1.0,
                    0.0,
                )
            )
    tracking = cursor_to_polars(
        FakeCursor(
            [
                "match_id",
                "frame_id",
                "player_id",
                "team_id",
                "x",
                "y",
                "ball_x",
                "ball_y",
            ],
            tracking_rows,
        ),
        batch_size=16,
    )
    passes = cursor_to_polars(
        FakeCursor(
            ["pass_distance_m", "pass_duration_seconds"],
            [(float(distance), 0.2 + distance / 20) for distance in range(1, 11)],
        ),
        batch_size=4,
    )

    config = SQConfig(
        pitch_length_m=20.0,
        pitch_width_m=10.0,
        grid_resolution_m=5.0,
        frame_rate=25.0,
        movement_density_resolution_m=0.5,
        movement_density_extent_m=10.0,
        movement_kde_bandwidth_m=0.5,
        movement_horizons_seconds=(0.2, 0.4),
        movement_min_samples=2,
        batch_frames=4,
    )
    movement = fit_player_movement_model(tracking, config)
    horizon = PassHorizonModel.fit(passes)
    output = list(
        iter_player_influence(
            tracking,
            movement,
            config,
            horizon_model=horizon,
            frame_stride=4,
        )
    )

    assert output
    assert all("player_influence" in batch.columns for batch in output)
    assert all(
        batch.get_column("player_influence").is_finite().all() for batch in output
    )
