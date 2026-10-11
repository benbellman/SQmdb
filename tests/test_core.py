import numpy as np
import polars as pl
import pytest

from sqmdb import (
    PassHorizonModel,
    PitchValueModel,
    SQConfig,
    defensive_influence_samples,
    fit_player_movement_model,
    iter_defensive_influence_samples,
    iter_pitch_value,
    iter_player_influence,
    space_quality,
)
from sqmdb._data import tracking_batches


def sample_tracking(frame_count: int = 31) -> pl.DataFrame:
    rows = []
    for frame_id in range(frame_count):
        for player_id, team_id, x, y in (
            ("runner", "attack", frame_id * 0.1, 0.0),
            ("static", "defend", 2.0, 1.0),
        ):
            rows.append(
                {
                    "match_id": "match-1",
                    "frame_id": frame_id,
                    "player_id": player_id,
                    "team_id": team_id,
                    "x": x,
                    "y": y,
                    "ball_x": 1.0,
                    "ball_y": 0.0,
                    "possession_team_id": "attack",
                }
            )
    return pl.DataFrame(rows)


def small_config(**overrides: object) -> SQConfig:
    defaults: dict[str, object] = {
        "pitch_length_m": 20.0,
        "pitch_width_m": 10.0,
        "grid_resolution_m": 5.0,
        "frame_rate": 25.0,
        "velocity_lag_seconds": 0.2,
        "frame_stride": 1,
        "batch_frames": 4,
        "movement_density_resolution_m": 0.5,
        "movement_density_extent_m": 10.0,
        "movement_kde_bandwidth_m": 0.5,
        "movement_horizons_seconds": (0.2, 0.4),
        "movement_min_samples": 2,
    }
    defaults.update(overrides)
    return SQConfig(**defaults)


def test_grid_has_stable_indices_and_pitch_cell_centers() -> None:
    grid = small_config().grid()
    assert grid.height == 8
    assert grid.columns == ["grid_i", "grid_j", "x", "y"]
    assert grid.get_column("x").min() == -7.5
    assert grid.get_column("x").max() == 7.5
    assert grid.get_column("y").min() == -2.5
    assert grid.get_column("y").max() == 2.5


def test_pitch_dimensions_can_come_from_metadata_or_defaults() -> None:
    from_metadata = SQConfig.from_metadata(
        {"pitch_length_m": 110.0},
        grid_resolution_m=2.0,
    )
    defaults = SQConfig.from_metadata({})
    assert from_metadata.pitch_length_m == 110.0
    assert from_metadata.pitch_width_m == 68.0
    assert defaults.pitch_length_m == 105.0
    assert defaults.pitch_width_m == 68.0


def test_fit_and_stream_player_influence_with_frame_stride() -> None:
    tracking = sample_tracking()
    config = small_config()
    model = fit_player_movement_model(tracking, config)

    batches = list(
        iter_player_influence(
            tracking.lazy(),
            model,
            config,
            horizon_seconds=0.3,
            frame_stride=4,
        )
    )
    result = pl.concat(batches)
    assert result.get_column("frame_id").unique().sort().to_list() == list(range(0, 31, 4))
    assert result.height == 8 * 2 * 8
    assert result.get_column("player_influence").min() >= 0.0
    assert result.get_column("player_influence").max() <= 1.0
    assert all(batch.height <= config.batch_frames * 2 * config.grid().height for batch in batches)


def test_pitch_value_training_and_space_quality() -> None:
    tracking = sample_tracking()
    config = small_config(frame_stride=4)
    movement_model = fit_player_movement_model(tracking, config)
    surfaces = pl.concat(
        list(
            iter_player_influence(
                tracking,
                movement_model,
                config,
                horizon_seconds=0.2,
            )
        )
    )
    labels = defensive_influence_samples(surfaces, tracking)
    assert labels.height == surfaces.get_column("frame_id").n_unique() * config.grid().height
    assert labels.get_column("defensive_influence").min() >= 0.0
    assert labels.get_column("defensive_influence").max() <= 1.0

    label_batches = list(
        iter_defensive_influence_samples(
            (batch for batch in surfaces.partition_by("frame_id")),
            tracking,
        )
    )
    assert sum(batch.height for batch in label_batches) == labels.height

    model = PitchValueModel.fit(
        labels,
        hidden_layer_sizes=(8,),
        max_iter=300,
        max_training_samples=20,
    )
    pitch_batches = list(iter_pitch_value(tracking, model, config))
    values = pl.concat(pitch_batches)
    assert values.height == surfaces.get_column("frame_id").n_unique() * config.grid().height
    assert values.get_column("pitch_value").min() >= 0.0
    assert values.get_column("pitch_value").max() <= 1.0

    quality = space_quality(surfaces, values)
    assert quality.height == surfaces.get_column("frame_id").n_unique() * 2 * config.grid().height
    assert quality.get_column("space_quality").is_finite().all()
    assert np.allclose(
        quality.get_column("space_quality").to_numpy(),
        quality.get_column("team_pitch_control").to_numpy()
        * quality.get_column("pitch_value").to_numpy(),
    )


def test_rejects_duplicate_tracking_rows() -> None:
    tracking = sample_tracking()
    duplicate = pl.concat([tracking, tracking.head(1)])
    with pytest.raises(ValueError, match="unique"):
        fit_player_movement_model(duplicate, small_config())


def test_streaming_batches_keep_whole_frames_across_chunk_boundaries() -> None:
    rows = [
        {
            "match_id": "match-1",
            "frame_id": frame_id,
            "player_id": player_id,
            "team_id": str(player_id),
            "x": float(frame_id),
            "y": 0.0,
        }
        for frame_id in range(520)
        for player_id in range(2)
    ]
    batches = list(tracking_batches(pl.DataFrame(rows).lazy(), small_config(batch_frames=10)))
    assert sum(batch.height for batch in batches) == len(rows)
    assert all(batch.get_column("frame_id").n_unique() <= 10 for batch in batches)
    assert all(
        count == 2
        for batch in batches
        for count in batch.group_by("frame_id").len().get_column("len").to_list()
    )


def test_streaming_batches_reject_out_of_order_frames() -> None:
    tracking = sample_tracking(3).sort("frame_id", descending=True)
    with pytest.raises(ValueError, match="ordered by frame_id"):
        list(tracking_batches(tracking, small_config()))


def test_pitch_value_rejects_invalid_targets() -> None:
    samples = pl.DataFrame(
        {
            "ball_x": [0.0] * 10,
            "ball_y": [0.0] * 10,
            "grid_x": np.arange(10, dtype=float),
            "grid_y": [0.0] * 10,
            "defensive_influence": [1.5] * 10,
        }
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PitchValueModel.fit(samples, hidden_layer_sizes=(2,), max_iter=2)


def test_pass_horizon_model_fits_and_bounds_predictions() -> None:
    passes = pl.DataFrame(
        {
            "pass_distance_m": np.linspace(2.0, 40.0, 40),
            "pass_duration_seconds": np.linspace(0.2, 2.0, 40),
        }
    )
    model = PassHorizonModel.fit(passes)
    predictions = model.predict(np.array([1.0, 20.0, 50.0]))
    assert predictions.shape == (3,)
    assert predictions.min() >= 0.2
    assert predictions.max() <= 2.0

    config = small_config()
    movement = fit_player_movement_model(sample_tracking(), config)
    first_batch = next(
        iter_player_influence(
            sample_tracking(),
            movement,
            config,
            horizon_model=model,
        )
    )
    assert first_batch.height == (
        config.batch_frames * 2 * config.grid().height
    )
