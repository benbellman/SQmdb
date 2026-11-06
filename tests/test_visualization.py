import numpy as np
import polars as pl
import pytest

from sqmdb import SQConfig, plot_pitch_surfaces
from sqmdb.visualization import _prepare_pitch_data


def visualization_data():
    config = SQConfig(pitch_length_m=20.0, pitch_width_m=10.0, grid_resolution_m=5.0)
    grid = config.grid()
    indices = grid.select("grid_i", "grid_j", "x", "y")
    pitch_value = indices.with_columns(
        pl.Series("match_id", ["sample"] * grid.height),
        pl.Series("frame_id", [12] * grid.height),
        pl.Series("pitch_value", np.linspace(0.1, 0.8, grid.height)),
    )
    control_rows = []
    quality_rows = []
    for team_id, control in (("home", 0.5), ("away", -0.5)):
        control_rows.append(
            indices.with_columns(
                pl.Series("match_id", ["sample"] * grid.height),
                pl.Series("frame_id", [12] * grid.height),
                pl.Series("team_id", [team_id] * grid.height),
                pl.Series("team_pitch_control", [control] * grid.height),
            )
        )
        quality_rows.append(
            indices.with_columns(
                pl.Series("match_id", ["sample"] * grid.height),
                pl.Series("frame_id", [12] * grid.height),
                pl.Series("team_id", [team_id] * grid.height),
                pl.Series("space_quality", pitch_value["pitch_value"] * control),
            )
        )
    tracking = pl.DataFrame(
        {
            "match_id": ["sample", "sample"],
            "frame_id": [12, 12],
            "player_id": ["p1", "p2"],
            "team_id": ["home", "away"],
            "x": [-3.0, 4.0],
            "y": [-1.0, 2.0],
            "vx": [2.0, -1.0],
            "vy": [1.0, 0.5],
            "ball_x": [0.0, 0.0],
            "ball_y": [0.5, 0.5],
        }
    )
    return (
        pl.concat(control_rows),
        pitch_value,
        pl.concat(quality_rows),
        tracking,
        config,
    )


def test_surface_arrays_follow_grid_i_x_and_grid_j_y_orientation():
    control, value, quality, tracking, config = visualization_data()
    prepared = _prepare_pitch_data(
        control,
        value,
        quality,
        tracking,
        12,
        match_id="sample",
        team_id="home",
        config=config,
        velocity_x="vx",
        velocity_y="vy",
    )

    assert prepared.x.tolist() == [-7.5, -2.5, 2.5, 7.5]
    assert prepared.y.tolist() == [-2.5, 2.5]
    assert prepared.pitch_value.shape == (2, 4)
    np.testing.assert_allclose(
        prepared.pitch_value,
        np.array([[0.1, 0.3, 0.5, 0.7], [0.2, 0.4, 0.6, 0.8]]),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        prepared.space_quality,
        prepared.pitch_control * prepared.pitch_value,
    )
    assert prepared.ball_x == 0.0
    assert prepared.ball_y == 0.5
    home_player = prepared.players.filter(pl.col("player_id") == "p1")
    assert home_player.select("x", "y", "vx", "vy").row(0) == (-3.0, -1.0, 2.0, 1.0)


def test_rejects_misaligned_surface_coordinates():
    control, value, quality, tracking, config = visualization_data()
    value = value.with_columns(
        pl.when(pl.col("grid_i") == 0)
        .then(pl.col("y") + 1.0)
        .otherwise(pl.col("y"))
        .alias("y")
    )

    with pytest.raises(ValueError, match="grid coordinates"):
        _prepare_pitch_data(
            control,
            value,
            quality,
            tracking,
            12,
            match_id="sample",
            team_id="home",
            config=config,
            velocity_x="vx",
            velocity_y="vy",
        )


def test_rejects_surfaces_from_different_matches():
    control, value, quality, tracking, config = visualization_data()
    value = value.with_columns(pl.lit("other").alias("match_id"))

    with pytest.raises(ValueError, match="same match"):
        _prepare_pitch_data(
            control,
            value,
            quality,
            tracking,
            12,
            match_id=None,
            team_id="home",
            config=config,
            velocity_x="vx",
            velocity_y="vy",
        )


def test_validates_selected_frame_and_overlay_values():
    control, value, quality, tracking, config = visualization_data()
    with pytest.raises(ValueError, match="team_id must be specified"):
        _prepare_pitch_data(
            control,
            value,
            quality,
            tracking,
            12,
            match_id="sample",
            team_id=None,
            config=config,
            velocity_x="vx",
            velocity_y="vy",
        )

    with pytest.raises(ValueError, match="constant"):
        inconsistent_ball = tracking.with_columns(
            pl.when(pl.col("player_id") == "p2")
            .then(pl.lit(1.0))
            .otherwise(pl.col("ball_x"))
            .alias("ball_x")
        )
        _prepare_pitch_data(
            control,
            value,
            quality,
            inconsistent_ball,
            12,
            match_id="sample",
            team_id="home",
            config=config,
            velocity_x="vx",
            velocity_y="vy",
        )


def test_render_is_repeatable_when_matplotlib_is_available():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control, value, quality, tracking, config = visualization_data()
    first, first_axes = plot_pitch_surfaces(
        control,
        value,
        quality,
        tracking,
        12,
        match_id="sample",
        team_id="home",
        config=config,
    )
    first.canvas.draw()
    first_pixels = np.asarray(first.canvas.buffer_rgba()).copy()
    second, second_axes = plot_pitch_surfaces(
        control,
        value,
        quality,
        tracking,
        12,
        match_id="sample",
        team_id="home",
        config=config,
    )
    second.canvas.draw()
    second_pixels = np.asarray(second.canvas.buffer_rgba()).copy()
    assert len(first_axes) == len(second_axes) == 3
    np.testing.assert_array_equal(first_pixels, second_pixels)
    plt.close(first)
    plt.close(second)


def test_rejects_invalid_velocity_arrow_duration():
    control, value, quality, tracking, config = visualization_data()
    with pytest.raises(ValueError, match="velocity_arrow_seconds"):
        plot_pitch_surfaces(
            control,
            value,
            quality,
            tracking,
            12,
            match_id="sample",
            team_id="home",
            config=config,
            velocity_arrow_seconds=0,
        )
