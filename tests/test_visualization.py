import numpy as np
import polars as pl
import pytest

from sqmdb import SQConfig, add_plot_velocities, plot_pitch_surfaces
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
            "ball_vx": [-2.0, -2.0],
            "ball_vy": [1.25, 1.25],
        }
    )
    return (
        pl.concat(control_rows),
        pitch_value,
        pl.concat(quality_rows),
        tracking,
        config,
    )


def test_add_plot_velocities_uses_config_lag_and_isolates_matches():
    tracking = pl.DataFrame(
        {
            "match_id": ["a", "a", "b", "b"],
            "frame_id": [0, 2, 0, 2],
            "player_id": ["p1", "p1", "p1", "p1"],
            "team_id": ["home"] * 4,
            "x": [0.0, 2.0, 100.0, 104.0],
            "y": [0.0, 1.0, 0.0, 2.0],
            "ball_x": [0.0, 1.0, 10.0, 12.0],
            "ball_y": [0.0, 0.5, 0.0, 1.0],
        }
    )
    config = SQConfig(frame_rate=10.0, velocity_lag_seconds=0.2)

    velocities = add_plot_velocities(tracking, config).collect().sort(
        "match_id", "frame_id"
    )

    result = velocities.select(
        "match_id", "frame_id", "vx", "vy", "ball_vx", "ball_vy"
    )
    assert result.select("match_id", "frame_id").rows() == [
        ("a", 0),
        ("a", 2),
        ("b", 0),
        ("b", 2),
    ]
    np.testing.assert_allclose(
        result.select("vx", "vy", "ball_vx", "ball_vy").to_numpy(),
        [
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 5.0, 5.0, 2.5],
            [0.0, 0.0, 0.0, 0.0],
            [20.0, 10.0, 10.0, 5.0],
        ],
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
    assert prepared.ball_vx == -2.0
    assert prepared.ball_vy == 1.25
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


def test_missing_surface_frame_error_identifies_input_and_match():
    control, value, quality, tracking, config = visualization_data()
    control = control.filter(pl.col("frame_id") != 12)

    with pytest.raises(
        ValueError,
        match="No pitch-control surface rows found for frame_id=12, match_id='sample'",
    ):
        plot_pitch_surfaces(
            control,
            value,
            quality,
            tracking,
            12,
            match_id="sample",
            team_id="home",
            config=config,
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

    with pytest.raises(ValueError, match="Ball velocity.*constant"):
        inconsistent_ball_velocity = tracking.with_columns(
            pl.when(pl.col("player_id") == "p2")
            .then(pl.lit(1.0))
            .otherwise(pl.col("ball_vx"))
            .alias("ball_vx")
        )
        _prepare_pitch_data(
            control,
            value,
            quality,
            inconsistent_ball_velocity,
            12,
            match_id="sample",
            team_id="home",
            config=config,
            velocity_x="vx",
            velocity_y="vy",
        )


def test_render_is_repeatable_when_matplotlib_is_available():
    pytest.importorskip("matplotlib")
    pytest.importorskip("mplsoccer")
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
        pitch_length=config.pitch_length_m,
        pitch_width=config.pitch_width_m,
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
        pitch_length=config.pitch_length_m,
        pitch_width=config.pitch_width_m,
    )
    second.canvas.draw()
    second_pixels = np.asarray(second.canvas.buffer_rgba()).copy()
    assert len(first_axes) == len(second_axes) == 3
    np.testing.assert_array_equal(first_pixels, second_pixels)
    plt.close(first)
    plt.close(second)


def test_pitch_defaults_and_attacking_defending_colors():
    pytest.importorskip("matplotlib")
    pytest.importorskip("mplsoccer")
    import matplotlib
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.collections import PathCollection
    from matplotlib.colors import to_rgba

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control, value, quality, tracking, config = visualization_data()
    fig, axes = plot_pitch_surfaces(
        control,
        value,
        quality,
        tracking,
        12,
        match_id="sample",
        team_id="home",
        config=config,
        attacking_team_color="#123456",
        defending_team_color="#654321",
    )

    assert len(axes) == 3
    assert axes[0].get_xlim() == (-52.5, 52.5)
    assert axes[0].get_ylim() == (-34.5, 34.5)
    player_and_ball = [
        collection
        for collection in axes[0].collections
        if isinstance(collection, PathCollection)
    ]
    assert len(player_and_ball) == 2
    assert player_and_ball[0].get_facecolors()[0] == pytest.approx(to_rgba("#654321"))
    assert player_and_ball[1].get_facecolors()[0] == pytest.approx(to_rgba("#123456"))
    ball_icons = [
        artist for artist in axes[0].artists if isinstance(artist, AnnotationBbox)
    ]
    assert len(ball_icons) == 1
    assert ball_icons[0].xy == (0.0, 0.5)
    ball_image = ball_icons[0].offsetbox
    assert isinstance(ball_image, OffsetImage)
    assert ball_image.get_data().shape == (32, 32, 4)
    assert ball_image.get_zoom() == pytest.approx(0.27)
    control_mesh = next(
        collection
        for collection in axes[0].collections
        if collection is not player_and_ball[0]
        and collection is not player_and_ball[1]
        and isinstance(collection, matplotlib.collections.QuadMesh)
    )
    assert control_mesh.cmap(0.0) == pytest.approx(to_rgba("#654321"))
    assert control_mesh.cmap(1.0) == pytest.approx(to_rgba("#123456"))
    assert control_mesh.cmap(0.5) == pytest.approx(to_rgba("#f7f7f7"))
    value_mesh = next(
        collection
        for collection in axes[1].collections
        if isinstance(collection, matplotlib.collections.QuadMesh)
    )
    for position, color in (
        (0.0, "#f6eff7"),
        (0.25, "#bdc9e1"),
        (0.5, "#67a9cf"),
        (0.75, "#1c9099"),
        (1.0, "#016c59"),
    ):
        assert value_mesh.cmap(position) == pytest.approx(to_rgba(color))
    assert all(line.get_color() == "black" for line in axes[1].lines)
    assert not any(label.get_visible() for label in axes[0].get_xticklabels())
    assert not any(label.get_visible() for label in axes[0].get_yticklabels())
    plt.close(fig)


def test_default_team_colors_and_velocity_arrow_scale():
    pytest.importorskip("matplotlib")
    pytest.importorskip("mplsoccer")
    import matplotlib
    from matplotlib.collections import PathCollection
    from matplotlib.colors import to_rgba
    from matplotlib.quiver import Quiver

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control, value, quality, tracking, config = visualization_data()
    fig, axes = plot_pitch_surfaces(
        control,
        value,
        quality,
        tracking,
        12,
        match_id="sample",
        team_id="home",
        config=config,
        pitch_length=config.pitch_length_m,
        pitch_width=config.pitch_width_m,
    )
    fig.canvas.draw()
    player_dots = [
        collection
        for collection in axes[0].collections
        if isinstance(collection, PathCollection)
    ]
    assert player_dots[0].get_facecolors()[0] == pytest.approx(to_rgba("#0072B2"))
    assert player_dots[1].get_facecolors()[0] == pytest.approx(to_rgba("#D55E00"))
    arrows = [collection for collection in axes[0].collections if isinstance(collection, Quiver)]
    fig.canvas.draw()
    np.testing.assert_allclose(arrows[0].U, [-1.5])
    np.testing.assert_allclose(arrows[1].U, [3.0])
    np.testing.assert_allclose(arrows[2].U, [-3.0])
    np.testing.assert_allclose(arrows[2].V, [1.875])
    assert arrows[2].get_facecolors()[0] == pytest.approx(to_rgba("black"))

    legend = fig.legends[0]
    fig.canvas.draw()
    legend_bottom = legend.get_window_extent().transformed(
        fig.transFigure.inverted()
    ).y0
    title_bottom = fig._suptitle.get_window_extent().transformed(
        fig.transFigure.inverted()
    ).y0
    legend_top = legend.get_window_extent().transformed(
        fig.transFigure.inverted()
    ).y1
    assert legend_bottom < title_bottom
    assert title_bottom > legend_top
    assert title_bottom - legend_top < 0.025
    assert legend_bottom > axes[0].get_position().y1
    assert legend_bottom - axes[0].get_position().y1 < 0.05
    assert [text.get_text() for text in legend.get_texts()] == ["away", "home"]
    legend_left = legend.get_window_extent().transformed(
        fig.transFigure.inverted()
    ).x0
    panel_left = axes[0].get_position().x0
    assert legend_left == pytest.approx(panel_left)
    assert fig._suptitle.get_ha() == "left"
    assert fig._suptitle.get_position()[0] == pytest.approx(panel_left)
    assert fig._suptitle.get_text() == "Frame 12"
    caption = next(
        text for text in fig.texts if text.get_text().startswith("The possession team")
    )
    assert " ".join(caption.get_text().split()) == (
        "The possession team represents positive values and is attacking "
        "left-to-right. Velocity arrows project locations in 1.5 seconds."
    )
    assert caption.get_fontsize() == pytest.approx(10)
    assert caption.get_position() == (panel_left, 0.005)
    assert caption.get_ha() == "left"
    assert caption.get_transform() == fig.transFigure
    caption_bbox = caption.get_window_extent().transformed(fig.transFigure.inverted())
    assert "\n" in caption.get_text()
    assert caption_bbox.x0 == pytest.approx(panel_left)
    assert caption_bbox.x1 <= axes[0].get_position().x1
    caption_bottom = caption_bbox.y1
    axis_bottom = axes[-1].get_position().y0
    assert caption_bottom < axis_bottom
    assert axis_bottom - caption_bottom < 0.02
    plt.close(fig)


def test_plot_title_can_be_overridden():
    pytest.importorskip("matplotlib")
    pytest.importorskip("mplsoccer")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control, value, quality, tracking, config = visualization_data()
    fig, _ = plot_pitch_surfaces(
        control,
        value,
        quality,
        tracking,
        12,
        title="Match overview",
        match_id="sample",
        team_id="home",
        config=config,
        pitch_length=config.pitch_length_m,
        pitch_width=config.pitch_width_m,
    )

    assert fig._suptitle.get_text() == "Match overview"
    plt.close(fig)


def test_missing_ball_velocity_warns_and_omits_only_ball_arrow():
    pytest.importorskip("matplotlib")
    pytest.importorskip("mplsoccer")
    import matplotlib
    from matplotlib.offsetbox import AnnotationBbox
    from matplotlib.quiver import Quiver

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control, value, quality, tracking, config = visualization_data()
    tracking = tracking.drop("ball_vx", "ball_vy")
    with pytest.warns(
        UserWarning, match="ball_vx, ball_vy.*add_plot_velocities"
    ):
        fig, axes = plot_pitch_surfaces(
            control,
            value,
            quality,
            tracking,
            12,
            match_id="sample",
            team_id="home",
            config=config,
            pitch_length=config.pitch_length_m,
            pitch_width=config.pitch_width_m,
        )

    fig.canvas.draw()
    arrows = [collection for collection in axes[0].collections if isinstance(collection, Quiver)]
    assert len(arrows) == 2
    assert len([artist for artist in axes[0].artists if isinstance(artist, AnnotationBbox)]) == 1
    plt.close(fig)


def test_missing_player_velocity_warns_and_omits_only_player_arrows():
    pytest.importorskip("matplotlib")
    pytest.importorskip("mplsoccer")
    import matplotlib
    from matplotlib.quiver import Quiver

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control, value, quality, tracking, config = visualization_data()
    tracking = tracking.drop("vx", "vy")
    with pytest.warns(UserWarning, match="vx, vy.*add_plot_velocities"):
        fig, axes = plot_pitch_surfaces(
            control,
            value,
            quality,
            tracking,
            12,
            match_id="sample",
            team_id="home",
            config=config,
            pitch_length=config.pitch_length_m,
            pitch_width=config.pitch_width_m,
        )

    fig.canvas.draw()
    arrows = [
        collection
        for collection in axes[0].collections
        if isinstance(collection, Quiver)
    ]
    assert len(arrows) == 1
    assert arrows[0].get_facecolors()[0] == pytest.approx(
        matplotlib.colors.to_rgba("black")
    )
    plt.close(fig)


def test_velocity_helper_output_plots_without_missing_velocity_warning():
    pytest.importorskip("matplotlib")
    pytest.importorskip("mplsoccer")
    import matplotlib
    import warnings
    from matplotlib.quiver import Quiver

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control, value, quality, tracking, config = visualization_data()
    tracking = tracking.drop("vx", "vy", "ball_vx", "ball_vy")
    previous_frame = tracking.with_columns(
        pl.lit(7).cast(pl.Int64).alias("frame_id"),
        (pl.col("x") - 1.0).alias("x"),
        (pl.col("y") - 1.0).alias("y"),
        (pl.col("ball_x") - 1.0).alias("ball_x"),
        (pl.col("ball_y") - 1.0).alias("ball_y"),
    )
    tracking_with_velocities = add_plot_velocities(
        pl.concat([previous_frame, tracking]),
        config,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, axes = plot_pitch_surfaces(
            control,
            value,
            quality,
            tracking_with_velocities,
            12,
            match_id="sample",
            team_id="home",
            config=config,
            pitch_length=config.pitch_length_m,
            pitch_width=config.pitch_width_m,
        )

    assert not any("Velocity column(s) not found" in str(item.message) for item in caught)
    arrows = [
        collection
        for collection in axes[0].collections
        if isinstance(collection, Quiver)
    ]
    assert len(arrows) == 3
    np.testing.assert_allclose([arrow.U[0] for arrow in arrows], [7.5, 7.5, 7.5])
    np.testing.assert_allclose([arrow.V[0] for arrow in arrows], [7.5, 7.5, 7.5])
    plt.close(fig)


def test_plot_title_legend_and_symbology_fonts_are_two_points_larger():
    pytest.importorskip("matplotlib")
    pytest.importorskip("mplsoccer")
    import matplotlib
    from matplotlib.font_manager import FontProperties

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control, value, quality, tracking, config = visualization_data()
    fig, axes = plot_pitch_surfaces(
        control,
        value,
        quality,
        tracking,
        12,
        match_id="sample",
        team_id="home",
        config=config,
        velocity_arrow_seconds=2.25,
        pitch_length=config.pitch_length_m,
        pitch_width=config.pitch_width_m,
    )

    axes_title_size = (
        FontProperties(size=plt.rcParams["axes.titlesize"]).get_size_in_points()
    )
    figure_title_size = (
        FontProperties(size=plt.rcParams["figure.titlesize"]).get_size_in_points()
    )
    legend_font_size = (
        FontProperties(size=plt.rcParams["legend.fontsize"]).get_size_in_points()
    )
    assert axes[0]._left_title.get_fontsize() == pytest.approx(axes_title_size + 2)
    assert fig._suptitle.get_fontsize() == pytest.approx(figure_title_size + 5)
    assert fig.legends[0].get_texts()[0].get_fontsize() == pytest.approx(
        legend_font_size + 2
    )
    assert fig.legends[0].legend_handles[0].get_markersize() == pytest.approx(8)
    assert " ".join(
        next(
            text.get_text()
            for text in fig.texts
            if text.get_text().startswith("The possession team")
        ).split()
    ).endswith("in 2.25 seconds.")
    plt.close(fig)


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
