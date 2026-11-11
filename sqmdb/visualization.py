from __future__ import annotations

import warnings
from dataclasses import dataclass
from importlib.resources import as_file, files
from typing import Any

import numpy as np
import polars as pl

from ._data import as_lazy
from .config import SQConfig


@dataclass(frozen=True, slots=True)
class _PreparedPitchData:
    x: np.ndarray
    y: np.ndarray
    pitch_control: np.ndarray
    pitch_value: np.ndarray
    space_quality: np.ndarray
    players: pl.DataFrame
    ball_x: float
    ball_y: float
    ball_vx: float | None
    ball_vy: float | None
    player_velocity_x: str | None
    player_velocity_y: str | None
    teams: tuple[Any, ...]
    attacking_team: Any


def _frame_rows(
    data: pl.DataFrame | pl.LazyFrame,
    frame_id: int,
    match_id: Any | None,
    required: set[str],
    data_label: str,
) -> pl.DataFrame:
    lazy = as_lazy(data, required | {"frame_id"})
    selected = lazy.filter(pl.col("frame_id") == frame_id)
    if match_id is not None:
        selected = selected.filter(pl.col("match_id") == match_id)
    rows = selected.collect(engine="streaming")
    if rows.is_empty():
        identifiers = [f"frame_id={frame_id!r}"]
        if match_id is not None:
            identifiers.append(f"match_id={match_id!r}")
        raise ValueError(
            f"No {data_label} rows found for {', '.join(identifiers)}."
        )
    if match_id is None and rows.get_column("match_id").n_unique() != 1:
        raise ValueError("match_id must be specified when a frame exists in multiple matches.")
    return rows


def _surface_grid(
    rows: pl.DataFrame,
    value_column: str,
    config: SQConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numeric = rows.with_columns(
        pl.col("grid_i").cast(pl.Int64, strict=True),
        pl.col("grid_j").cast(pl.Int64, strict=True),
        pl.col("x").cast(pl.Float64, strict=True),
        pl.col("y").cast(pl.Float64, strict=True),
        pl.col(value_column).cast(pl.Float64, strict=True),
    ).sort("grid_i", "grid_j")
    if numeric.select(
        pl.any_horizontal(
            pl.col("grid_i").is_null(),
            pl.col("grid_j").is_null(),
            pl.col("x").is_null(),
            pl.col("y").is_null(),
            pl.col(value_column).is_null(),
        ).any()
    ).item():
        raise ValueError(f"{value_column} surface values and grid coordinates cannot be null.")

    grid_i = numeric.get_column("grid_i").to_numpy()
    grid_j = numeric.get_column("grid_j").to_numpy()
    if len(set(zip(grid_i.tolist(), grid_j.tolist()))) != numeric.height:
        raise ValueError(f"{value_column} surface contains duplicate grid cells.")
    i_values = np.unique(grid_i)
    j_values = np.unique(grid_j)
    if (
        not np.array_equal(i_values, np.arange(len(i_values)))
        or not np.array_equal(j_values, np.arange(len(j_values)))
        or numeric.height != len(i_values) * len(j_values)
    ):
        raise ValueError(f"{value_column} surface must contain a complete zero-based grid.")

    x = numeric.get_column("x").to_numpy().reshape(len(i_values), len(j_values))[:, 0]
    y = numeric.get_column("y").to_numpy().reshape(len(i_values), len(j_values))[0, :]
    expected_x = np.linspace(
        -config.pitch_length_m / 2 + config.pitch_length_m / (2 * len(i_values)),
        config.pitch_length_m / 2 - config.pitch_length_m / (2 * len(i_values)),
        len(i_values),
    )
    expected_y = np.linspace(
        -config.pitch_width_m / 2 + config.pitch_width_m / (2 * len(j_values)),
        config.pitch_width_m / 2 - config.pitch_width_m / (2 * len(j_values)),
        len(j_values),
    )
    all_x = numeric.get_column("x").to_numpy()
    all_y = numeric.get_column("y").to_numpy()
    if (
        not np.all(np.isfinite(all_x))
        or not np.all(np.isfinite(all_y))
        or not np.allclose(x, expected_x)
        or not np.allclose(y, expected_y)
        or not np.allclose(all_x, np.repeat(x, len(j_values)))
        or not np.allclose(all_y, np.tile(y, len(i_values)))
    ):
        raise ValueError(
            f"{value_column} grid coordinates do not match the configured SQmdb grid."
        )
    values = numeric.get_column(value_column).to_numpy()
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{value_column} surface must contain only finite values.")
    return x, y, values.reshape(len(i_values), len(j_values)).T


def _prepare_pitch_data(
    pitch_control: pl.DataFrame | pl.LazyFrame,
    pitch_value: pl.DataFrame | pl.LazyFrame,
    space_quality: pl.DataFrame | pl.LazyFrame,
    tracking: pl.DataFrame | pl.LazyFrame,
    frame_id: int,
    *,
    match_id: Any | None,
    team_id: Any | None,
    config: SQConfig,
    velocity_x: str | None = "vx",
    velocity_y: str | None = "vy",
    ball_velocity_x: str | None = "ball_vx",
    ball_velocity_y: str | None = "ball_vy",
) -> _PreparedPitchData:
    control_rows = _frame_rows(
        pitch_control,
        frame_id,
        match_id,
        {"team_id", "grid_i", "grid_j", "x", "y", "team_pitch_control"},
        "pitch-control surface",
    )
    if team_id is None:
        teams = control_rows.get_column("team_id").unique().to_list()
        if len(teams) != 1:
            raise ValueError("team_id must be specified when pitch control has multiple teams.")
        team_id = teams[0]
    control_rows = control_rows.filter(pl.col("team_id") == team_id)
    if control_rows.is_empty():
        raise ValueError(f"No pitch-control surface found for team_id={team_id!r}.")

    value_rows = _frame_rows(
        pitch_value,
        frame_id,
        match_id,
        {"grid_i", "grid_j", "x", "y", "pitch_value"},
        "pitch-value surface",
    )
    quality_rows = _frame_rows(
        space_quality,
        frame_id,
        match_id,
        {"team_id", "grid_i", "grid_j", "x", "y", "space_quality"},
        "space-quality surface",
    ).filter(pl.col("team_id") == team_id)
    if quality_rows.is_empty():
        raise ValueError(f"No space-quality surface found for team_id={team_id!r}.")

    grids = [
        _surface_grid(rows, field, config)
        for rows, field in (
            (control_rows, "team_pitch_control"),
            (value_rows, "pitch_value"),
            (quality_rows, "space_quality"),
        )
    ]
    x, y, control = grids[0]
    for other_x, other_y, _ in grids[1:]:
        if not np.array_equal(x, other_x) or not np.array_equal(y, other_y):
            raise ValueError("All three surfaces must use identical grid coordinates.")
    if np.any((control < -1.0) | (control > 1.0)):
        raise ValueError("team_pitch_control values must be in [-1, 1].")
    value = grids[1][2]
    quality = grids[2][2]
    if np.any((value < 0.0) | (value > 1.0)):
        raise ValueError("pitch_value values must be in [0, 1].")
    if not np.allclose(quality, control * value, rtol=1e-5, atol=1e-7):
        raise ValueError("space_quality must equal team_pitch_control multiplied by pitch_value.")

    player_velocity_columns = (
        (velocity_x, velocity_y)
        if velocity_x is not None and velocity_y is not None
        else ()
    )
    ball_velocity_columns = (
        (ball_velocity_x, ball_velocity_y)
        if ball_velocity_x is not None and ball_velocity_y is not None
        else ()
    )
    tracking_columns = (
        "x",
        "y",
        "ball_x",
        "ball_y",
        *player_velocity_columns,
        *ball_velocity_columns,
    )
    players = _frame_rows(
        tracking,
        frame_id,
        match_id,
        {"player_id", "team_id", *tracking_columns},
        "tracking",
    ).with_columns(
        *[
            pl.col(column).cast(pl.Float64, strict=True)
            for column in tracking_columns
        ]
    )
    match_ids = [
        rows.get_column("match_id").unique().to_list()
        for rows in (control_rows, value_rows, quality_rows, players)
    ]
    if any(ids != match_ids[0] for ids in match_ids[1:]):
        raise ValueError("All surfaces and tracking rows must refer to the same match.")
    if players.select(
        pl.any_horizontal(
            pl.col("player_id").is_null(),
            pl.col("team_id").is_null(),
            *[
                pl.col(column).is_null()
                for column in tracking_columns
            ],
        ).any()
    ).item():
        raise ValueError("Player, ball, and velocity values cannot be null.")
    if players.select(pl.struct("player_id").n_unique()).item() != players.height:
        raise ValueError("Tracking rows must have one row per player in the selected frame.")
    player_values = players.select(*tracking_columns).to_numpy()
    if not np.all(np.isfinite(player_values)):
        raise ValueError("Player, ball, and velocity values must be finite.")
    ball = players.select("ball_x", "ball_y").unique()
    if ball.height != 1:
        raise ValueError("ball_x and ball_y must be constant within the selected frame.")
    ball_vx = None
    ball_vy = None
    if ball_velocity_columns:
        ball_velocity = players.select(*ball_velocity_columns).unique()
        if ball_velocity.height != 1:
            raise ValueError("Ball velocity must be constant within the selected frame.")
        ball_vx = float(ball_velocity.get_column(ball_velocity_x)[0])
        ball_vy = float(ball_velocity.get_column(ball_velocity_y)[0])

    selected_teams = tuple(
        sorted(players.get_column("team_id").unique().to_list(), key=str)
    )
    return _PreparedPitchData(
        x=x,
        y=y,
        pitch_control=control,
        pitch_value=value,
        space_quality=quality,
        players=players,
        ball_x=float(ball.get_column("ball_x")[0]),
        ball_y=float(ball.get_column("ball_y")[0]),
        ball_vx=ball_vx,
        ball_vy=ball_vy,
        player_velocity_x=velocity_x if player_velocity_columns else None,
        player_velocity_y=velocity_y if player_velocity_columns else None,
        teams=selected_teams,
        attacking_team=team_id,
    )


def add_plot_velocities(
    tracking: pl.DataFrame | pl.LazyFrame,
    config: SQConfig = SQConfig(),
) -> pl.LazyFrame:
    """Add player and ball velocity components for pitch visualization.

    Velocities are finite differences over the nearest whole-frame lag to
    ``config.velocity_lag_seconds``. Frames without a matching lagged position
    (such as the beginning of a match) receive zero velocity. The returned
    value is lazy.
    """
    rows = as_lazy(
        tracking,
        {"frame_id", "player_id", "x", "y", "ball_x", "ball_y"},
    )
    lag_frames = config.movement_lag_frames
    lag_seconds = lag_frames / config.frame_rate

    past_player_positions = rows.select(
        "match_id",
        "player_id",
        (pl.col("frame_id") + lag_frames).alias("frame_id"),
        pl.col("x").alias("_plot_past_x"),
        pl.col("y").alias("_plot_past_y"),
    )
    ball_positions = rows.group_by("match_id", "frame_id").agg(
        pl.col("ball_x").first().alias("_plot_ball_x"),
        pl.col("ball_y").first().alias("_plot_ball_y"),
    )
    past_ball_positions = ball_positions.select(
        "match_id",
        (pl.col("frame_id") + lag_frames).alias("frame_id"),
        pl.col("_plot_ball_x").alias("_plot_past_ball_x"),
        pl.col("_plot_ball_y").alias("_plot_past_ball_y"),
    )
    return (
        rows.join(
            past_player_positions,
            on=["match_id", "player_id", "frame_id"],
            how="left",
        )
        .join(past_ball_positions, on=["match_id", "frame_id"], how="left")
        .with_columns(
            (
                (pl.col("x") - pl.col("_plot_past_x")) / lag_seconds
            ).fill_null(0.0).alias("vx"),
            (
                (pl.col("y") - pl.col("_plot_past_y")) / lag_seconds
            ).fill_null(0.0).alias("vy"),
            (
                (pl.col("ball_x") - pl.col("_plot_past_ball_x")) / lag_seconds
            ).fill_null(0.0).alias("ball_vx"),
            (
                (pl.col("ball_y") - pl.col("_plot_past_ball_y")) / lag_seconds
            ).fill_null(0.0).alias("ball_vy"),
        )
        .drop(
            "_plot_past_x",
            "_plot_past_y",
            "_plot_past_ball_x",
            "_plot_past_ball_y",
        )
    )


def plot_pitch_surfaces(
    pitch_control: pl.DataFrame | pl.LazyFrame,
    pitch_value: pl.DataFrame | pl.LazyFrame,
    space_quality: pl.DataFrame | pl.LazyFrame,
    tracking: pl.DataFrame | pl.LazyFrame,
    frame_id: int,
    *,
    match_id: Any | None = None,
    team_id: Any | None = None,
    config: SQConfig = SQConfig(),
    velocity_x: str = "vx",
    velocity_y: str = "vy",
    ball_velocity_x: str = "ball_vx",
    ball_velocity_y: str = "ball_vy",
    velocity_arrow_seconds: float = 1.5,
    figsize: tuple[float, float] = (10.0, 15.0),
    attacking_team_color: Any = "#D55E00",
    defending_team_color: Any = "#0072B2",
    pitch_type: str = "secondspectrum",
    pitch_length: float = 105.0,
    pitch_width: float = 69.0,
    **pitch_kwargs: Any,
) -> tuple[Any, np.ndarray]:
    """Plot three aligned team surfaces with the selected frame's tracking.

    Tracking must contain one row per player with positions in metres and the
    ball position repeated consistently across the frame. Player velocity
    components (``vx``/``vy`` by default) and ball velocity components
    (``ball_vx``/``ball_vy`` by default) can be added with
    :func:`add_plot_velocities`. The surfaces use SQmdb's zero-based
    ``grid_i``/``grid_j`` cell-center coordinates. All three surfaces and the
    tracking input must contain the requested match and frame.

    Matplotlib and mplsoccer are optional dependencies; install
    ``sqmdb[visualization]`` to render charts. Pitch construction options can
    be passed as keyword arguments and are forwarded to ``mplsoccer.Pitch``.
    ``team_id`` is treated as the attacking team. The signed surface scale
    uses ``defending_team_color`` at -1, a neutral midpoint at 0, and
    ``attacking_team_color`` at +1; player markers use the corresponding team
    colors. Player and available ball velocity arrows are drawn 1.5 seconds
    ahead by default. The ball arrow is black. Missing player or ball velocity
    columns produce a warning and omit the respective arrows. The returned
    axes are ordered pitch control, pitch value, and space quality from top to
    bottom.
    """
    if not np.isfinite(velocity_arrow_seconds) or velocity_arrow_seconds <= 0:
        raise ValueError("velocity_arrow_seconds must be a positive finite number.")
    if len(figsize) != 2 or not all(np.isfinite(value) and value > 0 for value in figsize):
        raise ValueError("figsize must contain two positive finite numbers.")
    tracking_data = as_lazy(tracking)
    tracking_columns = set(tracking_data.collect_schema().names())
    missing_player_velocity = [
        column
        for column in (velocity_x, velocity_y)
        if column not in tracking_columns
    ]
    missing_ball_velocity = [
        column
        for column in (ball_velocity_x, ball_velocity_y)
        if column not in tracking_columns
    ]
    prepared_velocity_x: str | None = velocity_x
    prepared_velocity_y: str | None = velocity_y
    prepared_ball_velocity_x: str | None = ball_velocity_x
    prepared_ball_velocity_y: str | None = ball_velocity_y
    if missing_player_velocity or missing_ball_velocity:
        missing_columns = [*missing_player_velocity, *missing_ball_velocity]
        warnings.warn(
            "Velocity column(s) not found: "
            f"{', '.join(missing_columns)}. The plot will omit the corresponding "
            "velocity arrow(s). Add them with "
            "`add_plot_velocities(tracking, config)` or provide source velocity "
            "components.",
            UserWarning,
            stacklevel=2,
        )
        if missing_player_velocity:
            prepared_velocity_x = None
            prepared_velocity_y = None
        if missing_ball_velocity:
            prepared_ball_velocity_x = None
            prepared_ball_velocity_y = None
    prepared = _prepare_pitch_data(
        pitch_control,
        pitch_value,
        space_quality,
        tracking_data,
        frame_id,
        match_id=match_id,
        team_id=team_id,
        config=config,
        velocity_x=prepared_velocity_x,
        velocity_y=prepared_velocity_y,
        ball_velocity_x=prepared_ball_velocity_x,
        ball_velocity_y=prepared_ball_velocity_y,
    )
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.font_manager import FontProperties
        from matplotlib.lines import Line2D
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
        from mplsoccer import Pitch
    except ImportError as error:
        raise ImportError(
            "Pitch visualization requires Matplotlib and mplsoccer; "
            "install sqmdb[visualization]."
        ) from error
    with as_file(files("sqmdb").joinpath("assets", "soccer_ball.png")) as ball_icon_path:
        ball_icon = plt.imread(ball_icon_path)

    def cell_edges(centers: np.ndarray, extent: float) -> np.ndarray:
        if len(centers) == 1:
            return np.array([-extent / 2, extent / 2])
        midpoints = (centers[:-1] + centers[1:]) / 2
        return np.concatenate(
            (
                [centers[0] - (midpoints[0] - centers[0])],
                midpoints,
                [centers[-1] + (centers[-1] - midpoints[-1])],
            )
        )

    x_edges = cell_edges(prepared.x, pitch_length)
    y_edges = cell_edges(prepared.y, pitch_width)
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)
    panels = (
        ("Pitch control", prepared.pitch_control, None, -1.0, 1.0),
        ("Pitch value", prepared.pitch_value, "sqmdb_pitch_value", 0.0, 1.0),
        ("Space quality", prepared.space_quality, None, -1.0, 1.0),
    )
    team_colors = {
        team: (
            attacking_team_color
            if team == prepared.attacking_team
            else defending_team_color
        )
        for team in prepared.teams
    }
    control_cmap = LinearSegmentedColormap.from_list(
        "sqmdb_team_control",
        [
            (0.0, defending_team_color),
            (0.5, "#f7f7f7"),
            (1.0, attacking_team_color),
        ],
        N=257,
    )
    pitch_value_cmap = LinearSegmentedColormap.from_list(
        "sqmdb_pitch_value",
        [
            (0.0, "#f6eff7"),
            (0.25, "#bdc9e1"),
            (0.5, "#67a9cf"),
            (0.75, "#1c9099"),
            (1.0, "#016c59"),
        ],
        N=257,
    )
    pitch_options: dict[str, Any] = {
        "pitch_type": pitch_type,
        "pitch_length": pitch_length,
        "pitch_width": pitch_width,
        "line_color": "black",
        "line_zorder": 2,
        "axis": False,
        "label": False,
        "pad_left": 0,
        "pad_right": 0,
        "pad_bottom": 0,
        "pad_top": 0,
    }
    pitch_options.update(pitch_kwargs)
    pitch = Pitch(**pitch_options)
    panel_title_size = (
        FontProperties(size=plt.rcParams["axes.titlesize"]).get_size_in_points() + 2
    )
    main_title_size = (
        FontProperties(size=plt.rcParams["figure.titlesize"]).get_size_in_points() + 5
    )
    legend_font_size = (
        FontProperties(size=plt.rcParams["legend.fontsize"]).get_size_in_points() + 2
    )
    for ax, (title, values, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            cmap=(
                control_cmap
                if cmap is None
                else pitch_value_cmap
                if cmap == "sqmdb_pitch_value"
                else cmap
            ),
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )
        pitch.draw(ax=ax)
        ax.set_title(title, loc="left", fontsize=panel_title_size)
        for team in prepared.teams:
            team_players = prepared.players.filter(pl.col("team_id") == team)
            x = team_players.get_column("x").to_numpy()
            y = team_players.get_column("y").to_numpy()
            ax.scatter(
                x,
                y,
                color=[team_colors[team]],
                edgecolor="black",
                linewidth=0.7,
                s=48,
                zorder=4,
            )
            if (
                prepared.player_velocity_x is not None
                and prepared.player_velocity_y is not None
            ):
                u = (
                    team_players.get_column(prepared.player_velocity_x).to_numpy()
                    * velocity_arrow_seconds
                )
                v = (
                    team_players.get_column(prepared.player_velocity_y).to_numpy()
                    * velocity_arrow_seconds
                )
                ax.quiver(
                    x,
                    y,
                    u,
                    v,
                    color=[team_colors[team]],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.004,
                    zorder=5,
                )
        if prepared.ball_vx is not None and prepared.ball_vy is not None:
            ax.quiver(
                prepared.ball_x,
                prepared.ball_y,
                prepared.ball_vx * velocity_arrow_seconds,
                prepared.ball_vy * velocity_arrow_seconds,
                color="black",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.004,
                zorder=5,
            )
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(ball_icon, zoom=0.27),
                (prepared.ball_x, prepared.ball_y),
                frameon=False,
                box_alignment=(0.5, 0.5),
                pad=0,
                zorder=6,
            )
        )
        fig.colorbar(image, ax=ax, pad=0.02, fraction=0.04)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=team_colors[team],
            markeredgecolor="black",
            markersize=Line2D([], []).get_markersize() + 2,
            label=str(team),
        )
        for team in prepared.teams
    ]
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.935))
    panel_left = axes[0].get_position().x0
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(panel_left, 0.96),
        ncol=max(1, len(handles)),
        fontsize=legend_font_size,
        borderaxespad=0,
    )
    fig.suptitle(
        f"Frame {frame_id}",
        x=panel_left,
        y=0.995,
        ha="left",
        fontsize=main_title_size,
    )
    return fig, axes
