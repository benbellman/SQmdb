from __future__ import annotations

from dataclasses import dataclass
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
    teams: tuple[Any, ...]


def _frame_rows(
    data: pl.DataFrame | pl.LazyFrame,
    frame_id: int,
    match_id: Any | None,
    required: set[str],
) -> pl.DataFrame:
    lazy = as_lazy(data, required | {"frame_id"})
    selected = lazy.filter(pl.col("frame_id") == frame_id)
    if match_id is not None:
        selected = selected.filter(pl.col("match_id") == match_id)
    rows = selected.collect(engine="streaming")
    if rows.is_empty():
        raise ValueError(f"No rows found for frame_id={frame_id!r}.")
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
    velocity_x: str,
    velocity_y: str,
) -> _PreparedPitchData:
    control_rows = _frame_rows(
        pitch_control,
        frame_id,
        match_id,
        {"team_id", "grid_i", "grid_j", "x", "y", "team_pitch_control"},
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
    )
    quality_rows = _frame_rows(
        space_quality,
        frame_id,
        match_id,
        {"team_id", "grid_i", "grid_j", "x", "y", "space_quality"},
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

    players = _frame_rows(
        tracking,
        frame_id,
        match_id,
        {"player_id", "team_id", "x", "y", "ball_x", "ball_y", velocity_x, velocity_y},
    ).with_columns(
        *[
            pl.col(column).cast(pl.Float64, strict=True)
            for column in ("x", "y", "ball_x", "ball_y", velocity_x, velocity_y)
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
                for column in ("x", "y", "ball_x", "ball_y", velocity_x, velocity_y)
            ],
        ).any()
    ).item():
        raise ValueError("Player, ball, and velocity values cannot be null.")
    if players.select(pl.struct("player_id").n_unique()).item() != players.height:
        raise ValueError("Tracking rows must have one row per player in the selected frame.")
    player_values = players.select(
        "x", "y", "ball_x", "ball_y", velocity_x, velocity_y
    ).to_numpy()
    if not np.all(np.isfinite(player_values)):
        raise ValueError("Player, ball, and velocity values must be finite.")
    ball = players.select("ball_x", "ball_y").unique()
    if ball.height != 1:
        raise ValueError("ball_x and ball_y must be constant within the selected frame.")

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
        teams=selected_teams,
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
    velocity_arrow_seconds: float = 0.5,
    figsize: tuple[float, float] = (10.0, 15.0),
) -> tuple[Any, np.ndarray]:
    """Plot three aligned team surfaces with the selected frame's tracking.

    Tracking must contain one row per player with positions in metres, player
    velocities in metres per second (``vx``/``vy`` by default), and the ball
    position repeated consistently across the frame. The surfaces use SQmdb's
    zero-based ``grid_i``/``grid_j`` cell-center coordinates.

    Matplotlib is an optional dependency; install ``sqmdb[visualization]`` to
    render charts. The returned axes are ordered pitch control, pitch value,
    and space quality from top to bottom.
    """
    if not np.isfinite(velocity_arrow_seconds) or velocity_arrow_seconds <= 0:
        raise ValueError("velocity_arrow_seconds must be a positive finite number.")
    if len(figsize) != 2 or not all(np.isfinite(value) and value > 0 for value in figsize):
        raise ValueError("figsize must contain two positive finite numbers.")
    prepared = _prepare_pitch_data(
        pitch_control,
        pitch_value,
        space_quality,
        tracking,
        frame_id,
        match_id=match_id,
        team_id=team_id,
        config=config,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
    )
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
    except ImportError as error:
        raise ImportError(
            "Pitch visualization requires Matplotlib; install sqmdb[visualization]."
        ) from error

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

    x_edges = cell_edges(prepared.x, config.pitch_length_m)
    y_edges = cell_edges(prepared.y, config.pitch_width_m)
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)
    panels = (
        ("Pitch control", prepared.pitch_control, "coolwarm", -1.0, 1.0),
        ("Pitch value", prepared.pitch_value, "viridis", 0.0, 1.0),
        ("Space quality", prepared.space_quality, "coolwarm", -1.0, 1.0),
    )
    team_colors = {
        team: plt.get_cmap("tab10")(index % 10)
        for index, team in enumerate(prepared.teams)
    }
    for ax, (title, values, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )
        ax.set_title(title, loc="left")
        ax.set_xlim(-config.pitch_length_m / 2, config.pitch_length_m / 2)
        ax.set_ylim(-config.pitch_width_m / 2, config.pitch_width_m / 2)
        ax.set_aspect("equal", adjustable="box")
        ax.plot(
            [0, 0],
            [-config.pitch_width_m / 2, config.pitch_width_m / 2],
            color="white",
            linewidth=1.0,
            alpha=0.9,
        )
        ax.add_patch(
            Circle(
                (0, 0),
                min(9.15, config.pitch_width_m / 2),
                fill=False,
                edgecolor="white",
                linewidth=1.0,
                alpha=0.9,
            )
        )
        for team in prepared.teams:
            team_players = prepared.players.filter(pl.col("team_id") == team)
            x = team_players.get_column("x").to_numpy()
            y = team_players.get_column("y").to_numpy()
            u = team_players.get_column(velocity_x).to_numpy() * velocity_arrow_seconds
            v = team_players.get_column(velocity_y).to_numpy() * velocity_arrow_seconds
            ax.scatter(
                x,
                y,
                color=[team_colors[team]],
                edgecolor="black",
                linewidth=0.7,
                s=48,
                zorder=4,
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
        ax.scatter(
            [prepared.ball_x],
            [prepared.ball_y],
            marker="*",
            color="#ffcc00",
            edgecolor="black",
            linewidth=0.7,
            s=130,
            zorder=6,
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
            label=str(team),
        )
        for team in prepared.teams
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            color="#ffcc00",
            markeredgecolor="black",
            label="Ball",
        )
    )
    fig.legend(handles=handles, loc="upper center", ncol=max(1, len(handles)))
    fig.supxlabel("x (m)")
    fig.supylabel("y (m)")
    fig.suptitle(f"Frame {frame_id}", y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.97))
    return fig, axes
