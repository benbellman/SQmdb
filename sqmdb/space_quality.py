from __future__ import annotations

import polars as pl

from ._data import as_lazy


def space_quality(
    player_surfaces: pl.DataFrame | pl.LazyFrame,
    pitch_values: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame:
    """Calculate team pitch control and space quality on each grid cell.

    Pitch control for each team is tanh(sum of its players' influence minus the
    sum of opponent influence); space quality is pitch control times pitch
    value. Player influence surfaces remain available separately.
    """
    influence = as_lazy(
        player_surfaces,
        {
            "frame_id",
            "team_id",
            "grid_i",
            "grid_j",
            "player_influence",
        },
    )
    values = as_lazy(
        pitch_values,
        {"frame_id", "grid_i", "grid_j", "pitch_value"},
    )
    keys = ["match_id", "frame_id", "grid_i", "grid_j"]
    totals = (
        influence.group_by(*keys, "team_id")
        .agg(pl.col("player_influence").sum().alias("_team_influence"))
    )
    all_teams = totals.group_by(keys).agg(
        pl.col("_team_influence").sum().alias("_total_influence")
    )
    control = totals.join(all_teams, on=keys).with_columns(
        (pl.col("_team_influence") * 2 - pl.col("_total_influence"))
        .tanh()
        .alias("team_pitch_control")
    )
    return (
        control.join(values, on=keys, how="inner")
        .with_columns(
            (pl.col("team_pitch_control") * pl.col("pitch_value"))
            .alias("space_quality")
        )
        .select(
            *keys,
            "team_id",
            "team_pitch_control",
            "pitch_value",
            "space_quality",
        )
        .collect()
    )
