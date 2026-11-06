# SQmdb public API flow

This diagram summarizes the normal path through SQmdb. The left-to-right
branches distinguish movement model training, optional pass-time estimation,
pitch-value training, and frame-level surface calculation. The charting step is
optional and requires the `visualization` extra.

```mermaid
flowchart LR
    tracking["Canonical tracking<br/>positions, teams, frames"]
    config["SQConfig<br/>pitch, grid, sampling"]
    movement_fit["fit_player_movement_model"]
    movement_model["PlayerMovementModel"]
    passes["Completed pass events<br/>(optional)"]
    pass_model["PassHorizonModel.fit"]
    influence["iter_player_influence<br/>or player_influence"]
    labels["iter_defensive_influence_samples<br/>or defensive_influence_samples"]
    value_model["PitchValueModel.fit"]
    pitch["iter_pitch_value<br/>or pitch_value"]
    quality["space_quality"]
    plot["plot_pitch_surfaces<br/>(optional)"]
    surfaces["Long-form surface tables"]

    tracking --> movement_fit
    config --> movement_fit
    movement_fit --> movement_model
    passes --> pass_model
    movement_model --> influence
    pass_model -. optional time horizon .-> influence
    tracking --> influence
    config --> influence
    influence --> surfaces
    influence --> labels
    tracking --> labels
    labels --> value_model
    tracking --> pitch
    value_model --> pitch
    config --> pitch
    pitch --> quality
    influence --> quality
    quality --> surfaces
    influence --> plot
    pitch --> plot
    quality --> plot
    tracking --> plot
    config --> plot
    plot --> surfaces
```

Iterator functions yield surface tables incrementally; their non-iterator
counterparts collect those batches into one table. `SQConfig.grid()` provides
the cell-center coordinates used by each surface. `space_quality` is a
team-level result; player-level influence remains available separately.
