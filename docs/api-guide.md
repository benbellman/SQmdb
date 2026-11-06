# SQmdb API guide

SQmdb turns soccer tracking and pass-event data into values on a football-pitch
grid. This guide describes the public Python API, how its pieces fit together,
and what to expect from its inputs and outputs. The examples use made-up
column names and data; use your own approved data source and never commit
private tracking rows, pass records, credentials, or query output.

For the API flow at a glance, see [the API flow diagram](api-flow.md). The
repository [README](../README.md) has a complete workflow example, canonical
column table, local database adapter example, and method limitations.

## Python and data terms

- **Package:** SQmdb is a collection of Python modules installed into a Python
  environment. Install it from the repository with `python -m pip install .`.
  The `pip` command installs packages; `import sqmdb` makes the installed
  package's public names available to a Python program.
- **Polars:** Polars is the table library SQmdb uses. A `DataFrame` is a table
  already held in memory. A `LazyFrame` is a description of table operations
  that Polars can plan before executing them. SQmdb accepts both in many
  functions, but a `LazyFrame` input does **not** make every operation lazy:
  model fitting and some results are collected before the function returns.
- **Grid and surface:** `SQConfig.grid()` lays out cell centers over a pitch.
  A surface is a table of values at those cells, generally one row per
  `match_id`, `frame_id`, `grid_i`, and `grid_j` (and, for player influence,
  `player_id`). Surfaces are in long/table form, not a 2-D image array.
- **Eager result:** A function that returns a complete `DataFrame` or fitted
  model computes that result before returning. Eager convenience functions
  may retain many frames' surface rows in memory.
- **Iterator:** A function whose name starts with `iter_` returns a Python
  iterator. Each iteration yields a Polars `DataFrame` batch, so a caller can
  write or inspect results incrementally instead of concatenating an entire
  match in memory. An iterator is not a promise that all upstream fitting or
  each individual frame is lazy; see each API entry below.

## Recommended workflow

1. Convert source data to the [canonical schema](../README.md#canonical-tracking-schema):
   coordinates in metres, pitch centered at `(0, 0)`, and the attacking
   direction toward positive x. Keep rows grouped by match and ordered by
   frame for iterator-based tracking calculations.
2. Set pitch size, grid resolution, frame rate, and batch behavior with
   `SQConfig`.
3. Fit `PlayerMovementModel` on historical tracking. Optionally fit
   `PassHorizonModel` from completed passes; otherwise use a fixed movement
   horizon.
4. Calculate player influence. Build defensive-influence training labels from
   historical player surfaces, fit `PitchValueModel`, then calculate pitch
   value for the frames you want to analyze.
5. Combine player influence and pitch value with `space_quality`. Optionally
   render all three surfaces for a selected frame with `plot_pitch_surfaces`.

## Public API reference

Every name in `sqmdb.__all__` is listed here. Imports use the package root,
for example `from sqmdb import SQConfig, fit_player_movement_model`.

| Public name | What it does | Inputs and result | Execution and important assumptions |
| --- | --- | --- | --- |
| `SQConfig` | Holds pitch, grid, sampling, movement-density, and batch settings. | Construct with keyword settings; `grid()` returns a Polars table of `grid_i`, `grid_j`, `x`, and `y` cell centers. `from_metadata(mapping, **overrides)` reads canonical pitch-dimension keys and uses defaults for absent dimensions. | Grid indices start at zero. Coordinates are centered in metres and the positive x direction is the attack direction. Defaults are 105 × 68 m, 1 m resolution, 25 frames/s, and one frame per output batch. |
| `PlayerMovementModel` | Fitted, player- and speed-conditioned movement distributions. | Returned by `fit_player_movement_model`; use it with `iter_player_influence` or `player_influence`. Its `horizons` property lists the horizons actually fitted. | Not a raw endpoint table; it holds smoothed binned density maps in memory. Missing player/speed-bin fits can fall back to a pooled distribution. |
| `fit_player_movement_model` | Learns likely player movement from historical trajectories. | `tracking` with `frame_id`, `player_id`, `team_id`, `x`, `y`; optional `config` and `sample_stride`. Returns `PlayerMovementModel`. | Eager model fitting. Uses exact frame IDs for configured time offsets and requires unique rows per match/frame/player. More trajectory data improves coverage of speed bins and horizons. |
| `PassHorizonModel` | Learns typical pass travel time from pass distance. | `PassHorizonModel.fit(passes)` expects `pass_distance_m` and `pass_duration_seconds`; `.predict(distance)` returns predicted seconds as a NumPy array. | Fit is eager and requires at least 10 finite completed passes, positive durations, non-negative distances, and at least two distinct distances. Predictions are clipped to observed duration limits. |
| `iter_player_influence` | Calculates each player's influence over the pitch grid for selected frames. | Tracking plus a fitted movement model and optional `config`; choose either fixed `horizon_seconds` (defaults to 1 second) or `horizon_model`, not both. `frame_stride` can select output frames. Yields batches with player/team, frame, grid coordinates, and `player_influence`. | Streaming-style output in complete frame batches; input rows must be grouped by match and ordered by frame. A pass horizon model also requires per-frame `ball_x` and `ball_y`. Output volume grows with frames × players × grid cells. |
| `player_influence` | Convenience form of `iter_player_influence`. | Same inputs; returns a single Polars `DataFrame`. | Eager: consumes and concatenates all iterator batches. Prefer the iterator for large inputs. |
| `iter_defensive_influence_samples` | Creates pitch-value training labels from batches of player surfaces. | An iterable of player-surface `DataFrame`s and tracking with `possession_team_id`, `ball_x`, and `ball_y` on each frame; tracking also needs the canonical identity columns. Yields label batches with `grid_x`, `grid_y`, `ball_x`, `ball_y`, and `defensive_influence`. | Yields per input batch, but each batch's label result is collected before yielding. Goalkeepers are excluded when tracking provides `is_goalkeeper`; defending influence is summed and capped at 1. |
| `defensive_influence_samples` | Convenience form that creates all defensive-influence labels together. | One surface `DataFrame`/`LazyFrame` or an iterable of surface batches plus tracking; returns one label `DataFrame`. | Eager: consumes all yielded batches and concatenates them. Use `iter_defensive_influence_samples` for larger histories. |
| `PitchValueModel` | Predicts defensive influence from ball and grid-cell locations, then supports pitch-value surface calculation. | `PitchValueModel.fit(samples, ...)` expects `ball_x`, `ball_y`, `grid_x`, `grid_y`, and `defensive_influence`. `.predict_defensive_influence(...)` accepts NumPy arrays for the four coordinates and returns predictions bounded to `[0, 1]`. | Fit is eager; at least 10 finite samples with labels in `[0, 1]` are required. Training is deterministically sampled above `max_training_samples`. |
| `iter_pitch_value` | Predicts a pitch-value surface for each selected tracking frame. | Tracking with canonical player columns plus `ball_x` and `ball_y`; a fitted `PitchValueModel`; optional `config` and `frame_stride`. Yields batches with frame/grid coordinates and `pitch_value`. | Yields bounded frame batches. Each frame needs consistent ball coordinates. |
| `pitch_value` | Convenience form of `iter_pitch_value`. | Same inputs; returns one Polars `DataFrame`. | Eager: consumes and concatenates all batches. Use the iterator for large inputs. |
| `space_quality` | Combines team influence and pitch value into team pitch control and space quality. | Player influence surfaces and pitch-value surfaces; returns a Polars table with `team_pitch_control`, `pitch_value`, and `space_quality` by frame, team, and grid cell. | Eager: returns a collected `DataFrame`. Pitch control is `tanh(team influence - opponent influence)`; space quality is pitch control × pitch value. |
| `plot_pitch_surfaces` | Draws a three-panel view of pitch control, pitch value, and space quality, with the same frame's players, velocities, and ball overlaid. | Three compatible surfaces, tracking for the selected `frame_id`, optional `match_id` and `team_id`, and `SQConfig`; returns `(figure, axes)`. Tracking needs `x`, `y`, `vx`, `vy`, `ball_x`, and `ball_y` by default; velocity-column names can be changed. | Materializes and validates only the selected frame before plotting. Requires the optional `matplotlib` dependency (`python -m pip install ".[visualization]"`). This is an original three-panel chart, not a reproduction of the paper's figure. |

### Surfaces and frame keys

Each calculation should use compatible match and frame IDs, pitch dimensions,
and grid settings. `match_id` is optional in input tables; when omitted,
SQmdb treats the input as a single match. When saving surface tables, retain
`match_id`, `frame_id`, `grid_i`, `grid_j`, `x`, and `y`, as well as the
`SQConfig` settings used to create them. The grid indices identify a cell;
`x` and `y` give its location in metres.

Tracking positions must be finite and non-null. A frame's ball columns and
other frame-level metadata must be constant across its player rows. Do not mix
coordinate origins, units, attacking directions, or grid configurations
between inputs. SQmdb does not connect to a data source or infer how source
columns should be interpreted; make those choices in your private adapter.

### Streaming does not mean zero memory

In this package, “streaming” means a result can be consumed in batches rather
than immediately concatenated as one large output. `iter_player_influence`
keeps tracking work grouped by frame and retains only a short velocity history,
but each yielded frame batch still includes every player and grid cell selected
for those frames. `iter_pitch_value` and `iter_defensive_influence_samples`
also yield batches. Model fitting, `space_quality`, plotting one selected
frame, and the non-iterator convenience functions produce eager results.
Choose `frame_stride`, `grid_resolution_m`, and `batch_frames` with output
volume and memory in mind.

## Current modeling limitations

The package implements practical approximations, not every detail of the
referenced research method. Movement densities are smoothed histograms
approximating KDEs; pitch value uses a scikit-learn neural network and does
not reproduce the paper's dropout or Bayesian hyperparameter search. The
README's [Method and scope](../README.md#method-and-scope) section describes
these choices and additional caveats. A successful small data smoke test is
not evidence that the models generalize to other teams, competitions, or
seasons.
