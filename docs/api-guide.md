# SQmdb API manual

SQmdb works with Polars tables. This manual describes the public names exported
from `sqmdb`, including every function argument, its type and unit, defaults,
results, and important limitations. It is written for readers who are new to
Python and Polars. For a narrated walkthrough of the complete workflow, see
the [example pipeline](api-flow.md).

## Basic terms

- A Polars **DataFrame** is a table held in memory. A **LazyFrame** describes
  table operations that Polars executes when needed. Some SQmdb operations
  collect data even when given a LazyFrame; the individual entries explain
  which results are eager or yielded in batches.
- A **pitch grid** is a set of cell centers laid over the pitch. A **surface**
  is a table containing a value for each selected frame and grid cell.
- An **iterator** yields a sequence of Polars DataFrame batches. It can avoid
  keeping the entire output in memory, but it does not make model fitting or
  every intermediate calculation lazy.
- Coordinates and distances are in metres unless stated otherwise.

## Input conventions

Tracking input generally needs `frame_id`, `player_id`, `team_id`, `x`, and
`y`; use a `match_id` when the table contains multiple matches. Coordinates
must be finite, in metres, centered at `(0, 0)`, and oriented so that the
attacking direction is positive x. Rows must be unique per match/frame/player.
Iterator-based calculations require match-grouped rows ordered by frame.

Pitch-value training additionally uses `ball_x`, `ball_y`, and
`possession_team_id`, consistent across all player rows in a frame. Pass
horizon fitting uses `pass_distance_m` in metres and
`pass_duration_seconds` in seconds. SQmdb does not connect to a data source or
infer the meaning of source columns; transform your data before passing it
to the package.

## `SQConfig`

`SQConfig` stores pitch geometry, sampling, density, and output-batching
settings. Construct it with keyword arguments; omitted values use these
defaults:

| Argument | Type and unit | Default | Meaning |
| --- | --- | --- | --- |
| `pitch_length_m` | `float`, metres | `105.0` | Pitch length along x |
| `pitch_width_m` | `float`, metres | `68.0` | Pitch width along y |
| `grid_resolution_m` | `float`, metres | `1.0` | Approximate spacing between grid-cell centers |
| `frame_rate` | `float`, frames/second | `25.0` | Sampling rate used to convert frames to time |
| `velocity_lag_seconds` | `float`, seconds | `0.2` | Time interval for velocity estimates |
| `frame_stride` | `int`, frames | `1` | Select every nth output frame by default |
| `batch_frames` | `int`, frames | `1` | Maximum selected frames in a yielded batch |
| `movement_density_resolution_m` | `float`, metres | `0.5` | Cell width for the movement-density approximation |
| `movement_density_extent_m` | `float`, metres | `30.0` | Half-width of the movement-density map around the origin |
| `movement_kde_bandwidth_m` | `float`, metres | `1.0` | Smoothing bandwidth for movement densities |
| `movement_horizons_seconds` | `tuple[float, ...]`, seconds | `0.2` to `5.0`, in `0.2`-second steps | Future movement horizons fitted from tracking |
| `movement_min_samples` | `int`, observations | `20` | Minimum observations in a player/bin/horizon distribution |

Constructing the class returns an `SQConfig` instance.

All lengths, rates, and horizons must be positive and finite. Stride and batch
sizes must be positive integers. Horizons must be sorted, unique, and positive.
Grid indices start at zero; `x` and `y` are cell-center coordinates in metres.
Grid dimensions are rounded to cover the configured pitch.

### `SQConfig.from_metadata`

```python
SQConfig.from_metadata(metadata, **overrides)
```

Use this class method when pitch dimensions are recorded alongside a tracking
dataset, for example in metadata supplied by your data provider or in a local
configuration file. Passing those values into SQmdb helps the pitch grid match
the coordinate system of the data instead of silently relying on the
105-by-68-metre defaults. The method reads only the canonical
`pitch_length_m` and `pitch_width_m` keys; it does not inspect a tracking
table or infer its dimensions.

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `metadata` | Mapping with optional numeric `pitch_length_m` and `pitch_width_m` keys | Required | Dataset or provider metadata containing pitch dimensions in metres |
| `**overrides` | `SQConfig` keyword settings | None | Explicit settings that replace metadata values or class defaults |

Returns an `SQConfig`. Missing or `None` dimensions use class defaults;
overrides take precedence. Invalid values or unknown setting names raise an
error.

### `SQConfig.grid`

```python
config.grid()
```

Takes no arguments. Returns a Polars DataFrame containing `grid_i`, `grid_j`,
`x`, and `y` for every configured pitch cell.

### `SQConfig.movement_lag_frames`

This read-only property takes no arguments. Returns an integer: the nearest
whole number of frames to `velocity_lag_seconds * frame_rate`, with a minimum
of one frame.

## Movement model and player influence

### `fit_player_movement_model`

```text
fit_player_movement_model(tracking, config=SQConfig(), *, sample_stride=1)
```

| Argument | Type and unit | Default | Meaning |
| --- | --- | --- | --- |
| `tracking` | `pl.DataFrame` or `pl.LazyFrame` | Required | Historical player positions used to learn movement distributions |
| `config` | `SQConfig` | `SQConfig()` | Frame rate, velocity lag, density, and horizon settings |
| `sample_stride` | `int`, frames | `1` | Use every nth eligible frame when forming training trajectories |

Returns a fitted `PlayerMovementModel`. Fitting is eager and may take
substantial time on long histories. Trajectories are paired by exact frame
IDs; missing frames do not create a training triplet. At least one usable
movement distribution must be available, and sparse player/speed groups may
use a pooled fallback. `sample_stride` must be positive.

### `PlayerMovementModel`

Use the fitted object returned by `fit_player_movement_model`; users normally
do not construct it directly. Pass it to `iter_player_influence` or
`player_influence`.

| Property | Type and unit | Meaning |
| --- | --- | --- |
| `horizons` | `tuple[float, ...]`, seconds | Movement horizons represented in the fitted distributions |
| `extent` | `float`, metres | Half-width of each stored density map |
| `resolution` | `float`, metres | Cell width of each stored density map |

The fitted distributions are stored in memory. Density values are smoothed
histogram approximations and normalized relative to their map peak; they are
not calibrated probabilities.

### `iter_player_influence` and `player_influence`

```text
iter_player_influence(
    tracking, model, config=SQConfig(), *, horizon_seconds=None,
    horizon_model=None, frame_stride=None
)
player_influence(
    tracking, model, config=SQConfig(), *, horizon_seconds=None,
    horizon_model=None, frame_stride=None
)
```

| Argument | Type and unit | Default | Meaning |
| --- | --- | --- | --- |
| `tracking` | `pl.DataFrame` or `pl.LazyFrame` | Required | Tracking rows for the frames to calculate |
| `model` | `PlayerMovementModel` | Required | Fitted movement distributions |
| `config` | `SQConfig` | `SQConfig()` | Pitch grid, frame rate, velocity lag, and batch settings |
| `horizon_seconds` | `float` or `None`, seconds | `None` | Fixed movement horizon; if both horizon options are omitted, `1.0` second is used |
| `horizon_model` | `PassHorizonModel` or `None` | `None` | Predict each player's horizon from their distance to the ball |
| `frame_stride` | `int` or `None`, frames | `None` | Override `config.frame_stride` for output selection |

`iter_player_influence` returns an iterator yielding Polars DataFrame batches.
`player_influence` returns one eagerly concatenated Polars DataFrame. Each
batch has `match_id`, `frame_id`, `player_id`, `team_id`, `grid_i`, `grid_j`,
`x`, `y`, and `player_influence`; it contains whole selected frames and all
players/grid cells for those frames.

Choose at most one of `horizon_seconds` and `horizon_model`. A horizon model
requires finite, frame-consistent `ball_x` and `ball_y` columns. The frame
stride must be positive. Tracking rows must be grouped by match and ordered by
frame. Prefer the iterator for large outputs.

## Pass timing

### `PassHorizonModel.fit`

```python
PassHorizonModel.fit(passes)
```

| Argument | Type and unit | Default | Meaning |
| --- | --- | --- | --- |
| `passes` | `pl.DataFrame` or `pl.LazyFrame` | Required | Completed passes with `pass_distance_m` (metres) and `pass_duration_seconds` (seconds) |

Returns a fitted `PassHorizonModel`. Fitting is eager. At least 10 rows are
required; distances must be finite and non-negative, durations finite and
positive, and there must be at least two distinct distances.

### `PassHorizonModel.predict`

```python
model.predict(pass_distance_m)
```

| Argument | Type and unit | Default | Meaning |
| --- | --- | --- | --- |
| `pass_distance_m` | `numpy.ndarray` or `float`, metres | Required | One or more non-negative pass distances |

Returns a NumPy array of predicted pass durations in seconds. Values are
clipped to the minimum and maximum durations seen during fitting. A scalar
input produces a one-element array. Non-finite or negative distances raise
`ValueError`.

### `PassHorizonModel.min_seconds` and `PassHorizonModel.max_seconds`

These read-only float properties take no arguments and return the shortest
and longest pass durations, respectively, in the rows used to fit the model.

## Pitch-value training labels and model

### `iter_defensive_influence_samples` and `defensive_influence_samples`

```text
iter_defensive_influence_samples(player_surface_batches, tracking)
defensive_influence_samples(player_surfaces, tracking)
```

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `player_surface_batches` / `player_surfaces` | Iterable of `pl.DataFrame`, or a `pl.DataFrame` / `pl.LazyFrame` | Required | Player-influence batches for the iterator form, or one table/collection of batches for the eager form |
| `tracking` | `pl.DataFrame` or `pl.LazyFrame` | Required | Matching tracking rows with ball position and `possession_team_id` |

`iter_defensive_influence_samples` returns an iterator yielding Polars
DataFrame label batches. `defensive_influence_samples` returns one
concatenated Polars DataFrame.

Tracking also needs the frame/player/team identity and position columns used
by the relevant operations. Frame-level possession and ball values must be
consistent. Each label row contains `match_id`, `frame_id`, `grid_i`,
`grid_j`, `grid_x`, `grid_y`, `ball_x`, `ball_y`, and
`defensive_influence`. The label is the summed influence of non-possessing
players, capped at `1.0`; when `is_goalkeeper` is present, goalkeepers are
excluded. Use the iterator for large histories.

### `PitchValueModel.fit`

```text
PitchValueModel.fit(
    samples, *, hidden_layer_sizes=(64, 64), random_state=0, max_iter=300,
    max_training_samples=1_000_000
)
```

| Argument | Type and unit | Default | Meaning |
| --- | --- | --- | --- |
| `samples` | `pl.DataFrame` or `pl.LazyFrame` | Required | Labels with `ball_x`, `ball_y`, `grid_x`, `grid_y`, and `defensive_influence` |
| `hidden_layer_sizes` | `tuple[int, ...]`, units/layer | `(64, 64)` | Number of units in each neural-network hidden layer |
| `random_state` | `int` | `0` | Seed for deterministic model initialization and sample selection |
| `max_iter` | `int`, training iterations | `300` | Maximum optimizer iterations |
| `max_training_samples` | `int`, rows | `1_000_000` | Upper bound on training rows; larger sets are deterministically subsampled |

Returns a fitted `PitchValueModel`; training is eager. At least 10 finite
training rows and target values in `[0, 1]` are required. The model uses a
scikit-learn multilayer perceptron; it does not implement every training
technique in the paper. Use `fit` rather than constructing the model
directly; the constructor expects a fitted estimator.

### `PitchValueModel.predict_defensive_influence`

```python
model.predict_defensive_influence(ball_x, ball_y, grid_x, grid_y)
```

| Argument | Type and unit | Default | Meaning |
| --- | --- | --- | --- |
| `ball_x`, `ball_y` | `numpy.ndarray`, metres | Required | Ball coordinates for each prediction |
| `grid_x`, `grid_y` | `numpy.ndarray`, metres | Required | Pitch-grid coordinates paired with the ball coordinates |

Returns a NumPy array of predicted defensive-influence values. The four input
arrays must have the same number of rows; predictions are clipped to `[0, 1]`.

### `iter_pitch_value` and `pitch_value`

```text
iter_pitch_value(tracking, model, config=SQConfig(), *, frame_stride=None)
pitch_value(tracking, model, config=SQConfig(), *, frame_stride=None)
```

| Argument | Type and unit | Default | Meaning |
| --- | --- | --- | --- |
| `tracking` | `pl.DataFrame` or `pl.LazyFrame` | Required | Tracking with `ball_x` and `ball_y` consistent within each frame |
| `model` | `PitchValueModel` | Required | Fitted pitch-value predictor |
| `config` | `SQConfig` | `SQConfig()` | Pitch grid and output-batch settings |
| `frame_stride` | `int` or `None`, frames | `None` | Override `config.frame_stride` |

`iter_pitch_value` returns an iterator yielding Polars DataFrame batches.
`pitch_value` returns one eagerly concatenated Polars DataFrame. Rows contain
`match_id`, `frame_id`, `grid_i`, `grid_j`, `x`, `y`, and `pitch_value`.
Pitch value combines predicted defensive influence with the package's
distance-to-opponent-goal scaling; it is a model output, not a calibrated
scoring probability.

## Team control and space quality

### `space_quality`

```python
space_quality(player_surfaces, pitch_values)
```

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `player_surfaces` | `pl.DataFrame` or `pl.LazyFrame` | Required | Player-level influence rows with team, frame, grid, and `player_influence` |
| `pitch_values` | `pl.DataFrame` or `pl.LazyFrame` | Required | Pitch-value rows with matching frame/grid keys and `pitch_value` |

Returns a collected Polars DataFrame with one row per match, frame, team, and
grid cell. `team_pitch_control` is `tanh(team influence - opponent
influence)`; `space_quality` is `team_pitch_control * pitch_value`. The
inputs must use compatible frame and grid keys. Original player-level
influence remains available separately.

## Visualization

### `add_plot_velocities`

```python
add_plot_velocities(tracking, config=SQConfig())
```

| Argument | Type and unit | Default | Meaning |
| --- | --- | --- | --- |
| `tracking` | `pl.DataFrame` or `pl.LazyFrame` | Required | Tracking with frame/player IDs, player coordinates, and ball coordinates |
| `config` | `SQConfig` | `SQConfig()` | Frame rate and velocity lag |

Returns a Polars LazyFrame with `vx`, `vy`, `ball_vx`, and `ball_vy` in
metres/second. It compares each position to the nearest whole-frame lag
specified by the configuration. Positions without a matching lagged value
(including the start of a match) receive zero velocity. Apply it to the full
tracking sequence before filtering to plot frames.

### `plot_pitch_surfaces`

```text
plot_pitch_surfaces(
    pitch_control, pitch_value, space_quality, tracking, frame_id, *,
    title=None, match_id=None, team_id=None, config=SQConfig(),
    velocity_x="vx", velocity_y="vy", ball_velocity_x="ball_vx",
    ball_velocity_y="ball_vy", velocity_arrow_seconds=1.5,
    figsize=(10.0, 15.0), attacking_team_color="#D55E00",
    defending_team_color="#0072B2", pitch_type="secondspectrum",
    pitch_length=105.0, pitch_width=69.0, **pitch_kwargs
)
```

| Argument | Type and unit | Default | Meaning |
| --- | --- | --- | --- |
| `pitch_control` | `pl.DataFrame` or `pl.LazyFrame` | Required | Team control surface for the selected frame |
| `pitch_value` | `pl.DataFrame` or `pl.LazyFrame` | Required | Pitch-value surface for the selected frame |
| `space_quality` | `pl.DataFrame` or `pl.LazyFrame` | Required | Space-quality surface for the selected frame |
| `tracking` | `pl.DataFrame` or `pl.LazyFrame` | Required | Player positions and frame-consistent ball position for the selected frame |
| `frame_id` | `int`, frame index | Required | Frame to render |
| `title` | `str` or `None` | `None` | Optional chart title; otherwise a frame title is used |
| `match_id` | Any value or `None` | `None` | Match selector when the frame ID is present in multiple matches |
| `team_id` | Any value or `None` | `None` | Team treated as attacking in the signed color scale |
| `config` | `SQConfig` | `SQConfig()` | Grid coordinates and pitch settings used by the surfaces |
| `velocity_x`, `velocity_y` | Column names (`str`) | `"vx"`, `"vy"` | Player velocity columns |
| `ball_velocity_x`, `ball_velocity_y` | Column names (`str`) | `"ball_vx"`, `"ball_vy"` | Ball velocity columns |
| `velocity_arrow_seconds` | `float`, seconds | `1.5` | Forward projection duration for velocity arrows |
| `figsize` | Pair of `float`, inches | `(10.0, 15.0)` | Matplotlib figure size |
| `attacking_team_color` | Matplotlib-compatible color | `"#D55E00"` | Color for the selected attacking team and positive control |
| `defending_team_color` | Matplotlib-compatible color | `"#0072B2"` | Color for the opponent and negative control |
| `pitch_type` | `str` | `"secondspectrum"` | Pitch definition passed to mplsoccer |
| `pitch_length`, `pitch_width` | `float`, metres | `105.0`, `69.0` | Pitch dimensions passed to mplsoccer |
| `**pitch_kwargs` | Keyword arguments | None | Additional options forwarded to `mplsoccer.Pitch` |

Returns a tuple `(figure, axes)` containing the Matplotlib figure and an array
of axes ordered pitch control, pitch value, and space quality. All surfaces
and tracking must contain the selected frame and compatible grid coordinates.
Missing velocity columns produce a warning and omit the relevant arrows.
Rendering requires `python -m pip install --upgrade "sqmdb[visualization]"`.

## Modeling scope

SQmdb's movement distributions are smoothed histogram approximations to KDEs;
pitch-value fitting uses a scikit-learn neural network; and `space_quality`
combines team control and pitch value. These functions implement useful
components, not all analyses in the research paper. See the
[pipeline walkthrough](api-flow.md) for a concise mapping to the paper's
sections.

## Reference

Martens, F., Dick, U., & Brefeld, U. (2021). *Space and Control in Soccer*.
Frontiers in Sports and Active Living, 3, 676179.
[https://doi.org/10.3389/fspor.2021.676179](https://doi.org/10.3389/fspor.2021.676179)
