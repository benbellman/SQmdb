# SQmdb

SQmdb is a Polars-first Python library for calculating soccer player-influence,
pitch-value, and space-quality surfaces using the data-driven method in
Martens, Dick, and Brefeld (2021), “Space and Control in Soccer.”

## Install

```bash
pip install .
```

The library accepts Polars `DataFrame` and `LazyFrame` inputs and produces
Polars outputs. Source-specific adapters are intentionally outside the core:
rename and convert source columns to the canonical schema before calling
SQmdb.

New to Python packages or Polars? Start with the
[beginner API guide](docs/api-guide.md) for plain-language explanations of the
public API, required inputs, model assumptions, and eager versus iterator
results. The [API flow diagram](docs/api-flow.md) is a visual map of how the
models and surfaces connect.

## Visualize pitch surfaces

Install the optional plotting dependency with
`python -m pip install ".[visualization]"` (Matplotlib and mplsoccer).
`plot_pitch_surfaces` accepts
already-computed, long-form pitch-control, pitch-value, and space-quality
surfaces plus one frame of tracking. It renders three vertically stacked
panels with shared player, ball, and velocity overlays; it does not fetch or
transform data from a specific database or tracking provider.

```python
import polars as pl
from sqmdb import add_plot_velocities, plot_pitch_surfaces

tracking_with_velocities = add_plot_velocities(tracking, config)
tracking_frame = tracking_with_velocities.filter(pl.col("frame_id") == 1200)
fig, axes = plot_pitch_surfaces(
    control_surface,
    pitch_value_surface,
    quality_surface,
    tracking_frame,
    frame_id=1200,
    match_id="match-1",
    team_id="home",
    config=config,
    attacking_team_color="#D55E00",
    defending_team_color="#0072B2",
)
```

The full tracking sequence should be passed through `add_plot_velocities`
before selecting individual frames. It adds `vx`, `vy`, `ball_vx`, and
`ball_vy` in metres per second from positions separated by the configured
velocity lag; frames without a matching lagged position receive zero. The
tracking frame passed to the plot must contain `player_id`, `team_id`, `x`,
`y`, `ball_x`, and `ball_y`; ball coordinates must be constant across the
frame. If either player or ball velocity components are absent, plotting warns
with the helper syntax and omits the corresponding arrows. Player velocity
columns can be renamed through `velocity_x` and `velocity_y`; ball velocity
columns can be renamed through `ball_velocity_x` and `ball_velocity_y`. Present
ball velocity columns must be constant across the frame. Surfaces must include
the SQmdb grid indices and cell-center coordinates; `team_pitch_control` and
`space_quality` are selected for the requested team, and space quality is
validated as pitch control multiplied by pitch value. Pass `match_id` when the
requested frame occurs in multiple matches. The chart is an original,
single-column visualization of the three surfaces, not a reproduction of the
paper's figure. The selected `team_id` is colored as the attacking team;
additional pitch-style options are forwarded to `mplsoccer.Pitch`, with
`pitch_type="secondspectrum"`, `pitch_length=105`, and `pitch_width=69` by
default. The pitch-control and space-quality surfaces use a custom diverging
scale: the defending color at -1, a neutral midpoint at 0, and the attacking
color at +1. Player dots use those same team colors. The default colors are
color-blind friendly, and player (and available ball) velocity arrows project
1.5 seconds ahead by default. Pitch value uses a five-stop scale at 0, 0.25,
0.5, 0.75, and 1: `#f6eff7`, `#bdc9e1`, `#67a9cf`, `#1c9099`, and `#016c59`.
Pitch lines are black; axes, ticks, and axis labels are hidden. The ball
overlay uses a compact bundled icon displayed slightly smaller than the player
markers, with a black velocity arrow.

## Data safety

Never expose or commit raw tracking/pass rows, credentials, connection strings,
query outputs, or other row-level source data to this repository or public
artifacts. Keep provider-specific queries and local data private. Public
changes may include reproducibility code, synthetic test data, aggregated
findings, analysis, and reviewed visualizations only.

## Canonical tracking schema

One row represents one player's position at one frame:

| Column | Required | Meaning |
| --- | --- | --- |
| `frame_id` | Yes | Integer frame index within a match |
| `player_id` | Yes | Stable player identifier across training matches |
| `team_id` | Yes | Player's team identifier |
| `x`, `y` | Yes | Player position in metres |
| `match_id` | No | Match identifier; omitted means one match |
| `ball_x`, `ball_y` | For pitch value | Ball position repeated consistently on each row in a frame |
| `possession_team_id` | For pitch-value training | Team in possession, repeated consistently per frame |
| `is_goalkeeper` | Optional | Boolean; goalkeepers are excluded from pitch-value labels when supplied |

Coordinates must be in metres, centered on `(0, 0)`, with attacking play
oriented toward positive x. Convert tracking feeds with different origins,
units, or attack directions before calculation. `frame_id` must advance by one
for each adjacent sampled frame; frame gaps simply do not create movement
training triplets. For bounded streaming, rows must be grouped by `match_id`
and ordered by `frame_id`; frames must not be interleaved. Pitch dimensions
default to 105 × 68 m and are configurable in `SQConfig`; bounds are not
guessed from player-coordinate extrema. `SQConfig.from_metadata` reads
`pitch_length_m` and `pitch_width_m` from canonical metadata keys and falls
back to those defaults when either is absent.

## Quick start

Fit movement distributions on historical tracking, then calculate player
influence. The examples below use a small, configurable grid for illustration:

```python
import polars as pl
from pathlib import Path
from sqmdb import (
    PassHorizonModel,
    PitchValueModel,
    SQConfig,
    defensive_influence_samples,
    fit_player_movement_model,
    iter_defensive_influence_samples,
    iter_player_influence,
    iter_pitch_value,
    space_quality,
)

training_tracking = pl.read_parquet("training-tracking.parquet")
test_tracking = pl.read_parquet("test-tracking.parquet")
passes = pl.read_parquet("passes.parquet")
config = SQConfig(grid_resolution_m=2.0, batch_frames=1, frame_stride=25)

movement_model = fit_player_movement_model(
    training_tracking,
    config,
    sample_stride=5,
)

# Optional: fit from canonical pass-event fields. Without this argument the
# influence function uses a fixed 1-second horizon (or your chosen duration).
pass_horizons = PassHorizonModel.fit(
    passes.select("pass_distance_m", "pass_duration_seconds")
)

# Stream batches to Parquet/Delta or another sink instead of retaining a full
# match's frame × player × grid-cell rows in memory.
influence_batches = iter_player_influence(
    test_tracking,
    movement_model,
    config,
    horizon_model=pass_horizons,
)
```

For pitch value, build observed defensive-influence labels from player
influences calculated on the training situations, train the feed-forward
predictor, then predict surfaces on tracking data:

```python
training_influence_batches = iter_player_influence(
    training_tracking, movement_model, config
)
label_batches = iter_defensive_influence_samples(
    training_influence_batches, training_tracking
)
# Persist each labels batch (e.g., as Parquet) for large histories, then scan
# the files lazily to fit the bounded-size neural model.
label_dir = Path("defensive-influence")
label_dir.mkdir(exist_ok=True)
for index, batch in enumerate(label_batches):
    batch.write_parquet(label_dir / f"part-{index:06d}.parquet")
pitch_value_model = PitchValueModel.fit(
    pl.scan_parquet("defensive-influence/*.parquet"),
    max_training_samples=250_000,
)

pitch_batches = iter_pitch_value(test_tracking, pitch_value_model, config)
```

For a small input, eager wrappers are also available:

```python
from sqmdb import pitch_value, player_influence

influence = player_influence(test_tracking, movement_model, config)
values = pitch_value(test_tracking, pitch_value_model, config)
quality = space_quality(influence, values)
```

## Local database smoke test

For a small extract (for example, 1–5 matches), keep database-specific SQL and
connection settings outside SQmdb. The helper in `examples/local_validation.py`
converts an executed DB-API cursor—including a `pyodbc` cursor—to a Polars
`DataFrame`; it reads in batches but concatenates the selected rows in memory.
From the repository root, install SQmdb in the Python environment used to run
the example with `python -m pip install -e .`. The `examples` helper is included
in the install, so `examples.local_validation` can also be imported outside the
repository root. In Positron, select this same Python environment. Install
`pyodbc` separately only if your local driver needs it. Do not commit
connection strings or credentials.

Alias selected source columns in SQL to the canonical names, filter to the
chosen matches, and preserve row order:

```python
import os
import pyodbc

from examples.local_validation import cursor_to_polars
from sqmdb import PassHorizonModel, SQConfig, fit_player_movement_model, iter_player_influence

connection = pyodbc.connect(os.environ["SQMDB_CONNECTION_STRING"])
try:
    tracking_cursor = connection.cursor()
    tracking_cursor.execute(
        """
        SELECT source_match_id AS match_id,
               source_frame AS frame_id,
               source_player_id AS player_id,
               source_team_id AS team_id,
               player_x_metres AS x, player_y_metres AS y,
               ball_x_metres AS ball_x, ball_y_metres AS ball_y
        FROM your_tracking_source
        WHERE source_match_id IN (?, ?)
        ORDER BY source_match_id, source_frame, source_player_id
        """,
        *match_ids,
    )
    tracking = cursor_to_polars(tracking_cursor)

    pass_cursor = connection.cursor()
    pass_cursor.execute(
        """
        SELECT completed_pass_distance_metres AS pass_distance_m,
               pass_travel_seconds AS pass_duration_seconds
        FROM your_completed_pass_source
        WHERE source_match_id IN (?, ?)
        """,
        *match_ids,
    )
    passes = cursor_to_polars(pass_cursor)
finally:
    connection.close()

config = SQConfig(frame_rate=25.0, grid_resolution_m=2.0)
movement = fit_player_movement_model(tracking, config, sample_stride=5)
horizons = PassHorizonModel.fit(passes)
for batch in iter_player_influence(
    tracking, movement, config, horizon_model=horizons, frame_stride=25
):
    print(batch.head())
```

Replace the illustrative source names and aliases with your schema; provide
`match_ids` as a tuple with one value per placeholder. Tracking coordinates
must be in metres with the pitch centered at `(0, 0)` and attack oriented toward
positive x. The returned tracking rows must be unique by match, frame, and
player; frame IDs must be integer-like, sampled consistently, grouped by match,
and ordered by frame. Set `frame_rate` and `frame_stride` to your feed's rate
(the example values assume 25 Hz). Include `ball_x` and `ball_y` in every frame
when using the pass horizon model.

The pass query must return completed-pass records with `pass_distance_m` in
metres and `pass_duration_seconds` as elapsed time from pass release to
reception, both finite and non-null. In the paper, pass distance is the
Euclidean separation between passer and intended receiver at pass initiation;
it is not the ball's path length. Reception is identified using a 1.5 m
ball-to-receiver proximity heuristic. At least 10 passes with two or more
distinct distances are required by `PassHorizonModel.fit`, and durations must
be positive. If reliable completed-pass times are unavailable, skip pass
fitting and supply a fixed `horizon_seconds` to `iter_player_influence`
instead. A one-to-five-match run is a useful integration smoke test; sparse
movement bins may require more tracking or a lower `movement_min_samples`
setting, which makes the fitted densities less reliable. A smoke test is not
by itself evidence of model generalization.

`iter_player_influence` and `iter_pitch_value` yield long-form Polars
DataFrames containing `match_id`, `frame_id`, `grid_i`, `grid_j`, grid
coordinates, and their respective values; player influence also contains
`player_id` and `team_id`. `space_quality` returns one row per team and grid
cell with team pitch control and space quality. Keep `grid_i`, `grid_j`, the
`SQConfig` pitch/grid settings, and coordinate orientation with persisted
surfaces to support later array reconstruction. `frame_stride` defaults to 1
(all frames); set it to e.g. 25 for 1 Hz data from 25 Hz tracking. Batches are
bounded by `batch_frames`, which defaults to one frame.

## Method and scope

- **Player influence:** Historical player triplets use a 0.2-second
  velocity estimate, are translated to the origin and rotated to align initial
  heading with +x, and are grouped by player, paper speed bin, and future time
  horizon. A binned Gaussian KDE approximates each endpoint density; the
  resulting grid values are normalized by the estimated density mode. The
  implementation fits pooled distributions as a fallback when a player-specific
  bin is unavailable.
- **Time horizon:** The paper learns a pass-distance-to-travel-time regression
  from event data, then chooses a movement horizon. `PassHorizonModel.fit`
  implements the power-transformed regression from canonical pass-event columns;
  pass it to `iter_player_influence` to choose a horizon from each player's
  distance to the ball. If event data is unavailable, use an explicit
  `horizon_seconds` (defaults to 1.0) and the nearest fitted movement horizon.
- **Pitch value:** `defensive_influence_samples` sums influence from the team
  defending against possession and caps the result at one. `PitchValueModel`
  learns defensive influence from `(ball_x, ball_y, grid_x, grid_y)` using a
  two-hidden-layer feed-forward network (64 units per layer by default), then
  applies the paper's distance-to-opponent-goal scaling. Training rows are
  deterministically hash-sampled if they exceed `max_training_samples`.
  The implementation uses scikit-learn's Adam-trained MLP; it does not
  reproduce the paper's dropout or Bayesian hyperparameter search.
- **Space quality:** For each team, pitch control is
  `tanh(team influence - opponent influence)`; space quality is pitch control
  multiplied by pitch value. The per-player surfaces expose the component
  influences, while the paper's space-quality surface is team-level.

The movement KDE is stored as a smoothed histogram for efficient grid queries;
its mode and the article's mean-shift mode are therefore approximated. The
configurable density resolution/bandwidth trade query speed for spatial
precision. Outputs grow with sampled frames × players × grid cells, so use
frame sampling and iterator APIs before increasing grid density.

## Reference

Martens, F., Dick, U., & Brefeld, U. (2021). *Space and Control in Soccer*.
Frontiers in Sports and Active Living, 3, 676179.
[https://doi.org/10.3389/fspor.2021.676179](https://doi.org/10.3389/fspor.2021.676179)
