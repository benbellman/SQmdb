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
