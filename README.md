# SQmdb

SQmdb is a Python package for estimating player influence, pitch control,
pitch value, and space quality from soccer tracking data. It uses Polars tables
and implements selected components of the method in Martens, Dick, and Brefeld
(2021); it is an approximation of that method, not a complete reproduction of the paper.

## Install from PyPI

Install or upgrade to the latest release:

```bash
python -m pip install --upgrade sqmdb
```

For pitch visualizations, install the optional dependencies:

```bash
python -m pip install --upgrade "sqmdb[visualization]"
```

To record the installed release in an analysis environment:

```bash
python -c "from importlib.metadata import version; print(version('sqmdb'))"
```

For reproducible analyses, record that version with the analysis and pin it
when recreating the environment, for example
`python -m pip install "sqmdb==0.1.0"`.

## Starting with Parquet

SQmdb accepts Polars `DataFrame` and `LazyFrame` objects. Your input adapter
should provide the canonical columns below, with coordinates in metres,
centered on `(0, 0)`, and attacking play oriented toward positive x (left-to-right). The
[example pipeline](https://github.com/benbellman/SQmdb/blob/main/docs/api-flow.md) assumes you have stored your tracking data as Parquet files in a directory called `queried-data`. Valid frames during live play are saved to `tracking-live.parquet` and the pass-horizon data is saved to `passes.parquet`.

```python
import polars as pl
from sqmdb import SQConfig, fit_player_movement_model, iter_player_influence

tracking = pl.scan_parquet("path/to/queried-data/tracking-live.parquet")
config = SQConfig(frame_rate=25.0, grid_resolution_m=2.0, frame_stride=25)
movement = fit_player_movement_model(tracking, config, sample_stride=5)

for batch in iter_player_influence(tracking, movement, config):
    print(batch.head())
```

The [API manual](https://github.com/benbellman/SQmdb/blob/main/docs/api-guide.md) defines every public argument, return
value, unit, default, and important assumption.

## Tracking input columns

One row represents one player's position at one frame.

| Column | Required | Meaning |
| --- | --- | --- |
| `frame_id` | Yes | Integer frame index within a match |
| `player_id` | Yes | Stable player identifier |
| `team_id` | Yes | Player's team identifier |
| `x`, `y` | Yes | Player position in metres |
| `match_id` | No | Match identifier; omitted means one match |
| `ball_x`, `ball_y` | For pitch value and pass horizons | Ball position, consistent across a frame |
| `possession_team_id` | For pitch-value training | Team in possession, consistent across a frame |
| `is_goalkeeper` | No | Boolean; goalkeepers are excluded from labels when supplied |

Pass-horizon fitting also needs completed-pass data with
`pass_distance_m` (metres) and `pass_duration_seconds` (seconds). Tracking
rows must be unique by match, frame, and player. For iterator-based
calculations, group rows by match and order them by frame. See the API manual
for additional input and modeling requirements.

## Visualize results

`add_plot_velocities` adds velocity columns to tracking data.
`plot_pitch_surfaces` can then draw pitch-control, pitch-value, and
space-quality surfaces with player and ball overlays. Both functions require
the optional visualization dependencies; the pipeline guide includes a usage
example.

## Keep your data private

SQmdb operates on data you provide locally; it does not connect to your
tracking provider or database. Keep raw tracking and pass Parquet files,
row-level data, credentials, connection strings, private queries, and
generated result directories out of public repositories and package
artifacts. The example accepts local paths and writes results to a directory
you choose; use synthetic data for public automated tests.

## Reference

Martens, F., Dick, U., & Brefeld, U. (2021). *Space and Control in Soccer*.
Frontiers in Sports and Active Living, 3, 676179.
[https://doi.org/10.3389/fspor.2021.676179](https://doi.org/10.3389/fspor.2021.676179)
