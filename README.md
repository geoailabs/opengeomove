# OpenGeoMove

OpenGeoMove is a Python library for scalable geospatial analysis of urban
mobility data. It combines lazy, file-backed tabular operations with mobility
analysis, GPS visualization, and traffic analysis zone (TAZ) workflows.

The project is currently alpha software. Public interfaces may change between
early releases.

## Features

- Open Parquet, CSV, TSV, and JSON files as a common `Dataset` backed by
  DuckDB.
- Keep large inputs file-backed and push compatible filters, projections, and
  aggregations into DuckDB before materializing results.
- Inspect schemas, missing values, descriptive summaries, cardinality, and
  duplicates through `Dataset` methods.
- Compose read-only filters for device IDs, accuracy, timestamps, daily ping
  counts, and polygons.
- Detect stops, form trips, and classify recurring home and work locations.
- Plot bounded samples of GPS pings, with optional XYZ basemap tiles.
- Create, compare, classify, and visualize traffic analysis zones using
  population, road, building, and land-use evidence.

## Installation

OpenGeoMove requires Python 3.10 or newer. Install it with:

```bash
python -m pip install opengeomove
```

To install the current checkout instead:

```bash
git clone https://github.com/geoailabs/opengeomove.git
cd opengeomove
python -m pip install .
```

## Quick start

```python
import opengeomove as ogm

dataset = ogm.open("pings.parquet")

print(dataset.columns)
print(dataset.dtypes)
print(dataset.head())

filtered = dataset.apply(
    ogm.processing.HorizontalAccuracyFilter(
        50,
        accuracy_col="horizontal_accuracy_m",
    ),
    ogm.processing.TimeRangeFilter(
        "2026-01-01",
        "2026-02-01",
        "observed_at",
        timezone="Asia/Kolkata",
    ),
)

print(filtered.missing_summary())
```

`ogm.open()` does not copy the complete source into Python memory. Metadata is
computed and cached on first access by default; pass `eager_metadata=True` to
compute it while opening the file. An exact row count can still require a full
source scan, especially for text formats.

### Mobility analysis

Mobility thresholds are supplied by the caller rather than being inferred from
a city or dataset:

```python
thresholds = ogm.MobilityThresholds(
    max_distance_m=100,
    min_stop_duration_s=300,
    max_time_gap_s=1_800,
    trip_max_time_gap_s=7_200,
)

stops = filtered.cluster_stops(
    device_id_col="device_id",
    timestamp_col="observed_at",
    lat_col="latitude",
    lon_col="longitude",
    thresholds=thresholds,
    source_timezone="UTC",
)
trips = stops.form_trips(thresholds=thresholds)
```

### Traffic analysis zones

TAZ operations are available under `ogm.taz`. They accept GeoPandas/Shapely
spatial inputs and return a validated `TAZFrame`, rather than the tabular
DuckDB-backed `Dataset`:

```python
zones = ogm.taz.create_taz(
    study_area,
    config=ogm.taz.TAZConfig(method="hex_grid", min_area_km2=0.5),
)

figure = ogm.taz.plot_taz(zones)
figure.show()
```

Automatic road- and building-aware TAZ generation can download external OSM
and Google Open Buildings data when explicitly configured. Those workflows may
also write caches, diagnostics, and final outputs beneath their configured
output directory. Ordinary dataset loading and analysis remain read-only with
respect to source files.

## Supported input formats

The common `ogm.open()` interface currently supports local Parquet, CSV, TSV,
and JSON files. Spatial vector and raster inputs used by TAZ workflows have
separate, explicit readers; they are not loaded through `ogm.open()`.

Timestamp comparisons are normalized to UTC. Naive source timestamps default
to UTC unless `source_timezone` is supplied, while naive filter bounds use the
caller's `timezone` argument. Daily ping-count groups use UTC calendar days.

## License

OpenGeoMove is licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
