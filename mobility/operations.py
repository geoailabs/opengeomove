"""Per-device stop clustering and trip-summary operations."""

from __future__ import annotations

from datetime import datetime, timezone
import itertools
import math
from typing import TYPE_CHECKING, Any

import duckdb

from opengeomove.data_cleaning.filters import _utc_timestamp_expression, _validate_timezone

if TYPE_CHECKING:
    from opengeomove.data_loading.dataset import Dataset

_OUTPUT_IDS = itertools.count()
_FETCH_BATCH_SIZE = 8192
_WRITE_BATCH_SIZE = 2048
_EARTH_RADIUS_M = 6_371_008.8


def _quote_identifier(name: str) -> str:
    """Safely quote a SQL identifier."""
    if not isinstance(name, str) or not name:
        raise TypeError("column names must be non-empty strings")
    return '"' + name.replace('"', '""') + '"'


def _require_columns(dataset: Dataset, columns: tuple[str, ...]) -> None:
    """Raise if any named columns are absent from a Dataset."""
    missing = [name for name in columns if name not in dataset.columns]
    if missing:
        raise ValueError(f"Columns do not exist in this Dataset: {missing!r}")


def _positive_number(value: object, name: str, *, allow_zero: bool = False) -> float:
    """Validate and normalize a positive finite distance or duration."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        qualifier = "non-negative" if allow_zero else "positive"
        raise TypeError(f"{name} must be a {qualifier} number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return value


def _utc_datetime(value: Any) -> datetime:
    """Normalize a DuckDB timestamp value to an aware UTC datetime."""
    if not isinstance(value, datetime):
        raise TypeError("timestamp values must resolve to a date or timestamp")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in meters using the haversine formula."""
    lat1_radians = math.radians(lat1)
    lat2_radians = math.radians(lat2)
    delta_latitude = lat2_radians - lat1_radians
    delta_longitude = math.radians(lon2 - lon1)
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(lat1_radians)
        * math.cos(lat2_radians)
        * math.sin(delta_longitude / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(haversine)))


def _timestamp_seconds(value: datetime) -> float:
    """Return an aware datetime's Unix timestamp in seconds."""
    return _utc_datetime(value).timestamp()


def _create_result_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    schema: str,
) -> None:
    """Create a temporary DuckDB table for operation output."""
    connection.execute(f"CREATE TEMP TABLE {_quote_identifier(table_name)} ({schema})")


def _write_rows(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    rows: list[tuple[Any, ...]],
    column_count: int,
) -> None:
    """Append a bounded buffer of output rows to a temporary table."""
    if not rows:
        return
    placeholders = ", ".join("?" for _ in range(column_count))
    connection.executemany(
        f"INSERT INTO {_quote_identifier(table_name)} VALUES ({placeholders})", rows
    )
    rows.clear()


def _result_dataset(source: Dataset, table_name: str) -> Dataset:
    """Wrap a temporary result table in a Dataset sharing the source connection."""
    from opengeomove.data_loading.dataset import Dataset

    result = Dataset(
        relation=source._connection.table(table_name),
        connection=source._connection,
        source_path=source.source,
        data_format=source.format,
    )
    result._filters = source.filters
    return result


def cluster_stops(
    dataset: Dataset,
    *,
    device_id_col: str,
    timestamp_col: str,
    lat_col: str,
    lon_col: str,
    max_distance_m: float,
    min_stop_duration_s: float,
    max_time_gap_s: float,
    source_timezone: str = "UTC",
    timestamp_unit: str = "auto",
) -> Dataset:
    """Cluster each device's consecutive pings into stop summaries.

    This is a sequential staypoint detector. Pings are ordered per device by
    normalized UTC timestamp. A candidate stop continues while each ping stays
    within ``max_distance_m`` of the candidate's first ping and consecutive
    pings are no more than ``max_time_gap_s`` apart. A candidate is returned
    when its observed duration is at least ``min_stop_duration_s``. Invalid
    coordinates, null device IDs, and unparseable timestamps are ignored.

    Args:
        dataset: Source or filtered Dataset containing device pings.
        device_id_col: Column containing per-device identifiers.
        timestamp_col: Timestamp or numeric epoch column.
        lat_col: Latitude column in WGS84 degrees.
        lon_col: Longitude column in WGS84 degrees.
        max_distance_m: Maximum distance from the first ping in a stop
            candidate, in meters.
        min_stop_duration_s: Minimum observed candidate duration, in seconds.
        max_time_gap_s: A larger gap between consecutive pings ends a
            candidate, in seconds.
        source_timezone: Timezone for naive source timestamps. Defaults to UTC.
        timestamp_unit: ``"auto"``, ``"ms"``, or ``"s"`` for numeric epochs.

    Returns:
        A Dataset with one row per detected stop and columns ``cluster_id``,
        ``device_id``, ``start_time``, ``end_time``, ``latitude``,
        ``longitude``, ``ping_count``, and ``duration_s``. Results are stored
        in a temporary DuckDB table tied to the source Dataset's connection.

    Raises:
        TypeError: If a Dataset, column name, or threshold has an invalid type.
        ValueError: If columns are missing, thresholds are invalid, or a
            numeric timestamp lacks a supported unit.
        duckdb.Error: If DuckDB cannot order or read the input relation.
    """
    from opengeomove.data_loading.dataset import Dataset

    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be an opengeomove.Dataset")
    for name, value in (
        ("device_id_col", device_id_col),
        ("timestamp_col", timestamp_col),
        ("lat_col", lat_col),
        ("lon_col", lon_col),
    ):
        _quote_identifier(value)
    _require_columns(dataset, (device_id_col, timestamp_col, lat_col, lon_col))
    max_distance_m = _positive_number(max_distance_m, "max_distance_m")
    min_stop_duration_s = _positive_number(
        min_stop_duration_s, "min_stop_duration_s", allow_zero=True
    )
    max_time_gap_s = _positive_number(max_time_gap_s, "max_time_gap_s")
    source_timezone = _validate_timezone(source_timezone, argument="source_timezone")

    source_relation = dataset._relation
    device = _quote_identifier(device_id_col)
    latitude = _quote_identifier(lat_col)
    longitude = _quote_identifier(lon_col)
    timestamp = _utc_timestamp_expression(
        source_relation,
        timestamp_col,
        source_timezone=source_timezone,
        timestamp_unit=timestamp_unit,
    )
    alias = f"_ogm_stop_source_{next(_OUTPUT_IDS)}"
    query = (
        f"SELECT {device} AS device_key, {timestamp} AS observed_at, "
        f"TRY_CAST({latitude} AS DOUBLE) AS latitude, "
        f"TRY_CAST({longitude} AS DOUBLE) AS longitude "
        f"FROM {alias} WHERE {device} IS NOT NULL "
        f"AND {timestamp} IS NOT NULL "
        f"AND isfinite(TRY_CAST({latitude} AS DOUBLE)) "
        f"AND isfinite(TRY_CAST({longitude} AS DOUBLE)) "
        f"AND TRY_CAST({latitude} AS DOUBLE) BETWEEN -90 AND 90 "
        f"AND TRY_CAST({longitude} AS DOUBLE) BETWEEN -180 AND 180 "
        "ORDER BY device_key, observed_at, latitude, longitude"
    )
    ordered = source_relation.query(alias, query)

    device_type = str(
        source_relation.types[source_relation.columns.index(device_id_col)]
    )
    table_name = f"_ogm_stop_clusters_{next(_OUTPUT_IDS)}"
    schema = (
        f"cluster_id BIGINT, device_id {device_type}, start_time TIMESTAMPTZ, "
        "end_time TIMESTAMPTZ, latitude DOUBLE, longitude DOUBLE, "
        "ping_count BIGINT, duration_s DOUBLE"
    )
    _create_result_table(dataset._connection, table_name, schema)
    output_rows: list[tuple[Any, ...]] = []
    cluster_id = 0
    candidate: dict[str, Any] | None = None

    def finish_candidate() -> None:
        """Emit the current candidate when it meets the duration threshold."""
        nonlocal cluster_id, candidate
        if candidate is None:
            return
        duration = _timestamp_seconds(candidate["last_time"]) - _timestamp_seconds(
            candidate["start_time"]
        )
        if duration >= min_stop_duration_s:
            cluster_id += 1
            output_rows.append(
                (
                    cluster_id,
                    candidate["device"],
                    _utc_datetime(candidate["start_time"]),
                    _utc_datetime(candidate["last_time"]),
                    candidate["latitude_sum"] / candidate["ping_count"],
                    candidate["longitude_sum"] / candidate["ping_count"],
                    candidate["ping_count"],
                    duration,
                )
            )
            if len(output_rows) >= _WRITE_BATCH_SIZE:
                _write_rows(dataset._connection, table_name, output_rows, 8)
        candidate = None

    def start_candidate(device_value: Any, observed_at: datetime, lat: float, lon: float) -> dict[str, Any]:
        """Start a stop candidate at one ping."""
        return {
            "device": device_value,
            "start_time": observed_at,
            "last_time": observed_at,
            "anchor_latitude": lat,
            "anchor_longitude": lon,
            "latitude_sum": lat,
            "longitude_sum": lon,
            "ping_count": 1,
        }

    try:
        while batch := ordered.fetchmany(_FETCH_BATCH_SIZE):
            for device_value, observed_at, lat, lon in batch:
                if observed_at is None or lat is None or lon is None:
                    continue
                observed_at = _utc_datetime(observed_at)
                lat = float(lat)
                lon = float(lon)
                if candidate is None:
                    candidate = start_candidate(device_value, observed_at, lat, lon)
                    continue
                gap = _timestamp_seconds(observed_at) - _timestamp_seconds(
                    candidate["last_time"]
                )
                same_device = device_value == candidate["device"]
                within_gap = gap <= max_time_gap_s
                within_radius = (
                    same_device
                    and _distance_m(
                        candidate["anchor_latitude"],
                        candidate["anchor_longitude"],
                        lat,
                        lon,
                    )
                    <= max_distance_m
                )
                if not same_device or not within_gap or not within_radius:
                    finish_candidate()
                    candidate = start_candidate(device_value, observed_at, lat, lon)
                else:
                    candidate["last_time"] = observed_at
                    candidate["latitude_sum"] += lat
                    candidate["longitude_sum"] += lon
                    candidate["ping_count"] += 1
        finish_candidate()
        _write_rows(dataset._connection, table_name, output_rows, 8)
    except Exception:
        dataset._connection.execute(f"DROP TABLE IF EXISTS {_quote_identifier(table_name)}")
        raise
    return _result_dataset(dataset, table_name)


def form_trips(stops: Dataset, *, max_time_gap_s: float) -> Dataset:
    """Form trip summaries between consecutive stops for each device.

    Consecutive stop rows are paired per ``device_id`` in start-time order. A
    trip is emitted when the time between the preceding stop's end and the
    next stop's start is positive and no greater than ``max_time_gap_s``.

    Args:
        stops: Stop-summary Dataset returned by :meth:`Dataset.cluster_stops`.
        max_time_gap_s: Maximum allowed time between consecutive stops, in
            seconds. Larger gaps do not form a trip.

    Returns:
        A Dataset with one row per formed trip and columns ``trip_id``,
        ``device_id``, ``origin_cluster_id``, ``destination_cluster_id``,
        ``start_time``, ``end_time``, origin and destination coordinates,
        ``duration_s``, and great-circle ``distance_m``. Results are stored in
        a temporary DuckDB table tied to the source connection.

    Raises:
        TypeError: If ``stops`` is not a Dataset or the threshold is invalid.
        ValueError: If the Dataset lacks the required stop-summary columns.
    """
    from opengeomove.data_loading.dataset import Dataset

    if not isinstance(stops, Dataset):
        raise TypeError("stops must be an opengeomove.Dataset")
    max_time_gap_s = _positive_number(max_time_gap_s, "max_time_gap_s")
    required = (
        "cluster_id",
        "device_id",
        "start_time",
        "end_time",
        "latitude",
        "longitude",
    )
    _require_columns(stops, required)

    alias = f"_ogm_trip_source_{next(_OUTPUT_IDS)}"
    ordered = stops._relation.query(
        alias,
        f"SELECT cluster_id, device_id, start_time, end_time, latitude, longitude "
        f"FROM {alias} WHERE start_time IS NOT NULL AND end_time IS NOT NULL "
        "AND latitude IS NOT NULL AND longitude IS NOT NULL "
        "ORDER BY device_id, start_time, cluster_id",
    )
    device_type = str(stops._relation.types[stops._relation.columns.index("device_id")])
    table_name = f"_ogm_trips_{next(_OUTPUT_IDS)}"
    schema = (
        f"trip_id BIGINT, device_id {device_type}, origin_cluster_id BIGINT, "
        "destination_cluster_id BIGINT, start_time TIMESTAMPTZ, end_time TIMESTAMPTZ, "
        "origin_latitude DOUBLE, origin_longitude DOUBLE, "
        "destination_latitude DOUBLE, destination_longitude DOUBLE, "
        "duration_s DOUBLE, distance_m DOUBLE"
    )
    _create_result_table(stops._connection, table_name, schema)
    output_rows: list[tuple[Any, ...]] = []
    trip_id = 0
    previous: tuple[Any, ...] | None = None

    try:
        while batch := ordered.fetchmany(_FETCH_BATCH_SIZE):
            for row in batch:
                cluster_id, device_id, start_time, end_time, latitude, longitude = row
                start_time = _utc_datetime(start_time)
                end_time = _utc_datetime(end_time)
                current = (
                    cluster_id,
                    device_id,
                    start_time,
                    end_time,
                    float(latitude),
                    float(longitude),
                )
                if previous is not None and previous[1] == device_id:
                    duration = _timestamp_seconds(start_time) - _timestamp_seconds(
                        previous[3]
                    )
                    if 0 < duration <= max_time_gap_s:
                        trip_id += 1
                        output_rows.append(
                            (
                                trip_id,
                                device_id,
                                previous[0],
                                cluster_id,
                                previous[3],
                                start_time,
                                previous[4],
                                previous[5],
                                current[4],
                                current[5],
                                duration,
                                _distance_m(previous[4], previous[5], current[4], current[5]),
                            )
                        )
                        if len(output_rows) >= _WRITE_BATCH_SIZE:
                            _write_rows(stops._connection, table_name, output_rows, 12)
                previous = current
        _write_rows(stops._connection, table_name, output_rows, 12)
    except Exception:
        stops._connection.execute(f"DROP TABLE IF EXISTS {_quote_identifier(table_name)}")
        raise
    return _result_dataset(stops, table_name)
