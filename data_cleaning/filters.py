"""Lazy DuckDB-backed filters for large mobility datasets."""

from __future__ import annotations

from datetime import date, datetime, timezone
from itertools import count
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import duckdb

_QUERY_IDS = count()


def _identifier(name: str) -> str:
    """Quote a SQL identifier after validating its public type."""
    if not isinstance(name, str) or not name:
        raise TypeError("column names must be non-empty strings")
    return '"' + name.replace('"', '""') + '"'


def _literal(value: str) -> str:
    """Render a safely escaped SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def _relation_query(relation: duckdb.DuckDBPyRelation, sql: str) -> duckdb.DuckDBPyRelation:
    """Run a SQL query over a relation without materializing it in Python."""
    alias = f"_ogm_filter_source_{next(_QUERY_IDS)}"
    return relation.query(alias, sql.replace("ogm_filter_source", alias))


def _column_type(relation: duckdb.DuckDBPyRelation, column: str) -> str:
    """Return a DuckDB column type or raise for an unknown column."""
    try:
        position = relation.columns.index(column)
    except ValueError as exc:
        raise ValueError(f"Column {column!r} does not exist in this Dataset") from exc
    return str(relation.types[position]).upper()


def _utc_timestamp_expression(
    relation: duckdb.DuckDBPyRelation,
    column: str,
    *,
    source_timezone: str,
    timestamp_unit: str,
    qualifier: str | None = None,
) -> str:
    """Build a SQL expression that interprets the column as an instant in UTC."""
    quoted = f"{qualifier}.{_identifier(column)}" if qualifier else _identifier(column)
    column_type = _column_type(relation, column)
    if column_type in {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "HUGEINT", "UHUGEINT", "FLOAT", "DOUBLE", "DECIMAL"}:
        numeric_unit = timestamp_unit
        if numeric_unit == "auto" and column.lower().endswith("_ms"):
            numeric_unit = "ms"
        divisor = {"ms": "1000.0", "s": "1.0"}.get(numeric_unit)
        if divisor is None:
            raise ValueError("numeric timestamps require timestamp_unit='ms' or 's' (or a column ending in '_ms')")
        return f"to_timestamp(CAST({quoted} AS DOUBLE) / {divisor})"
    if timestamp_unit != "auto":
        raise ValueError("timestamp_unit='ms' or 's' is only valid for numeric timestamp columns")

    if "WITH TIME ZONE" in column_type or column_type == "TIMESTAMPTZ":
        return f"CAST({quoted} AS TIMESTAMPTZ)"
    if column_type.startswith("TIMESTAMP"):
        return f"timezone({_literal(source_timezone)}, CAST({quoted} AS TIMESTAMP))"
    if column_type == "DATE":
        return f"timezone({_literal(source_timezone)}, CAST({quoted} AS TIMESTAMP))"
    # VARCHAR/JSON-style columns can contain either explicit offsets or naive
    # timestamps. Honor offsets when present; interpret naive strings using the
    # caller's explicit source timezone (UTC by default).
    has_offset = f"regexp_matches(trim(CAST({quoted} AS VARCHAR)), '(Z|[+-][0-9]{{2}}:?[0-9]{{2}})$', 'i')"
    return (
        f"CASE WHEN {has_offset} THEN try_cast(CAST({quoted} AS VARCHAR) AS TIMESTAMPTZ) "
        f"ELSE timezone({_literal(source_timezone)}, "
        f"try_cast(CAST({quoted} AS VARCHAR) AS TIMESTAMP)) END"
    )


def _bound_utc(value: Any, timezone_name: str) -> datetime:
    """Convert a date/datetime bound to an aware UTC datetime."""
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid ISO date/time bound: {value!r}") from exc
    if not isinstance(value, datetime):
        raise TypeError("time bounds must be date, datetime, or ISO-8601 strings")
    if value.tzinfo is None:
        try:
            value = value.replace(tzinfo=ZoneInfo(timezone_name))
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown timezone: {timezone_name!r}") from exc
    return value.astimezone(timezone.utc)


def _validate_timezone(timezone_name: str, *, argument: str) -> str:
    """Validate and return an IANA timezone identifier."""
    if not isinstance(timezone_name, str):
        raise TypeError(f"{argument} must be a timezone name string")
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {timezone_name!r}") from exc
    return timezone_name


class _Filter:
    """Base class for immutable filter specifications applied by Dataset."""

    def _apply(
        self,
        relation: duckdb.DuckDBPyRelation,
        connection: duckdb.DuckDBPyConnection,
    ) -> duckdb.DuckDBPyRelation:
        """Return a relation with this filter composed into its SQL plan."""
        raise NotImplementedError


class DeviceIDFilter(_Filter):
    """Keep rows whose device identifier is in a supplied list of IDs.

    Args:
        device_ids: One or more device identifiers to retain.
        device_id_col: Column containing device identifiers.
    """

    def __init__(self, device_ids: list[Any] | tuple[Any, ...], device_id_col: str = "device_id") -> None:
        """Create a filter for a finite set of device identifiers."""
        if isinstance(device_ids, (str, bytes)) or not isinstance(device_ids, (list, tuple)):
            raise TypeError("device_ids must be a list or tuple")
        if not device_ids:
            raise ValueError("device_ids cannot be empty")
        self.device_ids = tuple(device_ids)
        self.device_id_col = device_id_col
        _identifier(device_id_col)

    def _apply(self, relation: duckdb.DuckDBPyRelation, connection: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
        """Compose device membership into the DuckDB relation."""
        _column_type(relation, self.device_id_col)
        values = ", ".join("NULL" if value is None else _literal(str(value)) for value in self.device_ids)
        return _relation_query(relation, f"SELECT * FROM ogm_filter_source WHERE {_identifier(self.device_id_col)} IN ({values})")


class HorizontalAccuracyFilter(_Filter):
    """Keep rows at or below a horizontal-accuracy threshold.

    Missing accuracy values are retained by default, matching the agreed
    Gurgaon cleaning behavior.

    Args:
        max_accuracy_m: Maximum accepted horizontal accuracy in meters.
        accuracy_col: Column containing accuracy values.
        keep_missing: Whether rows with NULL accuracy should be retained.
    """

    def __init__(self, max_accuracy_m: float, accuracy_col: str = "horizontal_accuracy_m", *, keep_missing: bool = True) -> None:
        """Create a threshold filter for horizontal accuracy."""
        if isinstance(max_accuracy_m, bool) or not isinstance(max_accuracy_m, (int, float)):
            raise TypeError("max_accuracy_m must be a number")
        if max_accuracy_m < 0:
            raise ValueError("max_accuracy_m cannot be negative")
        if not isinstance(keep_missing, bool):
            raise TypeError("keep_missing must be a boolean")
        _identifier(accuracy_col)
        self.max_accuracy_m = float(max_accuracy_m)
        self.accuracy_col = accuracy_col
        self.keep_missing = keep_missing

    def _apply(self, relation: duckdb.DuckDBPyRelation, connection: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
        """Compose the accuracy comparison into the DuckDB relation."""
        _column_type(relation, self.accuracy_col)
        col = _identifier(self.accuracy_col)
        predicate = f"{col} <= {self.max_accuracy_m!r}"
        if self.keep_missing:
            predicate = f"({predicate} OR {col} IS NULL)"
        return _relation_query(relation, f"SELECT * FROM ogm_filter_source WHERE {predicate}")


class TimeRangeFilter(_Filter):
    """Keep timestamps in the half-open interval ``[start, end)``.

    Timezone-aware bounds retain their supplied offsets. Naive bounds are
    interpreted in ``timezone``. Naive source timestamps are interpreted in
    ``source_timezone`` (UTC by default); source values with explicit offsets
    are respected. Numeric timestamps require the caller to identify seconds
    or milliseconds.

    Args:
        start: Inclusive lower date/time bound.
        end: Exclusive upper date/time bound.
        timestamp_col: Timestamp or epoch column.
        timezone: Timezone for naive user-provided bounds.
        source_timezone: Timezone for naive source timestamp values.
        timestamp_unit: ``"auto"`` for date/time columns, or ``"ms"``/``"s"``
            for numeric epoch columns.
    """

    def __init__(self, start: Any, end: Any, timestamp_col: str, *, timezone: str = "UTC", source_timezone: str = "UTC", timestamp_unit: str = "auto") -> None:
        """Create a half-open timestamp range filter."""
        self.timezone = _validate_timezone(timezone, argument="timezone")
        self.source_timezone = _validate_timezone(source_timezone, argument="source_timezone")
        if timestamp_unit not in {"auto", "ms", "s"}:
            raise ValueError("timestamp_unit must be 'auto', 'ms', or 's'")
        self.timestamp_unit = timestamp_unit
        _identifier(timestamp_col)
        self.timestamp_col = timestamp_col
        self.start = _bound_utc(start, self.timezone)
        self.end = _bound_utc(end, self.timezone)
        if self.start >= self.end:
            raise ValueError("start must be earlier than end")

    def _apply(self, relation: duckdb.DuckDBPyRelation, connection: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
        """Compose normalized UTC bounds into the DuckDB relation."""
        timestamp = _utc_timestamp_expression(relation, self.timestamp_col, source_timezone=self.source_timezone, timestamp_unit=self.timestamp_unit)
        start = _literal(self.start.isoformat())
        end = _literal(self.end.isoformat())
        sql = f"SELECT * FROM ogm_filter_source WHERE {timestamp} >= CAST({start} AS TIMESTAMPTZ) AND {timestamp} < CAST({end} AS TIMESTAMPTZ)"
        return _relation_query(relation, sql)


class DailyPingCountFilter(_Filter):
    """Retain pings belonging to device/UTC-day groups within count limits.

    Day boundaries are always UTC, independent of local display preferences.
    For a local use case, use timezone-aware time bounds in a separate
    :class:`TimeRangeFilter`; all comparisons are normalized to UTC.

    Args:
        min_pings: Inclusive minimum pings in a device/day group, optional.
        max_pings: Inclusive maximum pings in a device/day group, optional.
        device_ids: Optional device IDs to consider.
        device_id_col: Device identifier column.
        timestamp_col: Timestamp or epoch column.
        source_timezone: Timezone for naive source timestamps (default UTC).
        timestamp_unit: ``"auto"``, ``"ms"``, or ``"s"`` for numeric epochs.
    """

    def __init__(self, *, min_pings: int | None = None, max_pings: int | None = None, device_ids: list[Any] | tuple[Any, ...] | None = None, device_id_col: str = "device_id", timestamp_col: str = "observed_at", source_timezone: str = "UTC", timestamp_unit: str = "auto") -> None:
        """Create a filter for per-device daily ping-count thresholds."""
        if min_pings is None and max_pings is None:
            raise ValueError("at least one of min_pings or max_pings is required")
        for name, value in (("min_pings", min_pings), ("max_pings", max_pings)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")
        if min_pings is not None and max_pings is not None and min_pings > max_pings:
            raise ValueError("min_pings cannot be greater than max_pings")
        if device_ids is not None and (isinstance(device_ids, (str, bytes)) or not isinstance(device_ids, (list, tuple))):
            raise TypeError("device_ids must be a list or tuple when provided")
        if timestamp_unit not in {"auto", "ms", "s"}:
            raise ValueError("timestamp_unit must be 'auto', 'ms', or 's'")
        self.min_pings = min_pings
        self.max_pings = max_pings
        self.device_ids = tuple(device_ids) if device_ids is not None else None
        self.device_id_col = device_id_col
        self.timestamp_col = timestamp_col
        self.source_timezone = _validate_timezone(source_timezone, argument="source_timezone")
        self.timestamp_unit = timestamp_unit
        _identifier(device_id_col)
        _identifier(timestamp_col)

    def _apply(self, relation: duckdb.DuckDBPyRelation, connection: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
        """Aggregate device/UTC-day counts and retain matching source pings."""
        device = _identifier(self.device_id_col)
        _column_type(relation, self.device_id_col)
        timestamp = _utc_timestamp_expression(relation, self.timestamp_col, source_timezone=self.source_timezone, timestamp_unit=self.timestamp_unit)
        conditions: list[str] = []
        if self.device_ids is not None:
            if not self.device_ids:
                raise ValueError("device_ids cannot be empty when provided")
            values = ", ".join("NULL" if value is None else _literal(str(value)) for value in self.device_ids)
            conditions.append(f"{device} IN ({values})")
        having: list[str] = []
        if self.min_pings is not None:
            having.append(f"count(*) >= {self.min_pings}")
        if self.max_pings is not None:
            having.append(f"count(*) <= {self.max_pings}")
        source_where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        utc_day = f"CAST(timezone('UTC', {timestamp}) AS DATE)"
        src_timestamp = _utc_timestamp_expression(
            relation,
            self.timestamp_col,
            source_timezone=self.source_timezone,
            timestamp_unit=self.timestamp_unit,
            qualifier="src",
        )
        src_utc_day = f"CAST(timezone('UTC', {src_timestamp}) AS DATE)"
        sql = (
            f"WITH eligible AS (SELECT {device} AS device_key, {utc_day} AS utc_day "
            f"FROM ogm_filter_source {source_where} GROUP BY {device}, utc_day "
            f"HAVING {' AND '.join(having)}) "
            f"SELECT src.* FROM ogm_filter_source AS src WHERE EXISTS ("
            f"SELECT 1 FROM eligible AS e WHERE src.{device} = e.device_key "
            f"AND {src_utc_day} = e.utc_day)"
        )
        return _relation_query(relation, sql)


class PolygonFilter(_Filter):
    """Keep points inside or on the boundary of a supplied polygon.

    Polygon-like GeoPandas objects may be a GeoDataFrame, GeoSeries, or
    Shapely geometry. Inputs carrying a CRS are reprojected to EPSG:4326;
    geometry without CRS metadata is assumed to use longitude/latitude in
    WGS84. DuckDB's spatial extension is loaded on first use.

    Args:
        polygon: A polygon geometry, GeoSeries, or GeoDataFrame.
        latitude_col: Column containing latitude (Y).
        longitude_col: Column containing longitude (X).
    """

    def __init__(self, polygon: Any, *, latitude_col: str, longitude_col: str) -> None:
        """Create a boundary-inclusive spatial filter."""
        _identifier(latitude_col)
        _identifier(longitude_col)
        self.polygon = polygon
        self.latitude_col = latitude_col
        self.longitude_col = longitude_col

    def _polygon_wkb(self) -> bytes:
        """Extract polygon WKB, reprojecting CRS-aware inputs to WGS84."""
        geometry = self.polygon
        crs = getattr(geometry, "crs", None)
        if hasattr(geometry, "geometry") and not hasattr(geometry, "wkb"):
            geometries = geometry.geometry
            if len(geometries) != 1:
                raise ValueError("polygon input must contain exactly one geometry")
            geometry = geometries.iloc[0]
        elif hasattr(geometry, "iloc") and not hasattr(geometry, "wkb"):
            if len(geometry) != 1:
                raise ValueError("polygon input must contain exactly one geometry")
            geometry = geometry.iloc[0]
        if crs is not None and hasattr(self.polygon, "to_crs"):
            if hasattr(self.polygon, "geometry"):
                projected = self.polygon.to_crs("EPSG:4326")
                geometry = projected.geometry.iloc[0]
            else:
                projected = self.polygon.to_crs("EPSG:4326")
                geometry = projected.iloc[0]
        if not hasattr(geometry, "wkb"):
            raise TypeError("polygon must be a Shapely geometry or GeoPandas object")
        if getattr(geometry, "is_empty", False):
            raise ValueError("polygon cannot be empty")
        return bytes(geometry.wkb)

    def _apply(self, relation: duckdb.DuckDBPyRelation, connection: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyRelation:
        """Compose the boundary-inclusive DuckDB spatial predicate."""
        _column_type(relation, self.latitude_col)
        _column_type(relation, self.longitude_col)
        try:
            connection.execute("LOAD spatial")
        except duckdb.Error:
            connection.execute("INSTALL spatial")
            connection.execute("LOAD spatial")
        polygon_hex = self._polygon_wkb().hex()
        geom = f"ST_GeomFromWKB(from_hex('{polygon_hex}'))"
        point = f"ST_Point(TRY_CAST({_identifier(self.longitude_col)} AS DOUBLE), TRY_CAST({_identifier(self.latitude_col)} AS DOUBLE))"
        return _relation_query(relation, f"SELECT * FROM ogm_filter_source WHERE ST_Covers({geom}, {point})")
