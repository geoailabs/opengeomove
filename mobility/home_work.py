"""Recurring home and work location classification from stop summaries."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import math
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from opengeomove.data_cleaning.filters import _validate_timezone
from opengeomove.mobility.operations import (
    _FETCH_BATCH_SIZE,
    _OUTPUT_IDS,
    _distance_m,
    _positive_number,
    _quote_identifier,
    _require_columns,
    _result_dataset,
    _spherical_centroid,
    _write_rows,
)

if TYPE_CHECKING:
    from opengeomove.data_loading.dataset import Dataset

_WRITE_BATCH_SIZE = 2048


def _parse_window(window: tuple[str, str], name: str) -> tuple[time, time]:
    """Validate and parse a pair of local-time window boundaries."""
    if not isinstance(window, tuple) or len(window) != 2:
        raise TypeError(f"{name} must be a pair of 'HH:MM' strings")
    parsed: list[time] = []
    for value in window:
        if not isinstance(value, str):
            raise TypeError(f"{name} boundaries must be 'HH:MM' strings")
        if len(value) != 5:
            raise ValueError(f"Invalid {name} boundary {value!r}; expected HH:MM")
        try:
            parsed.append(time.fromisoformat(value))
        except ValueError as exc:
            raise ValueError(f"Invalid {name} boundary {value!r}; expected HH:MM") from exc
        if parsed[-1].second or parsed[-1].microsecond or parsed[-1].tzinfo:
            raise ValueError(f"{name} boundaries must use local HH:MM values")
    if parsed[0] == parsed[1]:
        raise ValueError(f"{name} start and end must differ")
    return parsed[0], parsed[1]


def _window_evidence(
    start_utc: datetime,
    end_utc: datetime,
    *,
    window: tuple[time, time],
    zone: ZoneInfo,
    allowed_weekdays: set[int] | None,
) -> tuple[set[date], float]:
    """Return local window days and overlapping dwell seconds."""
    local_start = start_utc.astimezone(zone)
    local_end = end_utc.astimezone(zone)
    anchor = local_start.date() - timedelta(days=1)
    last_day = local_end.date()
    start_clock, end_clock = window
    days: set[date] = set()
    dwell = 0.0
    while anchor <= last_day:
        if allowed_weekdays is None or anchor.weekday() in allowed_weekdays:
            window_start = datetime.combine(anchor, start_clock, zone)
            end_day = anchor + timedelta(days=1) if end_clock <= start_clock else anchor
            window_end = datetime.combine(end_day, end_clock, zone)
            overlap_start = max(start_utc, window_start.astimezone(timezone.utc))
            overlap_end = min(end_utc, window_end.astimezone(timezone.utc))
            seconds = (overlap_end - overlap_start).total_seconds()
            if seconds > 0:
                days.add(anchor)
                dwell += seconds
        anchor += timedelta(days=1)
    return days, dwell


def classify_home_work(
    stops: Dataset,
    *,
    timezone_name: str,
    min_visits: int,
    min_distinct_days: int,
    min_total_dwell_s: float,
    merge_nearby: bool = True,
    merge_distance_m: float | None = None,
    home_window: tuple[str, str] = ("20:00", "06:00"),
    work_window: tuple[str, str] = ("09:00", "17:00"),
    work_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> Dataset:
    """Classify each device's recurring home and work stop locations.

    Stop intervals are evaluated against local-time home and work windows.
    By default, nearby stop summaries are merged using the radius recorded by
    :meth:`Dataset.cluster_stops`; callers can override the radius or keep
    every stop as a separate location. For each role, the leading candidate is
    classified only when it meets all evidence minimums and is not tied with
    another candidate. Otherwise it is returned with ``uncertain`` status.
    The operation streams stop rows in bounded batches, retaining aggregated
    location candidates rather than source pings in Python memory.

    Args:
        stops: Stop-summary Dataset, normally returned by ``cluster_stops``.
        timezone_name: IANA timezone used to interpret the daily windows.
        min_visits: Minimum number of stop visits required for classification.
        min_distinct_days: Minimum number of local window days with evidence.
        min_total_dwell_s: Minimum dwell seconds within the role's windows.
        merge_nearby: Merge stops within ``merge_distance_m``. Defaults true.
        merge_distance_m: Merge radius in meters. Defaults to the radius used
            by ``cluster_stops`` when available.
        home_window: Local-time home window as ``(start, end)`` in HH:MM;
            the default is 20:00 through 06:00 and crosses midnight.
        work_window: Local-time work window as ``(start, end)`` in HH:MM.
        work_weekdays: Weekdays eligible for work evidence, Monday=0.

    Returns:
        A Dataset with at most two rows per device, one per role. Each row
        includes candidate coordinates, classification status, candidate
        count, visit count, distinct days, and dwell seconds. Uncertain rows
        retain the leading candidate and its evidence when one exists; rows
        with no evidence have null coordinates and zero evidence.

    Raises:
        TypeError: If the Dataset or an argument has an invalid type.
        ValueError: If required columns, timezone, windows, weekdays, or
            evidence thresholds are invalid.
    """
    from opengeomove.data_loading.dataset import Dataset

    if not isinstance(stops, Dataset):
        raise TypeError("stops must be an opengeomove.Dataset")
    _require_columns(
        stops,
        ("cluster_id", "device_id", "start_time", "end_time", "latitude", "longitude"),
    )
    timezone_name = _validate_timezone(timezone_name, argument="timezone_name")
    for name, value in (
        ("min_visits", min_visits),
        ("min_distinct_days", min_distinct_days),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive integer")
        if value < 1:
            raise ValueError(f"{name} must be a positive integer")
    min_total_dwell_s = _positive_number(min_total_dwell_s, "min_total_dwell_s")
    if not isinstance(merge_nearby, bool):
        raise TypeError("merge_nearby must be a bool")
    if merge_distance_m is not None:
        merge_distance_m = _positive_number(
            merge_distance_m, "merge_distance_m", allow_zero=True
        )
    elif merge_nearby:
        merge_distance_m = getattr(stops, "_stop_merge_distance_m", None)
        if merge_distance_m is None:
            raise ValueError(
                "merge_distance_m is required when stops do not record their clustering radius"
            )
    home_window = _parse_window(home_window, "home_window")
    work_window = _parse_window(work_window, "work_window")
    if not isinstance(work_weekdays, tuple) or any(
        isinstance(day, bool) or not isinstance(day, int) or day < 0 or day > 6
        for day in work_weekdays
    ):
        raise ValueError("work_weekdays must be a tuple of weekday numbers from 0 to 6")
    if len(set(work_weekdays)) != len(work_weekdays):
        raise ValueError("work_weekdays cannot contain duplicates")

    alias = f"_ogm_home_work_source_{next(_OUTPUT_IDS)}"
    ordered = stops._relation.query(
        alias,
        f"SELECT cluster_id, device_id, start_time, end_time, latitude, longitude "
        f"FROM {alias} WHERE device_id IS NOT NULL AND start_time IS NOT NULL "
        "AND end_time IS NOT NULL AND latitude IS NOT NULL AND longitude IS NOT NULL "
        "ORDER BY device_id, start_time, cluster_id",
    )
    device_type = str(stops._relation.types[stops._relation.columns.index("device_id")])
    table_name = f"_ogm_home_work_{next(_OUTPUT_IDS)}"
    columns = (
        "device_id",
        "role",
        "latitude",
        "longitude",
        "classification_status",
        "candidate_count",
        "visit_count",
        "distinct_days",
        "total_dwell_s",
    )
    schema = (
        f"device_id {device_type}, role VARCHAR, latitude DOUBLE, longitude DOUBLE, "
        "classification_status VARCHAR, candidate_count BIGINT, visit_count BIGINT, "
        "distinct_days BIGINT, total_dwell_s DOUBLE"
    )
    stops._connection.execute(
        f"CREATE TEMP TABLE {_quote_identifier(table_name)} ({schema})"
    )
    zone = ZoneInfo(timezone_name)
    weekdays = set(work_weekdays)
    candidates: dict[Any, list[dict[str, Any]]] = {}
    output_rows: list[tuple[Any, ...]] = []

    def evidence_for(
        start: datetime, end: datetime, role: str
    ) -> tuple[set[date], float]:
        if role == "home":
            return _window_evidence(
                start, end, window=home_window, zone=zone, allowed_weekdays=None
            )
        return _window_evidence(
            start, end, window=work_window, zone=zone, allowed_weekdays=weekdays
        )

    def update_centroid(candidate: dict[str, Any], latitude: float, longitude: float) -> None:
        """Add a coordinate to spherical centroid accumulators."""
        latitude_radians = math.radians(latitude)
        longitude_radians = math.radians(longitude)
        candidate["x_sum"] += math.cos(latitude_radians) * math.cos(longitude_radians)
        candidate["y_sum"] += math.cos(latitude_radians) * math.sin(longitude_radians)
        candidate["z_sum"] += math.sin(latitude_radians)
        candidate["latitude"], candidate["longitude"] = _spherical_centroid(
            candidate["x_sum"], candidate["y_sum"], candidate["z_sum"],
            (latitude, longitude),
        )

    def flush_device(device: Any) -> None:
        """Write the best home and work candidate rows for one device."""
        locations = candidates.get(device, [])
        for role in ("home", "work"):
            ranked = sorted(
                locations,
                key=lambda item: (
                    item[role]["days"], item[role]["visits"], item[role]["dwell"]
                ),
                reverse=True,
            )
            if not ranked or ranked[0][role]["visits"] == 0:
                output_rows.append(
                    (device, role, None, None, "uncertain", len(ranked), 0, 0, 0.0)
                )
                continue
            top = ranked[0]
            metrics = top[role]
            tied = len(ranked) > 1 and (
                metrics["days"], metrics["visits"], metrics["dwell"]
            ) == (
                ranked[1][role]["days"],
                ranked[1][role]["visits"],
                ranked[1][role]["dwell"],
            )
            enough = (
                metrics["visits"] >= min_visits
                and metrics["days"] >= min_distinct_days
                and metrics["dwell"] >= min_total_dwell_s
            )
            status = "classified" if enough and not tied else "uncertain"
            output_rows.append(
                (
                    device,
                    role,
                    top["latitude"],
                    top["longitude"],
                    status,
                    len(ranked),
                    metrics["visits"],
                    metrics["days"],
                    metrics["dwell"],
                )
            )
        if len(output_rows) >= _WRITE_BATCH_SIZE:
            _write_rows(stops._connection, table_name, output_rows, len(columns))

    try:
        current_device: Any = None
        have_device = False
        while batch := ordered.fetchmany(_FETCH_BATCH_SIZE):
            for _cluster_id, device, start_value, end_value, latitude, longitude in batch:
                if not have_device or device != current_device:
                    if have_device:
                        flush_device(current_device)
                        candidates.pop(current_device, None)
                    current_device = device
                    have_device = True
                    candidates[device] = []
                start = (
                    start_value.replace(tzinfo=timezone.utc)
                    if start_value.tzinfo is None
                    else start_value.astimezone(timezone.utc)
                )
                end = (
                    end_value.replace(tzinfo=timezone.utc)
                    if end_value.tzinfo is None
                    else end_value.astimezone(timezone.utc)
                )
                if end <= start:
                    continue
                latitude = float(latitude)
                longitude = float(longitude)
                if not math.isfinite(latitude) or not math.isfinite(longitude):
                    continue
                if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                    continue
                location_list = candidates[device]
                match = None
                if merge_nearby:
                    for candidate in location_list:
                        if _distance_m(
                            candidate["latitude"],
                            candidate["longitude"],
                            latitude,
                            longitude,
                        ) <= merge_distance_m:
                            match = candidate
                            break
                if match is None:
                    match = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "x_sum": 0.0,
                        "y_sum": 0.0,
                        "z_sum": 0.0,
                        "home": {"visits": 0, "days": 0, "day_set": set(), "dwell": 0.0},
                        "work": {"visits": 0, "days": 0, "day_set": set(), "dwell": 0.0},
                    }
                    location_list.append(match)
                update_centroid(match, latitude, longitude)
                for role in ("home", "work"):
                    overlapping_days, dwell = evidence_for(start, end, role)
                    if overlapping_days:
                        metric = match[role]
                        metric["visits"] += 1
                        metric["day_set"].update(overlapping_days)
                        metric["days"] = len(metric["day_set"])
                        metric["dwell"] += dwell
        if have_device:
            flush_device(current_device)
        _write_rows(stops._connection, table_name, output_rows, len(columns))
    except Exception:
        stops._connection.execute(f"DROP TABLE IF EXISTS {_quote_identifier(table_name)}")
        raise
    return _result_dataset(stops, table_name)
