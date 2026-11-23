"""Weather-to-TAZ aggregation operations."""

from __future__ import annotations

from datetime import date as date_type, datetime, time, timedelta, timezone as utc_timezone
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import geopandas as gpd
import numpy as np
import pandas as pd

from opengeomove.data_loading.weather import OpenMeteoHistoricalSource
from opengeomove.taz.exceptions import TAZGeometryError
from opengeomove.taz.models import TAZFrame
from opengeomove.weather.exceptions import WeatherDataError
from opengeomove.weather.models import (
    VARIABLES,
    WeatherOverlay,
    WeatherSource,
    resolve_variable,
)


def _validate_timezone(timezone_name: str) -> ZoneInfo:
    """Validate an IANA timezone and return its ZoneInfo object."""
    if not isinstance(timezone_name, str) or not timezone_name.strip():
        raise TypeError("timezone must be an IANA timezone string")
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {timezone_name!r}") from exc


def _parse_date(value: Any, timezone_name: str) -> date_type:
    """Normalize a date-like value to a local calendar date."""
    zone = _validate_timezone(timezone_name)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.date()
        return value.astimezone(zone).date()
    if isinstance(value, date_type):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return date_type.fromisoformat(text)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("date must be an ISO date or datetime") from exc
            if parsed.tzinfo is None:
                return parsed.date()
            return parsed.astimezone(zone).date()
    raise TypeError("date must be a date, datetime, or ISO-8601 string")


def _metric_zones(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return zones in a metric CRS for area metadata calculation."""
    units = {
        (axis.unit_name or "").lower()
        for axis in getattr(zones.crs, "axis_info", ())
    }
    if (
        not getattr(zones.crs, "is_geographic", False)
        and units
        and units <= {"metre", "meter"}
    ):
        return zones.copy()
    metric_crs = zones.estimate_utm_crs()
    if metric_crs is None:
        raise TAZGeometryError("Could not estimate a metric CRS for weather zones")
    return zones.to_crs(metric_crs)


def _normalize_zones(
    zones: TAZFrame | gpd.GeoDataFrame,
    id_column: str,
) -> TAZFrame:
    """Copy and validate arbitrary polygon TAZ input."""
    if isinstance(zones, TAZFrame):
        data = zones.to_geopandas()
        metadata = zones.metadata
    elif isinstance(zones, gpd.GeoDataFrame):
        data = zones.copy()
        metadata = {}
    else:
        raise TypeError("zones must be a TAZFrame or GeoDataFrame")
    if not isinstance(id_column, str) or not id_column.strip():
        raise TypeError("id_column must be a non-empty string")
    if data.empty or data.crs is None:
        raise TAZGeometryError("Weather zones must be non-empty and have a CRS")
    if id_column not in data:
        raise ValueError(f"Zone ID column {id_column!r} is missing")
    if data[id_column].isna().any() or data[id_column].astype(str).duplicated().any():
        raise ValueError(f"Zone ID column {id_column!r} must be non-null and unique")
    data["taz_id"] = data[id_column].astype(str)
    metric = _metric_zones(data)
    data["area_km2"] = metric.geometry.area.to_numpy(dtype=float) / 1_000_000
    return TAZFrame(data, {**metadata, "source_id_column": id_column})


def _requested_variables(
    variable: str | None,
    variables: Iterable[str] | None,
) -> tuple[str, ...]:
    """Normalize one or more requested variable names."""
    if variable is not None and variables is not None:
        raise TypeError("Pass either variable or variables, not both")
    if variable is not None:
        requested: Sequence[str] = (variable,)
    elif variables is None:
        requested = ("precipitation_sum",)
    elif isinstance(variables, str):
        requested = (variables,)
    else:
        try:
            requested = tuple(variables)
        except TypeError as exc:
            raise TypeError("variables must be an iterable of strings") from exc
    if not requested:
        raise ValueError("At least one weather variable is required")
    canonical = tuple(resolve_variable(item).name for item in requested)
    if len(set(canonical)) != len(canonical):
        raise ValueError("Weather variables cannot contain duplicates")
    return canonical


def _source_instance(source: WeatherSource | str | None) -> WeatherSource:
    """Resolve the built-in provider name or validate a custom source."""
    if source is None:
        return OpenMeteoHistoricalSource()
    if isinstance(source, str):
        if source in {"open_meteo", "open-meteo"}:
            return OpenMeteoHistoricalSource()
        raise ValueError(f"Unsupported weather source {source!r}")
    fetch = getattr(source, "fetch", None)
    if not callable(fetch):
        raise TypeError("source must be a provider name or an object with fetch()")
    return source


def overlay(
    zones: TAZFrame | gpd.GeoDataFrame,
    date: Any,
    *,
    variable: str | None = None,
    variables: Iterable[str] | None = None,
    source: WeatherSource | str | None = None,
    timezone: str = "UTC",
    spatial_method: str = "centroid",
    id_column: str = "taz_id",
) -> WeatherOverlay:
    """Fetch daily weather and attach it to every TAZ polygon.

    The default provider requests one representative point inside each TAZ.
    This is a point-sampled overlay, not an area-weighted average; provider
    coordinates and sampling metadata are retained in the result metadata.

    Args:
        zones: Validated TAZFrame or polygon GeoDataFrame. A GeoDataFrame may
            use a caller-specific identifier selected by ``id_column``.
        date: Local calendar date as a ``date``, ``datetime``, or ISO string.
        variable: One canonical variable or short alias. Defaults to daily
            precipitation when neither ``variable`` nor ``variables`` is set.
        variables: Optional iterable of multiple canonical variables or aliases.
        source: ``"open_meteo"``/``"open-meteo"`` or an object implementing
            the :class:`~opengeomove.weather.WeatherSource` protocol.
        timezone: IANA timezone defining the requested local day.
        spatial_method: Currently ``"centroid"`` or ``"representative_point"``.
        id_column: Input zone identifier column.

    Returns:
        A new :class:`WeatherOverlay`; the supplied zones are not modified.

    Raises:
        TypeError: If inputs or provider arguments have unsupported types.
        ValueError: If the date, timezone, variables, or spatial method is invalid.
        TAZGeometryError: If the zone layer is invalid.
        WeatherDataError: If the provider returns incomplete or invalid data.
    """
    if spatial_method not in {"centroid", "representative_point"}:
        raise ValueError(
            "spatial_method must be 'centroid' or 'representative_point'; "
            "area-weighted gridded sources are not available yet"
        )
    local_date = _parse_date(date, timezone)
    canonical_variables = _requested_variables(variable, variables)
    normalized = _normalize_zones(zones, id_column)
    source_instance = _source_instance(source)
    frame = normalized.to_geopandas()
    representative = frame.geometry.representative_point()
    points = gpd.GeoDataFrame(
        {
            "_weather_point_id": np.arange(len(frame), dtype=np.int64),
            "taz_id": frame.taz_id.to_numpy(),
        },
        geometry=representative.to_numpy(),
        crs=frame.crs,
    ).to_crs("EPSG:4326")
    field = source_instance.fetch(
        points,
        date=local_date,
        variables=canonical_variables,
        timezone=timezone,
    )
    if not hasattr(field, "data") or not hasattr(field, "variables"):
        raise WeatherDataError("Weather source fetch() must return a WeatherField")
    if tuple(field.variables) != canonical_variables:
        raise WeatherDataError(
            "Weather source returned variables different from the requested variables"
        )
    field_data = field.data.set_index("_weather_point_id", drop=False)
    expected_ids = list(range(len(frame)))
    if set(field_data.index) != set(expected_ids):
        raise WeatherDataError("Weather source did not return one row for every TAZ")
    result = frame.copy()
    ordered = field_data.loc[expected_ids]
    for variable_name in canonical_variables:
        result[variable_name] = pd.to_numeric(
            ordered[variable_name], errors="coerce"
        ).to_numpy(dtype=float)

    zone = _validate_timezone(timezone)
    local_start = datetime.combine(local_date, time.min, tzinfo=zone)
    local_end = local_start + timedelta(days=1)
    metadata = {
        **normalized.metadata,
        "object_type": "WeatherOverlay",
        "weather_date": local_date.isoformat(),
        "timezone": timezone,
        "utc_start": local_start.astimezone(utc_timezone.utc).isoformat(),
        "utc_end": local_end.astimezone(utc_timezone.utc).isoformat(),
        "variables": canonical_variables,
        "variable_definitions": {
            name: {
                "label": VARIABLES[name].label,
                "unit": VARIABLES[name].unit,
                "colorscale": VARIABLES[name].colorscale,
                "categorical": VARIABLES[name].categorical,
            }
            for name in canonical_variables
        },
        "source": getattr(source_instance, "name", type(source_instance).__name__),
        "source_metadata": dict(getattr(field, "metadata", {})),
        "spatial_method": "representative_point",
        "source_sample_count": len(field),
    }
    return WeatherOverlay(result, canonical_variables, metadata)


weather_overlay = overlay
