"""Historical weather providers used by OpenGeoMove weather analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
import json
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from opengeomove.weather.exceptions import WeatherDataError
from opengeomove.weather.models import WeatherField, resolve_variable


def _format_coordinate(value: Any) -> str:
    """Format one coordinate without emitting scientific notation."""
    return f"{float(value):.8f}"


def _response_error(payload: Any) -> str | None:
    """Extract an API error message from a decoded provider response."""
    if not isinstance(payload, dict) or not payload.get("error"):
        return None
    reason = payload.get("reason") or payload.get("message") or "unknown provider error"
    return str(reason)


@dataclass
class OpenMeteoHistoricalSource:
    """Fetch daily historical weather from Open-Meteo's archive API.

    Values are requested at one representative point per TAZ. Open-Meteo
    snaps requested coordinates to the provider's weather model grid; the
    returned source coordinates are retained in the :class:`WeatherField`.

    Args:
        api_url: Historical API endpoint.
        batch_size: Maximum number of coordinates per HTTP request.
        timeout: Per-request timeout in seconds.
        user_agent: HTTP user-agent sent to the provider.

    Raises:
        ValueError: If configuration values are invalid.
        WeatherDataError: If the provider response cannot be decoded or does
            not contain the requested daily data.
    """

    api_url: str = "https://archive-api.open-meteo.com/v1/archive"
    batch_size: int = 100
    timeout: float = 60.0
    user_agent: str = "opengeomove-weather/0.1"
    name: str = "open_meteo_historical"

    def __post_init__(self) -> None:
        """Validate provider request settings."""
        if not isinstance(self.api_url, str) or not self.api_url.strip():
            raise ValueError("api_url must be a non-empty string")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be a positive integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise TypeError("timeout must be a positive number")
        if not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        if not isinstance(self.user_agent, str) or not self.user_agent.strip():
            raise ValueError("user_agent must be a non-empty string")

    def _request(self, parameters: dict[str, str]) -> Any:
        """Issue one provider request and decode its JSON response."""
        url = f"{self.api_url}?{urlencode(parameters)}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urlopen(request, timeout=float(self.timeout)) as response:
                raw = response.read()
        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            raise WeatherDataError(
                f"Open-Meteo request failed with HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise WeatherDataError(f"Could not fetch Open-Meteo weather data: {exc}") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WeatherDataError("Open-Meteo returned invalid JSON") from exc
        message = _response_error(payload)
        if message is not None:
            raise WeatherDataError(f"Open-Meteo rejected the request: {message}")
        return payload

    def fetch(
        self,
        points: gpd.GeoDataFrame,
        *,
        date: date_type,
        variables: Sequence[str],
        timezone: str,
    ) -> WeatherField:
        """Fetch one daily row for every requested WGS84 point.

        Args:
            points: GeoDataFrame containing unique ``_weather_point_id``
                values and point geometry in any CRS.
            date: Local calendar date requested from the provider.
            variables: Canonical daily variable names.
            timezone: IANA timezone used for daily aggregation.

        Returns:
            A normalized :class:`WeatherField` with one row per point.

        Raises:
            TypeError: If the point layer or date is invalid.
            WeatherDataError: If the provider response is incomplete.
        """
        if not isinstance(points, gpd.GeoDataFrame):
            raise TypeError("points must be a GeoDataFrame")
        if points.empty:
            raise WeatherDataError("At least one weather point is required")
        if points.crs is None:
            raise WeatherDataError("Weather points require an explicit CRS")
        if "_weather_point_id" not in points:
            raise WeatherDataError("Weather points require _weather_point_id values")
        if points["_weather_point_id"].duplicated().any():
            raise WeatherDataError("Weather point IDs must be unique")
        if not isinstance(date, date_type):
            raise TypeError("date must be a datetime.date value")
        if not isinstance(timezone, str) or not timezone.strip():
            raise TypeError("timezone must be an IANA timezone string")
        normalized_variables = tuple(resolve_variable(variable).name for variable in variables)
        if not normalized_variables:
            raise ValueError("At least one weather variable is required")
        if len(set(normalized_variables)) != len(normalized_variables):
            raise ValueError("Weather variables cannot contain duplicates")

        wgs84 = points.to_crs("EPSG:4326").copy()
        if not set(wgs84.geometry.geom_type).issubset({"Point"}):
            raise WeatherDataError("Open-Meteo points must contain point geometry")
        records: list[dict[str, Any]] = []
        units: dict[str, str] = {}
        requests = 0
        point_records = list(
            wgs84[["_weather_point_id", "geometry"]].itertuples(index=False, name=None)
        )
        for offset in range(0, len(point_records), self.batch_size):
            batch = point_records[offset : offset + self.batch_size]
            latitudes = ",".join(_format_coordinate(geometry.y) for _, geometry in batch)
            longitudes = ",".join(_format_coordinate(geometry.x) for _, geometry in batch)
            parameters = {
                "latitude": latitudes,
                "longitude": longitudes,
                "start_date": date.isoformat(),
                "end_date": date.isoformat(),
                "daily": ",".join(normalized_variables),
                "timezone": timezone,
                "temperature_unit": "celsius",
                "precipitation_unit": "mm",
                "wind_speed_unit": "kmh",
            }
            payload = self._request(parameters)
            responses = payload if isinstance(payload, list) else [payload]
            if len(responses) != len(batch):
                raise WeatherDataError(
                    "Open-Meteo returned a different number of locations than requested"
                )
            requests += 1
            for (point_id, _), response in zip(batch, responses):
                if not isinstance(response, dict):
                    raise WeatherDataError("Open-Meteo returned an invalid location object")
                daily = response.get("daily")
                if not isinstance(daily, dict):
                    raise WeatherDataError("Open-Meteo response does not contain daily data")
                dates = daily.get("time")
                if not isinstance(dates, list) or date.isoformat() not in dates:
                    raise WeatherDataError(
                        f"Open-Meteo response does not contain requested date {date.isoformat()}"
                    )
                day_index = dates.index(date.isoformat())
                row: dict[str, Any] = {
                    "_weather_point_id": point_id,
                    "source_latitude": response.get("latitude"),
                    "source_longitude": response.get("longitude"),
                    "source_elevation_m": response.get("elevation"),
                }
                response_units = response.get("daily_units", {})
                if isinstance(response_units, dict):
                    units.update({str(key): str(value) for key, value in response_units.items()})
                for variable in normalized_variables:
                    values = daily.get(variable)
                    if not isinstance(values, list) or day_index >= len(values):
                        raise WeatherDataError(
                            f"Open-Meteo response is missing daily variable {variable!r}"
                        )
                    row[variable] = values[day_index]
                records.append(row)

        field_data = pd.DataFrame.from_records(records)
        geometries = []
        for latitude, longitude in zip(
            field_data["source_latitude"], field_data["source_longitude"]
        ):
            if latitude is None or longitude is None:
                raise WeatherDataError("Open-Meteo did not return source coordinates")
            try:
                latitude_value = float(latitude)
                longitude_value = float(longitude)
            except (TypeError, ValueError) as exc:
                raise WeatherDataError("Open-Meteo returned invalid source coordinates") from exc
            if not np.isfinite(latitude_value) or not np.isfinite(longitude_value):
                raise WeatherDataError("Open-Meteo returned invalid source coordinates")
            geometries.append(Point(longitude_value, latitude_value))
        field_frame = gpd.GeoDataFrame(field_data, geometry=geometries, crs="EPSG:4326")
        return WeatherField(
            field_frame,
            normalized_variables,
            {
                "provider": self.name,
                "endpoint": self.api_url,
                "date": date.isoformat(),
                "timezone": timezone,
                "request_count": requests,
                "requested_points": len(points),
                "units": units,
            },
        )
