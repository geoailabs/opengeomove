"""Models and variable definitions for TAZ weather overlays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import geopandas as gpd

from opengeomove.weather.exceptions import WeatherDataError


@dataclass(frozen=True)
class WeatherVariable:
    """Description of one daily weather variable.

    Args:
        name: Canonical column and provider variable name.
        label: Human-readable label used in plots.
        unit: Display unit returned by the provider.
        colorscale: Plotly colorscale suitable for the variable.
        categorical: Whether the variable represents categories rather than a
            continuous physical quantity.
    """

    name: str
    label: str
    unit: str
    colorscale: str
    categorical: bool = False


_VARIABLE_DEFINITIONS: tuple[WeatherVariable, ...] = (
    WeatherVariable("precipitation_sum", "Precipitation", "mm", "Blues"),
    WeatherVariable("rain_sum", "Rainfall", "mm", "Blues"),
    WeatherVariable("snowfall_sum", "Snowfall", "cm", "Blues"),
    WeatherVariable("precipitation_hours", "Precipitation duration", "h", "Blues"),
    WeatherVariable("temperature_2m_mean", "Mean temperature", "°C", "RdYlBu_r"),
    WeatherVariable("temperature_2m_max", "Maximum temperature", "°C", "RdYlBu_r"),
    WeatherVariable("temperature_2m_min", "Minimum temperature", "°C", "RdYlBu_r"),
    WeatherVariable("apparent_temperature_max", "Maximum apparent temperature", "°C", "RdYlBu_r"),
    WeatherVariable("apparent_temperature_min", "Minimum apparent temperature", "°C", "RdYlBu_r"),
    WeatherVariable("wind_speed_10m_max", "Maximum wind speed", "km/h", "Viridis"),
    WeatherVariable("wind_gusts_10m_max", "Maximum wind gusts", "km/h", "Viridis"),
    WeatherVariable("wind_direction_10m_dominant", "Dominant wind direction", "°", "Phase"),
    WeatherVariable("shortwave_radiation_sum", "Shortwave radiation", "MJ/m²", "YlOrRd"),
    WeatherVariable("et0_fao_evapotranspiration", "Reference evapotranspiration", "mm", "YlOrBr"),
    WeatherVariable("weather_code", "Weather code", "WMO code", "Turbo", categorical=True),
)


_VARIABLE_ALIASES: dict[str, str] = {
    "precipitation": "precipitation_sum",
    "rain": "rain_sum",
    "snowfall": "snowfall_sum",
    "temperature": "temperature_2m_mean",
    "temperature_mean": "temperature_2m_mean",
    "temperature_max": "temperature_2m_max",
    "temperature_min": "temperature_2m_min",
    "apparent_temperature_max": "apparent_temperature_max",
    "apparent_temperature_min": "apparent_temperature_min",
    "wind_speed_max": "wind_speed_10m_max",
    "wind_gusts_max": "wind_gusts_10m_max",
    "wind_direction": "wind_direction_10m_dominant",
    "radiation": "shortwave_radiation_sum",
    "evapotranspiration": "et0_fao_evapotranspiration",
    "weather": "weather_code",
}

VARIABLES: Mapping[str, WeatherVariable] = {
    definition.name: definition for definition in _VARIABLE_DEFINITIONS
}


def resolve_variable(variable: str) -> WeatherVariable:
    """Resolve a canonical weather variable or a documented alias.

    Args:
        variable: Canonical Open-Meteo daily variable name or a short alias.

    Returns:
        The normalized :class:`WeatherVariable` definition.

    Raises:
        TypeError: If ``variable`` is not a string.
        ValueError: If the variable is not supported.
    """
    if not isinstance(variable, str):
        raise TypeError("weather variable names must be strings")
    key = variable.strip().lower()
    key = _VARIABLE_ALIASES.get(key, key)
    try:
        return VARIABLES[key]
    except KeyError as exc:
        supported = ", ".join(sorted((*VARIABLES, *_VARIABLE_ALIASES)))
        raise ValueError(
            f"Unsupported weather variable {variable!r}; supported variables: {supported}"
        ) from exc


def available_variables() -> tuple[str, ...]:
    """Return canonical daily weather variable names in stable order."""
    return tuple(definition.name for definition in _VARIABLE_DEFINITIONS)


class WeatherSource(Protocol):
    """Protocol implemented by weather providers used by :func:`overlay`."""

    name: str

    def fetch(
        self,
        points: gpd.GeoDataFrame,
        *,
        date: Any,
        variables: Sequence[str],
        timezone: str,
    ) -> WeatherField:
        """Return one weather row for every requested point."""


@dataclass
class WeatherField:
    """Normalized weather values sampled at points or grid cells.

    The current public provider returns one row per representative point. The
    model deliberately keeps geometry and provider metadata so gridded,
    area-weighted sources can be added without changing the overlay result.

    Args:
        data: GeoDataFrame with ``_weather_point_id``, point geometry, and one
            column per requested canonical variable.
        variables: Canonical variables present in ``data``.
        metadata: Provider and query provenance.

    Raises:
        TypeError: If ``data`` is not a GeoDataFrame.
        WeatherDataError: If the field has no CRS, duplicate point IDs, or
            missing variable columns.
    """

    data: gpd.GeoDataFrame
    variables: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy and validate the normalized weather field."""
        if not isinstance(self.data, gpd.GeoDataFrame):
            raise TypeError("WeatherField.data must be a GeoDataFrame")
        self.data = self.data.copy()
        self.variables = tuple(self.variables)
        self.metadata = dict(self.metadata)
        if not self.variables:
            raise WeatherDataError("WeatherField must contain at least one variable")
        if self.data.crs is None:
            raise WeatherDataError("WeatherField requires an explicit CRS")
        if "_weather_point_id" not in self.data:
            raise WeatherDataError("WeatherField requires _weather_point_id values")
        if self.data["_weather_point_id"].isna().any():
            raise WeatherDataError("WeatherField point IDs cannot be null")
        if self.data["_weather_point_id"].duplicated().any():
            raise WeatherDataError("WeatherField point IDs must be unique")
        missing = set(self.variables) - set(self.data.columns)
        if missing:
            raise WeatherDataError(f"WeatherField is missing variables: {sorted(missing)}")

    def __len__(self) -> int:
        """Return the number of weather samples."""
        return len(self.data)


@dataclass
class WeatherOverlay:
    """TAZ polygons enriched with weather values for one local calendar day.

    Args:
        data: GeoDataFrame containing the original polygons and weather
            columns named by canonical variable.
        variables: Canonical weather variables included in ``data``.
        metadata: Date, source, units, sampling, and provenance details.

    Raises:
        TypeError: If ``data`` is not a GeoDataFrame.
        WeatherDataError: If the result lacks valid TAZ polygon geometry or
            one of the requested variable columns.
    """

    data: gpd.GeoDataFrame
    variables: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy and validate the weather-enriched TAZ layer."""
        if not isinstance(self.data, gpd.GeoDataFrame):
            raise TypeError("WeatherOverlay.data must be a GeoDataFrame")
        self.data = self.data.copy()
        self.variables = tuple(self.variables)
        self.metadata = dict(self.metadata)
        if self.data.empty:
            raise WeatherDataError("WeatherOverlay cannot be empty")
        if self.data.crs is None:
            raise WeatherDataError("WeatherOverlay requires an explicit CRS")
        if "taz_id" not in self.data:
            raise WeatherDataError("WeatherOverlay requires a taz_id column")
        if self.data.taz_id.isna().any() or self.data.taz_id.duplicated().any():
            raise WeatherDataError("WeatherOverlay TAZ identifiers must be unique")
        geometry = self.data.geometry
        if geometry.isna().any() or geometry.is_empty.any() or (~geometry.is_valid).any():
            raise WeatherDataError("WeatherOverlay polygons must be valid and non-empty")
        if not set(geometry.geom_type).issubset({"Polygon", "MultiPolygon"}):
            raise WeatherDataError("WeatherOverlay geometry must contain polygons only")
        missing = set(self.variables) - set(self.data.columns)
        if missing:
            raise WeatherDataError(f"WeatherOverlay is missing variables: {sorted(missing)}")

    def __len__(self) -> int:
        """Return the number of TAZ polygons in the overlay."""
        return len(self.data)

    def to_geopandas(self) -> gpd.GeoDataFrame:
        """Return a copy of the weather-enriched TAZ GeoDataFrame."""
        return self.data.copy()

    def plot(self, *, variable: str | None = None, **options: Any) -> Any:
        """Return an interactive quantitative weather map.

        Args:
            variable: Canonical variable or alias to render. Required only
                when this overlay contains multiple variables.
            **options: Display options accepted by
                :func:`opengeomove.weather.plot_weather_overlay`.

        Returns:
            A Plotly figure. Construction does not fetch basemap tiles.
        """
        from opengeomove.weather.plotting import plot_weather_overlay

        return plot_weather_overlay(self, variable=variable, **options)
