"""Historical weather overlays on traffic analysis zones."""

from opengeomove.data_loading.weather import OpenMeteoHistoricalSource
from opengeomove.weather.exceptions import (
    WeatherDataError,
    WeatherDependencyError,
    WeatherError,
)
from opengeomove.weather.models import (
    VARIABLES,
    WeatherField,
    WeatherOverlay,
    WeatherSource,
    WeatherVariable,
    available_variables,
    resolve_variable,
)
from opengeomove.weather.operations import overlay, weather_overlay
from opengeomove.weather.plotting import plot, plot_weather_overlay

__all__ = [
    "OpenMeteoHistoricalSource",
    "VARIABLES",
    "WeatherDataError",
    "WeatherDependencyError",
    "WeatherError",
    "WeatherField",
    "WeatherOverlay",
    "WeatherSource",
    "WeatherVariable",
    "available_variables",
    "overlay",
    "plot",
    "plot_weather_overlay",
    "resolve_variable",
    "weather_overlay",
]
