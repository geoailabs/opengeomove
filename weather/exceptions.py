"""Errors raised by weather acquisition, aggregation, and plotting."""


class WeatherError(ValueError):
    """Base error for invalid weather requests or results."""


class WeatherDataError(WeatherError):
    """Raised when a weather provider cannot return a valid field."""


class WeatherDependencyError(ImportError):
    """Raised when an optional weather plotting dependency is unavailable."""
