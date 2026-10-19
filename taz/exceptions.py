"""Errors raised by traffic analysis zone operations."""


class TAZError(ValueError):
    """Base error for invalid TAZ input or generated geometry."""


class TAZGeometryError(TAZError):
    """Raised when a study boundary or generated zone partition is invalid."""


class TAZDependencyError(ImportError):
    """Raised when an optional TAZ algorithm dependency is unavailable."""
