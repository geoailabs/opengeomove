"""Read-only, DuckDB-backed data loading for OpenGeoMove."""

from opengeomove.data_loading.dataset import Dataset
from opengeomove.data_loading.open import open

__all__ = ["Dataset", "open"]
