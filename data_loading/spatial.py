"""Explicit GeoPandas-backed vector loading for spatial analysis inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd


def read_spatial_layer(
    path: str | Path,
    *,
    layer: str | None = None,
    driver: str | None = None,
    **options: Any,
) -> gpd.GeoDataFrame:
    """Read one vector layer into a GeoDataFrame for spatial analysis.

    This explicit spatial reader is separate from :func:`ogm.open`, whose
    initial scope remains DuckDB-supported tabular formats. It is used for
    TAZ boundaries and road layers that need geometry operations.

    Args:
        path: Existing vector data path accepted by GeoPandas, such as KML,
            GeoJSON, GeoPackage, Shapefile, or GeoParquet.
        layer: Optional named layer within a multi-layer source.
        driver: Optional GDAL vector driver override.
        **options: Additional keyword options forwarded to GeoPandas.

    Returns:
        A materialized GeoDataFrame for the selected spatial layer.

    Raises:
        TypeError: If the path has an invalid type.
        FileNotFoundError: If the path does not exist.
        ValueError: If the loaded layer is empty or lacks a declared CRS.
        Exception: Propagates backend errors when the vector source cannot be
            read or decoded.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path")
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"No such spatial data file: {source}")
    if source.suffix.lower() in {".parquet", ".geoparquet"}:
        if layer is not None or driver is not None:
            raise ValueError("layer and driver options do not apply to GeoParquet")
        frame = gpd.read_parquet(source, **options)
    else:
        frame = gpd.read_file(source, layer=layer, driver=driver, **options)
    if frame.empty:
        raise ValueError(f"Spatial layer is empty: {source}")
    if frame.crs is None:
        raise ValueError(f"Spatial layer has no CRS: {source}")
    return frame
