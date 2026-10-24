"""Interactive basemap plotting for traffic analysis zones."""

from __future__ import annotations

import html
import math
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

from opengeomove.taz.exceptions import TAZDependencyError, TAZGeometryError
from opengeomove.taz.models import TAZFrame


def _plot_frame(zones: TAZFrame | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Copy and validate polygon zones for display.

    Args:
        zones: Validated TAZFrame or polygon GeoDataFrame.

    Returns:
        A copied GeoDataFrame in its original CRS.

    Raises:
        TypeError: If ``zones`` is not supported.
        TAZGeometryError: If the layer is empty, lacks a CRS, or contains
            unusable geometry.
    """
    if isinstance(zones, TAZFrame):
        frame = zones.to_geopandas()
    elif isinstance(zones, gpd.GeoDataFrame):
        frame = zones.copy()
    else:
        raise TypeError("zones must be a TAZFrame or GeoDataFrame")
    if frame.empty:
        raise TAZGeometryError("zones cannot be empty")
    if frame.crs is None:
        raise TAZGeometryError("zones must have an explicit CRS")
    geometry = frame.geometry
    if geometry.isna().any() or geometry.is_empty.any() or (~geometry.is_valid).any():
        raise TAZGeometryError("zones must contain non-empty, valid geometry")
    if not set(geometry.geom_type).issubset({"Polygon", "MultiPolygon"}):
        raise TAZGeometryError("zones must contain polygon geometry only")
    return frame


def _hover_value(value: Any) -> str:
    """Format one scalar value for a stable Plotly hover label."""
    if value is None:
        return "—"
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.3f}"
    return str(value)


def _auto_map_view(frame: gpd.GeoDataFrame) -> tuple[dict[str, float], float]:
    """Calculate a Web-map center and zoom from WGS84 layer bounds."""
    min_lon, min_lat, max_lon, max_lat = frame.total_bounds
    center = {
        "lon": float((min_lon + max_lon) / 2.0),
        "lat": float((min_lat + max_lat) / 2.0),
    }
    latitude_scale = max(math.cos(math.radians(center["lat"])), 0.1)
    span = max(
        float(max_lon - min_lon),
        float(max_lat - min_lat) / latitude_scale,
        1e-6,
    )
    # Roughly four 256-pixel tiles across the larger map dimension, with a
    # little context around the study area.
    zoom = float(np.clip(math.log2(1_440.0 / span) - 0.5, 0.0, 20.0))
    return center, zoom


def plot_taz(
    zones: TAZFrame | gpd.GeoDataFrame,
    *,
    opacity: float = 0.75,
    fill_color: str = "#2878B5",
    boundary_color: str = "#FFFFFF",
    boundary_width: float = 0.7,
    basemap_style: str = "open-street-map",
    hover_columns: Iterable[str] = ("taz_id", "area_km2"),
) -> Any:
    """Plot traffic analysis zones over an interactive basemap.

    The function does not mutate the supplied layer or write output. It
    reprojects a copy to WGS84, frames the full layer, and returns a Plotly
    figure that callers may display with ``show()`` or save with
    ``write_html()``. Basemap tiles are requested by the rendered figure, not
    while this function constructs it.

    Args:
        zones: TAZFrame or polygon GeoDataFrame with an explicit CRS.
        opacity: Polygon fill opacity from 0 (transparent) to 1 (opaque).
        fill_color: Plotly-compatible polygon fill color.
        boundary_color: Plotly-compatible zone boundary color.
        boundary_width: Non-negative boundary width in display pixels.
        basemap_style: Plotly Mapbox style, default ``"open-street-map"``.
        hover_columns: Existing columns to display when hovering over a zone.

    Returns:
        A ``plotly.graph_objects.Figure`` containing the basemap and TAZ layer.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If opacity, boundary width, style, or hover columns are
            invalid.
        TAZGeometryError: If the zone layer has unusable geometry or CRS.
        TAZDependencyError: If Plotly is unavailable.
    """
    if isinstance(opacity, bool) or not isinstance(opacity, (int, float)):
        raise TypeError("opacity must be a number from 0 through 1")
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("opacity must be between 0 and 1")
    if isinstance(boundary_width, bool) or not isinstance(boundary_width, (int, float)):
        raise TypeError("boundary_width must be a non-negative number")
    if not math.isfinite(boundary_width) or boundary_width < 0:
        raise ValueError("boundary_width must be finite and non-negative")
    for name, value in (
        ("fill_color", fill_color),
        ("boundary_color", boundary_color),
        ("basemap_style", basemap_style),
    ):
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{name} must be a non-empty string")
    if isinstance(hover_columns, str):
        raise TypeError("hover_columns must be an iterable of column names, not a string")
    try:
        columns = tuple(hover_columns)
    except TypeError as exc:
        raise TypeError("hover_columns must be an iterable of column names") from exc
    if not columns or any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("hover_columns must contain non-empty column names")
    if len(set(columns)) != len(columns):
        raise ValueError("hover_columns cannot contain duplicates")

    frame = _plot_frame(zones)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Hover columns do not exist: {sorted(missing)}")
    wgs84 = frame.to_crs("EPSG:4326").reset_index(drop=True)
    feature_ids = [str(index) for index in wgs84.index]
    features = [
        {
            "type": "Feature",
            "id": feature_id,
            "properties": {},
            "geometry": mapping(geometry),
        }
        for feature_id, geometry in zip(feature_ids, wgs84.geometry)
    ]
    geojson = {"type": "FeatureCollection", "features": features}
    customdata = np.asarray([
        [_hover_value(row[column]) for column in columns]
        for _, row in wgs84.iterrows()
    ], dtype=object)
    hover_template = "<br>".join(
        f"{html.escape(column)}: %{{customdata[{index}]}}"
        for index, column in enumerate(columns)
    ) + "<extra></extra>"
    center, zoom = _auto_map_view(wgs84)

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise TAZDependencyError("TAZ basemap plotting requires plotly") from exc
    figure = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=feature_ids,
        featureidkey="id",
        z=np.ones(len(wgs84)),
        colorscale=[[0.0, fill_color], [1.0, fill_color]],
        showscale=False,
        marker={
            "opacity": float(opacity),
            "line": {"color": boundary_color, "width": float(boundary_width)},
        },
        customdata=customdata,
        hovertemplate=hover_template,
        name="Traffic analysis zones",
    ))
    figure.update_layout(
        title="Traffic analysis zones",
        mapbox={"style": basemap_style, "center": center, "zoom": zoom},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        height=760,
        showlegend=False,
    )
    return figure
