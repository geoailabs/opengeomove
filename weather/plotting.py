"""Interactive plotting for quantitative weather overlays."""

from __future__ import annotations

import html
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from opengeomove.taz.plotting import _auto_map_view, _hover_value, _wgs84_plot_frame
from opengeomove.weather.exceptions import WeatherDependencyError
from opengeomove.weather.models import WeatherOverlay, resolve_variable


def _validate_display_number(name: str, value: Any, *, minimum: float) -> float:
    """Validate one finite non-negative plotting number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return float(value)


def plot_weather_overlay(
    overlay: WeatherOverlay,
    *,
    variable: str | None = None,
    opacity: float = 0.82,
    boundary_color: str = "#FFFFFF",
    boundary_width: float = 0.8,
    basemap_style: str = "open-street-map",
    colorscale: str | None = None,
    hover_columns: Iterable[str] | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> Any:
    """Plot one weather variable over the overlay's TAZ polygons.

    Args:
        overlay: WeatherOverlay returned by :func:`weather.overlay`.
        variable: Canonical variable or alias to render. If omitted, the only
            variable is selected automatically.
        opacity: Polygon fill opacity from 0 through 1.
        boundary_color: TAZ boundary color.
        boundary_width: Non-negative boundary width in display pixels.
        basemap_style: Plotly Mapbox style.
        colorscale: Optional Plotly colorscale name overriding the variable
            default.
        hover_columns: Existing columns to show in each zone's hover label;
            defaults to ``("taz_id", variable)``.
        zmin: Optional lower bound for the color scale.
        zmax: Optional upper bound for the color scale.

    Returns:
        A Plotly ``Figure``. Basemap tiles are requested only when displayed.

    Raises:
        TypeError: If the overlay or display arguments are invalid.
        ValueError: If the requested variable, columns, or color range is invalid.
        WeatherDependencyError: If Plotly is unavailable.
    """
    if not isinstance(overlay, WeatherOverlay):
        raise TypeError("overlay must be a WeatherOverlay")
    if variable is None:
        if len(overlay.variables) != 1:
            raise ValueError("variable is required when the overlay has multiple variables")
        selected = overlay.variables[0]
    else:
        selected = resolve_variable(variable).name
    if selected not in overlay.variables:
        raise ValueError(f"Variable {selected!r} is not present in the overlay")
    opacity = _validate_display_number("opacity", opacity, minimum=0.0)
    if opacity > 1:
        raise ValueError("opacity must be between 0 and 1")
    boundary_width = _validate_display_number("boundary_width", boundary_width, minimum=0.0)
    for name, value in (("boundary_color", boundary_color), ("basemap_style", basemap_style)):
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{name} must be a non-empty string")
    if colorscale is not None and (not isinstance(colorscale, str) or not colorscale.strip()):
        raise TypeError("colorscale must be None or a non-empty string")
    if hover_columns is None:
        columns = ("taz_id", selected)
    elif isinstance(hover_columns, str):
        raise TypeError("hover_columns must be an iterable of column names, not a string")
    else:
        try:
            columns = tuple(hover_columns)
        except TypeError as exc:
            raise TypeError("hover_columns must be an iterable of column names") from exc
    if not columns or any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("hover_columns must contain non-empty column names")
    if len(set(columns)) != len(columns):
        raise ValueError("hover_columns cannot contain duplicates")
    missing = set(columns) - set(overlay.data.columns)
    if missing:
        raise ValueError(f"Hover columns do not exist: {sorted(missing)}")
    if zmin is not None:
        zmin = _validate_display_number("zmin", zmin, minimum=-float("inf"))
    if zmax is not None:
        zmax = _validate_display_number("zmax", zmax, minimum=-float("inf"))
    if zmin is not None and zmax is not None and zmax <= zmin:
        raise ValueError("zmax must be greater than zmin")

    wgs84 = _wgs84_plot_frame(overlay.data)
    feature_ids = [str(index) for index in wgs84.index]
    features = [
        {
            "type": "Feature",
            "id": feature_id,
            "properties": {},
            "geometry": geometry.__geo_interface__,
        }
        for feature_id, geometry in zip(feature_ids, wgs84.geometry)
    ]
    geojson = {"type": "FeatureCollection", "features": features}
    customdata = np.asarray(
        [
            [_hover_value(row[column]) for column in columns]
            for _, row in wgs84.iterrows()
        ],
        dtype=object,
    )
    hover_template = "<br>".join(
        f"{html.escape(column)}: %{{customdata[{index}]}}"
        for index, column in enumerate(columns)
    ) + "<extra></extra>"
    values = pd.to_numeric(wgs84[selected], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if zmin is None:
        zmin = float(finite.min()) if finite.size else 0.0
    if zmax is None:
        zmax = float(finite.max()) if finite.size else 1.0
    if zmax <= zmin:
        padding = max(abs(zmin) * 0.01, 0.5)
        zmin -= padding
        zmax += padding
    definition = resolve_variable(selected)
    title = str(overlay.metadata.get("weather_date", "requested date"))
    title = f"{definition.label} by TAZ — {title}"
    center, zoom = _auto_map_view(wgs84)
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise WeatherDependencyError("Weather plotting requires plotly") from exc
    figure = go.Figure(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=feature_ids,
            featureidkey="id",
            z=values,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale or definition.colorscale,
            showscale=True,
            colorbar={"title": f"{definition.label} ({definition.unit})"},
            marker={
                "opacity": float(opacity),
                "line": {"color": boundary_color, "width": float(boundary_width)},
            },
            customdata=customdata,
            hovertemplate=hover_template,
            name=definition.label,
        )
    )
    figure.update_layout(
        title=title,
        mapbox={"style": basemap_style, "center": center, "zoom": zoom},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        height=760,
        showlegend=False,
    )
    return figure


plot = plot_weather_overlay
