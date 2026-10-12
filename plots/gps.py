"""Static plotting of GPS ping coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from opengeomove.data_loading.dataset import Dataset

_WEB_MERCATOR_LIMIT = 85.0511287798066
_EARTH_RADIUS_M = 6_378_137.0
_DEFAULT_BASEMAP: "_Basemap | None" = None


@dataclass(frozen=True)
class _Basemap:
    """Tile URL and attribution configured as the global plotting default."""

    url: str
    attribution: str | None


def _validate_tile_url(tile_url: str) -> str:
    """Validate a Web Mercator XYZ tile URL template."""
    if not isinstance(tile_url, str):
        raise TypeError("tile_url must be a string")
    if not tile_url.startswith(("https://", "http://")):
        raise ValueError("tile_url must start with http:// or https://")
    if not all(token in tile_url for token in ("{z}", "{x}", "{y}")):
        raise ValueError("tile_url must contain {z}, {x}, and {y} placeholders")
    return tile_url


def set_basemap(tile_url: str | None, *, attribution: str | None = None) -> None:
    """Set the default XYZ tile URL used by newly created GPS plots.

    No network request is made until :meth:`GPSPlot.show` is called. Pass
    ``None`` to clear the global default. Tile providers may require
    attribution; when supplied, it is rendered below the map.

    Args:
        tile_url: HTTP(S) URL template containing ``{z}``, ``{x}``, and
            ``{y}``, or ``None`` to clear the default.
        attribution: Optional provider attribution text displayed on the map.

    Raises:
        TypeError: If arguments are not strings or ``None`` as appropriate.
        ValueError: If the URL is not an HTTP(S) XYZ tile template.
    """
    global _DEFAULT_BASEMAP
    if tile_url is None:
        if attribution is not None:
            raise TypeError("attribution must be None when tile_url is None")
        _DEFAULT_BASEMAP = None
        return
    if attribution is not None and not isinstance(attribution, str):
        raise TypeError("attribution must be a string or None")
    _DEFAULT_BASEMAP = _Basemap(_validate_tile_url(tile_url), attribution)


def _quote_identifier(identifier: str) -> str:
    """Quote a DuckDB SQL identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def _mercator_xy(longitude: float, latitude: float) -> tuple[float, float]:
    """Project a longitude/latitude pair to spherical Web Mercator meters."""
    latitude = max(-_WEB_MERCATOR_LIMIT, min(_WEB_MERCATOR_LIMIT, latitude))
    x = _EARTH_RADIUS_M * math.radians(longitude)
    y = _EARTH_RADIUS_M * math.log(math.tan(math.pi / 4 + math.radians(latitude) / 2))
    return x, y


def _tile_indices(x: float, y: float, zoom: int) -> tuple[int, int]:
    """Return XYZ tile indices for projected Web Mercator coordinates."""
    count = 1 << zoom
    world = math.pi * _EARTH_RADIUS_M
    tile_x = math.floor((x + world) / (2 * world) * count)
    tile_y = math.floor((world - y) / (2 * world) * count)
    return max(0, min(count - 1, tile_x)), max(0, min(count - 1, tile_y))


class GPSPlot:
    """Render a bounded sample of GPS pings on a static map.

    Coordinates are read from the Dataset only when :meth:`show` is called.
    DuckDB performs a reservoir sample and returns at most ``sampling`` valid
    coordinates to Python. The sample is random and may differ between calls.
    Coordinates are assumed to be WGS84 longitude/latitude.

    Args:
        dataset: Source or filtered Dataset containing the coordinates.
        latitude_col: Column containing latitude values in degrees.
        longitude_col: Column containing longitude values in degrees.
        sampling: Required maximum number of points to render. This explicit
            limit bounds Python memory and rendering work.
        basemap: Optional XYZ tile URL override. If omitted, use the global
            default set by :func:`set_basemap`; clear that default with
            ``set_basemap(None)`` to plot without tiles.
        attribution: Optional attribution override for this plot.
        zoom: XYZ tile zoom level, from 0 through 22.
        max_tiles: Maximum number of map tiles fetched for one plot.
        point_color: Matplotlib color for ping markers.
        point_size: Marker area in points squared.
        title: Optional plot title.

    Raises:
        TypeError: If required values have the wrong type.
        ValueError: If values are outside their supported ranges or a
            coordinate column is missing.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        latitude_col: str,
        longitude_col: str,
        sampling: int,
        basemap: str | None = None,
        attribution: str | None = None,
        zoom: int = 12,
        max_tiles: int = 64,
        point_color: Any = "crimson",
        point_size: float = 4.0,
        title: str | None = None,
    ) -> None:
        """Create a GPS plot specification without reading source rows."""
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be an opengeomove.Dataset")
        for name, value in (("latitude_col", latitude_col), ("longitude_col", longitude_col)):
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be a non-empty string")
            if value not in dataset.columns:
                raise ValueError(f"Column {value!r} does not exist in this Dataset")
        if isinstance(sampling, bool) or not isinstance(sampling, int):
            raise TypeError("sampling must be a positive integer")
        if sampling <= 0:
            raise ValueError("sampling must be a positive integer")
        if isinstance(zoom, bool) or not isinstance(zoom, int):
            raise TypeError("zoom must be an integer from 0 through 22")
        if not 0 <= zoom <= 22:
            raise ValueError("zoom must be between 0 and 22")
        if isinstance(max_tiles, bool) or not isinstance(max_tiles, int):
            raise TypeError("max_tiles must be a positive integer")
        if max_tiles <= 0:
            raise ValueError("max_tiles must be a positive integer")
        if isinstance(point_size, bool) or not isinstance(point_size, (int, float)):
            raise TypeError("point_size must be a non-negative number")
        if point_size < 0:
            raise ValueError("point_size must be a non-negative number")
        if basemap is not None:
            _validate_tile_url(basemap)
            chosen_basemap = _Basemap(basemap, attribution)
        elif _DEFAULT_BASEMAP is not None:
            chosen_basemap = _DEFAULT_BASEMAP
            if attribution is not None:
                chosen_basemap = _Basemap(chosen_basemap.url, attribution)
        else:
            chosen_basemap = None
        if attribution is not None and not isinstance(attribution, str):
            raise TypeError("attribution must be a string or None")
        if title is not None and not isinstance(title, str):
            raise TypeError("title must be a string or None")

        self.dataset = dataset
        self.latitude_col = latitude_col
        self.longitude_col = longitude_col
        self.sampling = sampling
        self.basemap = chosen_basemap
        self.zoom = zoom
        self.max_tiles = max_tiles
        self.point_color = point_color
        self.point_size = float(point_size)
        self.title = title

    def _sample_coordinates(self) -> list[tuple[float, float]]:
        """Fetch at most the requested number of valid coordinates."""
        lat = _quote_identifier(self.latitude_col)
        lon = _quote_identifier(self.longitude_col)
        alias = "_ogm_gps_plot_source"
        sql = (
            f"SELECT TRY_CAST({lat} AS DOUBLE) AS latitude, "
            f"TRY_CAST({lon} AS DOUBLE) AS longitude FROM {alias} "
            f"WHERE TRY_CAST({lat} AS DOUBLE) IS NOT NULL "
            f"AND TRY_CAST({lon} AS DOUBLE) IS NOT NULL "
            f"AND isfinite(TRY_CAST({lat} AS DOUBLE)) "
            f"AND isfinite(TRY_CAST({lon} AS DOUBLE)) "
            f"AND TRY_CAST({lat} AS DOUBLE) BETWEEN -90 AND 90 "
            f"AND TRY_CAST({lon} AS DOUBLE) BETWEEN -180 AND 180 "
            f"USING SAMPLE reservoir({self.sampling} ROWS)"
        )
        relation = self.dataset._relation.query(alias, sql)
        return [(float(latitude), float(longitude)) for latitude, longitude in relation.fetchall()]

    def _draw_basemap(self, axes: Any, extent: tuple[float, float, float, float]) -> None:
        """Fetch and draw XYZ tiles covering the current map extent."""
        if self.basemap is None:
            return
        from matplotlib.image import imread

        min_x, max_x, min_y, max_y = extent
        min_tile_x, max_tile_y = _tile_indices(min_x, min_y, self.zoom)
        max_tile_x, min_tile_y = _tile_indices(max_x, max_y, self.zoom)
        count = 1 << self.zoom
        tile_xs = range(min_tile_x, max_tile_x + 1)
        tile_ys = range(min_tile_y, max_tile_y + 1)
        if len(tile_xs) * len(tile_ys) > self.max_tiles:
            raise ValueError(
                f"Map extent needs more than {self.max_tiles} tiles at zoom "
                f"{self.zoom}; lower zoom or increase max_tiles"
            )
        world = math.pi * _EARTH_RADIUS_M
        tile_span = 2 * world / count
        for tile_y in tile_ys:
            for tile_x in tile_xs:
                url = self.basemap.url.format(z=self.zoom, x=tile_x, y=tile_y)
                request = Request(url, headers={"User-Agent": "OpenGeoMove/0.1"})
                try:
                    with urlopen(request, timeout=15) as response:
                        image = imread(BytesIO(response.read()), format="png")
                except (OSError, URLError, ValueError) as exc:
                    raise RuntimeError(f"Could not fetch basemap tile {url!r}: {exc}") from exc
                west = -world + tile_x * tile_span
                east = west + tile_span
                north = world - tile_y * tile_span
                south = north - tile_span
                axes.imshow(image, extent=(west, east, south, north), origin="upper", zorder=0)
        if self.basemap.attribution:
            axes.text(
                0.995,
                0.005,
                self.basemap.attribution,
                transform=axes.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="#222222",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
                zorder=3,
            )

    def show(self) -> Any:
        """Render the sampled GPS pings and display the Matplotlib figure.

        The source is sampled only when this method is called. At most
        ``sampling`` coordinate pairs are materialized in Python. If a tile
        basemap is configured, its network tiles are fetched at this point.

        Returns:
            The Matplotlib Axes object, allowing further customization.

        Raises:
            ImportError: If Matplotlib is not installed.
            ValueError: If no valid coordinate rows exist or the tile count
                exceeds ``max_tiles``.
            RuntimeError: If a requested basemap tile cannot be fetched.
        """
        import matplotlib.pyplot as plt

        coordinates = self._sample_coordinates()
        if not coordinates:
            raise ValueError("No valid GPS coordinates are available to plot")
        projected = [_mercator_xy(longitude, latitude) for latitude, longitude in coordinates]
        xs = [x for x, _ in projected]
        ys = [y for _, y in projected]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        padding_x = max(span_x * 0.05, 100.0)
        padding_y = max(span_y * 0.05, 100.0)
        extent = (
            min(xs) - padding_x,
            max(xs) + padding_x,
            min(ys) - padding_y,
            max(ys) + padding_y,
        )

        figure, axes = plt.subplots()
        self._draw_basemap(axes, extent)
        axes.scatter(xs, ys, s=self.point_size, c=self.point_color, alpha=0.75, zorder=2)
        axes.set_xlim(extent[0], extent[1])
        axes.set_ylim(extent[2], extent[3])
        axes.set_aspect("equal", adjustable="box")
        axes.set_xlabel("Web Mercator X (m)")
        axes.set_ylabel("Web Mercator Y (m)")
        if self.title:
            axes.set_title(self.title)
        figure.tight_layout()
        plt.show()
        return axes
