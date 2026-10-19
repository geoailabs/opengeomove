"""Automatic road-aware TAZ creation.

It keeps the reference algorithm's stages explicit: major-road and natural
barriers, coarse polygon cleanup, METIS road partitions, optional Google 2.5D
building-aware splits, oriented fallback splits, failed-hole repair, and final
geometry cleanup. Heavy integrations are imported only when requested.
"""

from __future__ import annotations

import logging
import json
import pickle
import base64
import io
import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon
from shapely.ops import nearest_points, polygonize_full, split, unary_union

from opengeomove.taz.exceptions import TAZDependencyError, TAZGeometryError
from opengeomove.taz.models import StudyArea, TAZFrame
from opengeomove.data_loading.spatial import read_spatial_layer


@dataclass
class AutomaticTAZConfig:
    """Configuration for road-aware and optional building-aware TAZ creation.

    This configuration controls road filtering, corridor buffers, recursive
    split sizes, raster bands, geometry cleanup, and explicit output paths.
    Set ``download_google_buildings=True`` to fetch public Google 2.5D tiles
    for ``building_data_year`` (2016–2023; default 2023) when
    ``building_raster_path`` is not supplied. Downloads are cached under
    ``building_cache_dir`` or ``output_dir/building_rasters``. Outputs are
    written only when :func:`create_automatic_taz` is called. Plot output
    supports PNG, PDF, and self-contained HTML previews.
    """

    city_name: str = "city"
    output_dir: Path = Path("taz_metis_output")
    network_type: str = "drive"
    osm_cache_path: Path | None = None
    building_raster_path: Path | None = None
    download_google_buildings: bool = False
    building_data_year: int = 2023
    building_cache_dir: Path | None = None
    building_count_band: int = 1
    building_height_band: int = 2
    building_presence_band: int = 3
    building_presence_threshold: float = 0.5
    building_kmeans_n_init: int = 20
    building_max_samples: int = 100_000
    export_crs: str = "EPSG:4326"
    save_plots: bool = True
    plot_formats: tuple[str, ...] = ("html", "png")
    show_plots: bool = False
    major_road_types: tuple[Any, ...] = (
        "trunk", "motorway", "primary", "secondary", "road",
        ("tertiary", "residential"), ("secondary", "residential"),
        ("trunk", "motorway"), ("primary", "trunk"), ("trunk", "primary"),
    )
    local_road_types: tuple[Any, ...] = (
        "residential", "tertiary", "unclassified", "living_street",
        ("living_street", "residential"), ("residential", "unclassified"),
        ("tertiary", "residential"), ("tertiary", "unclassified"),
    )
    railway_types: tuple[str, ...] = ("rail", "subway", "construction")
    fetch_railways_and_waterways: bool = True
    major_road_buffer_m: float = 50.0
    barrier_buffer_m: float = 100.0
    coarse_boundary_buffer_m: float = 200.0
    centerline_interpolation_m: float = 5.0
    max_dangle_length_m: float = 5000.0
    # The cleanup candidate can contain dense sliver polygonization from
    # detailed OSM road geometry. Above this size, retain the valid coarse
    # partition instead of repeatedly merging tens of thousands of slivers.
    max_coarse_cleanup_polygons: int = 10_000
    node_rounding_decimals: int = 2
    retry_failed_splits_with_enclosed_zones: bool = True
    max_holes_for_failed_split_retry: int = 1
    final_corridor_min_width_m: float = 10.0
    final_corridor_min_component_area_m2: float = 25.0
    barrier_min_area_km2: float = 0.5
    coarse_min_area_km2: float = 0.5
    final_min_area_km2: float = 0.5
    final_max_area_km2: float = 1.5
    oriented_fallback_trigger_km2: float = 1.5
    oriented_fallback_min_area_km2: float = 0.5
    oriented_fallback_max_area_km2: float = 2.0
    divider_extension_m: float = 20_000.0
    road_snap_max_distance_m: float = 75.0
    divider_sample_spacing_m: float = 25.0
    max_split_depth: int = 15
    min_child_area_m2: float = 1.0
    figsize: tuple[int, int] = (14, 14)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate raster bands, area targets, output formats, and split limits."""
        if not isinstance(self.city_name, str) or not self.city_name.strip():
            raise ValueError("city_name must be a non-empty string")
        if not isinstance(self.download_google_buildings, bool):
            raise TypeError("download_google_buildings must be a bool")
        if (
            isinstance(self.building_data_year, bool)
            or not isinstance(self.building_data_year, int)
            or self.building_data_year not in range(2016, 2024)
        ):
            raise ValueError("building_data_year must be an integer from 2016 through 2023")
        if not 0 <= self.building_presence_threshold <= 1:
            raise ValueError("building_presence_threshold must be between 0 and 1")
        if min(
            self.building_count_band,
            self.building_height_band,
            self.building_presence_band,
        ) < 1:
            raise ValueError("building raster band indexes are 1-based positive integers")
        if self.building_max_samples < 2 or self.building_kmeans_n_init < 1:
            raise ValueError("building sample and KMeans initialization counts are too small")
        if not self.plot_formats or not set(self.plot_formats).issubset({"html", "png", "pdf"}):
            raise ValueError("plot_formats must contain html, png, or pdf")
        area_values = (
            self.barrier_min_area_km2,
            self.coarse_min_area_km2,
            self.final_min_area_km2,
            self.final_max_area_km2,
            self.oriented_fallback_min_area_km2,
            self.oriented_fallback_max_area_km2,
        )
        if any(not math.isfinite(value) or value <= 0 for value in area_values):
            raise ValueError("TAZ area thresholds must be finite and positive")
        if self.final_max_area_km2 < self.final_min_area_km2:
            raise ValueError("final_max_area_km2 cannot be smaller than final_min_area_km2")
        if self.oriented_fallback_max_area_km2 < self.oriented_fallback_min_area_km2:
            raise ValueError(
                "oriented_fallback_max_area_km2 cannot be smaller than its minimum"
            )
        if self.max_split_depth < 1 or self.max_coarse_cleanup_polygons < 1:
            raise ValueError("split depth and coarse cleanup limit must be positive")


Config = AutomaticTAZConfig


def _require_networkx():
    """Import graph helpers only when road-network processing is requested."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise TAZDependencyError("Automatic TAZ creation requires networkx") from exc
    return nx


def _require_pymetis():
    """Import the METIS binding only when recursive road splitting needs it."""
    try:
        import pymetis
    except ImportError as exc:
        raise TAZDependencyError(
            "METIS TAZ splitting requires the pymetis package"
        ) from exc
    return pymetis


def _area(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a copy of a metric GeoDataFrame with area in square kilometers."""
    result = frame.copy()
    result["area_km2"] = result.geometry.area / 1_000_000.0
    return result


def _lines(geometry: Any) -> list[LineString]:
    """Recursively collect line components from mixed Shapely geometry."""
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [line for part in geometry.geoms for line in _lines(part)]
    return []


def _polygons(geometry: Any) -> list[Polygon]:
    """Recursively collect polygon components from mixed Shapely geometry."""
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return [part for item in geometry.geoms for part in _polygons(item)]
    return []


def _gdf(geometries, crs) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame from non-empty geometries with the given CRS."""
    return gpd.GeoDataFrame(
        geometry=[g for g in geometries if g is not None and not g.is_empty], crs=crs
    )


def _export_layer(layer: gpd.GeoDataFrame, path: Path, cfg: AutomaticTAZConfig) -> Path:
    """Write one intermediate vector layer in the configured export CRS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output = layer.to_crs(cfg.export_crs)
    output.to_file(path, driver="GeoJSON")
    logging.info("Saved %d features: %s", len(layer), path)
    return path


def _plot_layers(stem: Path, cfg: AutomaticTAZConfig, layers, title: str) -> None:
    """Render a diagnostic layer stack to a static PNG when requested."""
    if not cfg.save_plots and not cfg.show_plots:
        return
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=cfg.figsize)
    for layer, style in layers:
        if layer is not None and not layer.empty:
            layer.plot(ax=axis, **style)
    axis.set_title(title)
    axis.set_axis_off()
    stem.parent.mkdir(parents=True, exist_ok=True)
    if cfg.save_plots and "png" in cfg.plot_formats:
        figure.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    if cfg.save_plots and "pdf" in cfg.plot_formats:
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    if cfg.save_plots and "html" in cfg.plot_formats:
        image = io.BytesIO()
        figure.savefig(image, format="png", dpi=160, bbox_inches="tight")
        encoded = base64.b64encode(image.getvalue()).decode("ascii")
        stem.with_suffix(".html").write_text(
            "<!doctype html><meta charset='utf-8'>"
            f"<title>{title}</title><h1>{title}</h1>"
            f"<img alt='{title}' style='max-width:100%' src='data:image/png;base64,{encoded}'>",
            encoding="utf-8",
        )
    if cfg.show_plots:
        plt.show()
    plt.close(figure)


def _centerlines(lines: gpd.GeoDataFrame, buffer_m: float, cfg: AutomaticTAZConfig):
    """Build buffered corridors and medial-axis centerlines from roads."""
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty]
    if lines.empty:
        return _gdf([], lines.crs), _gdf([], lines.crs)
    corridors = _gdf(_polygons(lines.geometry.buffer(buffer_m).union_all()), lines.crs)
    try:
        from centerline.geometry import Centerline
    except ImportError:
        # A buffered footprint boundary is a deterministic fallback when the
        # optional medial-axis package is unavailable.
        return corridors, _gdf(list(corridors.boundary), lines.crs)
    output = []
    for corridor in corridors.geometry:
        try:
            output.append(Centerline(
                corridor, interpolation_distance=cfg.centerline_interpolation_m
            ).geometry)
        except Exception as exc:
            logging.warning("Skipping centerline extraction: %s", exc)
    return corridors, _gdf(output, lines.crs)


def _polygonize(lines: gpd.GeoDataFrame, boundary, crs):
    """Polygonize barrier linework and retain faces inside the study boundary."""
    linework = boundary.boundary if lines.empty else lines.geometry.union_all().union(boundary.boundary)
    polygons, cuts, dangles, invalid = polygonize_full(linework)
    logging.info(
        "Polygonize: %d rings, %d cuts, %d dangles, %d invalid",
        len(polygons.geoms), len(cuts.geoms), len(dangles.geoms), len(invalid.geoms),
    )
    zones = _gdf(polygons.geoms, crs)
    zones = zones[zones.geometry.representative_point().within(boundary)].reset_index(drop=True)
    if zones.empty:
        zones = _gdf(_polygons(boundary), crs)
    return _area(zones), _gdf(dangles.geoms, crs)


def _merge_small(zones: gpd.GeoDataFrame, minimum: float):
    """Merge small touching coarse polygons into their longest-boundary neighbor."""
    zones = _area(zones).reset_index(drop=True)
    while True:
        small_ids = sorted(zones.index[zones.area_km2 < minimum], key=lambda i: zones.loc[i, "area_km2"])
        if not small_ids:
            return _area(zones)
        changed = False
        for index in small_ids:
            geom = zones.loc[index, "geometry"]
            neighbors = zones[(zones.index != index) & zones.geometry.touches(geom)].copy()
            if neighbors.empty:
                continue
            neighbors["shared_m"] = neighbors.geometry.apply(
                lambda other: geom.boundary.intersection(other.boundary).length
            )
            target = neighbors.shared_m.idxmax()
            zones.loc[target, "geometry"] = zones.loc[target, "geometry"].union(geom)
            zones = _area(zones.drop(index=index).reset_index(drop=True))
            changed = True
            break
        if not changed:
            logging.warning("Some polygons below %.2f km² have no touching neighbor", minimum)
            return _area(zones)


def _prune_terminal_branches(lines: gpd.GeoDataFrame, maximum: float, decimals: int):
    """Remove short dead-end branches from a centerline road graph."""
    nx = _require_networkx()
    working = _gdf(_lines(unary_union(list(lines.geometry))), lines.crs).reset_index(drop=True)
    while not working.empty:
        graph = nx.Graph()
        for index, line in working.geometry.items():
            coords = list(line.coords)
            a = tuple(round(v, decimals) for v in coords[0])
            b = tuple(round(v, decimals) for v in coords[-1])
            if a != b:
                graph.add_edge(a, b, line_id=index, length_m=line.length)
        remove = set()
        for start in (node for node, degree in graph.degree() if degree == 1):
            current, previous, branch, length = start, None, [], 0.0
            while True:
                next_nodes = [node for node in graph.neighbors(current) if node != previous]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                edge = graph.get_edge_data(current, nxt)
                branch.append(edge["line_id"])
                length += edge["length_m"]
                previous, current = current, nxt
                if graph.degree(current) != 2:
                    break
            if length <= maximum:
                remove.update(branch)
        if not remove:
            break
        working = working.drop(index=list(remove)).reset_index(drop=True)
    return working


def _sample_line(line: LineString, spacing: float):
    """Interpolate points at a bounded spacing along a line."""
    return [line.interpolate(distance) for distance in np.linspace(
        0, line.length, max(2, int(np.ceil(line.length / spacing)) + 1)
    )]


def _aligned_divider(divider: LineString, roads: gpd.GeoDataFrame, cfg):
    """Snap divider samples to nearby roads while preserving the line path."""
    if roads.empty:
        return divider
    road_union = roads.geometry.union_all()
    coords = []
    for point in _sample_line(divider, cfg.divider_sample_spacing_m):
        if point.distance(road_union) <= cfg.road_snap_max_distance_m:
            _, snapped = nearest_points(point, road_union)
            coords.append((snapped.x, snapped.y))
        else:
            coords.append((point.x, point.y))
    deduped = [coord for i, coord in enumerate(coords) if i == 0 or coord != coords[i - 1]]
    return LineString(deduped) if len(deduped) >= 2 else divider


def _split_all(parent: Polygon, divider: LineString, minimum: float):
    """Split a parent polygon and return polygonal children above a size floor."""
    try:
        pieces = split(parent, divider).geoms
    except Exception:
        return []
    children = []
    for piece in pieces:
        children.extend([poly for poly in _polygons(piece.intersection(parent)) if poly.area > minimum])
    return children


def _merge_children(children, crs, minimum):
    """Merge undersized split children until no useful small merge remains."""
    zones = _area(_gdf(children, crs)).reset_index(drop=True)
    while len(zones) > 1:
        small = zones[zones.area_km2 < minimum]
        if small.empty:
            break
        source = small.area_km2.idxmin()
        geom = zones.loc[source, "geometry"]
        candidates = zones[(zones.index != source) & zones.geometry.touches(geom)].copy()
        if not candidates.empty:
            candidates["shared_m"] = candidates.geometry.apply(
                lambda other: geom.boundary.intersection(other.boundary).length
            )
            candidates = candidates[candidates.shared_m > 0]
        target = (candidates.shared_m.idxmax() if not candidates.empty
                  else zones.drop(index=source).geometry.distance(geom).idxmin())
        zones.loc[target, "geometry"] = zones.loc[target, "geometry"].union(geom)
        zones = _area(zones.drop(index=source).reset_index(drop=True))
    return list(zones.geometry)


def _building_raster_paths(source: Path | str | None) -> list[Path]:
    """Resolve one raster or a directory of Google 2.5D GeoTIFF tiles."""
    if source is None:
        return []
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            (item for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
             for item in path.glob(pattern)),
            key=lambda item: item.name,
        )
    raise FileNotFoundError(f"Building raster path does not exist: {path}")


@lru_cache(maxsize=32)
def _building_raster_coverages(paths: tuple[str, ...], metric_crs: str):
    """Cache tile footprints so each split opens only intersecting TIFFs."""
    try:
        import rasterio
    except ImportError as exc:
        raise TAZDependencyError(
            "Building-aware TAZ creation requires rasterio and scikit-learn"
        ) from exc
    coverages = []
    for path in paths:
        with rasterio.open(path) as source:
            if source.crs is None:
                logging.warning("Skipping building raster without a CRS: %s", path)
                continue
            coverage = gpd.GeoSeries(
                [Polygon.from_bounds(*source.bounds)], crs=source.crs
            ).to_crs(metric_crs).iloc[0]
            coverages.append((path, coverage))
    return tuple(coverages)


def _building_cluster_centers(parent: Polygon, metric_crs, cfg: AutomaticTAZConfig):
    """Return two height-weighted building clusters from intersecting tiles.

    Google 2.5D tiles have fractional-count, height, and presence bands. The
    reader also supports a legacy one-band positive-weight raster. Large
    0.5-metre windows are downsampled before clustering so a city run does not
    materialize millions of pixels for each candidate TAZ.
    """
    paths = _building_raster_paths(cfg.building_raster_path)
    if not paths:
        return None
    if cfg.building_max_samples < 2:
        raise ValueError("building_max_samples must be at least 2")
    try:
        import rasterio
        from affine import Affine
        from rasterio.enums import Resampling
        from rasterio.features import geometry_mask
        from rasterio.windows import Window, from_bounds
        from rasterio.warp import transform as transform_coordinates
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise TAZDependencyError(
            "Building-aware TAZ creation requires rasterio and scikit-learn"
        ) from exc

    coverages = _building_raster_coverages(
        tuple(str(path) for path in paths), str(metric_crs)
    )
    paths = [Path(path) for path, coverage in coverages if coverage.intersects(parent)]
    if not paths:
        return None
    coordinate_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    samples_per_tile = max(2, int(np.ceil(cfg.building_max_samples / len(paths))))
    for path in paths:
        with rasterio.open(path) as source:
            if source.crs is None:
                logging.warning("Skipping building raster without a CRS: %s", path)
                continue
            parent_raster = gpd.GeoSeries([parent], crs=metric_crs).to_crs(source.crs).iloc[0]
            raster_bounds = Polygon.from_bounds(*source.bounds)
            if not raster_bounds.intersects(parent_raster):
                continue
            window = from_bounds(
                *parent_raster.bounds, transform=source.transform
            ).round_offsets().round_lengths()
            try:
                window = window.intersection(Window(0, 0, source.width, source.height))
            except rasterio.errors.WindowError:
                continue
            if window.width <= 0 or window.height <= 0:
                continue

            scale = max(1.0, np.sqrt(window.width * window.height / samples_per_tile))
            output_width = max(1, int(np.ceil(window.width / scale)))
            output_height = max(1, int(np.ceil(window.height / scale)))
            if source.count >= max(
                cfg.building_count_band,
                cfg.building_height_band,
                cfg.building_presence_band,
            ):
                indexes = [
                    cfg.building_count_band,
                    cfg.building_height_band,
                    cfg.building_presence_band,
                ]
            else:
                indexes = [1]
            values = source.read(
                indexes,
                window=window,
                out_shape=(len(indexes), output_height, output_width),
                masked=True,
                resampling=Resampling.average,
            )
            transform = source.window_transform(window) @ Affine.scale(
                window.width / output_width, window.height / output_height
            )
            inside = geometry_mask(
                [parent_raster.__geo_interface__],
                out_shape=(output_height, output_width),
                transform=transform,
                invert=True,
            )
            valid = inside & ~np.any(np.ma.getmaskarray(values), axis=0)
            if len(indexes) == 3:
                count, height, presence = (np.asarray(values[index].data) for index in range(3))
                valid &= (
                    np.isfinite(count)
                    & np.isfinite(height)
                    & np.isfinite(presence)
                    & (height > 0)
                    & (presence >= cfg.building_presence_threshold)
                )
                weights = height * np.maximum(count, np.finfo(float).eps)
            else:
                weights = np.asarray(values[0].data)
                valid &= np.isfinite(weights) & (weights > 0)
            rows, columns = np.where(valid)
            if not len(rows):
                continue
            xs, ys = rasterio.transform.xy(transform, rows, columns, offset="center")
            if str(source.crs) != str(metric_crs):
                xs, ys = transform_coordinates(source.crs, metric_crs, xs, ys)
            coordinate_parts.append(np.column_stack([xs, ys]))
            weight_parts.append(np.asarray(weights[valid], dtype=float))

    if not coordinate_parts:
        return None
    coordinates = np.concatenate(coordinate_parts)
    weights = np.concatenate(weight_parts)
    if len(coordinates) > cfg.building_max_samples:
        keep = np.linspace(
            0, len(coordinates) - 1, cfg.building_max_samples, dtype=int
        )
        coordinates, weights = coordinates[keep], weights[keep]
    if len(coordinates) < 2 or np.unique(coordinates, axis=0).shape[0] < 2:
        return None
    labels = KMeans(
        n_clusters=2, random_state=42, n_init=cfg.building_kmeans_n_init
    ).fit(coordinates, sample_weight=weights).labels_
    centers = [
        Point(
            np.average(coordinates[labels == label, 0], weights=weights[labels == label]),
            np.average(coordinates[labels == label, 1], weights=weights[labels == label]),
        )
        for label in (0, 1)
    ]
    return {
        "b0": centers[0],
        "b1": centers[1],
        "n_building_cells": len(coordinates),
    }


def _divider_from_axis(parent: Polygon, midpoint: Point, axis, cfg):
    """Extend a line perpendicular to an axis through a polygon midpoint."""
    axis = np.asarray(axis, dtype=float)
    norm = np.hypot(axis[0], axis[1])
    if norm == 0:
        return None
    midpoint = midpoint if parent.covers(midpoint) else parent.representative_point()
    ux, uy = axis / norm
    px, py = -uy, ux
    return LineString([
        (midpoint.x - px * cfg.divider_extension_m, midpoint.y - py * cfg.divider_extension_m),
        (midpoint.x + px * cfg.divider_extension_m, midpoint.y + py * cfg.divider_extension_m),
    ])


def _combined_divider(parent, c0, c1, building, cfg):
    """Combine aligned METIS-road and building-cluster directions."""
    metis_axis = np.array([c1.x - c0.x, c1.y - c0.y], dtype=float)
    if np.hypot(*metis_axis) == 0:
        return None, None
    metis_midpoint = Point((c0.x + c1.x) / 2, (c0.y + c1.y) / 2)
    info = {
        "metis_midpoint": metis_midpoint,
        "building_midpoint": None,
        "b0": None,
        "b1": None,
        "building_used": building is not None,
    }
    midpoint, axis = metis_midpoint, metis_axis
    if building is not None:
        b0, b1 = building["b0"], building["b1"]
        building_axis = np.array([b1.x - b0.x, b1.y - b0.y], dtype=float)
        building_midpoint = Point((b0.x + b1.x) / 2, (b0.y + b1.y) / 2)
        if np.hypot(*building_axis) == 0:
            info["building_used"] = False
        else:
            if np.dot(metis_axis, building_axis) < 0:
                building_axis = -building_axis
            midpoint = Point(
                (metis_midpoint.x + building_midpoint.x) / 2,
                (metis_midpoint.y + building_midpoint.y) / 2,
            )
            axis = (
                metis_axis / np.hypot(*metis_axis)
                + building_axis / np.hypot(*building_axis)
            )
            info.update({"building_midpoint": building_midpoint, "b0": b0, "b1": b1})
    divider = _divider_from_axis(parent, midpoint, axis, cfg)
    if divider is not None:
        info["divider_midpoint"] = (
            midpoint if parent.covers(midpoint) else parent.representative_point()
        )
    return divider, info


def _metis_split(parent, local_roads, crs, cfg):
    """Partition a clipped road graph and construct a road-aligned divider."""
    nx = _require_networkx()
    pymetis = _require_pymetis()
    roads = gpd.clip(local_roads, _gdf([parent], crs))
    segments = _lines(unary_union(list(roads.geometry))) if not roads.empty else []
    graph = nx.Graph()
    for segment in segments:
        coords = list(segment.coords)
        a = tuple(round(v, cfg.node_rounding_decimals) for v in coords[0])
        b = tuple(round(v, cfg.node_rounding_decimals) for v in coords[-1])
        if a != b:
            graph.add_edge(a, b)
    if graph.number_of_nodes() < 2:
        return None, "no_usable_local_road_graph", roads
    component = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(component).copy()
    if graph.number_of_nodes() < 2:
        return None, "local_road_component_too_small", roads
    nodes = list(graph.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    try:
        _, membership = pymetis.part_graph(
            2, adjacency=[[node_index[n] for n in graph.neighbors(node)] for node in nodes]
        )
    except Exception as exc:
        return None, f"metis_error:{exc}", roads
    groups = [[Point(nodes[i]) for i in range(len(nodes)) if membership[i] == label] for label in (0, 1)]
    if not groups[0] or not groups[1]:
        return None, "empty_metis_partition", roads
    centers = [unary_union(group).centroid for group in groups]
    building = _building_cluster_centers(parent, crs, cfg)
    divider, divider_info = _combined_divider(
        parent, centers[0], centers[1], building, cfg
    )
    if divider is None:
        return None, "identical_metis_centroids", roads
    hybrid = _aligned_divider(divider, roads, cfg)
    children = _split_all(parent, hybrid, cfg.min_child_area_m2)
    if len(children) == 2:
        return {
            "children": children,
            "method": "metis_building_road_aligned" if divider_info["building_used"] else "metis_road_aligned",
            "divider": hybrid,
            "reason": "",
            **divider_info,
        }, "", roads
    children = _split_all(parent, divider, cfg.min_child_area_m2)
    if len(children) == 2:
        return {
            "children": children,
            "method": "metis_building_straight" if divider_info["building_used"] else "metis_straight",
            "divider": divider,
            "reason": "road_aligned_divider_failed",
            **divider_info,
        }, "", roads
    return None, "metis_divider_did_not_make_two_children", roads


def _building_split(parent, local_roads, crs, cfg):
    """Split from building clusters when a usable road graph is unavailable."""
    building = _building_cluster_centers(parent, crs, cfg)
    if building is None:
        return None, "no_usable_building_clusters"
    b0, b1 = building["b0"], building["b1"]
    midpoint = Point((b0.x + b1.x) / 2, (b0.y + b1.y) / 2)
    divider = _divider_from_axis(
        parent, midpoint, [b1.x - b0.x, b1.y - b0.y], cfg
    )
    if divider is None:
        return None, "identical_building_centroids"
    roads = gpd.clip(local_roads, _gdf([parent], crs))
    hybrid = _aligned_divider(divider, roads, cfg)
    children = _split_all(parent, hybrid, cfg.min_child_area_m2)
    if len(children) == 2:
        return {
            "children": children,
            "method": "building_road_aligned",
            "divider": hybrid,
            "reason": "",
            "n_building_cells": building["n_building_cells"],
        }, ""
    children = _split_all(parent, divider, cfg.min_child_area_m2)
    if len(children) == 2:
        return {
            "children": children,
            "method": "building_straight",
            "divider": divider,
            "reason": "road_aligned_divider_failed",
            "n_building_cells": building["n_building_cells"],
        }, ""
    return None, "building_divider_did_not_make_two_children"


def _oriented_split(parent, local_roads, crs, cfg):
    """Split a polygon along its principal axis, optionally snapping to roads."""
    hull = parent.convex_hull
    coords = np.asarray(hull.exterior.coords)[:, :2]
    _, vectors = np.linalg.eigh(np.cov((coords - coords.mean(axis=0)).T))
    ux, uy = vectors[:, -1]
    point = parent.representative_point()
    px, py = -uy, ux
    straight = LineString([
        (point.x - px * cfg.divider_extension_m, point.y - py * cfg.divider_extension_m),
        (point.x + px * cfg.divider_extension_m, point.y + py * cfg.divider_extension_m),
    ])
    roads = gpd.clip(local_roads, _gdf([parent], crs))
    hybrid = _aligned_divider(straight, roads, cfg)
    raw = _split_all(parent, hybrid, cfg.min_child_area_m2)
    children = _merge_children(raw, crs, cfg.oriented_fallback_min_area_km2)
    if len(children) >= 2:
        return {"children": children, "method": "oriented_road_aligned", "divider": hybrid,
                "raw_child_count": len(raw), "merged_child_count": len(raw) - len(children),
                "roads": roads}
    raw = _split_all(parent, straight, cfg.min_child_area_m2)
    children = _merge_children(raw, crs, cfg.oriented_fallback_min_area_km2)
    if len(children) >= 2:
        return {"children": children, "method": "oriented_straight", "divider": straight,
                "raw_child_count": len(raw), "merged_child_count": len(raw) - len(children),
                "roads": roads}
    return None


def _oriented_recursive(parent, roads, crs, cfg, coarse_id, depth, failure):
    """Recursively apply the oriented fallback and retain failure metadata."""
    current = parent.area / 1_000_000.0
    base = {"coarse_zone_id": coarse_id, "depth": depth, "area_km2": current}
    if cfg.oriented_fallback_min_area_km2 <= current <= cfg.oriented_fallback_max_area_km2:
        return [{**base, "geometry": parent, "status": "oriented_valid", "split_method": "oriented", "failure_reason": failure}], []
    if current < cfg.oriented_fallback_min_area_km2 or depth >= cfg.max_split_depth:
        status = "oriented_undersized" if current < cfg.oriented_fallback_min_area_km2 else "oriented_max_depth"
        return [{**base, "geometry": parent, "status": status, "split_method": "oriented", "failure_reason": failure}], []
    info = _oriented_split(parent, roads, crs, cfg)
    if info is None:
        return [{**base, "geometry": parent, "status": "oriented_split_failed", "split_method": "oriented", "failure_reason": failure}], []
    divider = {"geometry": info["divider"], "coarse_zone_id": coarse_id, "depth": depth,
               "method": info["method"], "metis_failure_reason": failure,
               "raw_child_count": info["raw_child_count"], "merged_undersized_children": info["merged_child_count"]}
    records, dividers = [], [divider]
    child_roads = info.get("roads", roads)
    for child in info["children"]:
        child_records, child_dividers = _oriented_recursive(child, child_roads, crs, cfg, coarse_id, depth + 1, failure)
        records.extend(child_records)
        dividers.extend(child_dividers)
    return records, dividers


def _building_recursive(parent, roads, crs, cfg, coarse_id, depth, failure):
    """Try building-only recursion, then retain the PCA fallback."""
    current = parent.area / 1_000_000.0
    base = {"coarse_zone_id": coarse_id, "depth": depth, "area_km2": current}
    if cfg.final_min_area_km2 <= current <= cfg.final_max_area_km2:
        return [{
            **base,
            "geometry": parent,
            "status": "building_valid",
            "split_method": "building",
            "failure_reason": failure,
        }], []
    if current < cfg.final_min_area_km2:
        return [{
            **base,
            "geometry": parent,
            "status": "building_undersized",
            "split_method": "building",
            "failure_reason": failure,
        }], []
    if depth >= cfg.max_split_depth:
        return _oriented_recursive(
            parent, roads, crs, cfg, coarse_id, depth,
            f"{failure};building_max_depth",
        )
    info, building_failure = _building_split(parent, roads, crs, cfg)
    if info is None:
        return _oriented_recursive(
            parent, roads, crs, cfg, coarse_id, depth,
            f"{failure};{building_failure}",
        )
    divider = {
        "geometry": info["divider"],
        "coarse_zone_id": coarse_id,
        "depth": depth,
        "method": info["method"],
        "metis_failure_reason": failure,
        "n_building_cells": info["n_building_cells"],
    }
    records, dividers = [], [divider]
    for child in info["children"]:
        child_records, child_dividers = _building_recursive(
            child, roads, crs, cfg, coarse_id, depth + 1, failure
        )
        records.extend(child_records)
        dividers.extend(child_dividers)
    return records, dividers


def _recursive_split(parent, roads, crs, cfg, coarse_id, depth=0):
    """Recursively split one coarse polygon using METIS and fallback methods."""
    current = parent.area / 1_000_000.0
    base = {"coarse_zone_id": coarse_id, "depth": depth, "area_km2": current}
    if cfg.final_min_area_km2 <= current <= cfg.final_max_area_km2:
        return [{**base, "geometry": parent, "status": "valid", "split_method": "none", "failure_reason": ""}], []
    if current < cfg.final_min_area_km2:
        return [{**base, "geometry": parent, "status": "undersized", "split_method": "none", "failure_reason": "below_minimum_before_split"}], []
    if depth >= cfg.max_split_depth:
        return [{**base, "geometry": parent, "status": "max_depth", "split_method": "none", "failure_reason": "max_depth_reached"}], []
    info, reason, parent_roads = _metis_split(parent, roads, crs, cfg)
    if info is None:
        if current > cfg.oriented_fallback_trigger_km2:
            if cfg.building_raster_path is not None:
                return _building_recursive(
                    parent, parent_roads, crs, cfg, coarse_id, depth, reason
                )
            return _oriented_recursive(
                parent, parent_roads, crs, cfg, coarse_id, depth, reason
            )
        return [{**base, "geometry": parent, "status": "split_failed", "split_method": "none", "failure_reason": reason}], []
    divider = {"geometry": info["divider"], "coarse_zone_id": coarse_id, "depth": depth,
               "method": info["method"], "metis_failure_reason": info["reason"],
               "building_used": info["building_used"]}
    records, dividers = [], [divider]
    for child in info["children"]:
        child_records, child_dividers = _recursive_split(child, parent_roads, crs, cfg, coarse_id, depth + 1)
        records.extend(child_records)
        dividers.extend(child_dividers)
    return records, dividers


def _merge_final_undersized(zones, minimum, crs):
    """Absorb every final sub-minimum zone after recursive splitting."""
    working = _area(zones).reset_index(drop=True)
    audit = []
    while len(working) > 1:
        undersized = working[working.area_km2 < minimum]
        if undersized.empty:
            break
        source = undersized.area_km2.idxmin()
        source_geom = working.loc[source, "geometry"]
        candidates = working.drop(index=source).copy()
        touching = candidates[candidates.geometry.touches(source_geom)].copy()
        if not touching.empty:
            touching["shared_boundary_m"] = touching.geometry.apply(
                lambda geom: geom.boundary.intersection(source_geom.boundary).length
            )
            touching = touching[touching.shared_boundary_m > 0]
        if touching.empty:
            target = candidates.geometry.distance(source_geom).idxmin()
            merge_basis, shared_boundary = "nearest_fallback", 0.0
        else:
            target = touching.shared_boundary_m.idxmax()
            merge_basis = "longest_shared_boundary"
            shared_boundary = float(touching.loc[target, "shared_boundary_m"])
        audit.append({
            "geometry": source_geom,
            "area_km2_before_merge": working.loc[source, "area_km2"],
            "final_min_area_km2": minimum,
            "merged_into_premerge_index": int(target),
            "merge_basis": merge_basis,
            "shared_boundary_m": shared_boundary,
        })
        working.loc[target, "geometry"] = working.loc[target, "geometry"].union(source_geom)
        working = _area(working.drop(index=source).reset_index(drop=True))
    audit_frame = (
        gpd.GeoDataFrame(audit, geometry="geometry", crs=crs)
        if audit else _gdf([], crs)
    )
    return working, audit_frame


def _narrow_corridors(zones, width, minimum, crs):
    """Transfer narrow corridor slivers to adjacent zones and audit changes."""
    working = _area(zones).reset_index(drop=True)
    audit, protected = [], set()
    for source in list(working.index):
        if source not in working.index or source in protected or len(working) <= 1:
            continue
        source_geom = unary_union(_polygons(make_valid(working.loc[source, "geometry"])))
        if source_geom.is_empty:
            continue
        core_parts = _polygons(source_geom.buffer(-width / 2))
        if not core_parts:
            corridors = [source_geom]
        elif len(core_parts) == 1:
            continue
        else:
            eroded = make_valid(source_geom.buffer(-width / 2))
            restored = make_valid(eroded.buffer(width / 2).intersection(source_geom))
            candidates = _polygons(source_geom.difference(restored))
            cores = _polygons(restored)
            corridors = [part for part in candidates if part.area >= minimum and
                         sum(part.distance(core) <= 0.01 for core in cores) >= 2]
        remaining = source_geom
        for corridor in corridors:
            candidates = working.drop(index=source)
            candidates = candidates[candidates.geometry.notna() & ~candidates.geometry.is_empty]
            if candidates.empty:
                continue
            distances = candidates.geometry.distance(corridor)
            nearest_distance = distances.min()
            target = distances[np.isclose(distances, nearest_distance)].index[0]
            remaining = remaining.difference(corridor)
            working.loc[target, "geometry"] = working.loc[target, "geometry"].union(corridor)
            protected.add(target)
            audit.append({"geometry": corridor, "corridor_area_m2": corridor.area,
                          "corridor_width_test_m": width, "source_premerge_index": int(source),
                          "merged_into_premerge_index": int(target), "nearest_distance_m": float(nearest_distance)})
        if remaining.is_empty:
            working = working.drop(index=source)
        else:
            working.loc[source, "geometry"] = remaining
    working = _area(working.explode(index_parts=False).reset_index(drop=True))
    audit_gdf = gpd.GeoDataFrame(audit, geometry="geometry", crs=crs) if audit else _gdf([], crs)
    return working[working.geometry.notna() & ~working.geometry.is_empty], audit_gdf


def _retry_failed(records, dividers, roads, crs, cfg):
    """Retry failed polygons after absorbing zones trapped in small holes."""
    result = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    absorbed = []
    failed_ids = result.index[result.status.isin(["split_failed", "oriented_split_failed"]) & result.geometry.apply(
        lambda geom: 0 < len(getattr(geom, "interiors", [])) <= cfg.max_holes_for_failed_split_retry
    )].tolist()
    for failed_id in failed_ids:
        if failed_id not in result.index:
            continue
        failed = result.loc[failed_id]
        holes = [Polygon(ring) for ring in failed.geometry.interiors]
        inner_ids = [index for index, candidate in result.drop(index=failed_id).iterrows()
                     if any(hole.contains(candidate.geometry.representative_point()) for hole in holes)]
        if not inner_ids:
            continue
        for inner_id in inner_ids:
            audit = result.loc[inner_id].copy()
            audit["absorbed_for_failed_coarse_zone_id"] = failed.coarse_zone_id
            audit["absorbed_for_failed_status"] = failed.status
            absorbed.append(audit)
        filled = unary_union([failed.geometry, *result.loc[inner_ids].geometry.tolist()])
        retry_records, retry_dividers = _recursive_split(filled, roads, crs, cfg, int(failed.coarse_zone_id), int(failed.depth))
        result = result.drop(index=[failed_id, *inner_ids])
        result = gpd.GeoDataFrame(pd.concat([result, gpd.GeoDataFrame(retry_records, geometry="geometry", crs=crs)], ignore_index=True), geometry="geometry", crs=crs)
        dividers.extend(retry_dividers)
    absorbed_gdf = gpd.GeoDataFrame(absorbed, geometry="geometry", crs=crs) if absorbed else _gdf([], crs)
    return _area(result), dividers, absorbed_gdf


def _normalize_highway(value):
    """Normalize OSM highway tags from strings, lists, or cached JSON values."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return value[0] if len(value) == 1 else tuple(value)
    if isinstance(value, tuple):
        return value
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return tuple(parsed) if isinstance(parsed, list) else parsed
    return value


def _load_boundary(boundary):
    """Read or copy a boundary and dissolve it to one polygonal study extent."""
    if isinstance(boundary, StudyArea):
        frame = boundary.to_geopandas()
    elif isinstance(boundary, gpd.GeoDataFrame):
        frame = boundary.copy()
    else:
        frame = read_spatial_layer(boundary)
    if frame.empty or frame.crs is None:
        raise TAZGeometryError("Automatic TAZ boundary must be non-empty and have a CRS")
    dissolved = frame.geometry.union_all()
    return StudyArea(dissolved, frame.crs).to_geopandas()


def _cache_highway(value):
    """Serialize compound highway tags for a portable road-edge cache."""
    value = _normalize_highway(value)
    return json.dumps(list(value)) if isinstance(value, tuple) else value


def _road_edge_cache_path(cfg: AutomaticTAZConfig) -> Path:
    """Resolve the configured or default cached OSM edge file path."""
    return Path(cfg.osm_cache_path) if cfg.osm_cache_path else cfg.output_dir / "osm_road_edges.geojson"


def _download_osm_edges(boundary: gpd.GeoDataFrame, cfg: AutomaticTAZConfig) -> gpd.GeoDataFrame:
    """Download and cache road edges for the boundary using OSMnx."""
    try:
        import osmnx as ox
    except ImportError as exc:
        raise TAZDependencyError(
            "Boundary-only automatic TAZ creation requires the osmnx package"
        ) from exc
    polygon = boundary.to_crs("EPSG:4326").geometry.union_all()
    logging.info("Downloading %s OSM road network for the supplied boundary", cfg.network_type)
    graph = ox.graph_from_polygon(
        polygon, network_type=cfg.network_type, simplify=True, retain_all=True
    )
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True).reset_index(drop=True)
    if edges.empty or "highway" not in edges.columns:
        raise TAZGeometryError("OSM road network contains no edges with highway classifications")
    cache_path = _road_edge_cache_path(cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = edges[["highway", "geometry"]].copy()
    cached["highway"] = cached["highway"].map(_cache_highway)
    cached.to_file(cache_path, driver="GeoJSON")
    logging.info("Cached %d OSM road edges at %s", len(cached), cache_path)
    return edges


def _load_edges(graph, boundary: gpd.GeoDataFrame, cfg: AutomaticTAZConfig):
    """Resolve road edges from a layer, graph, cache, or boundary download."""
    if isinstance(graph, gpd.GeoDataFrame):
        return graph.copy()
    if graph is None:
        cache_path = _road_edge_cache_path(cfg)
        if cache_path.exists():
            logging.info("Loading cached OSM road edges from %s", cache_path)
            return read_spatial_layer(cache_path)
        return _download_osm_edges(boundary, cfg)
    if isinstance(graph, (str, Path)):
        graph_path = Path(graph)
        if graph_path.suffix.lower() in {".geojson", ".json", ".gpkg", ".shp", ".parquet"}:
            return read_spatial_layer(graph_path)
    try:
        import osmnx as ox
    except ImportError as exc:
        raise TAZDependencyError(
            "Graph input requires the osmnx package"
        ) from exc
    if isinstance(graph, (str, Path)):
        with Path(graph).open("rb") as file:
            graph = pickle.load(file)
    return ox.graph_to_gdfs(graph, nodes=False, edges=True)


def create_automatic_taz(
    boundary: Any,
    graph: Any = None,
    *,
    config: AutomaticTAZConfig | None = None,
    **overrides: Any,
) -> TAZFrame:
    """Create road-aware automatic TAZs from a boundary.

    ``boundary`` may be a StudyArea, GeoDataFrame, or vector file accepted by
    GeoPandas (including the KML workflow in the reference bundle). ``graph``
    may be an OSMnx graph, a pickled OSMnx graph, a vector path, or a road-edge
    GeoDataFrame. If omitted, OSM roads are downloaded and cached in the
    configured output directory. A local building raster or tile directory
    enables building-aware recursive splits. Alternatively,
    ``download_google_buildings=True`` downloads intersecting Google 2.5D
    tiles for the configured year into a reusable local cache.

    Args:
        boundary: Polygonal study boundary with a declared CRS.
        graph: Optional road graph or road edge layer.
        config: Automatic TAZ settings. A copy is used; the caller's object is
            not mutated.
        **overrides: Fields that replace values on ``config``.

    Returns:
        A TAZFrame. Intermediate GeoJSON layers and requested static plots are
        written under ``config.output_dir`` as part of this explicit workflow.

    Raises:
        TypeError: If an override is not a configuration field.
        ValueError: If raster bands, plot formats, or split thresholds are
            invalid.
        TAZDependencyError: If an optional road, raster, or clustering
            dependency is unavailable.
        TAZGeometryError: If input or generated geometry is unusable.
        FileNotFoundError: If Google publishes no matching tiles for the
            configured boundary and year.
    """
    if config is not None and not isinstance(config, AutomaticTAZConfig):
        raise TypeError("config must be an AutomaticTAZConfig instance")
    valid_fields = set(AutomaticTAZConfig.__dataclass_fields__)
    unknown = set(overrides) - valid_fields
    if unknown:
        raise TypeError(f"Unknown AutomaticTAZConfig option(s): {sorted(unknown)}")
    base_config = config if config is not None else AutomaticTAZConfig()
    cfg = replace(base_config, **overrides)
    cfg.output_dir = Path(cfg.output_dir)
    if cfg.building_presence_threshold < 0 or cfg.building_presence_threshold > 1:
        raise ValueError("building_presence_threshold must be between 0 and 1")
    if cfg.final_min_area_km2 <= 0 or cfg.final_max_area_km2 < cfg.final_min_area_km2:
        raise ValueError("final area limits must be positive and ordered")
    if cfg.building_raster_path is not None:
        cfg.building_raster_path = Path(cfg.building_raster_path)
        raster_paths = _building_raster_paths(cfg.building_raster_path)
        if not raster_paths:
            raise FileNotFoundError(
                f"No TIFF files found under building raster path: {cfg.building_raster_path}"
            )
        logging.info("Using %d building raster tile(s) from %s", len(raster_paths), cfg.building_raster_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    plots = cfg.output_dir / "plots"
    city = _load_boundary(boundary)
    building_source = "local" if cfg.building_raster_path is not None else None
    building_download_info = None
    if cfg.download_google_buildings and cfg.building_raster_path is None:
        from opengeomove.data_loading.google_buildings import (
            download_open_buildings_25d,
        )

        cache_root = (
            Path(cfg.building_cache_dir)
            if cfg.building_cache_dir is not None
            else cfg.output_dir / "building_rasters"
        )
        raster_cache = cache_root / str(cfg.building_data_year)
        downloaded_tiles = download_open_buildings_25d(
            city, cache_root, year=cfg.building_data_year
        )
        cfg.building_raster_path = raster_cache
        building_source = "google_open_buildings_2_5d"
        building_download_info = {
            "source": building_source,
            "year": cfg.building_data_year,
            "tile_paths": [str(path) for path in downloaded_tiles],
            "cache_dir": str(raster_cache),
            "license_options": ["CC-BY-4.0", "ODbL-1.0"],
            "source_url": "https://sites.research.google/gr/open-buildings/temporal",
        }
        logging.info(
            "Using %d cached Google 2.5D raster tile(s) from %s",
            len(downloaded_tiles), raster_cache,
        )
    metric_crs = city.estimate_utm_crs()
    if metric_crs is None:
        raise TAZGeometryError("Could not estimate a metric CRS for the boundary")
    city_m = city.to_crs(metric_crs)
    city_geom = city_m.geometry.union_all()
    edges = _load_edges(graph, city, cfg)
    if edges.crs is None:
        raise TAZGeometryError("Road graph edges must have a CRS")
    edges = edges.to_crs(metric_crs).copy()
    highway = edges.get("highway", pd.Series(index=edges.index, dtype="object"))
    edges["highway_type"] = highway.map(_normalize_highway)
    major = edges[edges.highway_type.isin(cfg.major_road_types)].copy()
    local = edges[edges.highway_type.isin(cfg.local_road_types)].copy()
    _export_layer(city_m, cfg.output_dir / "00_boundary.geojson", cfg)
    _export_layer(major, cfg.output_dir / "01_major_roads.geojson", cfg)
    _export_layer(local, cfg.output_dir / "02_local_roads.geojson", cfg)
    major_corridors, major_centerlines = _centerlines(major, cfg.major_road_buffer_m, cfg)
    _export_layer(major_corridors, cfg.output_dir / "03_major_road_corridors.geojson", cfg)
    _export_layer(major_centerlines, cfg.output_dir / "04_major_road_centerlines.geojson", cfg)
    road_polygons, road_dangles = _polygonize(major_centerlines, city_geom, metric_crs)
    road_barriers = road_polygons[road_polygons.area_km2 >= cfg.barrier_min_area_km2].copy()
    road_barriers.geometry = road_barriers.boundary
    _export_layer(road_polygons, cfg.output_dir / "05_major_road_polygons.geojson", cfg)
    _export_layer(road_dangles, cfg.output_dir / "05a_major_road_dangles.geojson", cfg)
    _export_layer(road_barriers, cfg.output_dir / "06_major_road_barriers.geojson", cfg)
    natural = _gdf([], metric_crs)
    if cfg.fetch_railways_and_waterways:
        try:
            import osmnx as ox
            features = ox.features_from_polygon(city.to_crs(cfg.export_crs).geometry.iloc[0], tags={"railway": True, "waterway": True})
            if features is not None and not features.empty:
                features = gpd.clip(features.to_crs(metric_crs), city_m)
                linear = features.geometry.geom_type.isin(["LineString", "MultiLineString"])
                railway = features[linear & features.get("railway", pd.Series(index=features.index, dtype="object")).isin(cfg.railway_types)]
                water = features[linear & features.get("waterway", pd.Series(index=features.index, dtype="object")).notna()]
                natural = gpd.GeoDataFrame(pd.concat([railway[["geometry"]], water[["geometry"]]], ignore_index=True), geometry="geometry", crs=metric_crs)
        except Exception as exc:
            logging.warning("Natural-feature retrieval failed; continuing with roads: %s", exc)
    natural_corridors, natural_centerlines = _centerlines(natural, cfg.barrier_buffer_m, cfg)
    natural_polygons, natural_dangles = _polygonize(natural_centerlines, city_geom, metric_crs)
    natural_barriers = natural_polygons[natural_polygons.area_km2 >= cfg.barrier_min_area_km2].copy()
    natural_barriers.geometry = natural_barriers.boundary
    for name, frame in [("07_raw_railway_waterway_lines", natural), ("08_railway_waterway_corridors", natural_corridors),
                        ("09_railway_waterway_centerlines", natural_centerlines), ("09a_railway_waterway_dangles", natural_dangles),
                        ("10_railway_waterway_barriers", natural_barriers)]:
        _export_layer(frame, cfg.output_dir / f"{name}.geojson", cfg)
    barrier_lines = gpd.GeoDataFrame(pd.concat([road_barriers[["geometry"]], natural_barriers[["geometry"]]], ignore_index=True), geometry="geometry", crs=metric_crs)
    first_coarse, first_dangles = _polygonize(barrier_lines, city_geom, metric_crs)
    first_coarse = _merge_small(first_coarse, cfg.coarse_min_area_km2)
    boundary_gdf = _gdf(list(first_coarse.boundary), metric_crs)
    coarse_corridors, coarse_centerlines = _centerlines(boundary_gdf, cfg.coarse_boundary_buffer_m, cfg)
    cleaned_centerlines = _prune_terminal_branches(coarse_centerlines, cfg.max_dangle_length_m, cfg.node_rounding_decimals)
    cleanup_candidate, cleaned_dangles = _polygonize(cleaned_centerlines, city_geom, metric_crs)
    if len(cleanup_candidate) > cfg.max_coarse_cleanup_polygons:
        coarse, save_candidate = first_coarse.copy().reset_index(drop=True), False
    else:
        coarse, save_candidate = _merge_small(cleanup_candidate, cfg.coarse_min_area_km2).reset_index(drop=True), True
    coarse["coarse_zone_id"] = range(len(coarse))
    for name, frame in [("11_initial_coarse_zones", first_coarse), ("11a_initial_coarse_dangles", first_dangles),
                        ("12_coarse_boundary_corridors", coarse_corridors), ("13_coarse_boundary_centerlines_raw", coarse_centerlines),
                        ("14_coarse_boundary_centerlines_dangle_pruned", cleaned_centerlines), ("14a_coarse_polygonize_dangles", cleaned_dangles)]:
        _export_layer(frame, cfg.output_dir / f"{name}.geojson", cfg)
    if save_candidate:
        _export_layer(cleanup_candidate, cfg.output_dir / "14b_coarse_cleanup_candidate.geojson", cfg)
    _export_layer(coarse, cfg.output_dir / "15_coarse_zones.geojson", cfg)
    records, divider_records = [], []
    for coarse_id, row in coarse.iterrows():
        zone_records, dividers = _recursive_split(row.geometry, local, metric_crs, cfg, coarse_id)
        records.extend(zone_records)
        divider_records.extend(dividers)
    if cfg.retry_failed_splits_with_enclosed_zones:
        final, divider_records, hole_absorptions = _retry_failed(records, divider_records, local, metric_crs, cfg)
    else:
        final, hole_absorptions = _area(gpd.GeoDataFrame(records, geometry="geometry", crs=metric_crs)), _gdf([], metric_crs)
    final, undersized_merges = _merge_final_undersized(
        final, cfg.final_min_area_km2, metric_crs
    )
    final, narrow_corridors = _narrow_corridors(final, cfg.final_corridor_min_width_m, cfg.final_corridor_min_component_area_m2, metric_crs)
    final, post_corridor_merges = _merge_final_undersized(
        final, cfg.final_min_area_km2, metric_crs
    )
    final = final.reset_index(drop=True)
    final["taz_id"] = [f"{cfg.city_name}_{i:05d}" for i in range(1, len(final) + 1)]
    dividers = (
        gpd.GeoDataFrame(divider_records, geometry="geometry", crs=metric_crs)
        if divider_records
        else _gdf([], metric_crs)
    )
    regular_target = final.area_km2.between(
        cfg.final_min_area_km2, cfg.final_max_area_km2
    )
    oriented_target = (
        final.status.eq("oriented_valid")
        & final.area_km2.between(
            cfg.oriented_fallback_min_area_km2,
            cfg.oriented_fallback_max_area_km2,
        )
    )
    failures = final[~(regular_target | oriented_target)].copy()
    for name, frame in [("16_recursive_split_dividers", dividers), ("16a_tazs_absorbed_for_failed_hole_retries", hole_absorptions),
                        ("16b_final_undersized_tazs_merged", undersized_merges),
                        ("16c_final_corridor_geometries_transferred", narrow_corridors),
                        ("16d_final_undersized_tazs_merged_after_corridor_cleanup", post_corridor_merges),
                        ("17_final_tazs_with_status", final),
                        ("17a_final_tazs_not_in_target_range", failures)]:
        _export_layer(frame, cfg.output_dir / f"{name}.geojson", cfg)
    _plot_layers(plots / "04_final_tazs_by_status", cfg, [(final, {"column": "status", "categorical": True, "legend": True, "edgecolor": "black", "linewidth": .25}), (city_m, {"facecolor": "none", "edgecolor": "black"})], "Final TAZs and recursive-split status")
    _plot_layers(plots / "05_final_tazs_and_dividers", cfg, [(final.boundary, {"color": "steelblue", "linewidth": .35}), (dividers, {"color": "red", "linewidth": .25}), (city_m, {"facecolor": "none", "edgecolor": "black"})], "Final TAZs with accepted recursive dividers")
    _plot_layers(plots / "05a_final_corridor_transfers", cfg, [(final.boundary, {"color": "lightgray", "linewidth": .3}), (narrow_corridors, {"color": "red", "edgecolor": "black", "alpha": .7}), (city_m, {"facecolor": "none", "edgecolor": "black"})], "Final corridor geometries transferred")
    metadata = {
        "automatic_taz_config": cfg,
        "output_dir": str(cfg.output_dir),
        "status_counts": final.status.value_counts().to_dict(),
        "building_raster_count": len(_building_raster_paths(cfg.building_raster_path)),
        "building_data_source": building_source,
        "building_data_year": (
            cfg.building_data_year
            if building_source == "google_open_buildings_2_5d"
            else None
        ),
        "building_download": building_download_info,
        **cfg.metadata,
    }
    return TAZFrame(final, metadata)


run = create_automatic_taz
