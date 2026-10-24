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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from hashlib import sha256
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import coverage_simplify, intersection, make_valid
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon
from shapely.ops import nearest_points, polygonize_full, split, unary_union

from opengeomove.taz.exceptions import TAZDependencyError, TAZGeometryError
from opengeomove.taz.cache import TAZStageCache, path_signature, stable_cache_key
from opengeomove.taz.models import StudyArea, TAZFrame
from opengeomove.taz.population import population_cells
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
    written only when :func:`create_automatic_taz` is called. ``cache_mode``
    controls content-aware GeoParquet caches under ``cache_dir``;
    ``diagnostics`` controls intermediate exports, and ``n_jobs`` controls
    independent coarse-zone workers. Workload limits pre-split unusually
    large or complex coarse zones before detailed recursion. Plot output
    supports PNG, PDF, and self-contained HTML previews in full diagnostics.
    """

    city_name: str = "city"
    output_dir: Path = Path("taz_metis_output")
    cache_dir: Path | None = None
    cache_mode: str = "reuse"
    diagnostics: str = "summary"
    n_jobs: int = 1
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
    metis_seed: int = 42
    adaptive_sizing: bool = False
    target_area_km2: float | None = None
    target_population: float | None = None
    population_column: str = "population"
    population_tolerance: float = 2.0
    building_density_exponent: float = 0.20
    adaptive_min_area_scale: float = 0.25
    adaptive_max_area_scale: float = 8.0
    split_compactness_weight: float = 1.0
    split_balance_weight: float = 1.0
    road_alignment_bonus: float = 0.15
    coverage_simplify_tolerance_m: float = 0.0
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
    max_coarse_area_km2: float | None = 40.0
    max_coarse_road_edges: int | None = 8_000
    max_coarse_vertices: int | None = 10_000
    max_coarse_split_depth: int = 8
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
    max_divider_samples: int = 256
    max_split_depth: int = 15
    min_child_area_m2: float = 1.0
    figsize: tuple[int, int] = (14, 14)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate raster bands, area targets, output formats, and split limits."""
        if not isinstance(self.city_name, str) or not self.city_name.strip():
            raise ValueError("city_name must be a non-empty string")
        if self.cache_mode not in {"reuse", "refresh", "off"}:
            raise ValueError("cache_mode must be 'reuse', 'refresh', or 'off'")
        if self.diagnostics not in {"none", "summary", "full"}:
            raise ValueError("diagnostics must be 'none', 'summary', or 'full'")
        if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, int) or self.n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer")
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
        if isinstance(self.metis_seed, bool) or not isinstance(self.metis_seed, int):
            raise TypeError("metis_seed must be an integer")
        if not isinstance(self.adaptive_sizing, bool):
            raise TypeError("adaptive_sizing must be a bool")
        if self.target_area_km2 is not None and (
            isinstance(self.target_area_km2, bool)
            or not math.isfinite(self.target_area_km2)
            or self.target_area_km2 <= 0
        ):
            raise ValueError("target_area_km2 must be finite and positive")
        if self.target_population is not None and (
            isinstance(self.target_population, bool)
            or not math.isfinite(self.target_population)
            or self.target_population <= 0
        ):
            raise ValueError("target_population must be finite and positive")
        if not math.isfinite(self.population_tolerance) or self.population_tolerance < 1:
            raise ValueError("population_tolerance must be finite and at least 1")
        if not math.isfinite(self.building_density_exponent) or self.building_density_exponent < 0:
            raise ValueError("building_density_exponent must be finite and non-negative")
        if (
            not math.isfinite(self.adaptive_min_area_scale)
            or not math.isfinite(self.adaptive_max_area_scale)
            or self.adaptive_min_area_scale <= 0
            or self.adaptive_max_area_scale < self.adaptive_min_area_scale
        ):
            raise ValueError("adaptive area scales must be finite, positive, and ordered")
        split_quality_values = (
            self.split_compactness_weight,
            self.split_balance_weight,
            self.road_alignment_bonus,
        )
        if any(not math.isfinite(value) or value < 0 for value in split_quality_values):
            raise ValueError("split-quality weights and bonuses must be finite and non-negative")
        if self.split_compactness_weight + self.split_balance_weight <= 0:
            raise ValueError("at least one split-quality weight must be positive")
        if (
            not math.isfinite(self.coverage_simplify_tolerance_m)
            or self.coverage_simplify_tolerance_m < 0
        ):
            raise ValueError("coverage_simplify_tolerance_m must be finite and non-negative")
        if self.adaptive_sizing and self.target_area_km2 is None:
            raise ValueError("adaptive_sizing requires target_area_km2")
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
        if self.max_coarse_area_km2 is not None and (
            isinstance(self.max_coarse_area_km2, bool)
            or not math.isfinite(self.max_coarse_area_km2)
            or self.max_coarse_area_km2 <= 0
        ):
            raise ValueError("max_coarse_area_km2 must be positive or None")
        for name, value in (
            ("max_coarse_road_edges", self.max_coarse_road_edges),
            ("max_coarse_vertices", self.max_coarse_vertices),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.max_coarse_split_depth < 1:
            raise ValueError("max_coarse_split_depth must be positive")
        if self.max_divider_samples < 2:
            raise ValueError("max_divider_samples must be at least 2")


Config = AutomaticTAZConfig

_PREPARED_STAGE_CACHE_VERSION = 2
_FINAL_RESULT_CACHE_VERSION = 6
_PYMETIS_LOCK = Lock()
_NON_ALGORITHM_CONFIG_FIELDS = {
    "output_dir",
    "cache_dir",
    "cache_mode",
    "diagnostics",
    "n_jobs",
    "save_plots",
    "plot_formats",
    "show_plots",
    "figsize",
    "metadata",
    "export_crs",
}


@dataclass
class _AdaptiveEvidence:
    """Reusable population and building evidence for recursive splitting."""

    population: gpd.GeoDataFrame | None = None
    buildings: gpd.GeoDataFrame | None = None
    city_building_density: float | None = None
    population_cache: dict[bytes, float] = field(default_factory=dict)
    building_density_cache: dict[bytes, float] = field(default_factory=dict)


def _frame_signature(frame: gpd.GeoDataFrame) -> dict[str, Any]:
    """Return a stable content signature for an in-memory spatial input."""
    digest = sha256()
    digest.update(str(frame.crs).encode("utf-8"))
    for geometry in frame.geometry:
        digest.update(b"<null>" if geometry is None else geometry.wkb)
    attributes = frame.drop(columns=[frame.geometry.name], errors="ignore")
    if len(attributes.columns):
        hashed = pd.util.hash_pandas_object(
            attributes.astype(str), index=True
        ).to_numpy()
        digest.update(hashed.tobytes())
    return {"kind": "geodataframe", "rows": len(frame), "sha256": digest.hexdigest()}


def _input_signature(value: Any) -> dict[str, Any]:
    """Describe a path-backed or in-memory automatic-TAZ input."""
    if value is None:
        return {"kind": "none"}
    if isinstance(value, (str, Path)):
        return {"kind": "path", **path_signature(value)}
    if isinstance(value, gpd.GeoDataFrame):
        return _frame_signature(value)
    cells = getattr(value, "cells", None)
    if isinstance(cells, gpd.GeoDataFrame):
        return {
            "kind": type(value).__name__,
            "cells": _frame_signature(cells),
            "value_column": getattr(value, "value_column", None),
            "source": getattr(value, "source", None),
        }
    return {"kind": type(value).__name__, "cacheable": False}


def _boundary_signature(boundary: gpd.GeoDataFrame) -> dict[str, Any]:
    """Fingerprint a normalized WGS84 study boundary."""
    geometry = boundary.to_crs("EPSG:4326").geometry.union_all().normalize()
    return {
        "crs": "EPSG:4326",
        "sha256": sha256(geometry.wkb).hexdigest(),
    }


def _config_signature(
    cfg: AutomaticTAZConfig,
    *,
    fields: set[str] | None = None,
) -> dict[str, Any]:
    """Select JSON-compatible algorithm settings for cache invalidation."""
    selected = fields or (
        set(AutomaticTAZConfig.__dataclass_fields__) - _NON_ALGORITHM_CONFIG_FIELDS
    )
    return {
        name: getattr(cfg, name)
        for name in sorted(selected)
    }


def _fresh_evidence(evidence: _AdaptiveEvidence) -> _AdaptiveEvidence:
    """Create task-local evidence caches while sharing immutable source rows."""
    return _AdaptiveEvidence(
        population=evidence.population,
        buildings=evidence.buildings,
        city_building_density=evidence.city_building_density,
    )


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


def _valid_polygonal(geometry: Any):
    """Return the valid polygonal portion of one geometry.

    Args:
        geometry: A polygonal Shapely geometry, possibly invalid.

    Returns:
        A valid Polygon or MultiPolygon, or ``None`` when no polygonal area
        remains after repair.
    """
    if geometry is None or geometry.is_empty:
        return None
    repaired = make_valid(geometry)
    polygons = _polygons(repaired)
    if not polygons:
        return None
    polygonal = unary_union(polygons)
    if not polygonal.is_valid:
        polygonal = make_valid(polygonal)
        polygons = _polygons(polygonal)
        polygonal = unary_union(polygons) if polygons else None
    return polygonal if polygonal is not None and not polygonal.is_empty else None


def _normalize_polygon_frame(
    frame: gpd.GeoDataFrame, *, stage: str
) -> gpd.GeoDataFrame:
    """Repair invalid polygon rows while retaining their attributes.

    Args:
        frame: GeoDataFrame whose active geometry must be polygonal.
        stage: Human-readable stage name included in logs and exceptions.

    Returns:
        A copy with valid Polygon or MultiPolygon geometries and refreshed
        ``area_km2`` values.

    Raises:
        TAZGeometryError: If a row has no polygonal area after repair.
    """
    result = frame.copy()
    needs_normalization = (
        result.geometry.isna()
        | result.geometry.is_empty
        | ~result.geometry.is_valid
        | ~result.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    )
    if needs_normalization.any():
        logging.warning(
            "Normalizing %d invalid, empty, or mixed polygon row(s) at %s",
            int(needs_normalization.sum()), stage,
        )
        result.geometry = result.geometry.map(_valid_polygonal)
    missing = result.geometry.isna() | result.geometry.is_empty
    if missing.any():
        raise TAZGeometryError(
            f"{stage} contains {int(missing.sum())} row(s) without polygonal area"
        )
    if (~result.geometry.is_valid).any():
        raise TAZGeometryError(f"{stage} still contains invalid polygon geometry")
    if not result.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise TAZGeometryError(f"{stage} still contains non-polygon geometry")
    return _area(result)


def _validate_partition(
    zones: gpd.GeoDataFrame, boundary: Any, *, stage: str
) -> None:
    """Validate non-overlap and complete boundary coverage in metric units.

    Args:
        zones: Valid polygon zones in the same metric CRS as ``boundary``.
        boundary: Polygonal study-area geometry.
        stage: Human-readable stage name included in exceptions.

    Raises:
        TAZGeometryError: If zones overlap, leave a material gap, or extend
            materially outside the study-area boundary.
    """
    zone_union = zones.geometry.union_all()
    total_area = float(zones.geometry.area.sum())
    tolerance = max(float(boundary.area), 1.0) * 1e-6
    overlap = max(0.0, total_area - float(zone_union.area))
    uncovered = float(boundary.difference(zone_union).area)
    outside = float(zone_union.difference(boundary).area)
    if overlap > tolerance or uncovered > tolerance or outside > tolerance:
        raise TAZGeometryError(
            f"{stage} is not a valid study-area partition: "
            f"overlap={overlap:.3f} m2, uncovered={uncovered:.3f} m2, "
            f"outside={outside:.3f} m2, tolerance={tolerance:.3f} m2"
        )


def _simplify_partition_coverage(
    zones: gpd.GeoDataFrame,
    tolerance_m: float,
    boundary: Any,
) -> gpd.GeoDataFrame:
    """Simplify shared edges while preserving zone coverage and attributes.

    Args:
        zones: Final polygon partition in a projected metric CRS.
        tolerance_m: Visvalingam-Whyatt coverage simplification tolerance.
        boundary: Original study-area boundary used for validation.

    Returns:
        A normalized partition with simplified internal edges. A zero
        tolerance returns a refreshed copy without modifying coordinates.

    Raises:
        TAZGeometryError: If simplification breaks the partition topology.
    """
    if tolerance_m <= 0:
        return _area(zones)
    simplified = zones.copy()
    simplified.geometry = coverage_simplify(
        zones.geometry.to_numpy(),
        tolerance_m,
        simplify_boundary=False,
    )
    simplified = _normalize_polygon_frame(
        simplified, stage="simplified final TAZ partition"
    )
    _validate_partition(
        simplified, boundary, stage="simplified final TAZ partition"
    )
    return simplified


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


def _export_layer(
    layer: gpd.GeoDataFrame,
    path: Path,
    cfg: AutomaticTAZConfig,
    *,
    level: str = "full",
) -> Path:
    """Write a vector layer when its diagnostic level is enabled.

    Args:
        layer: Spatial rows to export.
        path: GeoJSON destination.
        cfg: Automatic TAZ configuration.
        level: ``"full"`` for detailed intermediates, ``"summary"`` for
            compact diagnostics, or ``"always"`` for the final result.

    Returns:
        The requested destination path, whether or not it was written.
    """
    enabled = (
        level == "always"
        or cfg.diagnostics == "full"
        or (level == "summary" and cfg.diagnostics == "summary")
    )
    if not enabled:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    output = layer.to_crs(cfg.export_crs)
    output.to_file(path, driver="GeoJSON")
    logging.info("Saved %d features: %s", len(layer), path)
    return path


def _plot_layers(stem: Path, cfg: AutomaticTAZConfig, layers, title: str) -> None:
    """Render a diagnostic layer stack to a static PNG when requested."""
    if cfg.diagnostics != "full" or (not cfg.save_plots and not cfg.show_plots):
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


def _sample_line(line: LineString, spacing: float, maximum: int | None = None):
    """Interpolate a bounded number of evenly spaced points along a line."""
    count = max(2, int(np.ceil(line.length / spacing)) + 1)
    if maximum is not None:
        count = min(count, maximum)
    return [line.interpolate(distance) for distance in np.linspace(
        0, line.length, count
    )]


def _aligned_divider(divider: LineString, roads: gpd.GeoDataFrame, cfg):
    """Snap divider samples to nearby roads while preserving the line path."""
    if roads.empty:
        return divider
    road_union = roads.geometry.union_all()
    coords = []
    for point in _sample_line(
        divider,
        cfg.divider_sample_spacing_m,
        cfg.max_divider_samples,
    ):
        if point.distance(road_union) <= cfg.road_snap_max_distance_m:
            _, snapped = nearest_points(point, road_union)
            coords.append((snapped.x, snapped.y))
        else:
            coords.append((point.x, point.y))
    deduped = [coord for i, coord in enumerate(coords) if i == 0 or coord != coords[i - 1]]
    return LineString(deduped) if len(deduped) >= 2 else divider


def _clip_roads_to_geometry(
    roads: gpd.GeoDataFrame,
    geometry: Any,
) -> gpd.GeoDataFrame:
    """Return roads clipped to a polygon while preserving contained features.

    A spatial-index query first removes unrelated roads. Features already
    covered by the polygon are retained without an expensive intersection;
    only boundary-crossing features are geometrically clipped. This is much
    cheaper than applying ``geopandas.clip`` to every feature at every
    recursive node.

    Args:
        roads: Road features in the same CRS as ``geometry``.
        geometry: Polygonal parent or child zone.

    Returns:
        A reset-index copy containing non-empty line geometry inside the zone.
    """
    if roads.empty:
        return roads.copy().reset_index(drop=True)
    intersecting = np.sort(
        np.asarray(roads.sindex.query(geometry, predicate="intersects"), dtype=int)
    )
    if not len(intersecting):
        return roads.iloc[0:0].copy().reset_index(drop=True)
    contained = np.asarray(
        roads.sindex.query(geometry, predicate="covers"), dtype=int
    )
    candidates = roads.iloc[intersecting].copy()
    inside = np.isin(intersecting, contained, assume_unique=False)
    crossing_positions = np.flatnonzero(~inside)
    if len(crossing_positions):
        clipped = intersection(
            candidates.geometry.iloc[crossing_positions].array,
            geometry,
        )
        candidates.loc[
            candidates.index[crossing_positions], candidates.geometry.name
        ] = gpd.GeoSeries(
            clipped,
            index=candidates.index[crossing_positions],
            crs=roads.crs,
        )
    candidates = candidates[
        candidates.geometry.notna()
        & ~candidates.geometry.is_empty
        & candidates.geometry.geom_type.isin(["LineString", "MultiLineString"])
    ]
    return candidates.reset_index(drop=True)


def _split_all(parent: Polygon, divider: LineString, minimum: float):
    """Split a parent polygon and return polygonal children above a size floor."""
    parent = _valid_polygonal(parent)
    if parent is None:
        return []
    try:
        pieces = split(parent, divider).geoms
    except Exception:
        return []
    children = []
    for piece in pieces:
        repaired = _valid_polygonal(piece.intersection(parent))
        children.extend([
            poly for poly in _polygons(repaired) if poly.area > minimum
        ])
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


def _load_building_evidence(
    boundary: Any,
    metric_crs: Any,
    cfg: AutomaticTAZConfig,
) -> gpd.GeoDataFrame:
    """Read a bounded city-wide sample of weighted building cells.

    The sample is created once per run and reused through the recursive split
    tree. Each point carries a ``building_mass`` proportional to fractional
    building count, height, and represented ground area.

    Args:
        boundary: Study-area geometry in ``metric_crs``.
        metric_crs: Projected CRS used by the TAZ workflow.
        cfg: Automatic TAZ configuration and raster-band selection.

    Returns:
        Building-center points with a non-negative ``building_mass`` column.

    Raises:
        TAZDependencyError: If raster support is unavailable.
    """
    paths = _building_raster_paths(cfg.building_raster_path)
    if not paths:
        return gpd.GeoDataFrame(
            {"building_mass": pd.Series(dtype=float)},
            geometry=[],
            crs=metric_crs,
        )
    try:
        import rasterio
        from affine import Affine
        from rasterio.enums import Resampling
        from rasterio.features import geometry_mask
        from rasterio.windows import Window, from_bounds
        from rasterio.warp import transform as transform_coordinates
    except ImportError as exc:
        raise TAZDependencyError(
            "Building-aware TAZ creation requires rasterio"
        ) from exc

    windows: list[tuple[Path, Any, Any, float]] = []
    total_pixels = 0.0
    for path in paths:
        with rasterio.open(path) as source:
            if source.crs is None:
                logging.warning("Skipping building raster without a CRS: %s", path)
                continue
            boundary_raster = gpd.GeoSeries(
                [boundary], crs=metric_crs
            ).to_crs(source.crs).iloc[0]
            raster_bounds = Polygon.from_bounds(*source.bounds)
            clipped = boundary_raster.intersection(raster_bounds)
            if clipped.is_empty:
                continue
            window = from_bounds(
                *clipped.bounds, transform=source.transform
            ).round_offsets().round_lengths()
            try:
                window = window.intersection(
                    Window(0, 0, source.width, source.height)
                )
            except rasterio.errors.WindowError:
                continue
            if window.width <= 0 or window.height <= 0:
                continue
            clipped_metric_area = float(
                gpd.GeoSeries([clipped], crs=source.crs).to_crs(metric_crs).iloc[0].area
            )
            windows.append((path, window, clipped, clipped_metric_area))
            total_pixels += float(window.width * window.height)

    if not windows:
        return gpd.GeoDataFrame(
            {"building_mass": pd.Series(dtype=float)},
            geometry=[],
            crs=metric_crs,
        )
    scale = max(1.0, math.sqrt(total_pixels / cfg.building_max_samples))
    points: list[Point] = []
    masses: list[float] = []
    for path, window, clipped, clipped_metric_area in windows:
        with rasterio.open(path) as source:
            output_width = max(1, int(math.ceil(window.width / scale)))
            output_height = max(1, int(math.ceil(window.height / scale)))
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
                window.width / output_width,
                window.height / output_height,
            )
            inside = geometry_mask(
                [clipped.__geo_interface__],
                out_shape=(output_height, output_width),
                transform=transform,
                invert=True,
            )
            valid = inside & ~np.any(np.ma.getmaskarray(values), axis=0)
            if len(indexes) == 3:
                count, height, presence = (
                    np.asarray(values[index].data, dtype=float)
                    for index in range(3)
                )
                valid &= (
                    np.isfinite(count)
                    & np.isfinite(height)
                    & np.isfinite(presence)
                    & (count > 0)
                    & (height > 0)
                    # Presence is averaged into city-wide evidence cells, so
                    # any positive coverage is retained here. The configured
                    # threshold still applies to detailed raster clustering.
                    & (presence > 0)
                )
                raw_mass = count * height
            else:
                raw_mass = np.asarray(values[0].data, dtype=float)
                valid &= np.isfinite(raw_mass) & (raw_mass > 0)
            rows, columns = np.where(valid)
            if not len(rows):
                continue
            represented_area = clipped_metric_area / max(int(inside.sum()), 1)
            xs, ys = rasterio.transform.xy(
                transform, rows, columns, offset="center"
            )
            if str(source.crs) != str(metric_crs):
                xs, ys = transform_coordinates(source.crs, metric_crs, xs, ys)
            points.extend(Point(x, y) for x, y in zip(xs, ys))
            masses.extend((raw_mass[valid] * represented_area).tolist())
    frame = gpd.GeoDataFrame(
        {"building_mass": np.asarray(masses, dtype=float)},
        geometry=points,
        crs=metric_crs,
    )
    logging.info(
        "Prepared %d reusable building evidence point(s) for adaptive splitting",
        len(frame),
    )
    return frame


def _evidence_subset(
    points: gpd.GeoDataFrame | None, geometry: Any
) -> gpd.GeoDataFrame:
    """Return evidence points covered by one candidate zone."""
    if points is None or points.empty:
        return points.iloc[0:0] if points is not None else gpd.GeoDataFrame()
    positions = points.sindex.query(geometry, predicate="covers")
    if not len(positions):
        return points.iloc[0:0]
    return points.iloc[positions]


def _zone_population(geometry: Any, evidence: _AdaptiveEvidence | None) -> float | None:
    """Allocate population cells by intersection area to one candidate zone."""
    if evidence is None or evidence.population is None:
        return None
    key = geometry.wkb
    if key in evidence.population_cache:
        return evidence.population_cache[key]
    cells = evidence.population
    positions = cells.sindex.query(geometry, predicate="intersects")
    total = 0.0
    for position in positions:
        cell = cells.geometry.iloc[int(position)]
        if cell.area > 0:
            total += float(cells.population.iloc[int(position)]) * (
                geometry.intersection(cell).area / cell.area
            )
    evidence.population_cache[key] = total
    return total


def _zone_building_density(
    geometry: Any, evidence: _AdaptiveEvidence | None
) -> float | None:
    """Return sampled building mass per square metre for one candidate zone."""
    if evidence is None or evidence.buildings is None:
        return None
    key = geometry.wkb
    if key in evidence.building_density_cache:
        return evidence.building_density_cache[key]
    points = _evidence_subset(evidence.buildings, geometry)
    density = (
        float(points.building_mass.sum()) / geometry.area
        if geometry.area > 0 and not points.empty
        else 0.0
    )
    evidence.building_density_cache[key] = density
    return density


def _zone_building_mass(
    geometry: Any, evidence: _AdaptiveEvidence | None
) -> float | None:
    """Return sampled building mass within one candidate zone."""
    density = _zone_building_density(geometry, evidence)
    return None if density is None else density * geometry.area


def _split_quality(
    children: list[Any],
    cfg: AutomaticTAZConfig,
    evidence: _AdaptiveEvidence | None,
    *,
    road_aligned: bool,
) -> dict[str, float]:
    """Score one valid split from child compactness and allocation balance.

    Population balance is preferred when population evidence is available;
    otherwise sampled building mass is used. Area balance is always included,
    preventing a concentrated evidence sample from creating extreme slivers.
    A bounded bonus preserves the preference for road-aligned boundaries when
    their geometry is reasonably competitive.

    Args:
        children: Polygonal children produced by one divider candidate.
        cfg: Split-quality weights and road-alignment preference.
        evidence: Optional population and building evidence.
        road_aligned: Whether the divider follows the local road network.

    Returns:
        Component scores and the combined ``split_quality`` value. All
        component scores except the configured bonus are in the range 0–1.
    """
    areas = np.asarray([float(child.area) for child in children], dtype=float)
    compactness_values = np.asarray([
        4.0 * math.pi * child.area / (child.length ** 2)
        if child.area > 0 and child.length > 0 else 0.0
        for child in children
    ])
    compactness = float(np.average(compactness_values, weights=areas))
    area_balance = float(areas.min() / areas.max()) if areas.max() > 0 else 0.0

    masses: list[float] | None = None
    if evidence is not None and evidence.population is not None:
        masses = [float(_zone_population(child, evidence) or 0.0) for child in children]
    elif evidence is not None and evidence.buildings is not None:
        masses = [float(_zone_building_mass(child, evidence) or 0.0) for child in children]
    if masses is not None and max(masses, default=0.0) > 0:
        evidence_balance = min(masses) / max(masses)
        balance = (area_balance + evidence_balance) / 2.0
    else:
        evidence_balance = area_balance
        balance = area_balance

    road_bonus = cfg.road_alignment_bonus if road_aligned else 0.0
    quality = (
        cfg.split_compactness_weight * compactness
        + cfg.split_balance_weight * balance
        + road_bonus
    )
    return {
        "split_quality": float(quality),
        "split_compactness": compactness,
        "split_area_balance": area_balance,
        "split_evidence_balance": float(evidence_balance),
        "split_road_bonus": float(road_bonus),
    }


def _select_split_candidate(
    candidates: list[dict[str, Any]],
    cfg: AutomaticTAZConfig,
    evidence: _AdaptiveEvidence | None,
) -> dict[str, Any] | None:
    """Return the highest-quality valid divider candidate.

    Args:
        candidates: Candidate dictionaries containing ``children`` and a
            ``road_aligned`` flag, plus caller-specific divider metadata.
        cfg: Split-quality configuration.
        evidence: Optional population and building evidence.

    Returns:
        The winning candidate with quality diagnostics, or ``None`` when no
        candidate creates at least two polygonal children.
    """
    scored = []
    for candidate in candidates:
        if len(candidate.get("children", [])) < 2:
            continue
        scored.append({
            **candidate,
            **_split_quality(
                candidate["children"],
                cfg,
                evidence,
                road_aligned=bool(candidate.get("road_aligned")),
            ),
        })
    return max(scored, key=lambda item: item["split_quality"], default=None)


def _building_cluster_centers(
    parent: Polygon,
    metric_crs,
    cfg: AutomaticTAZConfig,
    evidence: _AdaptiveEvidence | None = None,
):
    """Return two height-weighted building clusters from intersecting tiles.

    Google 2.5D tiles have fractional-count, height, and presence bands. The
    reader also supports a legacy one-band positive-weight raster. Large
    0.5-metre windows are downsampled before clustering so a city run does not
    materialize millions of pixels for each candidate TAZ.
    """
    if evidence is not None and evidence.buildings is not None:
        points = _evidence_subset(evidence.buildings, parent)
        if len(points) < 2:
            return None
        coordinates = np.column_stack([points.geometry.x, points.geometry.y])
        weights = points.building_mass.to_numpy(dtype=float)
        if len(coordinates) > cfg.building_max_samples:
            keep = np.linspace(
                0, len(coordinates) - 1, cfg.building_max_samples, dtype=int
            )
            coordinates, weights = coordinates[keep], weights[keep]
        if np.unique(coordinates, axis=0).shape[0] < 2:
            return None
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise TAZDependencyError(
                "Building-aware TAZ creation requires scikit-learn"
            ) from exc
        labels = KMeans(
            n_clusters=2, random_state=42, n_init=cfg.building_kmeans_n_init
        ).fit(coordinates, sample_weight=weights).labels_
        centers = [
            Point(
                np.average(
                    coordinates[labels == label, 0], weights=weights[labels == label]
                ),
                np.average(
                    coordinates[labels == label, 1], weights=weights[labels == label]
                ),
            )
            for label in (0, 1)
        ]
        return {
            "b0": centers[0],
            "b1": centers[1],
            "n_building_cells": len(coordinates),
        }

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
    minx, miny, maxx, maxy = parent.bounds
    required = max(
        math.hypot(x - midpoint.x, y - midpoint.y)
        for x, y in (
            (minx, miny),
            (minx, maxy),
            (maxx, miny),
            (maxx, maxy),
        )
    ) + cfg.road_snap_max_distance_m
    extension = max(
        cfg.road_snap_max_distance_m * 2.0,
        min(cfg.divider_extension_m, required),
    )
    return LineString([
        (midpoint.x - px * extension, midpoint.y - py * extension),
        (midpoint.x + px * extension, midpoint.y + py * extension),
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


def _metis_split(
    parent,
    local_roads,
    crs,
    cfg,
    evidence=None,
    *,
    roads_clipped=False,
):
    """Partition a clipped road graph and construct a road-aligned divider."""
    nx = _require_networkx()
    pymetis = _require_pymetis()
    roads = (
        local_roads.reset_index(drop=True)
        if roads_clipped
        else _clip_roads_to_geometry(local_roads, parent)
    )
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
        adjacency = [
            [node_index[neighbor] for neighbor in graph.neighbors(node)]
            for node in nodes
        ]
        # PyMETIS exposes process-global native state in some builds. Serialize
        # calls across coarse-zone worker threads and seed every partition so
        # parallel execution remains reproducible.
        with _PYMETIS_LOCK:
            _, membership = pymetis.part_graph(
                2,
                adjacency=adjacency,
                options=pymetis.Options(seed=cfg.metis_seed),
            )
    except Exception as exc:
        return None, f"metis_error:{exc}", roads
    groups = [[Point(nodes[i]) for i in range(len(nodes)) if membership[i] == label] for label in (0, 1)]
    if not groups[0] or not groups[1]:
        return None, "empty_metis_partition", roads
    centers = [unary_union(group).centroid for group in groups]
    building = _building_cluster_centers(parent, crs, cfg, evidence)
    divider, divider_info = _combined_divider(
        parent, centers[0], centers[1], building, cfg
    )
    if divider is None:
        return None, "identical_metis_centroids", roads
    hybrid = _aligned_divider(divider, roads, cfg)
    aligned_method = (
        "metis_building_road_aligned"
        if divider_info["building_used"] else "metis_road_aligned"
    )
    straight_method = (
        "metis_building_straight"
        if divider_info["building_used"] else "metis_straight"
    )
    winner = _select_split_candidate([
        {
            "children": _split_all(parent, hybrid, cfg.min_child_area_m2),
            "method": aligned_method,
            "divider": hybrid,
            "road_aligned": True,
        },
        {
            "children": _split_all(parent, divider, cfg.min_child_area_m2),
            "method": straight_method,
            "divider": divider,
            "road_aligned": False,
        },
    ], cfg, evidence)
    if winner is not None and len(winner["children"]) == 2:
        winner["reason"] = (
            ""
            if winner["road_aligned"]
            else "road_aligned_divider_failed_or_lower_quality"
        )
        winner.update(divider_info)
        return winner, "", roads
    return None, "metis_divider_did_not_make_two_children", roads


def _building_split(
    parent,
    local_roads,
    crs,
    cfg,
    evidence=None,
    *,
    roads_clipped=False,
):
    """Split from building clusters when a usable road graph is unavailable."""
    building = _building_cluster_centers(parent, crs, cfg, evidence)
    if building is None:
        return None, "no_usable_building_clusters"
    b0, b1 = building["b0"], building["b1"]
    midpoint = Point((b0.x + b1.x) / 2, (b0.y + b1.y) / 2)
    divider = _divider_from_axis(
        parent, midpoint, [b1.x - b0.x, b1.y - b0.y], cfg
    )
    if divider is None:
        return None, "identical_building_centroids"
    roads = (
        local_roads.reset_index(drop=True)
        if roads_clipped
        else _clip_roads_to_geometry(local_roads, parent)
    )
    hybrid = _aligned_divider(divider, roads, cfg)
    winner = _select_split_candidate([
        {
            "children": _split_all(parent, hybrid, cfg.min_child_area_m2),
            "method": "building_road_aligned",
            "divider": hybrid,
            "road_aligned": True,
        },
        {
            "children": _split_all(parent, divider, cfg.min_child_area_m2),
            "method": "building_straight",
            "divider": divider,
            "road_aligned": False,
        },
    ], cfg, evidence)
    if winner is not None and len(winner["children"]) == 2:
        winner["reason"] = (
            ""
            if winner["road_aligned"]
            else "road_aligned_divider_failed_or_lower_quality"
        )
        winner["n_building_cells"] = building["n_building_cells"]
        return winner, ""
    return None, "building_divider_did_not_make_two_children"


def _oriented_split(
    parent,
    local_roads,
    crs,
    cfg,
    evidence=None,
    *,
    roads_clipped=False,
):
    """Split a polygon along its principal axis, optionally snapping to roads."""
    hull = parent.convex_hull
    coords = np.asarray(hull.exterior.coords)[:, :2]
    _, vectors = np.linalg.eigh(np.cov((coords - coords.mean(axis=0)).T))
    ux, uy = vectors[:, -1]
    point = parent.representative_point()
    straight = _divider_from_axis(parent, point, [ux, uy], cfg)
    if straight is None:
        return None
    roads = (
        local_roads.reset_index(drop=True)
        if roads_clipped
        else _clip_roads_to_geometry(local_roads, parent)
    )
    hybrid = _aligned_divider(straight, roads, cfg)
    aligned_raw = _split_all(parent, hybrid, cfg.min_child_area_m2)
    straight_raw = _split_all(parent, straight, cfg.min_child_area_m2)
    candidates = []
    for raw, children, method, candidate_divider, road_aligned in (
        (
            aligned_raw,
            _merge_children(aligned_raw, crs, cfg.oriented_fallback_min_area_km2),
            "oriented_road_aligned",
            hybrid,
            True,
        ),
        (
            straight_raw,
            _merge_children(straight_raw, crs, cfg.oriented_fallback_min_area_km2),
            "oriented_straight",
            straight,
            False,
        ),
    ):
        candidates.append({
            "children": children,
            "method": method,
            "divider": candidate_divider,
            "road_aligned": road_aligned,
            "raw_child_count": len(raw),
            "merged_child_count": len(raw) - len(children),
            "roads": roads,
        })
    return _select_split_candidate(candidates, cfg, evidence)


def _geometry_vertex_count(geometry: Any) -> int:
    """Count exterior and interior polygon coordinates recursively."""
    return sum(
        len(polygon.exterior.coords)
        + sum(len(ring.coords) for ring in polygon.interiors)
        for polygon in _polygons(geometry)
    )


def _coarse_workload_exceeded(
    geometry: Any,
    roads: gpd.GeoDataFrame,
    cfg: AutomaticTAZConfig,
) -> bool:
    """Return whether a coarse zone exceeds any configured workload limit."""
    return any((
        cfg.max_coarse_area_km2 is not None
        and geometry.area / 1_000_000.0 > cfg.max_coarse_area_km2,
        cfg.max_coarse_road_edges is not None
        and len(roads) > cfg.max_coarse_road_edges,
        cfg.max_coarse_vertices is not None
        and _geometry_vertex_count(geometry) > cfg.max_coarse_vertices,
    ))


def _balance_coarse_workloads(
    coarse: gpd.GeoDataFrame,
    local_roads: gpd.GeoDataFrame,
    crs: Any,
    cfg: AutomaticTAZConfig,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Road-align bisections until coarse-zone workloads are bounded.

    Args:
        coarse: Valid coarse study-area partition.
        local_roads: Local road features in the same projected CRS.
        crs: Projected workflow CRS.
        cfg: Coarse area, road-count, vertex-count, and depth limits.

    Returns:
        A balanced coarse partition and accepted workload dividers. A zone is
        retained with ``workload_limit_reached=True`` if no valid split is
        available before the configured depth limit.
    """
    records: list[dict[str, Any]] = []
    dividers: list[dict[str, Any]] = []

    def visit(
        geometry: Any,
        roads: gpd.GeoDataFrame,
        parent_id: int,
        depth: int,
    ) -> None:
        """Recursively split one coarse workload while retaining provenance."""
        exceeded = _coarse_workload_exceeded(geometry, roads, cfg)
        if not exceeded or depth >= cfg.max_coarse_split_depth:
            records.append({
                "geometry": geometry,
                "coarse_parent_id": parent_id,
                "coarse_split_depth": depth,
                "coarse_road_edges": len(roads),
                "coarse_vertices": _geometry_vertex_count(geometry),
                "workload_limit_reached": bool(exceeded),
            })
            return
        info = _oriented_split(
            geometry,
            roads,
            crs,
            cfg,
            evidence=None,
            roads_clipped=True,
        )
        if info is None or len(info["children"]) < 2:
            records.append({
                "geometry": geometry,
                "coarse_parent_id": parent_id,
                "coarse_split_depth": depth,
                "coarse_road_edges": len(roads),
                "coarse_vertices": _geometry_vertex_count(geometry),
                "workload_limit_reached": True,
            })
            return
        dividers.append({
            "geometry": info["divider"],
            "coarse_parent_id": parent_id,
            "coarse_split_depth": depth,
            "method": info["method"],
            "split_quality": info["split_quality"],
        })
        for child in info["children"]:
            visit(
                child,
                _clip_roads_to_geometry(roads, child),
                parent_id,
                depth + 1,
            )

    for parent_id, geometry in enumerate(coarse.geometry):
        visit(
            geometry,
            _clip_roads_to_geometry(local_roads, geometry),
            parent_id,
            0,
        )
    balanced = _normalize_polygon_frame(
        gpd.GeoDataFrame(records, geometry="geometry", crs=crs),
        stage="workload-balanced coarse partition",
    ).reset_index(drop=True)
    balanced["coarse_zone_id"] = range(len(balanced))
    divider_frame = (
        gpd.GeoDataFrame(dividers, geometry="geometry", crs=crs)
        if dividers else _gdf([], crs)
    )
    return balanced, divider_frame


def _adaptive_split_decision(
    parent: Any,
    cfg: AutomaticTAZConfig,
    evidence: _AdaptiveEvidence | None,
) -> dict[str, Any]:
    """Return adaptive area, population, and building split evidence.

    Fixed-area behavior is retained when ``adaptive_sizing`` is false. In
    adaptive mode, building density scales the preferred upper area while an
    optional population target can independently require another split.
    """
    area_km2 = float(parent.area / 1_000_000.0)
    population = _zone_population(parent, evidence)
    building_density = _zone_building_density(parent, evidence)
    target_area = cfg.final_max_area_km2
    area_scale = 1.0
    if cfg.adaptive_sizing:
        target_area = float(cfg.target_area_km2)
        global_density = evidence.city_building_density if evidence else None
        if (
            building_density is not None
            and global_density is not None
            and global_density > 0
        ):
            density_ratio = max(building_density / global_density, 1e-9)
            area_scale = float(np.clip(
                density_ratio ** (-cfg.building_density_exponent),
                cfg.adaptive_min_area_scale,
                cfg.adaptive_max_area_scale,
            ))
        target_area = float(np.clip(
            target_area * area_scale,
            cfg.final_min_area_km2,
            cfg.final_max_area_km2,
        ))
    area_score = area_km2 / target_area
    population_score = 0.0
    if population is not None and cfg.target_population is not None:
        population_score = population / (
            cfg.target_population * cfg.population_tolerance
        )
    split_score = max(area_score, population_score)
    should_split = (
        area_km2 > cfg.final_min_area_km2
        and (split_score > 1.0 or area_km2 > cfg.final_max_area_km2)
    )
    return {
        "area_km2": area_km2,
        "adaptive_target_area_km2": target_area,
        "adaptive_area_scale": area_scale,
        "population": population,
        "building_density": building_density,
        "split_score": split_score,
        "should_split": should_split,
    }


def _record_evidence(decision: dict[str, Any]) -> dict[str, Any]:
    """Select stable adaptive evidence fields for output zone records."""
    return {
        key: decision[key]
        for key in (
            "adaptive_target_area_km2",
            "adaptive_area_scale",
            "population",
            "building_density",
            "split_score",
        )
    }


def _oriented_recursive(
    parent,
    roads,
    crs,
    cfg,
    coarse_id,
    depth,
    failure,
    evidence=None,
    *,
    roads_clipped=False,
):
    """Recursively apply the oriented fallback and retain failure metadata."""
    decision = _adaptive_split_decision(parent, cfg, evidence)
    current = decision["area_km2"]
    base = {
        "coarse_zone_id": coarse_id,
        "depth": depth,
        "area_km2": current,
        **_record_evidence(decision),
    }
    if not decision["should_split"] and current >= cfg.final_min_area_km2:
        return [{**base, "geometry": parent, "status": "oriented_valid", "split_method": "oriented", "failure_reason": failure}], []
    if current < cfg.final_min_area_km2 or depth >= cfg.max_split_depth:
        status = "oriented_undersized" if current < cfg.final_min_area_km2 else "oriented_max_depth"
        return [{**base, "geometry": parent, "status": status, "split_method": "oriented", "failure_reason": failure}], []
    info = _oriented_split(
        parent,
        roads,
        crs,
        cfg,
        evidence,
        roads_clipped=roads_clipped,
    )
    if info is None:
        return [{**base, "geometry": parent, "status": "oriented_split_failed", "split_method": "oriented", "failure_reason": failure}], []
    divider = {"geometry": info["divider"], "coarse_zone_id": coarse_id, "depth": depth,
               "method": info["method"], "metis_failure_reason": failure,
               "raw_child_count": info["raw_child_count"], "merged_undersized_children": info["merged_child_count"]}
    divider.update({key: info[key] for key in (
        "split_quality", "split_compactness", "split_area_balance",
        "split_evidence_balance", "split_road_bonus",
    )})
    records, dividers = [], [divider]
    parent_roads = info.get("roads", roads)
    for child in info["children"]:
        child_roads = _clip_roads_to_geometry(parent_roads, child)
        child_records, child_dividers = _oriented_recursive(
            child,
            child_roads,
            crs,
            cfg,
            coarse_id,
            depth + 1,
            failure,
            evidence,
            roads_clipped=True,
        )
        records.extend(child_records)
        dividers.extend(child_dividers)
    return records, dividers


def _building_recursive(
    parent,
    roads,
    crs,
    cfg,
    coarse_id,
    depth,
    failure,
    evidence=None,
    *,
    roads_clipped=False,
):
    """Try building-only recursion, then retain the PCA fallback."""
    decision = _adaptive_split_decision(parent, cfg, evidence)
    current = decision["area_km2"]
    base = {
        "coarse_zone_id": coarse_id,
        "depth": depth,
        "area_km2": current,
        **_record_evidence(decision),
    }
    if not decision["should_split"] and current >= cfg.final_min_area_km2:
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
            evidence,
            roads_clipped=roads_clipped,
        )
    parent_roads = (
        roads.reset_index(drop=True)
        if roads_clipped
        else _clip_roads_to_geometry(roads, parent)
    )
    info, building_failure = _building_split(
        parent,
        parent_roads,
        crs,
        cfg,
        evidence,
        roads_clipped=True,
    )
    if info is None:
        return _oriented_recursive(
            parent, parent_roads, crs, cfg, coarse_id, depth,
            f"{failure};{building_failure}",
            evidence,
            roads_clipped=True,
        )
    divider = {
        "geometry": info["divider"],
        "coarse_zone_id": coarse_id,
        "depth": depth,
        "method": info["method"],
        "metis_failure_reason": failure,
        "n_building_cells": info["n_building_cells"],
    }
    divider.update({key: info[key] for key in (
        "split_quality", "split_compactness", "split_area_balance",
        "split_evidence_balance", "split_road_bonus",
    )})
    records, dividers = [], [divider]
    for child in info["children"]:
        child_roads = _clip_roads_to_geometry(parent_roads, child)
        child_records, child_dividers = _building_recursive(
            child,
            child_roads,
            crs,
            cfg,
            coarse_id,
            depth + 1,
            failure,
            evidence,
            roads_clipped=True,
        )
        records.extend(child_records)
        dividers.extend(child_dividers)
    return records, dividers


def _recursive_split(
    parent,
    roads,
    crs,
    cfg,
    coarse_id,
    depth=0,
    evidence=None,
    *,
    roads_clipped=False,
):
    """Recursively split one coarse polygon using METIS and fallback methods."""
    decision = _adaptive_split_decision(parent, cfg, evidence)
    current = decision["area_km2"]
    base = {
        "coarse_zone_id": coarse_id,
        "depth": depth,
        "area_km2": current,
        **_record_evidence(decision),
    }
    if not decision["should_split"] and current >= cfg.final_min_area_km2:
        return [{**base, "geometry": parent, "status": "valid", "split_method": "none", "failure_reason": ""}], []
    if current < cfg.final_min_area_km2:
        return [{**base, "geometry": parent, "status": "undersized", "split_method": "none", "failure_reason": "below_minimum_before_split"}], []
    if depth >= cfg.max_split_depth:
        return [{**base, "geometry": parent, "status": "max_depth", "split_method": "none", "failure_reason": "max_depth_reached"}], []
    info, reason, parent_roads = _metis_split(
        parent,
        roads,
        crs,
        cfg,
        evidence,
        roads_clipped=roads_clipped,
    )
    if info is None:
        if decision["should_split"] or current > cfg.oriented_fallback_trigger_km2:
            if cfg.building_raster_path is not None:
                return _building_recursive(
                    parent,
                    parent_roads,
                    crs,
                    cfg,
                    coarse_id,
                    depth,
                    reason,
                    evidence,
                    roads_clipped=True,
                )
            return _oriented_recursive(
                parent,
                parent_roads,
                crs,
                cfg,
                coarse_id,
                depth,
                reason,
                evidence,
                roads_clipped=True,
            )
        return [{**base, "geometry": parent, "status": "split_failed", "split_method": "none", "failure_reason": reason}], []
    divider = {"geometry": info["divider"], "coarse_zone_id": coarse_id, "depth": depth,
               "method": info["method"], "metis_failure_reason": info["reason"],
               "building_used": info["building_used"]}
    divider.update({key: info[key] for key in (
        "split_quality", "split_compactness", "split_area_balance",
        "split_evidence_balance", "split_road_bonus",
    )})
    records, dividers = [], [divider]
    for child in info["children"]:
        child_roads = _clip_roads_to_geometry(parent_roads, child)
        child_records, child_dividers = _recursive_split(
            child,
            child_roads,
            crs,
            cfg,
            coarse_id,
            depth + 1,
            evidence,
            roads_clipped=True,
        )
        records.extend(child_records)
        dividers.extend(child_dividers)
    return records, dividers


def _shared_boundary_length(geometry, reference_boundary) -> float:
    """Return shared boundary length, treating null/empty boundaries as zero."""
    if geometry is None or reference_boundary is None or geometry.is_empty:
        return 0.0
    boundary = geometry.boundary
    if boundary is None or boundary.is_empty or reference_boundary.is_empty:
        return 0.0
    return float(boundary.intersection(reference_boundary).length)


def _merge_final_undersized(zones, minimum, crs):
    """Absorb every final sub-minimum zone after recursive splitting."""
    working = _area(zones).reset_index(drop=True)
    invalid_geometry = working.geometry.isna() | working.geometry.is_empty
    if invalid_geometry.any():
        logging.warning(
            "Dropping %d final TAZ row(s) with null or empty geometry before merging",
            int(invalid_geometry.sum()),
        )
        working = working.loc[~invalid_geometry].reset_index(drop=True)
    audit = []
    while len(working) > 1:
        undersized = working[working.area_km2 < minimum]
        if undersized.empty:
            break
        source = undersized.area_km2.idxmin()
        source_geom = working.loc[source, "geometry"]
        candidates = working.drop(index=source).copy()
        candidates = candidates[
            candidates.geometry.notna() & ~candidates.geometry.is_empty
        ]
        touching = candidates[candidates.geometry.touches(source_geom)].copy()
        if not touching.empty:
            source_boundary = source_geom.boundary
            touching["shared_boundary_m"] = touching.geometry.apply(
                lambda geom: _shared_boundary_length(geom, source_boundary)
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


def _retry_failed(records, dividers, roads, crs, cfg, evidence=None):
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
        retry_records, retry_dividers = _recursive_split(
            filled,
            roads,
            crs,
            cfg,
            int(failed.coarse_zone_id),
            int(failed.depth),
            evidence,
        )
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


def _build_coarse_partition(
    city: gpd.GeoDataFrame,
    city_m: gpd.GeoDataFrame,
    city_geom: Any,
    major: gpd.GeoDataFrame,
    local: gpd.GeoDataFrame,
    metric_crs: Any,
    cfg: AutomaticTAZConfig,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build and workload-balance the road/natural-feature coarse partition.

    Args:
        city: Study boundary in its source CRS.
        city_m: Study boundary in the projected workflow CRS.
        city_geom: Dissolved projected study boundary.
        major: Major roads used as coarse barriers.
        local: Local roads used to bound recursive workloads.
        metric_crs: Projected workflow CRS.
        cfg: Automatic TAZ configuration and diagnostic settings.

    Returns:
        Workload-balanced coarse zones and the accepted balancing dividers.
    """
    _export_layer(city_m, cfg.output_dir / "00_boundary.geojson", cfg)
    _export_layer(major, cfg.output_dir / "01_major_roads.geojson", cfg)
    _export_layer(local, cfg.output_dir / "02_local_roads.geojson", cfg)
    major_corridors, major_centerlines = _centerlines(
        major, cfg.major_road_buffer_m, cfg
    )
    _export_layer(
        major_corridors,
        cfg.output_dir / "03_major_road_corridors.geojson",
        cfg,
    )
    _export_layer(
        major_centerlines,
        cfg.output_dir / "04_major_road_centerlines.geojson",
        cfg,
    )
    road_polygons, road_dangles = _polygonize(
        major_centerlines, city_geom, metric_crs
    )
    road_barriers = road_polygons[
        road_polygons.area_km2 >= cfg.barrier_min_area_km2
    ].copy()
    road_barriers.geometry = road_barriers.boundary
    _export_layer(road_polygons, cfg.output_dir / "05_major_road_polygons.geojson", cfg)
    _export_layer(road_dangles, cfg.output_dir / "05a_major_road_dangles.geojson", cfg)
    _export_layer(road_barriers, cfg.output_dir / "06_major_road_barriers.geojson", cfg)

    natural = _gdf([], metric_crs)
    if cfg.fetch_railways_and_waterways:
        try:
            import osmnx as ox

            features = ox.features_from_polygon(
                city.to_crs(cfg.export_crs).geometry.iloc[0],
                tags={"railway": True, "waterway": True},
            )
            if features is not None and not features.empty:
                features = gpd.clip(features.to_crs(metric_crs), city_m)
                linear = features.geometry.geom_type.isin(
                    ["LineString", "MultiLineString"]
                )
                railway = features[
                    linear
                    & features.get(
                        "railway", pd.Series(index=features.index, dtype="object")
                    ).isin(cfg.railway_types)
                ]
                water = features[
                    linear
                    & features.get(
                        "waterway", pd.Series(index=features.index, dtype="object")
                    ).notna()
                ]
                natural = gpd.GeoDataFrame(
                    pd.concat(
                        [railway[["geometry"]], water[["geometry"]]],
                        ignore_index=True,
                    ),
                    geometry="geometry",
                    crs=metric_crs,
                )
        except Exception as exc:
            logging.warning(
                "Natural-feature retrieval failed; continuing with roads: %s", exc
            )
    natural_corridors, natural_centerlines = _centerlines(
        natural, cfg.barrier_buffer_m, cfg
    )
    natural_polygons, natural_dangles = _polygonize(
        natural_centerlines, city_geom, metric_crs
    )
    natural_barriers = natural_polygons[
        natural_polygons.area_km2 >= cfg.barrier_min_area_km2
    ].copy()
    natural_barriers.geometry = natural_barriers.boundary
    for name, frame in [
        ("07_raw_railway_waterway_lines", natural),
        ("08_railway_waterway_corridors", natural_corridors),
        ("09_railway_waterway_centerlines", natural_centerlines),
        ("09a_railway_waterway_dangles", natural_dangles),
        ("10_railway_waterway_barriers", natural_barriers),
    ]:
        _export_layer(frame, cfg.output_dir / f"{name}.geojson", cfg)

    barrier_lines = gpd.GeoDataFrame(
        pd.concat(
            [road_barriers[["geometry"]], natural_barriers[["geometry"]]],
            ignore_index=True,
        ),
        geometry="geometry",
        crs=metric_crs,
    )
    first_coarse, first_dangles = _polygonize(
        barrier_lines, city_geom, metric_crs
    )
    first_coarse = _normalize_polygon_frame(
        _merge_small(first_coarse, cfg.coarse_min_area_km2),
        stage="initial coarse partition",
    )
    boundary_gdf = _gdf(list(first_coarse.boundary), metric_crs)
    coarse_corridors, coarse_centerlines = _centerlines(
        boundary_gdf, cfg.coarse_boundary_buffer_m, cfg
    )
    cleaned_centerlines = _prune_terminal_branches(
        coarse_centerlines,
        cfg.max_dangle_length_m,
        cfg.node_rounding_decimals,
    )
    cleanup_candidate, cleaned_dangles = _polygonize(
        cleaned_centerlines, city_geom, metric_crs
    )
    if len(cleanup_candidate) > cfg.max_coarse_cleanup_polygons:
        coarse, save_candidate = first_coarse.copy().reset_index(drop=True), False
    else:
        coarse = _merge_small(
            cleanup_candidate, cfg.coarse_min_area_km2
        ).reset_index(drop=True)
        save_candidate = True
    coarse = _normalize_polygon_frame(coarse, stage="coarse partition")
    _validate_partition(coarse, city_geom, stage="coarse partition")
    for name, frame in [
        ("11_initial_coarse_zones", first_coarse),
        ("11a_initial_coarse_dangles", first_dangles),
        ("12_coarse_boundary_corridors", coarse_corridors),
        ("13_coarse_boundary_centerlines_raw", coarse_centerlines),
        ("14_coarse_boundary_centerlines_dangle_pruned", cleaned_centerlines),
        ("14a_coarse_polygonize_dangles", cleaned_dangles),
    ]:
        _export_layer(frame, cfg.output_dir / f"{name}.geojson", cfg)
    if save_candidate:
        _export_layer(
            cleanup_candidate,
            cfg.output_dir / "14b_coarse_cleanup_candidate.geojson",
            cfg,
        )
    _export_layer(
        coarse,
        cfg.output_dir / "15a_unbalanced_coarse_zones.geojson",
        cfg,
    )
    balanced, workload_dividers = _balance_coarse_workloads(
        coarse, local, metric_crs, cfg
    )
    _validate_partition(
        balanced, city_geom, stage="workload-balanced coarse partition"
    )
    _export_layer(balanced, cfg.output_dir / "15_coarse_zones.geojson", cfg)
    _export_layer(
        workload_dividers,
        cfg.output_dir / "15b_coarse_workload_dividers.geojson",
        cfg,
    )
    return balanced, workload_dividers


def create_automatic_taz(
    boundary: Any,
    graph: Any = None,
    *,
    population: Any = None,
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
    tiles for the configured year into a reusable local cache. ``population``
    may be a raster path, population-cell GeoDataFrame, or PopulationSource;
    it is used only when supplied and is never inferred from a reference TAZ.

    Args:
        boundary: Polygonal study boundary with a declared CRS.
        graph: Optional road graph or road edge layer.
        population: Optional raster or vector population cells used by
            adaptive split targets.
        config: Automatic TAZ settings. A copy is used; the caller's object is
            not mutated.
        **overrides: Fields that replace values on ``config``.

    Returns:
        A TAZFrame. The final GeoJSON and run manifest are written under
        ``config.output_dir``. Additional intermediate layers and static plots
        follow the configured diagnostics level. Matching prepared/final cache
        entries are reused according to ``cache_mode``.

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
    cfg.cache_dir = (
        Path(cfg.cache_dir)
        if cfg.cache_dir is not None
        else cfg.output_dir / "cache"
    )
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
    started = time.perf_counter()
    timings: dict[str, float] = {}
    cache_hits: list[str] = []
    cache = TAZStageCache(cfg.cache_dir, cfg.cache_mode)
    city = _load_boundary(boundary)
    boundary_key = _boundary_signature(city)
    building_source = "local" if cfg.building_raster_path is not None else None
    building_download_info = None
    stage_started = time.perf_counter()
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
    timings["input_resolution_s"] = time.perf_counter() - stage_started
    metric_crs = city.estimate_utm_crs()
    if metric_crs is None:
        raise TAZGeometryError("Could not estimate a metric CRS for the boundary")
    city_m = city.to_crs(metric_crs)
    city_geom = city_m.geometry.union_all()

    graph_source = (
        _road_edge_cache_path(cfg) if graph is None else graph
    )
    graph_signature = _input_signature(graph_source)
    population_signature = _input_signature(population)
    building_signature = path_signature(cfg.building_raster_path)
    final_config_signature = _config_signature(cfg)
    final_key = stable_cache_key({
        "algorithm_version": _FINAL_RESULT_CACHE_VERSION,
        "boundary": boundary_key,
        "graph": graph_signature,
        "population": population_signature,
        "buildings": building_signature,
        "config": final_config_signature,
    })
    cached_final = cache.load_frame("final", final_key)
    if cached_final is not None:
        cached_metadata = cache.load_json("final", final_key) or {}
        metadata = dict(cached_metadata.get("metadata", {}))
        metadata.update({
            "automatic_taz_config": cfg,
            "output_dir": str(cfg.output_dir),
            "cache_mode": cfg.cache_mode,
            "cache_hit": "final",
            "cache_key": final_key,
            "elapsed_s": time.perf_counter() - started,
        })
        logging.info(
            "Reusing %d final TAZ(s) from cache key %s",
            len(cached_final), final_key,
        )
        _export_layer(
            cached_final,
            cfg.output_dir / "17_final_tazs_with_status.geojson",
            cfg,
            level="always",
        )
        cached_manifest = {
            "algorithm_version": _FINAL_RESULT_CACHE_VERSION,
            "prepared_stage_cache_version": _PREPARED_STAGE_CACHE_VERSION,
            "cache_key": final_key,
            "cache_hit": "final",
            "cache_hits": metadata.get("cache_hits", []),
            "config": final_config_signature,
            "timings_s": metadata.get("timings_s", {}),
            "coarse_timings": metadata.get("coarse_timings", []),
            "input_zone_count": metadata.get("coarse_zone_count"),
            "output_zone_count": int(len(cached_final)),
            "cache_lookup_elapsed_s": metadata["elapsed_s"],
        }
        (cfg.output_dir / "run_manifest.json").write_text(
            json.dumps(cached_manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return TAZFrame(cached_final, metadata)

    population_data = None
    if population is not None:
        population_key = stable_cache_key({
            "boundary": boundary_key,
            "source": population_signature,
            "column": cfg.population_column,
            "crs": str(metric_crs),
        })
        population_data = cache.load_frame("population_evidence", population_key)
        if population_data is None:
            population_data = population_cells(
                population,
                population_column=cfg.population_column,
                target_crs=metric_crs,
                boundary=city_geom,
            )
            cache.save_frame("population_evidence", population_key, population_data)
        else:
            cache_hits.append("population_evidence")
        logging.info(
            "Prepared %d population cell(s) for adaptive splitting",
            len(population_data),
        )
    stage_started = time.perf_counter()
    building_evidence = None
    if cfg.adaptive_sizing and cfg.building_raster_path is not None:
        building_key = stable_cache_key({
            "boundary": boundary_key,
            "source": building_signature,
            "crs": str(metric_crs),
            "config": _config_signature(cfg, fields={
                "building_count_band",
                "building_height_band",
                "building_presence_band",
                "building_presence_threshold",
                "building_max_samples",
            }),
        })
        building_evidence = cache.load_frame("building_evidence", building_key)
        if building_evidence is None:
            building_evidence = _load_building_evidence(city_geom, metric_crs, cfg)
            cache.save_frame("building_evidence", building_key, building_evidence)
        else:
            cache_hits.append("building_evidence")
            logging.info(
                "Reusing %d prepared building evidence point(s)",
                len(building_evidence),
            )
    timings["evidence_preparation_s"] = time.perf_counter() - stage_started
    evidence = _AdaptiveEvidence(
        population=population_data,
        buildings=building_evidence,
        city_building_density=(
            float(building_evidence.building_mass.sum()) / city_geom.area
            if building_evidence is not None
            and not building_evidence.empty
            and city_geom.area > 0
            else None
        ),
    )
    population_total = _zone_population(city_geom, evidence)
    if (
        cfg.adaptive_sizing
        and population_total is not None
        and cfg.target_population is None
    ):
        expected_zones = max(
            1.0, city_geom.area / (float(cfg.target_area_km2) * 1_000_000.0)
        )
        cfg.target_population = population_total / expected_zones
        logging.info(
            "Derived adaptive target population %.3f from %.3f total people",
            cfg.target_population,
            population_total,
        )
    stage_started = time.perf_counter()
    road_key = stable_cache_key({
        "boundary": boundary_key,
        "source": graph_signature,
        "network_type": cfg.network_type,
        "crs": str(metric_crs),
    })
    edges = cache.load_frame("prepared_roads", road_key)
    if edges is None:
        edges = _load_edges(graph, city, cfg)
        if edges.crs is None:
            raise TAZGeometryError("Road graph edges must have a CRS")
        edges = edges.to_crs(metric_crs).copy()
        highway = edges.get("highway", pd.Series(index=edges.index, dtype="object"))
        edges["highway_type"] = highway.map(_normalize_highway)
        # A first OSM download changes the source path signature. Persist under
        # the post-download key that subsequent runs will calculate.
        graph_signature = _input_signature(
            _road_edge_cache_path(cfg) if graph is None else graph
        )
        road_key = stable_cache_key({
            "boundary": boundary_key,
            "source": graph_signature,
            "network_type": cfg.network_type,
            "crs": str(metric_crs),
        })
        cache_edges = edges[["highway_type", "geometry"]].copy()
        cache_edges["highway_type"] = cache_edges["highway_type"].map(
            _cache_highway
        )
        cache.save_frame("prepared_roads", road_key, cache_edges)
    else:
        cache_hits.append("prepared_roads")
        logging.info("Reusing %d prepared road edge(s)", len(edges))
    edges["highway_type"] = edges["highway_type"].map(_normalize_highway)
    timings["road_preparation_s"] = time.perf_counter() - stage_started
    final_key = stable_cache_key({
        "algorithm_version": _FINAL_RESULT_CACHE_VERSION,
        "boundary": boundary_key,
        "graph": graph_signature,
        "population": population_signature,
        "buildings": building_signature,
        "config": final_config_signature,
    })
    major = edges[edges.highway_type.isin(cfg.major_road_types)].copy()
    local = edges[edges.highway_type.isin(cfg.local_road_types)].copy()
    stage_started = time.perf_counter()
    coarse_key = stable_cache_key({
        "algorithm_version": _PREPARED_STAGE_CACHE_VERSION,
        "boundary": boundary_key,
        "road_key": road_key,
        "config": _config_signature(cfg, fields={
            "major_road_types",
            "local_road_types",
            "railway_types",
            "fetch_railways_and_waterways",
            "major_road_buffer_m",
            "barrier_buffer_m",
            "coarse_boundary_buffer_m",
            "centerline_interpolation_m",
            "max_dangle_length_m",
            "max_coarse_cleanup_polygons",
            "node_rounding_decimals",
            "barrier_min_area_km2",
            "coarse_min_area_km2",
            "max_coarse_area_km2",
            "max_coarse_road_edges",
            "max_coarse_vertices",
            "max_coarse_split_depth",
            "divider_extension_m",
            "road_snap_max_distance_m",
            "divider_sample_spacing_m",
            "max_divider_samples",
            "oriented_fallback_min_area_km2",
        }),
    })
    coarse = cache.load_frame("coarse_partition", coarse_key)
    if coarse is None:
        coarse, workload_dividers = _build_coarse_partition(
            city,
            city_m,
            city_geom,
            major,
            local,
            metric_crs,
            cfg,
        )
        cache.save_frame("coarse_partition", coarse_key, coarse)
    else:
        cache_hits.append("coarse_partition")
        workload_dividers = _gdf([], metric_crs)
        logging.info("Reusing %d workload-balanced coarse zone(s)", len(coarse))
        _validate_partition(coarse, city_geom, stage="cached coarse partition")
        _export_layer(coarse, cfg.output_dir / "15_coarse_zones.geojson", cfg)
    timings["coarse_partition_s"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()

    def split_coarse(
        item: tuple[int, Any]
    ) -> tuple[int, list[Any], list[Any], dict[str, Any]]:
        """Split one coarse row with isolated evidence caches."""
        coarse_id, row = item
        coarse_started = time.perf_counter()
        logging.info(
            "Splitting coarse zone %d/%d (%.3f km2)",
            coarse_id + 1, len(coarse), float(row.area_km2),
        )
        parent_roads = _clip_roads_to_geometry(local, row.geometry)
        task_evidence = _fresh_evidence(evidence) if cfg.n_jobs > 1 else evidence
        zone_records, dividers = _recursive_split(
            row.geometry,
            parent_roads,
            metric_crs,
            cfg,
            coarse_id,
            evidence=task_evidence,
            roads_clipped=True,
        )
        logging.info(
            "Completed coarse zone %d/%d with %d final part(s)",
            coarse_id + 1, len(coarse), len(zone_records),
        )
        return coarse_id, zone_records, dividers, {
            "coarse_zone_id": int(coarse_id),
            "area_km2": float(row.area_km2),
            "road_edges": int(len(parent_roads)),
            "output_parts": int(len(zone_records)),
            "elapsed_s": time.perf_counter() - coarse_started,
        }

    items = list(coarse.iterrows())
    if cfg.n_jobs == 1:
        completed = [split_coarse(item) for item in items]
    else:
        logging.info("Splitting %d coarse zones with %d worker threads", len(items), cfg.n_jobs)
        with ThreadPoolExecutor(max_workers=cfg.n_jobs) as executor:
            completed = list(executor.map(split_coarse, items))
    records, divider_records, coarse_timings = [], [], []
    for _, zone_records, dividers, coarse_timing in completed:
        records.extend(zone_records)
        divider_records.extend(dividers)
        coarse_timings.append(coarse_timing)
    timings["recursive_splitting_s"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    if cfg.retry_failed_splits_with_enclosed_zones:
        final, divider_records, hole_absorptions = _retry_failed(
            records,
            divider_records,
            local,
            metric_crs,
            cfg,
            evidence,
        )
    else:
        final, hole_absorptions = _area(gpd.GeoDataFrame(records, geometry="geometry", crs=metric_crs)), _gdf([], metric_crs)
    final, undersized_merges = _merge_final_undersized(
        final, cfg.final_min_area_km2, metric_crs
    )
    final, narrow_corridors = _narrow_corridors(final, cfg.final_corridor_min_width_m, cfg.final_corridor_min_component_area_m2, metric_crs)
    final, post_corridor_merges = _merge_final_undersized(
        final, cfg.final_min_area_km2, metric_crs
    )
    final = _normalize_polygon_frame(
        final.reset_index(drop=True), stage="final TAZ partition"
    )
    _validate_partition(final, city_geom, stage="final TAZ partition")
    final = _simplify_partition_coverage(
        final,
        cfg.coverage_simplify_tolerance_m,
        city_geom,
    )
    timings["final_cleanup_s"] = time.perf_counter() - stage_started
    final_decisions = [
        _adaptive_split_decision(geometry, cfg, evidence)
        for geometry in final.geometry
    ]
    for key in (
        "adaptive_target_area_km2",
        "adaptive_area_scale",
        "population",
        "building_density",
        "split_score",
    ):
        final[key] = [decision[key] for decision in final_decisions]
    final["taz_id"] = [f"{cfg.city_name}_{i:05d}" for i in range(1, len(final) + 1)]
    dividers = (
        gpd.GeoDataFrame(divider_records, geometry="geometry", crs=metric_crs)
        if divider_records
        else _gdf([], metric_crs)
    )
    regular_target = (
        final.area_km2.ge(cfg.final_min_area_km2)
        & (
            final.split_score.le(1.0 + 1e-9)
            if cfg.adaptive_sizing
            else final.area_km2.le(cfg.final_max_area_km2)
        )
    )
    oriented_target = (
        final.status.eq("oriented_valid")
        & final.area_km2.between(
            cfg.oriented_fallback_min_area_km2,
            cfg.oriented_fallback_max_area_km2,
        )
    )
    failures = final[~(regular_target | oriented_target)].copy()
    for name, frame in [
        ("16_recursive_split_dividers", dividers),
        ("16a_tazs_absorbed_for_failed_hole_retries", hole_absorptions),
        ("16b_final_undersized_tazs_merged", undersized_merges),
        ("16c_final_corridor_geometries_transferred", narrow_corridors),
        ("16d_final_undersized_tazs_merged_after_corridor_cleanup", post_corridor_merges),
    ]:
        _export_layer(frame, cfg.output_dir / f"{name}.geojson", cfg, level="full")
    _export_layer(
        final,
        cfg.output_dir / "17_final_tazs_with_status.geojson",
        cfg,
        level="always",
    )
    _export_layer(
        failures,
        cfg.output_dir / "17a_final_tazs_not_in_target_range.geojson",
        cfg,
        level="summary",
    )
    _plot_layers(plots / "04_final_tazs_by_status", cfg, [(final, {"column": "status", "categorical": True, "legend": True, "edgecolor": "black", "linewidth": .25}), (city_m, {"facecolor": "none", "edgecolor": "black"})], "Final TAZs and recursive-split status")
    _plot_layers(plots / "05_final_tazs_and_dividers", cfg, [(final.boundary, {"color": "steelblue", "linewidth": .35}), (dividers, {"color": "red", "linewidth": .25}), (city_m, {"facecolor": "none", "edgecolor": "black"})], "Final TAZs with accepted recursive dividers")
    _plot_layers(plots / "05a_final_corridor_transfers", cfg, [(final.boundary, {"color": "lightgray", "linewidth": .3}), (narrow_corridors, {"color": "red", "edgecolor": "black", "alpha": .7}), (city_m, {"facecolor": "none", "edgecolor": "black"})], "Final corridor geometries transferred")
    timings["total_s"] = time.perf_counter() - started
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
        "adaptive_sizing": cfg.adaptive_sizing,
        "target_area_km2": cfg.target_area_km2,
        "target_population": cfg.target_population,
        "population_total": population_total,
        "population_cells": (
            int(len(population_data)) if population_data is not None else 0
        ),
        "building_evidence_points": (
            int(len(building_evidence)) if building_evidence is not None else 0
        ),
        "city_building_density": evidence.city_building_density,
        "building_density_exponent": cfg.building_density_exponent,
        "split_compactness_weight": cfg.split_compactness_weight,
        "split_balance_weight": cfg.split_balance_weight,
        "road_alignment_bonus": cfg.road_alignment_bonus,
        "coverage_simplify_tolerance_m": cfg.coverage_simplify_tolerance_m,
        "cache_mode": cfg.cache_mode,
        "cache_key": final_key,
        "cache_hits": cache_hits,
        "timings_s": timings,
        "coarse_timings": coarse_timings,
        "coarse_zone_count": int(len(coarse)),
        "coarse_workload_dividers": int(len(workload_dividers)),
        "n_jobs": cfg.n_jobs,
        "diagnostics": cfg.diagnostics,
        **cfg.metadata,
    }
    cache_metadata = {
        key: value
        for key, value in metadata.items()
        if key != "automatic_taz_config"
    }
    cache.save_frame("final", final_key, final)
    cache.save_json("final", final_key, {"metadata": cache_metadata})
    manifest = {
        "algorithm_version": _FINAL_RESULT_CACHE_VERSION,
        "prepared_stage_cache_version": _PREPARED_STAGE_CACHE_VERSION,
        "cache_key": final_key,
        "cache_hits": cache_hits,
        "config": final_config_signature,
        "timings_s": timings,
        "coarse_timings": coarse_timings,
        "input_zone_count": int(len(coarse)),
        "output_zone_count": int(len(final)),
        "failure_count": int(len(failures)),
    }
    (cfg.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return TAZFrame(final, metadata)


run = create_automatic_taz
