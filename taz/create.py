"""Stable hex, population-balanced, and road-constrained TAZ creation."""

from __future__ import annotations

from dataclasses import replace
from math import ceil, sqrt
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon, box
from shapely.ops import split, unary_union

from opengeomove.taz.config import TAZConfig
from opengeomove.taz.exceptions import TAZDependencyError, TAZGeometryError
from opengeomove.taz.models import StudyArea, TAZFrame, TAZReport
from opengeomove.taz.population import attach_population, population_cells
from opengeomove.data_loading.spatial import read_spatial_layer


def stable_id(prefix: str, *parts: Any) -> str:
    """Return a stable short identifier for a zone geometry."""
    import hashlib
    import json

    raw = json.dumps(parts, default=str, separators=(",", ":"), sort_keys=True)
    return f"{prefix}_{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def _hexagons(boundary, side):
    """Create clipped pointy-top hexagonal cells covering a boundary."""
    minx, miny, maxx, maxy = boundary.bounds
    dx, dy = 1.5 * side, sqrt(3) * side
    result = []
    x, column = minx - 2 * side, 0
    while x <= maxx + 2 * side:
        y = miny - 2 * dy + (dy / 2 if column % 2 else 0)
        while y <= maxy + 2 * dy:
            coordinates = [
                (x + side * np.cos(np.pi / 3 * k), y + side * np.sin(np.pi / 3 * k))
                for k in range(6)
            ]
            polygon = Polygon(coordinates).intersection(boundary)
            if not polygon.is_empty and polygon.area > 0:
                result.append(polygon)
            y += dy
        x += dx
        column += 1
    return result


def _squares(boundary, target_count):
    """Create clipped square seed cells at an approximate target count."""
    minx, miny, maxx, maxy = boundary.bounds
    side = sqrt(boundary.area / max(target_count * 4, 1))
    result = []
    for x in np.arange(minx, maxx + side, side):
        for y in np.arange(miny, maxy + side, side):
            clipped = box(x, y, x + side, y + side).intersection(boundary)
            if not clipped.is_empty and clipped.area > 0:
                result.append(clipped)
    return result


def _population_groups(cells, target_population, min_area, max_area=None):
    """Greedily merge nearby cells until population and area targets are met."""
    remaining = set(cells.index)
    groups = []
    while remaining:
        seed = min(remaining, key=lambda i: (cells.geometry[i].centroid.y, cells.geometry[i].centroid.x))
        group = [seed]
        remaining.remove(seed)
        population = float(cells.population.loc[seed])
        geometry = cells.geometry.loc[seed]
        while remaining and (population < target_population or geometry.area < min_area):
            adjacent = [i for i in remaining if cells.geometry.loc[i].intersects(geometry)]
            pool = adjacent or list(remaining)
            ordered = sorted(pool, key=lambda i: (geometry.distance(cells.geometry.loc[i]), i))
            if max_area is not None and geometry.area >= min_area:
                fitting = [i for i in ordered if unary_union([geometry, cells.geometry.loc[i]]).area <= max_area]
                if not fitting:
                    break
                next_index = fitting[0]
            else:
                next_index = ordered[0]
            group.append(next_index)
            remaining.remove(next_index)
            population += float(cells.population.loc[next_index])
            geometry = unary_union([geometry, cells.geometry.loc[next_index]])
        groups.append((group, geometry, population))
    # Merge undersized final islands into their nearest neighbor.
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for position, (indices, geometry, population) in enumerate(groups):
            if geometry.area >= min_area:
                continue
            choices = [i for i in range(len(groups)) if i != position]
            neighbor = min(choices, key=lambda i: geometry.distance(groups[i][1]))
            ni, ng, npop = groups[neighbor]
            groups[neighbor] = (ni + indices, unary_union([ng, geometry]), npop + population)
            groups.pop(position)
            changed = True
            break
    return groups


def _road_partitions(boundary, roads, crs):
    """Split a study boundary along supplied road lines when possible."""
    if roads is None:
        return [boundary]
    road_frame = (
        roads.copy()
        if isinstance(roads, gpd.GeoDataFrame)
        else read_spatial_layer(roads)
    )
    road_frame = road_frame.to_crs(crs)
    cutter = unary_union([
        line for line in road_frame.geometry if isinstance(line, (LineString, MultiLineString))
    ])
    if cutter.is_empty:
        return [boundary]
    try:
        return [part for part in split(boundary, cutter).geoms if part.area > 0]
    except Exception:
        return [boundary]


def _validate_partition(frame, boundary, tolerance=1e-6):
    """Reject zone sets that leave gaps, exceed the boundary, or overlap."""
    if frame.empty:
        raise TAZGeometryError("TAZ creation produced no zones")
    union = frame.geometry.union_all()
    missing_ratio = boundary.difference(union).area / boundary.area
    excess_ratio = union.difference(boundary).area / boundary.area
    sum_area = frame.geometry.area.sum()
    overlap_ratio = max(0.0, sum_area - union.area) / boundary.area
    if max(missing_ratio, excess_ratio, overlap_ratio) > tolerance:
        raise TAZGeometryError(
            f"Invalid TAZ partition: missing={missing_ratio:.3g}, excess={excess_ratio:.3g}, overlap={overlap_ratio:.3g}"
        )


def create(
    study_area: StudyArea | gpd.GeoDataFrame,
    *,
    population: Any = None,
    roads: gpd.GeoDataFrame | None = None,
    config: TAZConfig | None = None,
    **overrides: Any,
) -> TAZFrame:
    """Create city-agnostic TAZ polygons from a study boundary.

    The algorithm projects the boundary to a locally estimated metric CRS,
    creates hexagonal seed cells for ``hex_grid`` or clipped square seed cells
    for population-based methods, then greedily merges adjacent cells toward
    the requested population and area limits. Road-constrained mode uses the
    supplied road GeoDataFrame as a barrier when configured. Population may be
    a vector grid, a :class:`PopulationSource`, or a raster path.

    Args:
        study_area: Polygonal StudyArea or one-row/multi-row polygon
            GeoDataFrame whose union defines the analysis boundary.
        population: Optional population cells or raster. Required for
            population-balanced and road-constrained methods.
        roads: Optional road line GeoDataFrame in any CRS with CRS metadata.
        config: Optional immutable TAZ generation configuration.
        **overrides: TAZConfig fields that replace values in ``config``.

    Returns:
        A validated TAZFrame with stable IDs, area, population, geometry,
        configuration, and a generation report. Geometry is stored in the
        local metric CRS chosen for area and partition calculations.

    Raises:
        TypeError: If config or input objects have unsupported types.
        ValueError: If required population inputs or threshold values are
            invalid.
        TAZGeometryError: If no valid partition can be generated.
    """
    if config is not None and not isinstance(config, TAZConfig):
        raise TypeError("config must be a TAZConfig instance")
    cfg = replace(config) if config is not None else TAZConfig()
    if "road_network" in overrides:
        roads = overrides.pop("road_network")
    if "population_raster" in overrides:
        population = overrides.pop("population_raster")
    if overrides:
        config_keys = set(TAZConfig.__dataclass_fields__)
        allowed_aliases = {"roads", "population"}
        unknown = set(overrides) - config_keys - allowed_aliases
        if unknown:
            raise TypeError(f"Unknown TAZ configuration option(s): {sorted(unknown)}")
        config_values = {key: value for key, value in overrides.items() if key in config_keys}
        cfg = replace(cfg, **config_values)
        roads = overrides.get("roads", roads)
        population = overrides.get("population", population)
    if cfg.min_area_km2 <= 0 or cfg.target_population <= 0:
        raise ValueError("min_area_km2 and target_population must be positive")
    if cfg.max_area_km2 is not None and cfg.max_area_km2 < cfg.min_area_km2:
        raise ValueError("max_area_km2 cannot be smaller than min_area_km2")
    if isinstance(study_area, StudyArea):
        area_frame = study_area.to_geopandas()
    elif isinstance(study_area, gpd.GeoDataFrame):
        if study_area.empty or study_area.crs is None:
            raise TAZGeometryError("Study-area GeoDataFrame must be non-empty and have a CRS")
        dissolved = study_area.geometry.union_all()
        area_frame = StudyArea(dissolved, study_area.crs).to_geopandas()
    else:
        raise TypeError("study_area must be a StudyArea or GeoDataFrame")
    metric = area_frame.estimate_utm_crs()
    if metric is None:
        raise TAZGeometryError("Could not select a metric CRS for TAZ creation")
    boundary = area_frame.to_crs(metric).geometry.iloc[0]
    min_area = cfg.min_area_km2 * 1_000_000
    max_area = cfg.max_area_km2 * 1_000_000 if cfg.max_area_km2 is not None else None
    population_control = None
    if isinstance(roads, str) and roads == "osm":
        try:
            import osmnx as ox
        except ImportError as exc:
            raise TAZDependencyError(
                "OSM road download requires the osmnx package"
            ) from exc
        polygon = area_frame.to_crs("EPSG:4326").geometry.iloc[0]
        graph = ox.graph_from_polygon(polygon, network_type="drive", simplify=True)
        roads = ox.graph_to_gdfs(graph, nodes=False, edges=True).to_crs(metric)

    if cfg.method == "hex_grid":
        if cfg.grid_resolution is not None:
            side = max(50.0, 2_000_000 / (2 ** cfg.grid_resolution))
        else:
            side = sqrt(2 * max(min_area, 1) / (3 * sqrt(3)))
        geometries = _hexagons(boundary, side)
        base = gpd.GeoDataFrame(
            {"population": [0.0] * len(geometries)}, geometry=geometries, crs=metric
        )
        rows = [
            {"geometry": geometry, "population": pop}
            for _, geometry, pop in _population_groups(base, 0.0, min_area, max_area)
        ]
    else:
        if population is None:
            raise ValueError(f"population is required for {cfg.method} TAZ creation")
        source_cells = population_cells(
            population, target_crs=metric, boundary=boundary
        )
        population_total = float(
            sum(
                value * (cell.intersection(boundary).area / cell.area)
                if cell.area > 0
                else 0.0
                for cell, value in zip(
                    source_cells.geometry, source_cells.population
                )
            )
        )
        population_control = population_total
        target_count = max(1, ceil(population_total / cfg.target_population))
        rows = []
        partitions = _road_partitions(boundary, roads, metric) if (
            cfg.method == "road_constrained" and cfg.respect_major_road_barriers
        ) else [boundary]
        for partition in partitions:
            base = gpd.GeoDataFrame(
                geometry=_squares(partition, max(1, ceil(target_count * partition.area / boundary.area))),
                crs=metric,
            )
            temp = TAZFrame(gpd.GeoDataFrame({
                "taz_id": [str(i) for i in range(len(base))],
                "area_km2": base.geometry.area / 1e6,
                "geometry": base.geometry,
            }, geometry="geometry", crs=metric))
            base = attach_population(temp, source_cells).data
            for _, geometry, pop in _population_groups(
                base, cfg.target_population, min_area, max_area
            ):
                rows.append({"geometry": geometry, "population": pop})

    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=metric)
    result = result[~result.geometry.is_empty].copy()
    result["area_km2"] = result.geometry.area / 1_000_000
    if population_control is not None and result.population.sum() > 0:
        # Reprojection can introduce sub-pixel edge discrepancies. Preserve the
        # source control total while retaining each zone's allocated share.
        result["population"] *= population_control / result.population.sum()
    result["_sort_y"] = result.geometry.centroid.y.round(3)
    result["_sort_x"] = result.geometry.centroid.x.round(3)
    result = result.sort_values(["_sort_y", "_sort_x"], kind="stable").drop(
        columns=["_sort_y", "_sort_x"]
    ).reset_index(drop=True)
    result[cfg.id_column] = [stable_id("taz", geometry.wkb_hex) for geometry in result.geometry]
    if cfg.id_column != "taz_id":
        result["taz_id"] = result[cfg.id_column]
    columns = ["taz_id", *([] if cfg.id_column == "taz_id" else [cfg.id_column]), "area_km2", "population", "geometry"]
    result = result[columns]
    _validate_partition(result, boundary)
    report = TAZReport(
        "create_taz", 1, len(result), details={"config": cfg},
        total_area_km2=float(result.area_km2.sum()),
        population_total=float(result.population.sum()) if population is not None else None,
    )
    # TAZs remain in their local metric CRS so barriers and shared boundaries
    # stay numerically identical. Consumers can explicitly reproject for display.
    return TAZFrame(result, {"taz_config": cfg, "taz_report": report,
                             "study_area_crs": str(study_area.crs)})


create_taz = create


class TAZBuilder:
    """Reusable builder for repeated TAZ generation with one configuration."""

    def __init__(self, *, roads=None, **parameters):
        """Store road barriers and configuration parameters for later builds."""
        self.roads = roads
        self.config = TAZConfig(**parameters)

    def build(self, study_area, *, population=None, roads=None):
        """Generate a TAZFrame, optionally overriding the stored road layer."""
        return create(
            study_area, population=population,
            roads=self.roads if roads is None else roads, config=self.config,
        )
