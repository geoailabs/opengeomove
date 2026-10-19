"""Population sources and spatially-indexed zonal population attachment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import geopandas as gpd
import numpy as np
from shapely.geometry import box, mapping

from opengeomove.taz.exceptions import TAZDependencyError, TAZGeometryError
from opengeomove.taz.models import TAZFrame


@dataclass(frozen=True)
class PopulationSource:
    """Normalized population-cell source used by TAZ operations."""

    cells: gpd.GeoDataFrame
    value_column: str = "population"
    source: str | None = None

    def __post_init__(self):
        """Validate the cell layer and selected population field."""
        if self.cells.crs is None:
            raise TAZGeometryError("Population cells require a CRS")
        if self.value_column not in self.cells:
            raise TAZGeometryError(f"Population cells require {self.value_column!r}")


def population_cells(
    source,
    *,
    population_column: str = "population",
    target_crs=None,
    boundary: Any = None,
) -> gpd.GeoDataFrame:
    """Load vector or raster population cells and normalize values.

    Args:
        source: PopulationSource, GeoDataFrame, or raster file path.
        population_column: Numeric source field used for population values.
        target_crs: Optional CRS to which the returned cells are projected.
        boundary: Optional geometry in ``target_crs``. Raster reads are cropped
            to this extent and vector sources are spatially filtered before use.

    Returns:
        A copied GeoDataFrame containing valid geometry and non-negative
        numeric values in its normalized ``population`` column.

    Raises:
        TAZDependencyError: If raster input is requested without rasterio.
        TAZGeometryError: If source CRS or population values are missing.
    """
    source_column = population_column
    if isinstance(source, PopulationSource):
        cells = source.cells.copy()
        source_column = source.value_column
    elif isinstance(source, gpd.GeoDataFrame):
        cells = source.copy()
    else:
        try:
            import rasterio
            from rasterio.features import geometry_mask
            from rasterio.transform import xy
            from rasterio.windows import Window, from_bounds
        except ImportError as exc:
            raise TAZDependencyError(
                "Raster population input requires rasterio to be installed"
            ) from exc
        with rasterio.open(source) as dataset:
            if dataset.crs is None:
                raise TAZGeometryError("Population raster requires a CRS")
            window = Window(0, 0, dataset.width, dataset.height)
            boundary_in_source = None
            if boundary is not None:
                if target_crs is None:
                    raise ValueError("target_crs is required when cropping raster population cells")
                boundary_in_source = gpd.GeoSeries([boundary], crs=target_crs).to_crs(
                    dataset.crs
                ).iloc[0]
                requested = from_bounds(
                    *boundary_in_source.bounds, transform=dataset.transform
                ).round_offsets().round_lengths()
                try:
                    window = requested.intersection(
                        Window(0, 0, dataset.width, dataset.height)
                    )
                except rasterio.errors.WindowError:
                    values = np.ma.masked_all((0, 0))
                else:
                    values = dataset.read(1, window=window, masked=True)
            else:
                values = dataset.read(1, masked=True)
            mask = np.ma.getmaskarray(values)
            data = np.asarray(values.data)
            rows, cols = np.where(~mask & np.isfinite(data) & (data >= 0))
            geometries, populations = [], []
            transform = dataset.window_transform(window)
            if boundary_in_source is not None and values.size:
                inside = geometry_mask(
                    [mapping(boundary_in_source)],
                    out_shape=values.shape,
                    transform=transform,
                    invert=True,
                    all_touched=True,
                )
                rows, cols = np.where(
                    inside & ~mask & np.isfinite(data) & (data >= 0)
                )
            for row, col in zip(rows, cols):
                left, top = xy(transform, row, col, offset="ul")
                right, bottom = xy(transform, row, col, offset="lr")
                geometries.append(box(min(left, right), min(top, bottom), max(left, right), max(top, bottom)))
                populations.append(float(data[row, col]))
            cells = gpd.GeoDataFrame({population_column: populations}, geometry=geometries, crs=dataset.crs)
    if cells.crs is None:
        raise TAZGeometryError("Population cells require a CRS")
    if source_column not in cells:
        raise TAZGeometryError(f"Population cells require {source_column!r}")
    cells = cells[cells.geometry.notna() & ~cells.geometry.is_empty].copy()
    numeric_population = np.asarray(cells[source_column], dtype=float)
    numeric_population[~np.isfinite(numeric_population)] = 0.0
    cells["population"] = np.maximum(0.0, numeric_population)
    if boundary is not None and len(cells):
        if target_crs is None:
            raise ValueError("target_crs is required when filtering population cells")
        boundary_source = gpd.GeoSeries([boundary], crs=target_crs).to_crs(cells.crs).iloc[0]
        positions = cells.sindex.query(boundary_source, predicate="intersects")
        cells = cells.iloc[positions].copy()
    if target_crs is not None:
        cells = cells.to_crs(target_crs)
    return cells.reset_index(drop=True)


def _candidate_pairs(zones: gpd.GeoDataFrame, cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return only cell/zone pairs whose spatial indexes intersect."""
    joined = gpd.sjoin(cells[["geometry"]], zones[["geometry"]], how="inner", predicate="intersects")
    return joined.reset_index().rename(columns={"index": "cell_index"})


def attach_population(
    zones: TAZFrame,
    population,
    *,
    output_column: str = "population",
    allocation: Literal["area_weighted", "cell_centroid"] = "area_weighted",
    population_column: str = "population",
) -> TAZFrame:
    """Attach population to zones using a spatial-indexed candidate join.

    ``area_weighted`` conserves each cell proportionally across intersecting
    zones. ``cell_centroid`` assigns the whole cell to the zone containing its
    centroid. Complete partitions conserve the source control total.

    Args:
        zones: TAZFrame receiving a population column.
        population: PopulationSource, vector cells, or raster path.
        output_column: Name of the attached zone-level value.
        allocation: ``"area_weighted"`` or ``"cell_centroid"`` allocation.
        population_column: Source field containing population values.

    Returns:
        A new TAZFrame with population values and allocation metadata.
    """
    if allocation not in {"area_weighted", "cell_centroid"}:
        raise ValueError("allocation must be area_weighted or cell_centroid")
    zone_data = zones.to_geopandas()
    zones_boundary = zone_data.geometry.union_all()
    cells = population_cells(
        population,
        population_column=population_column,
        target_crs=zone_data.crs,
        boundary=zones_boundary,
    )
    totals = np.zeros(len(zone_data), dtype=float)
    if not zone_data.empty and not cells.empty:
        pairs = _candidate_pairs(zone_data, cells)
        contributions: list[tuple[int, float]] = []
        for pair in pairs.itertuples(index=False):
            cell_index = int(pair.cell_index)
            zone_index = int(pair.index_right)
            cell = cells.geometry.iloc[cell_index]
            value = float(cells["population"].iloc[cell_index])
            if allocation == "cell_centroid":
                if zone_data.geometry.iloc[zone_index].covers(cell.centroid):
                    contributions.append((zone_index, value))
            elif cell.area > 0:
                overlap = cell.intersection(zone_data.geometry.iloc[zone_index]).area
                if overlap > 0:
                    contributions.append((zone_index, value * overlap / cell.area))
        for zone_index, value in contributions:
            totals[zone_index] += value
    if allocation == "area_weighted" and not cells.empty:
        union = zone_data.geometry.union_all()
        coverage = [union.intersection(cell).area / cell.area for cell in cells.geometry if cell.area > 0]
        allocated = float(totals.sum())
        if coverage and min(coverage) >= 0.999 and allocated > 0:
            totals *= float(cells["population"].sum()) / allocated
    zone_data[output_column] = totals
    metadata = {
        **zones.metadata,
        "population_source": str(population),
        "population_total": float(totals.sum()),
        "population_allocation": allocation,
        "population_cells": int(len(cells)),
    }
    return TAZFrame(zone_data, metadata)
