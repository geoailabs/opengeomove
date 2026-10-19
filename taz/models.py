"""Spatial result objects used by the TAZ module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

from opengeomove.taz.exceptions import TAZGeometryError


@dataclass
class StudyArea:
    """A named polygonal study boundary with an explicit coordinate system.

    Args:
        geometry: Shapely Polygon or MultiPolygon in ``crs``.
        crs: Coordinate reference system for the supplied geometry.
        name: Descriptive boundary label.
        metadata: Optional provenance details retained with the boundary.

    Raises:
        TAZGeometryError: If the geometry is empty, invalid, or not polygonal.
    """

    geometry: Any
    crs: Any = "EPSG:4326"
    name: str = "study_area"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and safely repair the study boundary."""
        if self.geometry is None or self.geometry.is_empty:
            raise TAZGeometryError("StudyArea geometry cannot be empty")
        if self.geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise TAZGeometryError("StudyArea geometry must be a polygon or multipolygon")
        if self.crs is None:
            raise TAZGeometryError("StudyArea requires an explicit CRS")
        if not self.geometry.is_valid:
            repaired = self.geometry.buffer(0)
            if repaired.is_empty or not repaired.is_valid:
                raise TAZGeometryError("StudyArea geometry is invalid and cannot be repaired")
            self.geometry = repaired
        self.metadata = dict(self.metadata)

    def to_geopandas(self) -> gpd.GeoDataFrame:
        """Return a one-row GeoDataFrame without changing the stored geometry."""
        return gpd.GeoDataFrame(
            {"name": [self.name]}, geometry=[self.geometry], crs=self.crs
        )


@dataclass
class TAZFrame:
    """Validated polygon zones plus generation metadata.

    This spatial result remains separate from the file-backed tabular
    :class:`~opengeomove.Dataset`; TAZ geometry is represented by a
    GeoDataFrame because the current Dataset reader does not load vector files.

    Args:
        data: GeoDataFrame containing unique ``taz_id`` values and polygons.
        metadata: Configuration, provenance, and report details.

    Raises:
        TAZGeometryError: If the frame has an invalid CRS, key, or geometry.
    """

    data: gpd.GeoDataFrame
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy and validate zone data, then retain metadata separately."""
        if not isinstance(self.data, gpd.GeoDataFrame):
            raise TypeError("TAZFrame.data must be a GeoDataFrame")
        self.data = self.data.copy()
        self.metadata = dict(self.metadata)
        required = {"taz_id", "area_km2", self.data.geometry.name}
        missing = required - set(self.data.columns)
        if missing:
            raise TAZGeometryError(f"TAZFrame is missing columns: {sorted(missing)}")
        if self.data.crs is None:
            raise TAZGeometryError("TAZFrame requires an explicit CRS")
        if self.data.empty:
            raise TAZGeometryError("TAZFrame cannot be empty")
        if self.data.taz_id.isna().any() or self.data.taz_id.duplicated().any():
            raise TAZGeometryError("TAZ identifiers must be non-null and unique")
        geometry = self.data.geometry
        if geometry.isna().any() or geometry.is_empty.any() or (~geometry.is_valid).any():
            raise TAZGeometryError("TAZ polygons must be non-empty and valid")
        if not set(geometry.geom_type).issubset({"Polygon", "MultiPolygon"}):
            raise TAZGeometryError("TAZ geometry must contain polygons only")
        areas = pd.to_numeric(self.data.area_km2, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(areas).all() or (areas <= 0).any():
            raise TAZGeometryError("TAZ areas must be positive")
        metric = self.data
        if getattr(metric.crs, "is_geographic", False):
            metric_crs = metric.estimate_utm_crs()
            if metric_crs is None:
                raise TAZGeometryError("Could not estimate a metric CRS to validate zone overlap")
            metric = metric.to_crs(metric_crs)
        total_area = float(metric.geometry.area.sum())
        union_area = float(metric.geometry.union_all().area)
        if total_area - union_area > max(union_area, 1.0) * 1e-6:
            raise TAZGeometryError("TAZ polygons must not overlap")
        self.metadata.setdefault("object_type", "TAZFrame")
        self.metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        self.metadata.setdefault("crs", str(self.data.crs))

    def __len__(self) -> int:
        """Return the number of zones without further materialization."""
        return len(self.data)

    def to_geopandas(self) -> gpd.GeoDataFrame:
        """Return a copy of the zone GeoDataFrame."""
        return self.data.copy()

    def attach_population(self, population: Any, **options: Any) -> TAZFrame:
        """Return a copy of this TAZFrame with population attached.

        Args:
            population: PopulationSource, polygon cells, or raster path.
            **options: Options accepted by :func:`taz.attach_population`,
                such as ``allocation`` and ``population_column``.

        Returns:
            A new TAZFrame with a population column and provenance metadata.
        """
        from opengeomove.taz.population import attach_population

        return attach_population(self, population, **options)


@dataclass(frozen=True)
class TAZReport:
    """Small summary of a TAZ generation or comparison operation."""

    operation: str
    input_rows: int
    output_rows: int
    details: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    total_area_km2: float = 0.0
    population_total: float | None = None
