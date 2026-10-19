"""Configuration for reusable TAZ generation algorithms."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


@dataclass(frozen=True)
class TAZConfig:
    """Settings for grid, population-balanced, or road-constrained TAZs.

    Args:
        method: Generation strategy: ``"hex_grid"``,
            ``"population_balanced"``, or ``"road_constrained"``.
        target_population: Approximate people per generated zone for the two
            population-based strategies.
        min_area_km2: Minimum preferred zone area; undersized adjacent zones
            are merged when possible.
        max_area_km2: Optional preferred upper area limit.
        grid_resolution: Optional H3-like refinement hint for hex sizing.
        respect_major_road_barriers: Split population-balanced zones on road
            barriers when a road layer is provided.
        id_column: Name of a caller-specific duplicate of ``taz_id``.
    """

    method: Literal["population_balanced", "road_constrained", "hex_grid"] = "population_balanced"
    target_population: float = 50_000.0
    min_area_km2: float = 0.25
    max_area_km2: float | None = None
    grid_resolution: int | None = None
    respect_major_road_barriers: bool = True
    id_column: str = "taz_id"

    def __post_init__(self) -> None:
        """Reject invalid TAZ sizing and identifier settings."""
        if self.method not in {"population_balanced", "road_constrained", "hex_grid"}:
            raise ValueError(f"Unsupported TAZ method: {self.method!r}")
        for name, value in (
            ("target_population", self.target_population),
            ("min_area_km2", self.min_area_km2),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive number")
        if self.max_area_km2 is not None:
            if (
                isinstance(self.max_area_km2, bool)
                or not isinstance(self.max_area_km2, (int, float))
                or not math.isfinite(self.max_area_km2)
                or self.max_area_km2 < self.min_area_km2
            ):
                raise ValueError("max_area_km2 must be finite and at least min_area_km2")
        if self.grid_resolution is not None and (
            isinstance(self.grid_resolution, bool)
            or not isinstance(self.grid_resolution, int)
            or self.grid_resolution < 0
        ):
            raise ValueError("grid_resolution must be a non-negative integer")
        if not isinstance(self.respect_major_road_barriers, bool):
            raise TypeError("respect_major_road_barriers must be a bool")
        if not isinstance(self.id_column, str) or not self.id_column:
            raise ValueError("id_column must be a non-empty string")
