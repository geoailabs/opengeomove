"""Reusable, caller-selected thresholds for stop and trip detection."""

from __future__ import annotations

from dataclasses import dataclass
import math


def _validate_threshold(name: str, value: float, *, allow_zero: bool = False) -> float:
    """Validate one finite distance or duration threshold."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return number


@dataclass(frozen=True)
class MobilityThresholds:
    """Caller-defined thresholds that can be reused across city datasets.

    The configuration contains no city-specific assumptions. Dataset column
    names, timestamp timezone, and epoch unit remain arguments to the
    operation because those can vary between sources.

    Args:
        max_distance_m: Maximum distance from a stop candidate's first ping.
        min_stop_duration_s: Minimum observed duration required for a stop.
        max_time_gap_s: Maximum gap between consecutive pings in one stop
            candidate.
        trip_max_time_gap_s: Optional maximum elapsed time between one stop's
            end and the next stop's start for them to form a trip.

    Raises:
        TypeError: If a threshold is not numeric.
        ValueError: If a threshold is not finite or outside its allowed range.
    """

    max_distance_m: float
    min_stop_duration_s: float
    max_time_gap_s: float
    trip_max_time_gap_s: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalize configured thresholds."""
        object.__setattr__(
            self, "max_distance_m", _validate_threshold("max_distance_m", self.max_distance_m)
        )
        object.__setattr__(
            self,
            "min_stop_duration_s",
            _validate_threshold(
                "min_stop_duration_s", self.min_stop_duration_s, allow_zero=True
            ),
        )
        object.__setattr__(
            self, "max_time_gap_s", _validate_threshold("max_time_gap_s", self.max_time_gap_s)
        )
        if self.trip_max_time_gap_s is not None:
            object.__setattr__(
                self,
                "trip_max_time_gap_s",
                _validate_threshold("trip_max_time_gap_s", self.trip_max_time_gap_s),
            )
