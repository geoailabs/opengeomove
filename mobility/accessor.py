"""Dataset methods for mobility clustering and trip formation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from opengeomove.mobility.config import MobilityThresholds
from opengeomove.mobility.home_work import classify_home_work
from opengeomove.mobility.operations import cluster_stops, form_trips

if TYPE_CHECKING:
    from opengeomove.data_loading.dataset import Dataset


class MobilityAccessor:
    """Expose mobility analysis methods on Dataset objects."""

    def cluster_stops(
        self: Dataset,
        *,
        device_id_col: str,
        timestamp_col: str,
        lat_col: str,
        lon_col: str,
        max_distance_m: float | None = None,
        min_stop_duration_s: float | None = None,
        max_time_gap_s: float | None = None,
        thresholds: MobilityThresholds | None = None,
        source_timezone: str = "UTC",
        timestamp_unit: str = "auto",
    ) -> Dataset:
        """Return per-device stop clusters using spatial and time thresholds.

        Args:
            device_id_col: Column containing device identifiers.
            timestamp_col: Timestamp or numeric epoch column.
            lat_col: WGS84 latitude column.
            lon_col: WGS84 longitude column.
            max_distance_m: Maximum candidate radius from its first ping.
            min_stop_duration_s: Minimum observed stop duration in seconds.
            max_time_gap_s: Maximum gap between consecutive candidate pings.
            thresholds: Reusable caller-selected thresholds. Pass this instead
                of the three individual stop thresholds.
            source_timezone: Timezone for naive source timestamps.
            timestamp_unit: ``"auto"``, ``"ms"``, or ``"s"`` for epochs.

        Returns:
            A Dataset with one row per detected stop.

        Raises:
            TypeError: If a column name or threshold has an invalid type.
            ValueError: If a column is missing or a threshold is invalid.
        """
        return cluster_stops(
            self,
            device_id_col=device_id_col,
            timestamp_col=timestamp_col,
            lat_col=lat_col,
            lon_col=lon_col,
            max_distance_m=max_distance_m,
            min_stop_duration_s=min_stop_duration_s,
            max_time_gap_s=max_time_gap_s,
            thresholds=thresholds,
            source_timezone=source_timezone,
            timestamp_unit=timestamp_unit,
        )

    def form_trips(
        self: Dataset,
        *,
        max_time_gap_s: float | None = None,
        thresholds: MobilityThresholds | None = None,
    ) -> Dataset:
        """Return trips between consecutive stop clusters for each device.

        Args:
            max_time_gap_s: Maximum time between consecutive stops, in seconds.
            thresholds: Reusable threshold set with
                ``trip_max_time_gap_s`` configured.

        Returns:
            A Dataset with one summary row per qualifying trip.

        Raises:
            TypeError: If the Dataset or threshold has an invalid type.
            ValueError: If required stop columns are absent or the threshold
                is invalid.
        """
        return form_trips(
            self,
            max_time_gap_s=max_time_gap_s,
            thresholds=thresholds,
        )

    def classify_home_work(
        self: Dataset,
        *,
        timezone_name: str,
        min_visits: int,
        min_distinct_days: int,
        min_total_dwell_s: float,
        merge_nearby: bool = True,
        merge_distance_m: float | None = None,
        home_window: tuple[str, str] = ("20:00", "06:00"),
        work_window: tuple[str, str] = ("09:00", "17:00"),
        work_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4),
    ) -> Dataset:
        """Classify recurring stop locations as likely home and work.

        Args:
            timezone_name: IANA timezone used to interpret local-time windows.
            min_visits: Minimum stop visits required for classification.
            min_distinct_days: Minimum local window days with evidence.
            min_total_dwell_s: Minimum dwell within the role windows.
            merge_nearby: Merge nearby locations; defaults to true.
            merge_distance_m: Merge radius, defaulting to the stop-clustering
                radius recorded by ``cluster_stops``.
            home_window: Local-time ``(start, end)`` pair, default 20:00–06:00.
            work_window: Local-time ``(start, end)`` pair, default 09:00–17:00.
            work_weekdays: Eligible weekdays using Monday=0 through Sunday=6.

        Returns:
            A Dataset with at most one candidate row per device and role,
            including an ``uncertain`` classification status when evidence is
            insufficient or tied.

        Raises:
            TypeError: If a parameter or this Dataset has an invalid type.
            ValueError: If parameters or required stop-summary columns are
                invalid.
        """
        return classify_home_work(
            self,
            timezone_name=timezone_name,
            min_visits=min_visits,
            min_distinct_days=min_distinct_days,
            min_total_dwell_s=min_total_dwell_s,
            merge_nearby=merge_nearby,
            merge_distance_m=merge_distance_m,
            home_window=home_window,
            work_window=work_window,
            work_weekdays=work_weekdays,
        )
