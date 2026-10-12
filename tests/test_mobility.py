"""Tests for per-device stop clustering and trip formation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb

import opengeomove as ogm


class MobilityTests(unittest.TestCase):
    """Exercise stop and trip operations on a small local Parquet fixture."""

    def setUp(self) -> None:
        """Create two stop locations and movement pings for two devices."""
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary_directory.name) / "pings.parquet"
        connection = duckdb.connect()
        connection.execute(
            """COPY (SELECT * FROM (VALUES
                ('a', TIMESTAMP '2025-01-01 00:00:00', 28.61390::DOUBLE, 77.20900::DOUBLE),
                ('a', TIMESTAMP '2025-01-01 00:02:30', 28.61391::DOUBLE, 77.20901::DOUBLE),
                ('a', TIMESTAMP '2025-01-01 00:05:00', 28.61392::DOUBLE, 77.20902::DOUBLE),
                ('a', TIMESTAMP '2025-01-01 00:10:00', 28.61800::DOUBLE, 77.20900::DOUBLE),
                ('a', TIMESTAMP '2025-01-01 00:12:00', 28.61900::DOUBLE, 77.20900::DOUBLE),
                ('a', TIMESTAMP '2025-01-01 00:20:00', 28.63390::DOUBLE, 77.20900::DOUBLE),
                ('a', TIMESTAMP '2025-01-01 00:22:30', 28.63391::DOUBLE, 77.20901::DOUBLE),
                ('a', TIMESTAMP '2025-01-01 00:25:00', 28.63392::DOUBLE, 77.20902::DOUBLE),
                ('b', TIMESTAMP '2025-01-01 00:00:00', 28.70000::DOUBLE, 77.20000::DOUBLE),
                ('b', TIMESTAMP '2025-01-01 00:05:00', 28.70001::DOUBLE, 77.20001::DOUBLE)
            ) AS points(device_id, observed_at, lat, lon))
            TO ? (FORMAT PARQUET)""",
            [str(self.path)],
        )
        connection.close()
        self.dataset = ogm.open(self.path)

    def tearDown(self) -> None:
        """Close the source connection and remove temporary data."""
        self.dataset._connection.close()
        self._temporary_directory.cleanup()

    def _stops(self, **overrides: object) -> ogm.Dataset:
        """Create stop clusters with the test's common parameters."""
        options: dict[str, object] = {
            "device_id_col": "device_id",
            "timestamp_col": "observed_at",
            "lat_col": "lat",
            "lon_col": "lon",
            "max_distance_m": 100,
            "min_stop_duration_s": 300,
            "max_time_gap_s": 900,
        }
        options.update(overrides)
        return self.dataset.cluster_stops(**options)  # type: ignore[arg-type]

    def test_cluster_stops_returns_per_device_stop_summaries(self) -> None:
        """Find stops per device and leave the source Dataset untouched."""
        stops = self._stops()

        self.assertEqual(stops.shape, (3, 8))
        self.assertEqual(self.dataset.shape, (10, 4))
        rows = list(stops.rows)
        self.assertEqual([row[1] for row in rows], ["a", "a", "b"])
        self.assertEqual([row[6] for row in rows], [3, 3, 2])
        self.assertEqual([row[7] for row in rows], [300.0, 300.0, 300.0])

    def test_form_trips_pairs_consecutive_stops_within_time_gap(self) -> None:
        """Build one summary from device a's two consecutive stop clusters."""
        stops = self._stops()
        trips = stops.form_trips(max_time_gap_s=901)

        self.assertEqual(trips.shape, (1, 12))
        trip = next(iter(trips.rows))
        self.assertEqual(trip[1], "a")
        self.assertEqual(trip[2:4], (1, 2))
        self.assertEqual(trip[10], 900.0)
        self.assertGreater(trip[11], 2_000)

    def test_trip_gap_above_threshold_does_not_form_trip(self) -> None:
        """Omit adjacent stops whose between-stop time exceeds the limit."""
        trips = self._stops().form_trips(max_time_gap_s=899)

        self.assertEqual(trips.shape[0], 0)

    def test_home_work_classification_uses_local_windows_and_evidence(self) -> None:
        """Identify recurring home and weekday work locations from stop rows."""
        path = Path(self._temporary_directory.name) / "recurring-stops.parquet"
        connection = duckdb.connect()
        connection.execute(
            """COPY (SELECT * FROM (VALUES
                (1, 'a', TIMESTAMPTZ '2025-01-06 15:30:00+00', TIMESTAMPTZ '2025-01-06 16:00:00+00', 28.61000::DOUBLE, 77.20000::DOUBLE),
                (2, 'a', TIMESTAMPTZ '2025-01-07 04:30:00+00', TIMESTAMPTZ '2025-01-07 05:00:00+00', 28.62000::DOUBLE, 77.21000::DOUBLE),
                (3, 'a', TIMESTAMPTZ '2025-01-07 15:30:00+00', TIMESTAMPTZ '2025-01-07 16:00:00+00', 28.61001::DOUBLE, 77.20001::DOUBLE),
                (4, 'a', TIMESTAMPTZ '2025-01-08 04:30:00+00', TIMESTAMPTZ '2025-01-08 05:00:00+00', 28.62001::DOUBLE, 77.21001::DOUBLE),
                (5, 'a', TIMESTAMPTZ '2025-01-08 15:30:00+00', TIMESTAMPTZ '2025-01-08 16:00:00+00', 28.61002::DOUBLE, 77.20002::DOUBLE),
                (6, 'a', TIMESTAMPTZ '2025-01-09 04:30:00+00', TIMESTAMPTZ '2025-01-09 05:00:00+00', 28.62002::DOUBLE, 77.21002::DOUBLE)
            ) AS stops(cluster_id, device_id, start_time, end_time, latitude, longitude))
            TO ? (FORMAT PARQUET)""",
            [str(path)],
        )
        connection.close()
        stops = ogm.open(path)
        stops._stop_merge_distance_m = 100
        try:
            classified = stops.classify_home_work(
                timezone_name="Asia/Kolkata",
                min_visits=3,
                min_distinct_days=3,
                min_total_dwell_s=5_400,
            )

            rows = {row[1]: row for row in classified.rows}
            self.assertEqual(set(rows), {"home", "work"})
            self.assertEqual(rows["home"][4], "classified")
            self.assertEqual(rows["work"][4], "classified")
            self.assertEqual(rows["home"][6:8], (3, 3))
            self.assertEqual(rows["work"][6:8], (3, 3))
            self.assertAlmostEqual(rows["home"][2], 28.61001, places=4)
            self.assertAlmostEqual(rows["work"][2], 28.62001, places=4)
            self.assertEqual(stops.shape, (6, 6))
        finally:
            stops._connection.close()

    def test_home_work_uncertain_for_tied_or_weak_candidates(self) -> None:
        """Return uncertain candidates when evidence minima are not met."""
        stops = self._stops()
        classified = stops.classify_home_work(
            timezone_name="UTC",
            min_visits=5,
            min_distinct_days=2,
            min_total_dwell_s=600,
            merge_nearby=False,
        )

        rows = list(classified.rows)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row[4] == "uncertain" for row in rows))

        tied = stops.classify_home_work(
            timezone_name="UTC",
            min_visits=1,
            min_distinct_days=1,
            min_total_dwell_s=300,
            merge_nearby=False,
        )
        by_device_role = {(row[0], row[1]): row for row in tied.rows}
        self.assertEqual(by_device_role[("a", "home")][4], "uncertain")
        self.assertEqual(by_device_role[("b", "home")][4], "classified")

    def test_reusable_thresholds_apply_to_stops_and_trips(self) -> None:
        """Use one immutable threshold set for both analysis stages."""
        thresholds = ogm.MobilityThresholds(
            max_distance_m=100,
            min_stop_duration_s=300,
            max_time_gap_s=900,
            trip_max_time_gap_s=901,
        )
        stops = self.dataset.cluster_stops(
            device_id_col="device_id",
            timestamp_col="observed_at",
            lat_col="lat",
            lon_col="lon",
            thresholds=thresholds,
        )
        trips = stops.form_trips(thresholds=thresholds)

        self.assertEqual(stops.shape[0], 3)
        self.assertEqual(trips.shape[0], 1)

    def test_stop_centroid_handles_points_across_the_date_line(self) -> None:
        """Use a spherical center instead of averaging longitude through zero."""
        path = Path(self._temporary_directory.name) / "date-line.parquet"
        connection = duckdb.connect()
        connection.execute(
            """COPY (SELECT * FROM (VALUES
                ('c', TIMESTAMP '2025-01-01 00:00:00', 0.0::DOUBLE, 179.999::DOUBLE),
                ('c', TIMESTAMP '2025-01-01 00:02:00', 0.0::DOUBLE, -179.999::DOUBLE)
            ) AS points(device_id, observed_at, lat, lon)) TO ? (FORMAT PARQUET)""",
            [str(path)],
        )
        connection.close()
        dataset = ogm.open(path)
        try:
            stops = dataset.cluster_stops(
                device_id_col="device_id",
                timestamp_col="observed_at",
                lat_col="lat",
                lon_col="lon",
                max_distance_m=300,
                min_stop_duration_s=60,
                max_time_gap_s=300,
            )

            stop = next(iter(stops.rows))
            self.assertAlmostEqual(abs(stop[5]), 180.0, places=3)
        finally:
            dataset._connection.close()

    def test_invalid_thresholds_and_columns_are_rejected(self) -> None:
        """Validate user-specified columns and clustering parameters."""
        with self.assertRaises(ValueError):
            self._stops(max_distance_m=0)
        with self.assertRaises(ValueError):
            self._stops(device_id_col="unknown")


if __name__ == "__main__":
    unittest.main()
