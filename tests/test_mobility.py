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

    def test_invalid_thresholds_and_columns_are_rejected(self) -> None:
        """Validate user-specified columns and clustering parameters."""
        with self.assertRaises(ValueError):
            self._stops(max_distance_m=0)
        with self.assertRaises(ValueError):
            self._stops(device_id_col="unknown")


if __name__ == "__main__":
    unittest.main()
