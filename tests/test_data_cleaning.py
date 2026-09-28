"""Tests for lazy Dataset filter composition and UTC time behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb

import opengeomove as ogm


class DataCleaningTests(unittest.TestCase):
    """Exercise filters against a small local Parquet fixture."""

    def setUp(self) -> None:
        """Create a temporary Parquet dataset with mixed clean/noisy pings."""
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary_directory.name) / "pings.parquet"
        connection = duckdb.connect()
        connection.execute(
            """COPY (SELECT * FROM (VALUES
                ('a', TIMESTAMP '2025-01-01 00:00:00', 1000::BIGINT, 10.0::DOUBLE),
                ('a', TIMESTAMP '2025-01-01 12:00:00', 2000::BIGINT, NULL::DOUBLE),
                ('a', TIMESTAMP '2025-01-02 00:00:00', 3000::BIGINT, 30.0::DOUBLE),
                ('b', TIMESTAMP '2025-01-01 01:00:00', 4000::BIGINT, 20.0::DOUBLE)
            ) AS p(device_id, observed_at, timestamp_ms, horizontal_accuracy_m))
            TO ? (FORMAT PARQUET)""",
            [str(self.path)],
        )
        connection.close()
        self.dataset = ogm.open(self.path)

    def tearDown(self) -> None:
        """Close the Dataset connection and remove its temporary fixture."""
        self.dataset._connection.close()
        self._temporary_directory.cleanup()

    def test_chained_filters_are_lazy_views_and_leave_source_unchanged(self) -> None:
        """Verify filters compose and source row counts remain unchanged."""
        filtered = self.dataset.apply(
            ogm.processing.DeviceIDFilter(["a"])
        ).apply(ogm.processing.HorizontalAccuracyFilter(20))

        self.assertEqual(filtered.shape, (2, 4))
        self.assertEqual(self.dataset.shape, (4, 4))
        self.assertEqual(len(list(filtered.rows)), 2)
        self.assertEqual(len(filtered.filters), 2)

    def test_time_ranges_use_timezone_and_half_open_boundaries(self) -> None:
        """Interpret naive bounds in their declared timezone and use [start,end)."""
        filtered = self.dataset.apply(
            ogm.processing.TimeRangeFilter(
                "2025-01-01T05:30:00",
                "2025-01-01T06:30:00",
                timestamp_col="observed_at",
                timezone="Asia/Kolkata",
            )
        )

        self.assertEqual(filtered.shape[0], 1)

    def test_daily_counts_are_grouped_by_utc_day(self) -> None:
        """Retain device/day groups meeting a threshold using UTC day keys."""
        filtered = self.dataset.apply(
            ogm.processing.DailyPingCountFilter(
                min_pings=2,
                timestamp_col="observed_at",
            )
        )

        self.assertEqual(filtered.shape[0], 2)

    def test_epoch_milliseconds_are_supported(self) -> None:
        """Recognize the timestamp_ms column as epoch milliseconds."""
        filtered = self.dataset.apply(
            ogm.processing.DailyPingCountFilter(
                min_pings=2,
                timestamp_col="timestamp_ms",
            )
        )

        self.assertEqual(filtered.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
