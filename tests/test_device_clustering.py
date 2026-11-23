"""Tests for trajectory-derived device profiles and five-cluster KMeans."""

from pathlib import Path
import tempfile
import unittest

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import opengeomove as ogm


class DeviceClusteringTests(unittest.TestCase):
    """Exercise device feature aggregation and reproducible clustering."""

    def setUp(self) -> None:
        """Create a small trajectory-visit fixture with five devices."""
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary_directory.name) / "visits.parquet"
        connection = duckdb.connect()
        connection.execute(
            """COPY (SELECT * FROM (VALUES
                ('a','a-1',DATE '2025-01-01',TIMESTAMP '2025-01-01 08:00:00',1000.0,1,1,1,28.60,77.20),
                ('a','a-1',DATE '2025-01-01',TIMESTAMP '2025-01-01 08:00:00',1000.0,2,1,2,28.61,77.21),
                ('a','a-2',DATE '2025-01-02',TIMESTAMP '2025-01-02 09:00:00',3000.0,1,0,2,28.60,77.20),
                ('b','b-1',DATE '2025-01-01',TIMESTAMP '2025-01-01 08:00:00',6000.0,1,1,3,28.70,77.30),
                ('b','b-1',DATE '2025-01-01',TIMESTAMP '2025-01-01 08:00:00',6000.0,2,1,4,28.71,77.31),
                ('b','b-2',DATE '2025-01-02',TIMESTAMP '2025-01-02 08:00:00',40000.0,1,1,5,28.80,77.40),
                ('c','c-1',DATE '2025-01-01',TIMESTAMP '2025-01-01 08:00:00',100.0,1,1,6,28.90,77.50),
                ('d','d-1',DATE '2025-01-01',TIMESTAMP '2025-01-01 08:00:00',20000.0,1,0,7,28.95,77.55),
                ('e','e-1',DATE '2025-01-01',TIMESTAMP '2025-01-01 08:00:00',50000.0,1,1,8,29.00,77.60),
                ('e','e-2',DATE '2025-01-02',TIMESTAMP '2025-01-02 08:00:00',50000.0,1,1,9,29.10,77.70)
            ) AS visits(
                device_id, trajectory_id, trajectory_date,
                trajectory_start_time, trajectory_distance_m, visit_order,
                stay_point, spatial_cluster_label, lat, lon
            )) TO ? (FORMAT PARQUET)""",
            [str(self.path)],
        )
        connection.close()
        self.dataset = ogm.open(self.path)

    def tearDown(self) -> None:
        """Close DuckDB and remove the temporary fixture."""
        plt.close("all")
        self.dataset._connection.close()
        self._temporary_directory.cleanup()

    def test_build_device_features_deduplicates_trajectory_values(self) -> None:
        """Count repeated trajectory values once and expose all slide features."""
        features = self.dataset.build_device_features()

        self.assertEqual(features.columns, ("device_id", *ogm.DEVICE_CLUSTER_FEATURES))
        self.assertEqual(features.shape, (5, 21))
        rows = {row[0]: row for row in features.rows}
        device_a = rows["a"]
        self.assertEqual(device_a[1], 2)
        self.assertAlmostEqual(device_a[2], 2.0)
        self.assertAlmostEqual(device_a[5], 1.0)
        self.assertAlmostEqual(device_a[10], 0.5)
        self.assertAlmostEqual(device_a[11], 0.5)
        self.assertEqual(device_a[15], 2)
        self.assertAlmostEqual(device_a[16], 0.0)

    def test_cluster_devices_returns_five_assignments_and_profiles(self) -> None:
        """Fit five deterministic clusters and provide heatmap-ready profiles."""
        features = self.dataset.build_device_features()
        first = features.cluster_devices(n_clusters=5, random_state=42)
        second = features.cluster_devices(n_clusters=5, random_state=42)

        self.assertEqual(first.n_clusters, 5)
        self.assertEqual(first.assignments.shape, (5, 4))
        self.assertEqual(first.profiles.shape[0], 5)
        self.assertEqual(first.normalized_profiles.shape[0], 5)
        self.assertEqual(list(first.assignments.rows), list(second.assignments.rows))
        normalized = list(first.normalized_profiles.rows)
        self.assertTrue(
            all(0.0 <= value <= 1.0 for row in normalized for value in row[3:])
        )

        figure = first.plot_profiles()
        try:
            self.assertEqual(len(figure.axes[0].patches), 5)
            self.assertEqual(len(figure.axes[1].images), 1)
            self.assertEqual(len(figure.axes[1].get_yticklabels()), 20)
        finally:
            plt.close(figure)

    def test_cluster_devices_rejects_duplicate_device_rows(self) -> None:
        """Require a device feature table to contain one row per device."""
        features = self.dataset.build_device_features()
        duplicate_relation = features._relation.query(
            "_ogm_duplicate_device_features",
            "SELECT * FROM _ogm_duplicate_device_features "
            "UNION ALL SELECT * FROM _ogm_duplicate_device_features",
        )
        duplicated = features._clone_with_relation(duplicate_relation)
        with self.assertRaises(ValueError):
            duplicated.cluster_devices(n_clusters=2)


if __name__ == "__main__":
    unittest.main()
