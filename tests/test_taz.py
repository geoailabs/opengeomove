"""Tests for spatial TAZ creation and comparison."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, box

import opengeomove as ogm


class TAZTests(unittest.TestCase):
    """Exercise TAZ creation, population attachment, and comparisons."""

    def setUp(self) -> None:
        """Create a small projected study area and population cells."""
        self.boundary = box(77.0, 28.0, 77.04, 28.04)
        self.area = ogm.taz.StudyArea(self.boundary, "EPSG:4326", "test")
        cells = [
            box(x, y, x + 0.02, y + 0.02)
            for x in (77.0, 77.02)
            for y in (28.0, 28.02)
        ]
        self.population = gpd.GeoDataFrame(
            {"population": [100.0] * len(cells)}, geometry=cells, crs="EPSG:4326"
        )

    def test_population_balanced_zones_conserve_population_and_ids(self) -> None:
        """Generate deterministic, non-overlapping population zones."""
        config = ogm.taz.TAZConfig(
            method="population_balanced", target_population=100, min_area_km2=0.01
        )
        first = ogm.taz.create_taz(self.area, population=self.population, config=config)
        second = ogm.taz.create_taz(self.area, population=self.population, config=config)

        self.assertGreater(len(first), 1)
        self.assertEqual(first.data.taz_id.tolist(), second.data.taz_id.tolist())
        self.assertAlmostEqual(first.data.population.sum(), 400, delta=0.01)
        self.assertTrue(first.data.geometry.is_valid.all())

    def test_hex_grid_covers_boundary_and_population_can_be_attached(self) -> None:
        """Create grid zones and attach population using spatial allocation."""
        zones = ogm.taz.create_taz(
            self.area,
            config=ogm.taz.TAZConfig(
                method="hex_grid", grid_resolution=10, min_area_km2=0.01
            ),
        )
        source = ogm.taz.PopulationSource(
            self.population.rename(columns={"population": "people"}),
            value_column="people",
        )
        populated = zones.attach_population(source)
        metric_crs = self.area.to_geopandas().estimate_utm_crs()
        boundary = self.area.to_geopandas().to_crs(metric_crs).geometry.iloc[0]
        projected = zones.data.to_crs(metric_crs)

        self.assertGreater(len(zones), 1)
        self.assertLess(
            boundary.difference(projected.geometry.union_all()).area / boundary.area,
            1e-6,
        )
        self.assertAlmostEqual(populated.data.population.sum(), 400, delta=0.01)

    def test_road_constrained_zones_and_reference_comparison(self) -> None:
        """Respect a road barrier and report perfect overlap against itself."""
        roads = gpd.GeoDataFrame(
            geometry=[LineString([(77.02, 27.99), (77.02, 28.05)])],
            crs="EPSG:4326",
        )
        zones = ogm.taz.create_taz(
            self.area,
            population=self.population,
            roads=roads,
            config=ogm.taz.TAZConfig(
                method="road_constrained", target_population=200, min_area_km2=0.01
            ),
        )
        comparison = ogm.taz.compare_taz(zones, zones)

        self.assertGreaterEqual(len(zones), 2)
        self.assertAlmostEqual(
            comparison.global_metrics["intersection_over_union_pct"], 100, places=6
        )
        self.assertEqual(
            comparison.global_metrics["actual_zone_count"], len(zones)
        )
        with tempfile.TemporaryDirectory() as directory:
            report = comparison.write_report(Path(directory) / "comparison.html")
            self.assertTrue(report.is_file())
            self.assertTrue((Path(directory) / "comparison_overlay.html").is_file())
            self.assertTrue((Path(directory) / "taz_matches.geojson").is_file())

    def test_invalid_area_settings_are_rejected(self) -> None:
        """Reject contradictory minimum and maximum area constraints."""
        with self.assertRaises(ValueError):
            ogm.taz.TAZConfig(min_area_km2=2, max_area_km2=1)
        with self.assertRaises(ValueError):
            ogm.taz.AutomaticTAZConfig(building_data_year=2024)

    def test_google_building_downloader_selects_intersecting_tiles_and_caches(self) -> None:
        """Download only boundary-intersecting public tiles and reuse the cache."""
        import json
        from opengeomove.data_loading import google_buildings

        class FakeBlob:
            """Represent one manifest or public raster object in tests."""

            def __init__(self, name, payload=None):
                """Store blob identity, manifest payload, and download count."""
                self.name = name
                self.payload = payload
                self.download_count = 0

            def download_as_bytes(self):
                """Serialize the configured manifest payload."""
                return json.dumps(self.payload).encode("utf-8")

            def download_to_filename(self, filename, timeout):
                """Write a small placeholder tile without network access."""
                self.download_count += 1
                Path(filename).write_bytes(b"fake-cog")

        tile_manifest = {
            "uriPrefix": "gs://open-buildings-temporal-data/v1/2023/",
            "tilesets": [{
                "crs": "EPSG:4326",
                "sources": [
                    {
                        "affineTransform": {
                            "translateX": 77.0,
                            "translateY": 29.0,
                            "scaleX": 1.0,
                            "scaleY": -1.0,
                        },
                        "dimensions": {"width": 1, "height": 1},
                        "uris": ["intersecting.tif"],
                    },
                    {
                        "affineTransform": {
                            "translateX": 0.0,
                            "translateY": 1.0,
                            "scaleX": 1.0,
                            "scaleY": -1.0,
                        },
                        "dimensions": {"width": 1, "height": 1},
                        "uris": ["outside.tif"],
                    },
                ],
            }],
        }
        manifest = FakeBlob("v1/manifests/fake_2023_manifest.json", tile_manifest)

        class FakeBucket:
            """Provide fake GCS raster blobs."""

            def __init__(self):
                """Initialize the in-memory blob map."""
                self.tiles = {}

            def blob(self, name):
                """Return a stable fake object for a GCS object path."""
                return self.tiles.setdefault(name, FakeBlob(name))

        class FakeStorageClient:
            """Provide fake GCS manifest and bucket lookups."""

            def __init__(self):
                """Create a fresh fake bucket for each test."""
                self.fake_bucket = FakeBucket()

            def list_blobs(self, bucket, prefix):
                """Return the test manifest for any requested prefix."""
                self.requested_prefix = prefix
                return [manifest]

            def bucket(self, name):
                """Return the configured fake bucket."""
                return self.fake_bucket

        fake_client = FakeStorageClient()
        boundary = gpd.GeoDataFrame(
            geometry=[self.boundary], crs="EPSG:4326"
        )
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(google_buildings, "_new_storage_client", return_value=fake_client),
                patch.object(google_buildings, "_s2_covering_tokens", return_value=["fake"]),
            ):
                first = google_buildings.download_open_buildings_25d(
                    boundary, directory, year=2023
                )
                second = google_buildings.download_open_buildings_25d(
                    boundary, directory, year=2023
                )

            self.assertEqual(first, second)
            self.assertEqual(len(first), 1)
            self.assertTrue(first[0].is_file())
            self.assertEqual(
                fake_client.fake_bucket.tiles[
                    "v1/2023/intersecting.tif"
                ].download_count,
                1,
            )
            self.assertTrue((first[0].parent / "download_manifest.json").is_file())

    def test_raster_population_reads_only_cells_in_the_boundary(self) -> None:
        """Crop population raster reads to the supplied study-area extent."""
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "population.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=20,
                height=20,
                count=1,
                dtype="float32",
                crs="EPSG:32643",
                transform=from_origin(0, 20, 1, 1),
                nodata=-99,
            ) as output:
                output.write(np.ones((20, 20), dtype="float32"), 1)
            cells = ogm.taz.population_cells(
                path,
                target_crs="EPSG:32643",
                boundary=box(0, 0, 10, 10),
            )

        self.assertEqual(len(cells), 100)
        self.assertEqual(cells.population.sum(), 100)

    def test_google_building_raster_clusters_are_sampled(self) -> None:
        """Read 2.5D building bands and return two weighted centers."""
        import rasterio
        from rasterio.transform import from_origin
        from opengeomove.taz import automatic

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "building-tile.tif"
            data = np.zeros((3, 20, 20), dtype="float32")
            data[0, 4:8, 2:6] = 0.8
            data[1, 4:8, 2:6] = 10
            data[2, 4:8, 2:6] = 0.9
            data[0, 12:16, 14:18] = 0.7
            data[1, 12:16, 14:18] = 30
            data[2, 12:16, 14:18] = 0.95
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=20,
                height=20,
                count=3,
                dtype="float32",
                crs="EPSG:32643",
                transform=from_origin(0, 20, 1, 1),
                nodata=-99,
            ) as output:
                output.write(data)
            clusters = automatic._building_cluster_centers(
                box(0, 0, 20, 20),
                "EPSG:32643",
                ogm.taz.AutomaticTAZConfig(
                    building_raster_path=Path(directory), building_max_samples=1000
                ),
            )

        self.assertIsNotNone(clusters)
        assert clusters is not None
        self.assertEqual(clusters["n_building_cells"], 32)
        self.assertCountEqual(
            [round(clusters["b0"].x), round(clusters["b1"].x)], [4, 16]
        )

    def test_automatic_helpers_use_given_edges_and_merge_small_zones(self) -> None:
        """Keep automatic TAZ helpers independent of external OSM access."""
        from opengeomove.taz import automatic

        roads = gpd.GeoDataFrame(
            {"highway": ["primary"]},
            geometry=[LineString([(77.0, 28.0), (77.04, 28.0)])],
            crs="EPSG:4326",
        )
        with tempfile.TemporaryDirectory() as directory:
            config = ogm.taz.AutomaticTAZConfig(output_dir=Path(directory))
            boundary = gpd.GeoDataFrame(geometry=[self.boundary], crs="EPSG:4326")
            with patch.object(automatic, "_download_osm_edges", return_value=roads):
                loaded = automatic._load_edges(None, boundary, config)
            self.assertEqual(len(loaded), 1)

        zones = gpd.GeoDataFrame(
            geometry=[box(0, 0, 1000, 1000), box(1000, 0, 1100, 1000)],
            crs="EPSG:32643",
        )
        merged, audit = automatic._merge_final_undersized(zones, 0.5, zones.crs)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged.area_km2.iloc[0], 1.1)
        self.assertEqual(len(audit), 1)

    def test_automatic_taz_pipeline_handles_no_recursive_dividers(self) -> None:
        """Create/export a coarse partition when zones already meet size goals."""
        from opengeomove.taz import automatic

        boundary = gpd.GeoDataFrame(
            geometry=[box(77.0, 28.0, 77.001, 28.001)], crs="EPSG:4326"
        )
        roads = gpd.GeoDataFrame(
            {"highway": ["primary"]},
            geometry=[LineString([(76.999, 28.0005), (77.002, 28.0005)])],
            crs="EPSG:4326",
        )

        def no_centerlines(lines, buffer_m, config):
            """Skip medial-axis work while exercising pipeline orchestration."""
            empty = gpd.GeoDataFrame(geometry=[], crs=lines.crs)
            return empty, empty.copy()

        with tempfile.TemporaryDirectory() as directory:
            def fake_google_download(boundary, cache_dir, *, year):
                """Create a placeholder cache entry without network access."""
                path = Path(cache_dir) / str(year) / "tile.tif"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test placeholder")
                return [path]

            with (
                patch.object(automatic, "_centerlines", side_effect=no_centerlines),
                patch(
                    "opengeomove.data_loading.google_buildings.download_open_buildings_25d",
                    side_effect=fake_google_download,
                ),
            ):
                zones = ogm.taz.create_automatic_taz(
                    boundary,
                    roads,
                    config=ogm.taz.AutomaticTAZConfig(
                        output_dir=Path(directory),
                        download_google_buildings=True,
                        fetch_railways_and_waterways=False,
                        save_plots=False,
                        barrier_min_area_km2=0.001,
                        coarse_min_area_km2=0.001,
                        final_min_area_km2=0.001,
                        final_max_area_km2=1.5,
                    ),
                )

            self.assertEqual(len(zones), 1)
            self.assertEqual(
                zones.metadata["building_data_source"],
                "google_open_buildings_2_5d",
            )
            self.assertEqual(zones.metadata["building_data_year"], 2023)
            self.assertTrue(
                (Path(directory) / "17_final_tazs_with_status.geojson").is_file()
            )


if __name__ == "__main__":
    unittest.main()
