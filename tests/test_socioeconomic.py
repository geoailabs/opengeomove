"""Tests for TAZ socioeconomic feature extraction and classification."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

import opengeomove as ogm


class SocioeconomicTests(unittest.TestCase):
    """Exercise the approved feature and decision-tree public interface."""

    def _sources(self):
        """Build two small zones and their spatial feature sources."""
        zones = gpd.GeoDataFrame(
            {"zone": ["left", "right"], "population": [10.0, 20.0]},
            geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
            crs="EPSG:3857",
        )
        poi = gpd.GeoDataFrame(
            {"primaryType": ["school", "restaurant", "corporate_office"]},
            geometry=[Point(2, 2), Point(4, 4), Point(12, 2)],
            crs=zones.crs,
        )
        roads = gpd.GeoDataFrame(
            {"highway": ["residential", "primary"]},
            geometry=[LineString([(1, 1), (9, 9)]), LineString([(11, 1), (19, 9)])],
            crs=zones.crs,
        )
        footprints = gpd.GeoDataFrame(
            {"confidence": [0.9, 0.4, 0.9]},
            geometry=[box(1, 1, 3, 3), box(4, 4, 8, 8), box(11, 1, 13, 3)],
            crs=zones.crs,
        )
        return zones, poi, roads, footprints

    def test_extracts_features_from_arbitrary_zones_and_windowed_raster(self) -> None:
        """Aggregate POI, road, population, footprint, and height evidence."""
        zones, poi, roads, footprints = self._sources()
        with tempfile.TemporaryDirectory() as directory:
            raster_path = Path(directory) / "buildings.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=2,
                height=1,
                count=3,
                dtype="float32",
                crs=zones.crs,
                transform=from_origin(0, 10, 10, 10),
            ) as dataset:
                dataset.write(np.array([[2.0, 1.0]], dtype="float32"), 1)
                dataset.write(np.array([[5.0, 40.0]], dtype="float32"), 2)
                dataset.write(np.array([[0.8, 0.9]], dtype="float32"), 3)

            features = ogm.taz.extract_socioeconomic_features(
                zones,
                poi=poi,
                roads=roads,
                building_footprints=footprints,
                building_rasters=raster_path,
                id_col="zone",
            )

        self.assertIsInstance(features, ogm.taz.TAZFrame)
        self.assertEqual(features.data.taz_id.tolist(), ["left", "right"])
        self.assertEqual(features.data.poi_total.tolist(), [2, 1])
        self.assertEqual(features.data.poi_education.tolist(), [1, 0])
        self.assertEqual(features.data.poi_food.tolist(), [1, 0])
        self.assertEqual(features.data.poi_offices.tolist(), [0, 1])
        self.assertEqual(features.data.road_residential.tolist(), [1, 0])
        self.assertEqual(features.data.building_count_sum.tolist(), [2.0, 1.0])
        self.assertEqual(features.data.height_weighted_mean.tolist(), [5.0, 40.0])
        self.assertAlmostEqual(features.data.built_area_ratio.iloc[0], 0.04)
        self.assertAlmostEqual(features.data.empty_area_ratio.iloc[0], 0.96)
        self.assertAlmostEqual(features.data.PopDensity.iloc[0], 100_000.0)
        self.assertTrue(
            set(ogm.taz.SOCIOECONOMIC_FEATURE_COLUMNS).issubset(features.data.columns)
        )

    def test_classifier_matches_legacy_id_spelling_and_preserves_tazframe(self) -> None:
        """Fit all five labels and append named predictions to a TAZFrame."""
        rows = 10
        data = {
            column: np.arange(rows, dtype=float) + index
            for index, column in enumerate(ogm.taz.SOCIOECONOMIC_FEATURE_COLUMNS)
        }
        data["TAZ_ID"] = np.arange(1, rows + 1)
        data["taz_id"] = [str(value) for value in data["TAZ_ID"]]
        data["area_km2"] = np.ones(rows)
        features = ogm.taz.TAZFrame(
            gpd.GeoDataFrame(
                data,
                geometry=[box(index * 2, 0, index * 2 + 1, 1) for index in range(rows)],
                crs="EPSG:3857",
            )
        )
        annotations = pd.DataFrame(
            {"taz_id": np.arange(1, rows + 1), "label": np.repeat(np.arange(5), 2)}
        )

        classifier = ogm.taz.fit_socioeconomic_classifier(
            features,
            annotations,
            id_col="TAZ_ID",
            evaluation_folds=2,
        )
        classified = classifier.predict(features)

        self.assertIsInstance(classified, ogm.taz.TAZFrame)
        self.assertEqual(set(classified.data.socioeconomic_label), set(range(5)))
        self.assertEqual(
            classified.data.socioeconomic_class.iloc[0],
            "farmland_or_no_buildings",
        )
        self.assertEqual(classifier.metadata["annotation_rows"], rows)
        self.assertIn("cross_validation", classifier.metadata)
        self.assertFalse(classifier.metadata["probabilities_calibrated"])

    def test_classifier_rejects_leaky_or_unknown_inputs(self) -> None:
        """Reject missing approved features and labels outside the definition."""
        feature_data = {
            column: [0.0, 1.0]
            for column in ogm.taz.SOCIOECONOMIC_FEATURE_COLUMNS
        }
        feature_data["taz_id"] = ["a", "b"]
        features = pd.DataFrame(feature_data)
        with self.assertRaisesRegex(ValueError, "Unknown socioeconomic labels"):
            ogm.taz.fit_socioeconomic_classifier(
                features,
                pd.DataFrame({"taz_id": ["a", "b"], "label": [0, 9]}),
                evaluation_folds=None,
            )
        with self.assertRaisesRegex(ValueError, "Missing socioeconomic feature"):
            ogm.taz.fit_socioeconomic_classifier(
                features.drop(columns="poi_orgs"),
                pd.DataFrame({"taz_id": ["a", "b"], "label": [0, 1]}),
                evaluation_folds=None,
            )


if __name__ == "__main__":
    unittest.main()
