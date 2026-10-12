"""Tests for bounded GPS plotting and optional tile handling."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from urllib.error import URLError
from unittest.mock import patch

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

import opengeomove as ogm


class GPSPlotTests(unittest.TestCase):
    """Exercise GPS plotting with a small Parquet source and no live network."""

    def setUp(self) -> None:
        """Create a local coordinate dataset for plotting tests."""
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary_directory.name) / "points.parquet"
        connection = duckdb.connect()
        connection.execute(
            """COPY (SELECT * FROM (VALUES
                (28.6139::DOUBLE, 77.2090::DOUBLE),
                (28.6200::DOUBLE, 77.2150::DOUBLE),
                (28.6250::DOUBLE, 77.2200::DOUBLE),
                (NULL::DOUBLE, 77.2300::DOUBLE),
                (91.0::DOUBLE, 0.0::DOUBLE)
            ) AS points(lat, lon)) TO ? (FORMAT PARQUET)""",
            [str(self.path)],
        )
        connection.close()
        self.dataset = ogm.open(self.path)
        ogm.set_basemap(None)

    def tearDown(self) -> None:
        """Close figures, DuckDB, and the temporary source file."""
        plt.close("all")
        ogm.set_basemap(None)
        self.dataset._connection.close()
        self._temporary_directory.cleanup()

    def _tile_png(self) -> bytes:
        """Return a tiny valid PNG response for a mocked tile request."""
        image_buffer = BytesIO()
        Image.new("RGB", (256, 256), color=(230, 230, 230)).save(image_buffer, format="PNG")
        return image_buffer.getvalue()

    def test_show_materializes_no_more_than_the_requested_sample(self) -> None:
        """Use the required sampling limit and ignore invalid coordinates."""
        plot = ogm.GPSPlot(
            self.dataset,
            latitude_col="lat",
            longitude_col="lon",
            sampling=2,
        )

        with patch("matplotlib.pyplot.show"):
            axes = plot.show()

        self.assertEqual(len(axes.collections[0].get_offsets()), 2)

    def test_global_basemap_draws_tiles_and_attribution(self) -> None:
        """Fetch configured tiles only during show and display attribution."""
        ogm.set_basemap(
            "https://tiles.example/{z}/{x}/{y}.png",
            attribution="Example tile provider",
        )
        plot = ogm.GPSPlot(
            self.dataset,
            latitude_col="lat",
            longitude_col="lon",
            sampling=2,
            zoom=1,
        )

        with patch("opengeomove.plots.gps.urlopen", return_value=BytesIO(self._tile_png())) as fetch_tile:
            with patch("matplotlib.pyplot.show"):
                axes = plot.show()

        self.assertGreater(fetch_tile.call_count, 0)
        self.assertIn("Example tile provider", [text.get_text() for text in axes.texts])

    def test_tile_request_failure_raises_clear_error(self) -> None:
        """Surface failed tile requests instead of silently hiding the map."""
        plot = ogm.GPSPlot(
            self.dataset,
            latitude_col="lat",
            longitude_col="lon",
            sampling=2,
            basemap="https://tiles.example/{z}/{x}/{y}.png",
            zoom=1,
        )

        with patch("opengeomove.plots.gps.urlopen", side_effect=URLError("offline")):
            with self.assertRaisesRegex(RuntimeError, "Could not fetch basemap tile"):
                plot.show()

    def test_sampling_and_columns_are_validated(self) -> None:
        """Reject missing coordinate columns and invalid sample limits."""
        with self.assertRaises(ValueError):
            ogm.GPSPlot(
                self.dataset,
                latitude_col="missing",
                longitude_col="lon",
                sampling=2,
            )
        with self.assertRaises(ValueError):
            ogm.GPSPlot(
                self.dataset,
                latitude_col="lat",
                longitude_col="lon",
                sampling=0,
            )


if __name__ == "__main__":
    unittest.main()
