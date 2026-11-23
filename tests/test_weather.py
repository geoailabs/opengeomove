"""Tests for historical weather acquisition and TAZ overlays."""

from __future__ import annotations

import json
from datetime import date
from io import BytesIO
from unittest.mock import patch
import unittest
from urllib.parse import parse_qs, urlparse

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box

import opengeomove as ogm


class _FakeWeatherSource:
    """Deterministic provider used to test aggregation without networking."""

    name = "fake_weather"

    def __init__(self) -> None:
        self.calls = []

    def fetch(self, points, *, date, variables, timezone):
        """Return one deterministic row per requested point."""
        self.calls.append(
            {
                "points": points.copy(),
                "date": date,
                "variables": tuple(variables),
                "timezone": timezone,
            }
        )
        data = {
            "_weather_point_id": points["_weather_point_id"].tolist(),
            "source_latitude": points.geometry.y.to_numpy(),
            "source_longitude": points.geometry.x.to_numpy(),
        }
        for variable_index, variable in enumerate(variables):
            data[variable] = np.arange(len(points), dtype=float) + variable_index + 0.5
        field = gpd.GeoDataFrame(data, geometry=points.geometry, crs=points.crs)
        return ogm.weather.WeatherField(
            field,
            tuple(variables),
            {"provider": self.name, "test": True},
        )


class _Response:
    """Small context-managed response compatible with urllib.urlopen."""

    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


class WeatherTests(unittest.TestCase):
    """Exercise weather source normalization, overlays, and plotting."""

    def setUp(self) -> None:
        """Create two adjacent TAZs in a metric CRS."""
        self.zones = ogm.taz.TAZFrame(
            gpd.GeoDataFrame(
                {"taz_id": ["west", "east"], "area_km2": [1.0, 1.0]},
                geometry=[box(0, 0, 1_000, 1_000), box(1_000, 0, 2_000, 1_000)],
                crs="EPSG:3857",
            )
        )

    def test_overlay_is_read_only_and_uses_local_date(self) -> None:
        """Attach daily values without modifying the input TAZ frame."""
        source = _FakeWeatherSource()
        original_columns = tuple(self.zones.data.columns)
        overlay = ogm.weather.overlay(
            self.zones,
            date="2024-07-15T00:30:00+05:30",
            variable="precipitation",
            source=source,
            timezone="Asia/Kolkata",
        )

        self.assertIsInstance(overlay, ogm.weather.WeatherOverlay)
        self.assertEqual(overlay.data.precipitation_sum.tolist(), [0.5, 1.5])
        self.assertEqual(tuple(self.zones.data.columns), original_columns)
        self.assertNotIn("precipitation_sum", self.zones.data)
        self.assertEqual(overlay.metadata["weather_date"], "2024-07-15")
        self.assertEqual(overlay.metadata["timezone"], "Asia/Kolkata")
        self.assertEqual(overlay.metadata["spatial_method"], "representative_point")
        self.assertEqual(source.calls[0]["date"], date(2024, 7, 15))
        self.assertEqual(source.calls[0]["timezone"], "Asia/Kolkata")
        self.assertEqual(len(source.calls[0]["points"]), 2)

    def test_overlay_accepts_arbitrary_zone_id_column(self) -> None:
        """Normalize existing polygon layers with a caller-selected ID field."""
        source = _FakeWeatherSource()
        input_frame = gpd.GeoDataFrame(
            {"zone_code": [101, 102], "label": ["a", "b"]},
            geometry=self.zones.data.geometry,
            crs=self.zones.data.crs,
        )
        overlay = ogm.weather.overlay(
            input_frame,
            date=date(2024, 7, 15),
            source=source,
            id_column="zone_code",
        )

        self.assertEqual(overlay.data.taz_id.tolist(), ["101", "102"])
        self.assertEqual(overlay.data.area_km2.tolist(), [1.0, 1.0])
        self.assertEqual(overlay.data.precipitation_sum.tolist(), [0.5, 1.5])

    def test_multiple_variables_and_aliases_are_attached(self) -> None:
        """Request all supported daily fields through canonical names and aliases."""
        source = _FakeWeatherSource()
        requested = (
            "precipitation",
            "rain",
            "snowfall",
            "precipitation_hours",
            "temperature",
            "temperature_max",
            "temperature_min",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "wind_speed_max",
            "wind_gusts_max",
            "wind_direction",
            "radiation",
            "evapotranspiration",
            "weather",
        )
        overlay = ogm.weather.overlay(
            self.zones,
            date="2024-07-15",
            variables=requested,
            source=source,
            timezone="Asia/Kolkata",
        )

        expected = ogm.weather.available_variables()
        self.assertEqual(overlay.variables, expected)
        self.assertTrue(set(expected).issubset(overlay.data.columns))
        self.assertEqual(source.calls[0]["variables"], expected)
        self.assertEqual(overlay.data["weather_code"].tolist(), [14.5, 15.5])

    def test_plot_has_quantitative_colorbar_and_hover_data(self) -> None:
        """Render a weather variable with a colorbar and zone values."""
        overlay = ogm.weather.overlay(
            self.zones,
            date="2024-07-15",
            variables=("precipitation", "temperature"),
            source=_FakeWeatherSource(),
            timezone="Asia/Kolkata",
        )
        figure = overlay.plot(variable="temperature")

        trace = figure.data[0]
        self.assertEqual(trace.name, "Mean temperature")
        self.assertTrue(trace.showscale)
        self.assertEqual(trace.colorbar.title.text, "Mean temperature (°C)")
        self.assertEqual(list(trace.z), [1.5, 2.5])
        self.assertIn("temperature_2m_mean", trace.hovertemplate)
        self.assertIn("2024-07-15", figure.layout.title.text)

    def test_open_meteo_source_parses_multiple_locations_and_units(self) -> None:
        """Decode a multi-coordinate archive response into a WeatherField."""
        points = gpd.GeoDataFrame(
            {"_weather_point_id": [0, 1]},
            geometry=[Point(77.0, 28.4), Point(77.1, 28.5)],
            crs="EPSG:4326",
        )
        payload = [
            {
                "latitude": 28.4,
                "longitude": 77.0,
                "elevation": 210,
                "daily_units": {
                    "time": "iso8601",
                    "precipitation_sum": "mm",
                    "temperature_2m_max": "°C",
                },
                "daily": {
                    "time": ["2024-07-15"],
                    "precipitation_sum": [1.25],
                    "temperature_2m_max": [34.0],
                },
            },
            {
                "latitude": 28.5,
                "longitude": 77.1,
                "elevation": 211,
                "daily_units": {
                    "time": "iso8601",
                    "precipitation_sum": "mm",
                    "temperature_2m_max": "°C",
                },
                "daily": {
                    "time": ["2024-07-15"],
                    "precipitation_sum": [2.5],
                    "temperature_2m_max": [35.0],
                },
            },
        ]
        source = ogm.weather.OpenMeteoHistoricalSource()
        with patch(
            "opengeomove.data_loading.weather.urlopen",
            return_value=_Response(payload),
        ) as fetch:
            field = source.fetch(
                points,
                date=date(2024, 7, 15),
                variables=("precipitation_sum", "temperature_2m_max"),
                timezone="Asia/Kolkata",
            )

        self.assertEqual(field.data.precipitation_sum.tolist(), [1.25, 2.5])
        self.assertEqual(field.data.temperature_2m_max.tolist(), [34.0, 35.0])
        self.assertEqual(field.metadata["request_count"], 1)
        query = parse_qs(urlparse(fetch.call_args.args[0].full_url).query)
        self.assertEqual(query["daily"], ["precipitation_sum,temperature_2m_max"])
        self.assertEqual(query["timezone"], ["Asia/Kolkata"])
        self.assertEqual(query["wind_speed_unit"], ["kmh"])

    def test_invalid_requests_are_rejected(self) -> None:
        """Reject unsupported variables, duplicate variables, and methods."""
        source = _FakeWeatherSource()
        with self.assertRaises(ValueError):
            ogm.weather.overlay(
                self.zones,
                date="2024-07-15",
                variable="not-a-weather-variable",
                source=source,
            )
        with self.assertRaises(ValueError):
            ogm.weather.overlay(
                self.zones,
                date="2024-07-15",
                variables=("rain", "rain_sum"),
                source=source,
            )
        with self.assertRaises(ValueError):
            ogm.weather.overlay(
                self.zones,
                date="2024-07-15",
                source=source,
                spatial_method="area_weighted",
            )


if __name__ == "__main__":
    unittest.main()
