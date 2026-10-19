"""Spatial comparison metrics for generated and reference TAZ layers."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import html
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from opengeomove.taz.exceptions import TAZGeometryError
from opengeomove.taz.models import TAZFrame


@dataclass
class TAZComparison:
    """Best-overlap matches and aggregate metrics for two TAZ layers.

    Attributes:
        matches: One row per actual/reference zone, including its best
            generated match and per-zone overlap/IoU metrics.
        global_metrics: Aggregate coverage, count, area, overlap, and distance
            statistics.
        generated: Generated zones projected to the comparison CRS.
    """

    matches: gpd.GeoDataFrame
    global_metrics: dict[str, Any]
    generated: gpd.GeoDataFrame

    def to_json(self, path: str | Path) -> Path:
        """Write aggregate metrics as JSON to the requested path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.global_metrics, indent=2), encoding="utf-8")
        return destination

    def write_report(self, path: str | Path, *, title: str = "TAZ comparison") -> Path:
        """Write a self-contained HTML summary with static comparison maps.

        Args:
            path: Destination HTML file.
            title: Report title.

        Returns:
            The report file path. Beside it, the method writes reference and
            generated GeoJSON, per-reference-zone comparison GeoJSON, global
            metrics JSON, and a manifest. The report embeds its map images and
            metrics, so it does not fetch remote basemap tiles when opened.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            from opengeomove.taz.exceptions import TAZDependencyError

            raise TAZDependencyError(
                "TAZ report map rendering requires matplotlib"
            ) from exc
        figure, axes = plt.subplots(2, 2, figsize=(14, 12))
        self.matches.plot(ax=axes[0, 0], color="#27a8c7", edgecolor="white", linewidth=0.15)
        axes[0, 0].set_title("Reference zones")
        self.generated.plot(ax=axes[0, 1], color="#f06a4e", edgecolor="white", linewidth=0.15)
        axes[0, 1].set_title("Generated zones")
        self.matches.plot(
            ax=axes[1, 0], column="actual_overlap_pct", legend=True,
            cmap="viridis", edgecolor="white", linewidth=0.15,
        )
        axes[1, 0].set_title("Best-match reference overlap (%)")
        self.matches.plot(
            ax=axes[1, 1], column="iou_pct", legend=True,
            cmap="magma", edgecolor="white", linewidth=0.15,
        )
        axes[1, 1].set_title("Best-match intersection over union (%)")
        for axis in axes.flat:
            axis.set_axis_off()
        figure.tight_layout()
        image_buffer = BytesIO()
        figure.savefig(image_buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(figure)
        encoded_map = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        metrics = pd.DataFrame(
            {"metric": list(self.global_metrics), "value": list(self.global_metrics.values())}
        ).to_html(index=False, escape=True)
        worst = self.matches.sort_values("actual_overlap_pct").head(20)
        columns = [
            "actual_taz_id", "generated_taz_id", "actual_overlap_pct", "iou_pct",
            "centroid_distance_m",
        ]
        table = worst[columns].to_html(index=False, escape=True, float_format=lambda x: f"{x:,.3f}")
        reference_path = destination.with_name("reference_tazs.geojson")
        generated_path = destination.with_name("generated_tazs.geojson")
        matches_path = destination.with_name("taz_matches.geojson")
        metrics_path = destination.with_name("global_metrics.json")
        manifest_path = destination.with_name("comparison_manifest.json")
        self.matches[["actual_taz_id", "geometry"]].to_file(reference_path, driver="GeoJSON")
        self.generated.to_file(generated_path, driver="GeoJSON")
        self.matches.to_file(matches_path, driver="GeoJSON")
        metrics_path.write_text(json.dumps(self.global_metrics, indent=2), encoding="utf-8")
        manifest_path.write_text(
            json.dumps({"report": destination.name, "crs": str(self.matches.crs)}, indent=2),
            encoding="utf-8",
        )
        destination.write_text(
            "<!doctype html><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title><h1>{html.escape(title)}</h1>"
            "<p>Each reference zone is paired with the generated zone having its largest positive-area overlap.</p>"
            f"<img alt='TAZ comparison maps' style='max-width:100%' src='data:image/png;base64,{encoded_map}'>"
            f"<h2>Aggregate metrics</h2>{metrics}"
            f"<h2>Twenty lowest-overlap reference zones</h2>{table}"
            f"<h2>Data files</h2><ul><li>{reference_path.name}</li>"
            f"<li>{generated_path.name}</li><li>{matches_path.name}</li>"
            f"<li>{metrics_path.name}</li><li>{manifest_path.name}</li></ul>",
            encoding="utf-8",
        )
        return destination


def _as_geodataframe(value: TAZFrame | gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    """Copy and minimally validate an input TAZ layer."""
    frame = value.to_geopandas() if isinstance(value, TAZFrame) else value.copy()
    if not isinstance(frame, gpd.GeoDataFrame):
        raise TypeError(f"{name} must be a TAZFrame or GeoDataFrame")
    if frame.empty or frame.crs is None:
        raise TAZGeometryError(f"{name} must be non-empty and have a CRS")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any():
        raise TAZGeometryError(f"{name} contains null or empty geometry")
    if (~frame.geometry.is_valid).any():
        raise TAZGeometryError(f"{name} contains invalid geometry")
    if not set(frame.geometry.geom_type).issubset({"Polygon", "MultiPolygon"}):
        raise TAZGeometryError(f"{name} must contain polygon geometry only")
    return frame


def compare_taz(
    actual: TAZFrame | gpd.GeoDataFrame,
    generated: TAZFrame | gpd.GeoDataFrame,
    *,
    actual_id_col: str | None = None,
    generated_id_col: str = "taz_id",
) -> TAZComparison:
    """Compare generated zones with reference polygons by best positive overlap.

    For every reference zone, the generated zone with the largest
    intersection area is selected. Metrics are calculated in an estimated
    local UTM CRS; the comparison does not mutate either input layer.

    Args:
        actual: Reference/ground-truth polygons.
        generated: Generated TAZ polygons.
        actual_id_col: Reference identifier field, or ``None`` to use
            ``taz_id`` when present and otherwise the row index.
        generated_id_col: Generated identifier field, default ``taz_id``.

    Returns:
        A TAZComparison with per-reference-zone metrics and global metrics.

    Raises:
        TypeError: If either input is not spatial tabular data.
        ValueError: If the requested identifier field is absent.
        TAZGeometryError: If either layer is empty or lacks a CRS.
    """
    actual_frame = _as_geodataframe(actual, "actual")
    generated_frame = _as_geodataframe(generated, "generated")
    if actual_id_col is None:
        actual_id_col = "taz_id" if "taz_id" in actual_frame else None
    if actual_id_col is not None and actual_id_col not in actual_frame:
        raise ValueError(f"Reference identifier column {actual_id_col!r} does not exist")
    if generated_id_col not in generated_frame:
        raise ValueError(f"Generated identifier column {generated_id_col!r} does not exist")
    metric_crs = actual_frame.estimate_utm_crs()
    if metric_crs is None:
        raise TAZGeometryError("Could not estimate a metric CRS for TAZ comparison")
    actual_metric = actual_frame.to_crs(metric_crs).reset_index(drop=True)
    generated_metric = generated_frame.to_crs(metric_crs).reset_index(drop=True)
    actual_ids = (
        actual_metric[actual_id_col].astype(str).tolist()
        if actual_id_col
        else [str(index) for index in actual_metric.index]
    )
    generated_ids = generated_metric[generated_id_col].astype(str).tolist()
    spatial_index = generated_metric.sindex
    rows: list[dict[str, Any]] = []
    for index, geometry in enumerate(actual_metric.geometry):
        candidates = spatial_index.query(geometry, predicate="intersects")
        overlaps = []
        for candidate_index in candidates:
            other = generated_metric.geometry.iloc[int(candidate_index)]
            overlap_area = geometry.intersection(other).area
            if overlap_area > 0:
                overlaps.append((overlap_area, int(candidate_index), other))
        if not overlaps:
            rows.append({
                "actual_taz_id": actual_ids[index],
                "generated_taz_id": None,
                "actual_area_km2": geometry.area / 1_000_000,
                "generated_area_km2": 0.0,
                "overlap_area_km2": 0.0,
                "actual_overlap_pct": 0.0,
                "generated_overlap_pct": 0.0,
                "iou_pct": 0.0,
                "centroid_distance_m": None,
                "geometry": geometry,
            })
            continue
        overlap_area, best_index, best_geometry = max(overlaps, key=lambda item: item[0])
        union_area = geometry.union(best_geometry).area
        rows.append({
            "actual_taz_id": actual_ids[index],
            "generated_taz_id": generated_ids[best_index],
            "actual_area_km2": geometry.area / 1_000_000,
            "generated_area_km2": best_geometry.area / 1_000_000,
            "overlap_area_km2": overlap_area / 1_000_000,
            "actual_overlap_pct": 100 * overlap_area / geometry.area,
            "generated_overlap_pct": 100 * overlap_area / best_geometry.area,
            "iou_pct": 100 * overlap_area / union_area if union_area else 0.0,
            "centroid_distance_m": geometry.centroid.distance(best_geometry.centroid),
            "geometry": geometry,
        })
    matches = gpd.GeoDataFrame(rows, geometry="geometry", crs=metric_crs)
    actual_union = actual_metric.geometry.union_all()
    generated_union = generated_metric.geometry.union_all()
    intersection_area = actual_union.intersection(generated_union).area
    union_area = actual_union.union(generated_union).area
    distances = matches.centroid_distance_m.dropna()
    metrics = {
        "actual_zone_count": int(len(actual_metric)),
        "generated_zone_count": int(len(generated_metric)),
        "actual_total_area_km2": float(actual_union.area / 1_000_000),
        "generated_total_area_km2": float(generated_union.area / 1_000_000),
        "intersection_area_km2": float(intersection_area / 1_000_000),
        "union_area_km2": float(union_area / 1_000_000),
        "intersection_over_union_pct": float(100 * intersection_area / union_area) if union_area else 0.0,
        "mean_actual_zone_overlap_pct": float(matches.actual_overlap_pct.mean()),
        "median_actual_zone_overlap_pct": float(matches.actual_overlap_pct.median()),
        "zones_at_least_50pct_overlap": int((matches.actual_overlap_pct >= 50).sum()),
        "zones_at_least_75pct_overlap": int((matches.actual_overlap_pct >= 75).sum()),
        "mean_best_match_iou_pct": float(matches.iou_pct.mean()),
        "mean_centroid_distance_m": float(distances.mean()) if not distances.empty else None,
    }
    return TAZComparison(matches, metrics, generated_metric)
