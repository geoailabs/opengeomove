"""Download boundary-matched Open Buildings 2.5D GeoTIFF tiles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any
from urllib.parse import urlparse

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform


_BUCKET = "open-buildings-temporal-data"
_DATASET_VERSION = "v1"
_MANIFESTS_FOLDER = "manifests"
_MANIFEST_S2_LEVEL = 2
_MAX_DOWNLOAD_WORKERS = 4
_SUPPORTED_YEARS = range(2016, 2024)


def _new_storage_client():
    """Create an anonymous client for Google's public Cloud Storage bucket."""
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError(
            "Google 2.5D downloads require google-auth and google-cloud-storage"
        ) from exc
    return storage.Client.create_anonymous_client()


def _boundary_geometry(boundary: gpd.GeoDataFrame | Any) -> Any:
    """Normalize a polygon boundary to a single valid WGS84 geometry."""
    if isinstance(boundary, gpd.GeoDataFrame):
        if boundary.empty or boundary.crs is None:
            raise ValueError("boundary GeoDataFrame must be non-empty and have a CRS")
        geometry = boundary.geometry.union_all()
        source_crs = boundary.crs
    else:
        geometry = boundary
        source_crs = "EPSG:4326"
    if geometry is None or geometry.is_empty:
        raise ValueError("boundary geometry must be non-empty")
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("boundary must be a Polygon or MultiPolygon")
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("boundary geometry is invalid and could not be repaired")
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    return transform(transformer.transform, geometry)


def _s2_covering_tokens(geometry) -> list[str]:
    """Return Google's level-2 S2 manifest tokens covering the boundary bounds."""
    try:
        import s2sphere as s2
    except ImportError as exc:
        raise ImportError(
            "Google 2.5D downloads require s2sphere"
        ) from exc
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    rectangle = s2.LatLngRect.from_point_pair(
        s2.LatLng.from_degrees(min_lat, min_lon),
        s2.LatLng.from_degrees(max_lat, max_lon),
    )
    coverer = s2.RegionCoverer()
    coverer.min_level = _MANIFEST_S2_LEVEL
    coverer.max_level = _MANIFEST_S2_LEVEL
    coverer.level_mod = 1
    coverer.max_cells = 1_000_000
    return [cell.to_token() for cell in coverer.get_covering(rectangle)]


def _year_manifest_blobs(storage_client, tokens: list[str], year: int) -> list[Any]:
    """Find and de-duplicate manifest objects for the requested S2 cells/year."""
    manifests: dict[str, Any] = {}
    for token in tokens:
        prefix = f"{_DATASET_VERSION}/{_MANIFESTS_FOLDER}/{token}_"
        for blob in storage_client.list_blobs(_BUCKET, prefix=prefix):
            if re.search(rf"_{year}_", blob.name):
                manifests[blob.name] = blob
    return [manifests[name] for name in sorted(manifests)]


def _manifest_tiles(blob: Any, boundary_wgs84) -> list[str]:
    """Return Cloud Storage raster URIs intersecting the WGS84 boundary."""
    manifest = json.loads(blob.download_as_bytes())
    uris: list[str] = []
    manifest_crs = None
    candidates: list[tuple[str, Polygon]] = []
    for tileset in manifest.get("tilesets", []):
        manifest_crs = manifest_crs or tileset.get("crs")
        for source in tileset.get("sources", []):
            affine = source["affineTransform"]
            width = source["dimensions"]["width"]
            height = source["dimensions"]["height"]
            x0, y0 = affine["translateX"], affine["translateY"]
            sx, sy = affine["scaleX"], affine["scaleY"]
            corners = [
                (x0, y0),
                (x0 + sx * width, y0),
                (x0 + sx * width, y0 + sy * height),
                (x0, y0 + sy * height),
            ]
            object_uri = manifest["uriPrefix"] + source["uris"][0]
            candidates.append((object_uri, Polygon(corners)))
    if manifest_crs is None:
        return []
    transformer = Transformer.from_crs(
        "EPSG:4326", manifest_crs, always_xy=True
    )
    projected_boundary = transform(transformer.transform, boundary_wgs84)
    for object_uri, footprint in candidates:
        if footprint.intersects(projected_boundary):
            uris.append(object_uri)
    return uris


def _selected_tile_uris(storage_client, boundary_wgs84, year: int) -> list[str]:
    """Query year manifests and return unique raster URIs intersecting AOI."""
    tokens = _s2_covering_tokens(boundary_wgs84)
    manifests = _year_manifest_blobs(storage_client, tokens, year)
    uris: set[str] = set()
    for manifest_blob in manifests:
        uris.update(_manifest_tiles(manifest_blob, boundary_wgs84))
    return sorted(uris)


def _local_tile_path(cache_dir: Path, uri: str) -> Path:
    """Build a collision-resistant cache filename from a public GCS URI."""
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Unexpected Google Cloud Storage URI: {uri!r}")
    object_name = parsed.path.lstrip("/")
    original_name = PurePosixPath(object_name).name
    suffix = Path(original_name).suffix or ".tif"
    stem = Path(original_name).stem or "building_tile"
    digest = sha256(uri.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{stem}_{digest}{suffix}"


def _download_tile(storage_client, uri: str, destination: Path) -> Path:
    """Download one public COG atomically, reusing a non-empty cache file."""
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    parsed = urlparse(uri)
    bucket = storage_client.bucket(parsed.netloc)
    blob = bucket.blob(parsed.path.lstrip("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        blob.download_to_filename(str(temporary), timeout=300)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise OSError(f"Google returned an empty raster for {uri}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def download_open_buildings_25d(
    boundary: gpd.GeoDataFrame | Any,
    cache_dir: str | Path,
    *,
    year: int = 2023,
    max_workers: int = _MAX_DOWNLOAD_WORKERS,
) -> list[Path]:
    """Download Google Open Buildings 2.5D COG tiles intersecting a boundary.

    The public Google Cloud Storage manifests are queried anonymously. Only
    raster tiles whose footprints intersect the supplied polygon are
    downloaded. Existing non-empty files in ``cache_dir`` are reused. This
    stores complete Cloud Optimized GeoTIFF tiles; it does not clip or alter
    the source rasters.

    Args:
        boundary: Polygon or MultiPolygon in WGS84, or a GeoDataFrame in any
            CRS. Multiple GeoDataFrame features are dissolved before lookup.
        cache_dir: Directory for downloaded COGs and cache metadata.
        year: Dataset year, from 2016 through 2023 inclusive.
        max_workers: Maximum parallel downloads, between 1 and 8.

    Returns:
        Local paths to all intersecting GeoTIFF tiles, in stable URI order.

    Raises:
        ImportError: If Google Cloud Storage or S2 geometry dependencies are
            unavailable.
        ValueError: If the year, worker count, or boundary is invalid.
        FileNotFoundError: If Google publishes no matching tiles for this
            boundary and year.
        Exception: Propagates storage/network errors with any partial file
            removed; complete cached tiles are preserved.
    """
    if (
        isinstance(year, bool)
        or not isinstance(year, int)
        or year not in _SUPPORTED_YEARS
    ):
        raise ValueError("year must be an integer from 2016 through 2023")
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or not 1 <= max_workers <= 8
    ):
        raise ValueError("max_workers must be an integer from 1 through 8")
    boundary_wgs84 = _boundary_geometry(boundary)
    storage_client = _new_storage_client()
    uris = _selected_tile_uris(storage_client, boundary_wgs84, year)
    if not uris:
        raise FileNotFoundError(
            f"Google Open Buildings 2.5D has no raster tiles intersecting the "
            f"boundary for {year}; check dataset coverage and boundary CRS"
        )
    cache = Path(cache_dir).expanduser() / str(year)
    cache.mkdir(parents=True, exist_ok=True)
    destinations = [(uri, _local_tile_path(cache, uri)) for uri in uris]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(destinations))) as executor:
        paths = list(executor.map(
            lambda item: _download_tile(storage_client, item[0], item[1]),
            destinations,
        ))
    metadata = {
        "dataset": "Google Open Buildings 2.5D Temporal",
        "dataset_version": _DATASET_VERSION,
        "year": year,
        "boundary_crs": "EPSG:4326",
        "tile_uris": uris,
        "license_options": ["CC-BY-4.0", "ODbL-1.0"],
        "source_url": "https://sites.research.google/gr/open-buildings/temporal",
    }
    (cache / "download_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    logging.info(
        "Resolved %d Google Open Buildings 2.5D tile(s) for %s",
        len(paths), year,
    )
    return paths
