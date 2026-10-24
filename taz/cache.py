"""Content-aware caches for expensive automatic TAZ stages."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import geopandas as gpd


_CACHE_SCHEMA_VERSION = 1


def stable_cache_key(payload: Any) -> str:
    """Return a stable short SHA-256 key for JSON-compatible input metadata.

    Args:
        payload: Nested JSON-compatible cache-key content.

    Returns:
        A 20-character hexadecimal cache key.
    """
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return sha256(encoded).hexdigest()[:20]


def path_signature(path: str | Path | None) -> dict[str, Any]:
    """Describe file names, sizes, and modification times without reading data.

    Args:
        path: File or directory to describe, or ``None``.

    Returns:
        JSON-compatible metadata suitable for stage-cache invalidation.
    """
    if path is None:
        return {"path": None, "exists": False}
    source = Path(path).expanduser()
    if not source.exists():
        return {"path": str(source.resolve()), "exists": False}
    if source.is_file():
        stat = source.stat()
        return {
            "path": str(source.resolve()),
            "exists": True,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    files = []
    for item in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
        stat = item.stat()
        files.append({
            "name": str(item.relative_to(source)),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return {
        "path": str(source.resolve()),
        "exists": True,
        "files": files,
    }


@dataclass(frozen=True)
class TAZStageCache:
    """Read and write keyed GeoParquet/JSON stage artifacts.

    Args:
        root: Root directory for internal automatic-TAZ cache artifacts.
        mode: ``"reuse"`` reads and writes, ``"refresh"`` only writes, and
            ``"off"`` disables both operations.
    """

    root: Path
    mode: str = "reuse"

    @property
    def can_read(self) -> bool:
        """Return whether valid existing artifacts may be reused."""
        return self.mode == "reuse"

    @property
    def can_write(self) -> bool:
        """Return whether newly computed artifacts should be cached."""
        return self.mode in {"reuse", "refresh"}

    def _path(self, stage: str, key: str, suffix: str) -> Path:
        """Resolve one stage artifact without creating it."""
        return self.root / stage / f"{key}{suffix}"

    def load_frame(self, stage: str, key: str) -> gpd.GeoDataFrame | None:
        """Load a matching GeoParquet stage artifact when reuse is enabled."""
        path = self._path(stage, key, ".parquet")
        if not self.can_read or not path.is_file():
            return None
        return gpd.read_parquet(path)

    def save_frame(self, stage: str, key: str, frame: gpd.GeoDataFrame) -> Path | None:
        """Atomically persist a GeoDataFrame stage artifact when enabled."""
        if not self.can_write:
            return None
        path = self._path(stage, key, ".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        try:
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def load_json(self, stage: str, key: str) -> dict[str, Any] | None:
        """Load a matching JSON artifact when reuse is enabled."""
        path = self._path(stage, key, ".json")
        if not self.can_read or not path.is_file():
            return None
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if content.get("cache_schema_version") != _CACHE_SCHEMA_VERSION:
            return None
        return content

    def save_json(self, stage: str, key: str, content: dict[str, Any]) -> Path | None:
        """Atomically persist JSON stage metadata when caching is enabled."""
        if not self.can_write:
            return None
        path = self._path(stage, key, ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        payload = {"cache_schema_version": _CACHE_SCHEMA_VERSION, **content}
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path
