"""Public file-opening function and DuckDB reader dispatch."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import duckdb

from opengeomove.data_loading.dataset import Dataset

PathLike = Union[str, os.PathLike[str]]
_SUPPORTED_FORMATS = {".parquet", ".csv", ".tsv", ".json"}


def open(path: PathLike, *, eager_metadata: bool = False) -> Dataset:
    """Open a supported local file as a lazy, read-only Dataset.

    The format is selected from the file extension. The source is scanned by
    DuckDB and is not copied into Python memory. Metadata is resolved lazily by
    default and cached on the returned Dataset.

    Args:
        path: Path to a Parquet, CSV, TSV, or JSON file.
        eager_metadata: If true, compute and cache the dataset shape before
            returning. This can require a full scan for CSV, TSV, or JSON.

    Returns:
        A DuckDB-backed Dataset for the source file.

    Raises:
        TypeError: If ``path`` is not a string or path-like value, or if it
            resolves to a bytes path. Also raised if ``eager_metadata`` is not
            a boolean.
        FileNotFoundError: If the path does not exist or is not a file.
        ValueError: If the extension is not supported.
        duckdb.Error: If DuckDB cannot read the source file.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a string or path-like value")
    if not isinstance(eager_metadata, bool):
        raise TypeError("eager_metadata must be a boolean")

    raw_path = os.fspath(path)
    if not isinstance(raw_path, str):
        raise TypeError("bytes paths are not supported")

    source_path = Path(raw_path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"No such data file: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix not in _SUPPORTED_FORMATS:
        supported = ", ".join(sorted(_SUPPORTED_FORMATS))
        raise ValueError(
            f"Unsupported file extension {suffix!r}; supported extensions: {supported}"
        )

    resolved_path = source_path.resolve()
    connection = duckdb.connect(database=":memory:")
    try:
        # Make DuckDB's implicit timestamp casts deterministic across machines.
        connection.execute("SET TimeZone = 'UTC'")
        if suffix == ".parquet":
            relation = connection.read_parquet(str(resolved_path))
            data_format = "parquet"
        elif suffix == ".csv":
            relation = connection.read_csv(str(resolved_path), delim=",")
            data_format = "csv"
        elif suffix == ".tsv":
            relation = connection.read_csv(str(resolved_path), delim="\t")
            data_format = "tsv"
        else:
            relation = connection.read_json(str(resolved_path))
            data_format = "json"
    except Exception:
        connection.close()
        raise

    return Dataset(
        relation=relation,
        connection=connection,
        source_path=resolved_path,
        data_format=data_format,
        eager_metadata=eager_metadata,
    )
