"""Implementations of lazy, DuckDB-backed EDA operations."""

from __future__ import annotations

from operator import index
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opengeomove.data_loading.dataset import Dataset


def head(dataset: Dataset, n: int = 5) -> Dataset:
    """Return a new lazy Dataset limited to the first ``n`` rows.

    Args:
        dataset: Dataset to preview.
        n: Non-negative maximum number of rows in the result.

    Returns:
        A new Dataset backed by a DuckDB LIMIT relation.

    Raises:
        TypeError: If ``n`` does not implement the integer index protocol.
        ValueError: If ``n`` is negative.
    """
    if isinstance(n, bool):
        raise TypeError("n must be a non-negative integer")
    try:
        row_limit = index(n)
    except TypeError as error:
        raise TypeError("n must be a non-negative integer") from error
    if row_limit < 0:
        raise ValueError("n must be non-negative")
    return dataset._clone_with_relation(dataset._relation.limit(row_limit))


def isna(dataset: Dataset) -> Dataset:
    """Return a lazy boolean Dataset marking NULL and floating NaN values.

    Empty strings and positive/negative infinity are treated as present. The
    expression is evaluated by DuckDB when a result is fetched, rather than
    constructing a full boolean mask in Python memory.

    Args:
        dataset: Dataset whose values are checked.

    Returns:
        A new Dataset with the same rows and columns and boolean values.
    """
    expressions = []
    for name, data_type in zip(dataset._relation.columns, dataset._relation.types):
        column = _quote_identifier(name)
        missing = f"({column} IS NULL)"
        if _is_floating_type(data_type):
            missing = f"({missing} OR isnan({column}))"
        expressions.append(f"{missing} AS {column}")
    if not expressions:
        relation = dataset._relation
    else:
        relation = dataset._relation.project(", ".join(expressions))
    return dataset._clone_with_relation(relation)


def isnull(dataset: Dataset) -> Dataset:
    """Alias for :func:`isna`."""
    return isna(dataset)


def notna(dataset: Dataset) -> Dataset:
    """Return a lazy boolean Dataset marking present values.

    The returned values are the inverse of :func:`isna`; NULL and floating NaN
    are false, while empty strings and infinity are true.

    Args:
        dataset: Dataset whose values are checked.

    Returns:
        A new Dataset with the same rows and columns and boolean values.
    """
    expressions = []
    for name, data_type in zip(dataset._relation.columns, dataset._relation.types):
        column = _quote_identifier(name)
        missing = f"({column} IS NULL)"
        if _is_floating_type(data_type):
            missing = f"({missing} OR isnan({column}))"
        expressions.append(f"NOT {missing} AS {column}")
    if not expressions:
        relation = dataset._relation
    else:
        relation = dataset._relation.project(", ".join(expressions))
    return dataset._clone_with_relation(relation)


def notnull(dataset: Dataset) -> Dataset:
    """Alias for :func:`notna`."""
    return notna(dataset)


def _quote_identifier(identifier: str) -> str:
    """Quote a DuckDB column identifier, escaping embedded double quotes."""
    return '"' + identifier.replace('"', '""') + '"'


def _is_floating_type(data_type: object) -> bool:
    """Return whether a DuckDB type supports floating-point NaN values."""
    return str(data_type).upper() in {"FLOAT", "REAL", "DOUBLE", "FLOAT4", "FLOAT8"}
