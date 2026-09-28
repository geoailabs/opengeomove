"""Implementations of lazy, DuckDB-backed EDA operations."""

from __future__ import annotations

from operator import index
from typing import TYPE_CHECKING, Sequence

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


def info(dataset: Dataset, *, deep: bool = False) -> None:
    """Print schema and source information without scanning rows by default.

    Args:
        dataset: Dataset to describe.
        deep: If true, compute the exact row count. This may scan the complete
            source when the count is not already cached.

    Returns:
        None. The report is written to standard output.
    """
    if not isinstance(deep, bool):
        raise TypeError("deep must be a bool")
    print(f"Dataset: {dataset.source}")
    print(f"Format: {dataset.format}")
    if deep:
        row_count = dataset.shape[0]
        print(f"Rows: {row_count:,}")
    elif "shape" in dataset._metadata_cache:
        print(f"Rows: {dataset._metadata_cache['shape'][0]:,} (cached)")
    else:
        print("Rows: not computed (use deep=True to count)")
    print(f"Columns: {len(dataset.columns)}")
    print("Name | Type")
    for name, data_type in dataset.dtypes.items():
        print(f"{name} | {data_type}")


def missing_summary(dataset: Dataset) -> Dataset:
    """Return a lazy per-column NULL/NaN count and percentage summary.

    The aggregates are expressed in a single DuckDB query. Fetching the
    returned Dataset scans the source; the summary itself has one row per
    column and does not materialize source rows in Python.

    Args:
        dataset: Dataset to summarize.

    Returns:
        A Dataset with column name, DuckDB type, row, NULL, NaN, and missing
        counts, plus the percentage of rows that are missing.
    """
    columns = list(zip(dataset._relation.columns, dataset._relation.types))
    if not columns:
        relation = dataset._connection.sql(
            "SELECT CAST(NULL AS VARCHAR) AS column_name, "
            "CAST(NULL AS VARCHAR) AS dtype, CAST(0 AS BIGINT) AS row_count, "
            "CAST(0 AS BIGINT) AS null_count, CAST(0 AS BIGINT) AS nan_count, "
            "CAST(0 AS BIGINT) AS missing_count, "
            "CAST(NULL AS DOUBLE) AS missing_percent WHERE FALSE"
        )
        return dataset._clone_with_relation(relation)

    structs = []
    for name, data_type in columns:
        column = _quote_identifier(name)
        is_float = _is_floating_type(data_type)
        nan_count = (
            f"count(*) FILTER (WHERE {column} IS NOT NULL AND isnan({column}))"
            if is_float
            else "CAST(0 AS BIGINT)"
        )
        missing_condition = (
            f"{column} IS NULL OR isnan({column})" if is_float else f"{column} IS NULL"
        )
        structs.append(
            "struct_pack("
            f"column_name := {_quote_string(name)}, "
            f"dtype := {_quote_string(str(data_type))}, "
            "row_count := count(*), "
            f"null_count := count(*) FILTER (WHERE {column} IS NULL), "
            f"nan_count := {nan_count}, "
            f"missing_count := count(*) FILTER (WHERE {missing_condition}), "
            "missing_percent := 100.0 * count(*) FILTER "
            f"(WHERE {missing_condition}) / NULLIF(count(*), 0))"
        )
    query = "SELECT unnest([" + ", ".join(structs) + "]) AS summary FROM __ogm_missing_source"
    relation = dataset._relation.query("__ogm_missing_source", query).project(
        "summary.column_name, summary.dtype, summary.row_count, "
        "summary.null_count, summary.nan_count, summary.missing_count, "
        "summary.missing_percent"
    )
    return dataset._clone_with_relation(relation)


def describe(dataset: Dataset) -> Dataset:
    """Return DuckDB's type-aware summary statistics as a lazy Dataset.

    The summary includes approximate quantiles and approximate distinct counts
    where applicable. Floating-point NaN values are treated as missing before
    statistics are computed, matching this library's missing-value methods.
    Fetching the summary may scan the complete source.

    Args:
        dataset: Dataset to summarize.

    Returns:
        A Dataset containing one summary row per source column.
    """
    expressions = []
    for name, data_type in zip(dataset._relation.columns, dataset._relation.types):
        column = _quote_identifier(name)
        if _is_floating_type(data_type):
            expressions.append(f"CASE WHEN isnan({column}) THEN NULL ELSE {column} END AS {column}")
        else:
            expressions.append(column)
    relation = (
        dataset._relation.project(", ".join(expressions))
        if expressions
        else dataset._relation
    )
    summary = relation.query("__ogm_describe_source", "SUMMARIZE __ogm_describe_source")
    return dataset._clone_with_relation(summary)


def value_counts(
    dataset: Dataset,
    column: str,
    *,
    top_k: int = 20,
    dropna: bool = True,
) -> Dataset:
    """Return the most frequent values in one column as a lazy Dataset.

    Counts are exact. DuckDB may need to scan the source and retain distinct
    groups, even though the returned relation is limited to ``top_k`` rows.
    NULL and floating NaN groups are excluded by default.

    Args:
        dataset: Dataset containing the column.
        column: Column name to count.
        top_k: Maximum number of values to return; must be non-negative.
        dropna: If true, exclude NULL and floating NaN values.

    Returns:
        A Dataset with the selected value column and a ``count`` column.

    Raises:
        TypeError: If ``top_k`` is not an integer or ``dropna`` is not boolean.
        ValueError: If ``top_k`` is negative or the column does not exist.
    """
    limit = _validate_nonnegative_integer(top_k, "top_k")
    if not isinstance(dropna, bool):
        raise TypeError("dropna must be a bool")
    _require_column(dataset, column)
    quoted = _quote_identifier(column)
    where = ""
    if dropna:
        where = f" WHERE {quoted} IS NOT NULL"
        if _is_floating_type(_column_type(dataset, column)):
            where += f" AND NOT isnan({quoted})"
    query = (
        f"SELECT {quoted}, count(*) AS count FROM __ogm_value_counts_source"
        f"{where} GROUP BY {quoted} ORDER BY count DESC LIMIT {limit}"
    )
    relation = dataset._relation.query("__ogm_value_counts_source", query)
    return dataset._clone_with_relation(relation)


def unique(dataset: Dataset, column: str) -> Dataset:
    """Return a lazy one-column Dataset of distinct values.

    NULL, if present, is retained as one distinct value. Fetching the result
    may require DuckDB to scan and group the complete source column.

    Args:
        dataset: Dataset containing the column.
        column: Column name whose distinct values are requested.

    Returns:
        A lazy Dataset containing the selected column's distinct values.

    Raises:
        ValueError: If the column does not exist.
    """
    _require_column(dataset, column)
    relation = dataset._relation.project(_quote_identifier(column)).distinct()
    return dataset._clone_with_relation(relation)


def nunique(dataset: Dataset, column: str, *, approximate: bool = True) -> int:
    """Count distinct non-missing values in one column.

    Approximate mode uses DuckDB's HyperLogLog aggregate and is preferred for
    very large or high-cardinality columns. Exact mode may use more memory.

    Args:
        dataset: Dataset containing the column.
        column: Column name to count.
        approximate: If true, return an approximate count.

    Returns:
        An integer distinct-value count, excluding NULL and floating NaN.

    Raises:
        TypeError: If ``approximate`` is not boolean.
        ValueError: If the column does not exist.
    """
    if not isinstance(approximate, bool):
        raise TypeError("approximate must be a bool")
    _require_column(dataset, column)
    quoted = _quote_identifier(column)
    function = "approx_count_distinct" if approximate else "count"
    distinct = f"{function}(DISTINCT {quoted})" if not approximate else f"{function}({quoted})"
    where = f"{quoted} IS NOT NULL"
    if _is_floating_type(_column_type(dataset, column)):
        where += f" AND NOT isnan({quoted})"
    query = f"SELECT {distinct} FROM __ogm_nunique_source WHERE {where}"
    value = dataset._relation.query("__ogm_nunique_source", query).fetchone()[0]
    return int(value or 0)


def duplicate_count(dataset: Dataset, subset: Sequence[str] | None = None) -> int:
    """Count rows beyond the first occurrence of each duplicate group.

    Args:
        dataset: Dataset to inspect.
        subset: Columns defining duplicate identity. Defaults to all columns.

    Returns:
        Exact number of duplicate rows, excluding one retained row per group.

    Raises:
        TypeError: If ``subset`` is not a sequence of column names.
        ValueError: If a requested column does not exist.

    Note:
        Exact duplicate detection can require a full scan and substantial
        DuckDB memory for high-cardinality input.
    """
    if isinstance(subset, str):
        raise TypeError("subset must be a sequence of column names, not a string")
    try:
        names = list(dataset._relation.columns) if subset is None else list(subset)
    except TypeError as error:
        raise TypeError("subset must be a sequence of column names") from error
    if subset is not None and not names:
        raise ValueError("subset must contain at least one column")
    for name in names:
        _require_column(dataset, name)
    if not names:
        return 0
    group_columns = ", ".join(_quote_identifier(name) for name in names)
    query = (
        "SELECT coalesce(sum(group_count - 1), 0) "
        "FROM (SELECT count(*) AS group_count FROM __ogm_duplicate_source "
        f"GROUP BY {group_columns} HAVING count(*) > 1) duplicate_groups"
    )
    value = dataset._relation.query("__ogm_duplicate_source", query).fetchone()[0]
    return int(value or 0)


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


def _quote_string(value: str) -> str:
    """Quote a SQL string literal, escaping embedded single quotes."""
    return "'" + value.replace("'", "''") + "'"


def _validate_nonnegative_integer(value: object, name: str) -> int:
    """Validate and normalize a non-negative integer argument."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer")
    try:
        normalized = index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a non-negative integer") from error
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _require_column(dataset: Dataset, column: str) -> None:
    """Raise a clear error if a requested column is not present."""
    if not isinstance(column, str):
        raise TypeError("column must be a string")
    if column not in dataset._relation.columns:
        raise ValueError(f"column {column!r} does not exist")


def _column_type(dataset: Dataset, column: str) -> object:
    """Return a relation column's DuckDB type after validating its name."""
    _require_column(dataset, column)
    return dataset._relation.types[dataset._relation.columns.index(column)]


def _is_floating_type(data_type: object) -> bool:
    """Return whether a DuckDB type supports floating-point NaN values."""
    return str(data_type).upper() in {"FLOAT", "REAL", "DOUBLE", "FLOAT4", "FLOAT8"}
