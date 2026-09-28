"""Dataset method mixin that delegates EDA operations to this package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from opengeomove.eda.operations import (
    describe,
    duplicate_count,
    head,
    info,
    isna,
    isnull,
    missing_summary,
    notna,
    notnull,
    nunique,
    unique,
    value_counts,
)

if TYPE_CHECKING:
    from opengeomove.data_loading.dataset import Dataset


class EDAAccessor:
    """Expose lazy EDA methods on a Dataset without owning reader logic."""

    def head(self: Dataset, n: int = 5) -> Dataset:
        """Return a lazy Dataset limited to the first ``n`` rows.

        Args:
            n: Non-negative maximum number of rows in the result.

        Returns:
            A new Dataset with the same columns and at most ``n`` rows.

        Raises:
            TypeError: If ``n`` is not an integer.
            ValueError: If ``n`` is negative.
        """
        return head(self, n=n)

    def info(self: Dataset, *, deep: bool = False) -> None:
        """Print source and schema information; count rows only if requested.

        Args:
            deep: If true, compute the exact row count, which may scan input.

        Returns:
            None. The report is written to standard output.

        Raises:
            TypeError: If ``deep`` is not a boolean.
        """
        return info(self, deep=deep)

    def missing_summary(self: Dataset) -> Dataset:
        """Return lazy per-column NULL/NaN counts and percentages.

        Returns:
            A Dataset with one summary row per column.
        """
        return missing_summary(self)

    def describe(self: Dataset) -> Dataset:
        """Return DuckDB's type-aware summary statistics as a lazy Dataset.

        Returns:
            A Dataset with one summary row per source column.
        """
        return describe(self)

    def value_counts(
        self: Dataset,
        column: str,
        *,
        top_k: int = 20,
        dropna: bool = True,
    ) -> Dataset:
        """Return the most frequent values of a column as a lazy Dataset.

        Args:
            column: Column whose values are counted.
            top_k: Maximum number of values returned.
            dropna: Whether to exclude NULL and floating NaN values.

        Returns:
            A Dataset containing values and exact counts.
        """
        return value_counts(self, column, top_k=top_k, dropna=dropna)

    def unique(self: Dataset, column: str) -> Dataset:
        """Return a lazy Dataset containing distinct values of one column.

        Args:
            column: Column whose distinct values are returned.

        Returns:
            A one-column Dataset of distinct values.
        """
        return unique(self, column)

    def nunique(self: Dataset, column: str, *, approximate: bool = True) -> int:
        """Count distinct non-missing values in one column.

        Args:
            column: Column whose distinct values are counted.
            approximate: Whether to use an approximate aggregate.

        Returns:
            The distinct-value count.
        """
        return nunique(self, column, approximate=approximate)

    def duplicate_count(self: Dataset, subset: Sequence[str] | None = None) -> int:
        """Count rows beyond the first occurrence of each duplicate group.

        Args:
            subset: Columns defining duplicate identity; defaults to all columns.

        Returns:
            Exact number of duplicate rows.
        """
        return duplicate_count(self, subset=subset)

    def isna(self: Dataset) -> Dataset:
        """Return a lazy boolean Dataset marking NULL and floating NaN values."""
        return isna(self)

    def isnull(self: Dataset) -> Dataset:
        """Alias for :meth:`isna`."""
        return isnull(self)

    def notna(self: Dataset) -> Dataset:
        """Return the lazy boolean inverse of :meth:`isna`."""
        return notna(self)

    def notnull(self: Dataset) -> Dataset:
        """Alias for :meth:`notna`."""
        return notnull(self)
