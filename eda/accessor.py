"""Dataset method mixin that delegates EDA operations to this package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from opengeomove.eda.operations import head, isna, isnull, notna, notnull

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
