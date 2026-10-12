"""Common lazy dataset object returned by OpenGeoMove data readers."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import duckdb

from opengeomove.eda.accessor import EDAAccessor
from opengeomove.mobility.accessor import MobilityAccessor

_DISPLAY_MAX_ROWS = 5
_ITER_BATCH_SIZE = 1024


class Dataset(EDAAccessor, MobilityAccessor):
    """A read-only, lazy view over a source file scanned by DuckDB.

    Args:
        relation: DuckDB relation representing the source or a derived query.
        connection: Connection that owns the relation.
        source_path: Original source path, retained for derived datasets.
        data_format: Normalized source format name.
        eager_metadata: If true, resolve the shape during construction.

    The object retains a query relation rather than materializing every row in
    Python. Metadata properties and derived EDA operations execute only the
    work they require, and derived datasets do not modify their source.
    """

    def __init__(
        self,
        relation: duckdb.DuckDBPyRelation,
        connection: duckdb.DuckDBPyConnection,
        source_path: Path,
        data_format: str,
        eager_metadata: bool = False,
    ) -> None:
        """Create a Dataset around a DuckDB relation and its connection."""
        self._relation = relation
        self._connection = connection
        self._source_path = source_path
        self._format = data_format
        self._metadata_cache: dict[str, Any] = {}
        self._filters: tuple[Any, ...] = ()
        if eager_metadata:
            self._prime_metadata()

    @property
    def columns(self) -> tuple[str, ...]:
        """Return and cache the source column names as an immutable tuple."""
        if "columns" not in self._metadata_cache:
            self._metadata_cache["columns"] = tuple(self._relation.columns)
        return self._metadata_cache["columns"]

    @property
    def dtypes(self) -> OrderedDict[str, Any]:
        """Return column names and DuckDB types in source column order.

        The mapping is built from relation schema metadata and does not scan
        dataset rows. A fresh mapping is returned so caller edits cannot
        affect the Dataset's cached metadata.
        """
        if "dtypes" not in self._metadata_cache:
            self._metadata_cache["dtypes"] = tuple(
                zip(self._relation.columns, self._relation.types)
            )
        return OrderedDict(self._metadata_cache["dtypes"])

    @property
    def shape(self) -> tuple[int, int]:
        """Return and cache ``(row_count, column_count)``.

        DuckDB computes the row count when this property is first accessed.
        For large text files this can require scanning the complete source,
        though the result itself is only a scalar and is cached per Dataset.
        """
        if "shape" not in self._metadata_cache:
            row_count, _ = self._relation.shape
            self._metadata_cache["shape"] = (int(row_count), len(self.columns))
        return self._metadata_cache["shape"]

    @property
    def source(self) -> Path:
        """Return the local path from which this Dataset was opened."""
        return self._source_path

    @property
    def format(self) -> str:
        """Return the normalized source format name."""
        return self._format

    @property
    def filters(self) -> tuple[Any, ...]:
        """Return the filter objects that define this Dataset's current view."""
        return self._filters

    def apply(self, *filters: Any) -> Dataset:
        """Return a lazy Dataset with the supplied filters added to its view.

        Filters are executed by DuckDB when a later operation consumes the
        relation. They never delete or rewrite source data. The current Dataset
        is unchanged; the returned Dataset carries the added filters, and
        filters can be composed in one call or across chained calls.

        Args:
            *filters: Instances of filters exposed through ``ogm.processing``.

        Returns:
            A new Dataset whose query plan includes all existing and supplied
            filters.

        Raises:
            TypeError: If no filters are supplied or an argument is not a
                supported filter instance.
            ValueError: If a filter references a missing column or has invalid
                parameters.
        """
        if not filters:
            raise TypeError("apply() requires at least one filter instance")

        relation = self._relation
        for filter_obj in filters:
            apply_filter = getattr(filter_obj, "_apply", None)
            if not callable(apply_filter):
                raise TypeError(
                    "apply() arguments must be filter instances from ogm.processing"
                )
            relation = apply_filter(relation, self._connection)

        derived = self._clone_with_relation(relation)
        derived._filters = (*self._filters, *filters)
        return derived

    @property
    def rows(self) -> Iterator[tuple[Any, ...]]:
        """Return a lazy iterator yielding all rows as tuples.

        Rows are fetched from DuckDB in bounded batches, so accessing this
        property does not collect the full dataset in Python memory. Accessing
        ``rows`` again creates a fresh iterator over the Dataset relation.
        """
        relation = self._relation.project("*")
        while batch := relation.fetchmany(_ITER_BATCH_SIZE):
            yield from batch

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        """Iterate over all dataset rows as tuples in column order."""
        return self.rows

    def _prime_metadata(self) -> None:
        """Resolve and cache the metadata currently supported by Dataset."""
        _ = self.columns
        _ = self.shape

    def _clone_with_relation(self, relation: duckdb.DuckDBPyRelation) -> Dataset:
        """Create a derived Dataset sharing this object's connection."""
        derived = Dataset(
            relation=relation,
            connection=self._connection,
            source_path=self._source_path,
            data_format=self._format,
        )
        derived._filters = self._filters
        if hasattr(self, "_stop_merge_distance_m"):
            derived._stop_merge_distance_m = self._stop_merge_distance_m
        return derived

    def __str__(self) -> str:
        """Render up to five rows as readable text without counting all rows.

        Printing a Dataset materializes only a bounded preview. If the
        relation has more than five rows, the output notes that rows were
        omitted; use :meth:`head` to create a smaller Dataset to print.
        """
        preview = self._relation.limit(_DISPLAY_MAX_ROWS + 1)
        fetched_rows = preview.fetchall()
        rows = fetched_rows[:_DISPLAY_MAX_ROWS]
        columns = tuple(preview.columns)
        if not columns:
            rendered = "(no columns)"
        else:
            rendered = " | ".join(columns)
            for row in rows:
                rendered += "\n" + " | ".join(repr(value) for value in row)
        if not rows:
            rendered += "\n(no rows)"
        if len(fetched_rows) > _DISPLAY_MAX_ROWS:
            rendered += f"\n... (showing first {_DISPLAY_MAX_ROWS} rows)"
        return rendered

    def __repr__(self) -> str:
        """Render a bounded diagnostic summary without counting all rows."""
        preview = self._relation.limit(_DISPLAY_MAX_ROWS + 1)
        fetched_rows = preview.fetchall()
        has_more_rows = len(fetched_rows) > _DISPLAY_MAX_ROWS
        rows = fetched_rows[:_DISPLAY_MAX_ROWS]
        return (
            f"Dataset(format={self._format!r}, columns={tuple(preview.columns)!r}, "
            f"preview={rows!r}, more_rows={has_more_rows})"
        )
