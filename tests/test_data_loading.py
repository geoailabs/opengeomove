"""Regression checks for DuckDB-backed delimited text readers."""

import tempfile
import unittest
from pathlib import Path

import duckdb

import opengeomove as ogm


class DelimitedReaderTests(unittest.TestCase):
    """Exercise actual CSV and TSV reader calls with quoted text fields."""

    def test_csv_and_tsv_readers_preserve_rows_and_quoted_geometry(self):
        """Use DuckDB's supported delimiter argument for both text formats."""
        with tempfile.TemporaryDirectory() as directory:
            for suffix, delimiter in [("csv", ","), ("tsv", "\t")]:
                with self.subTest(format=suffix):
                    path = Path(directory) / f"footprints.{suffix}"
                    connection = duckdb.connect()
                    connection.execute(
                        "COPY (SELECT 'POLYGON ((0 0, 1 0, 0 0))' AS geometry, "
                        "0.75 AS confidence) TO ? (FORMAT CSV, HEADER, DELIMITER ?)",
                        [str(path), delimiter],
                    )
                    connection.close()
                    ds = ogm.open(path)
                    try:
                        self.assertEqual(ds.shape, (1, 2))
                        self.assertEqual(next(iter(ds.rows))[0], "POLYGON ((0 0, 1 0, 0 0))")
                        self.assertEqual(ds.format, suffix)
                    finally:
                        ds._connection.close()


if __name__ == "__main__":
    unittest.main()
