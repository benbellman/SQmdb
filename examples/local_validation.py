"""Helpers for moving small DB-API query results into Polars for SQmdb."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import polars as pl


class Cursor(Protocol):
    @property
    def description(self) -> Sequence[Sequence[object]] | None: ...

    def fetchmany(self, size: int) -> Sequence[Sequence[object]]: ...


def cursor_to_polars(cursor: Cursor, *, batch_size: int = 50_000) -> pl.DataFrame:
    """Consume an executed DB-API cursor into a Polars frame.

    SQL should alias columns to SQmdb's canonical names. Results are read in
    batches, but concatenated in memory; use this helper for small extracts.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if cursor.description is None:
        raise ValueError("The cursor must contain a query result with column names.")

    columns = [column[0] for column in cursor.description]
    if any(not isinstance(name, str) or not name for name in columns):
        raise ValueError("Query result columns must have non-empty string names.")
    if len(set(columns)) != len(columns):
        raise ValueError("Query result column names must be unique; alias duplicate columns.")

    batches: list[pl.DataFrame] = []
    while rows := cursor.fetchmany(batch_size):
        batches.append(
            pl.DataFrame(
                [tuple(row) for row in rows],
                schema=columns,
                orient="row",
                infer_schema_length=None,
            )
        )
    if not batches:
        return pl.DataFrame(schema=columns)
    return pl.concat(batches, how="vertical_relaxed")
