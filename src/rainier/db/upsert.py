"""Batch UPSERT helper for the ``market.*`` canonical store (task plan §4).

One reusable function, ``market_upsert``, drives every dual-write call site
(``backfill_thematic_universe`` + the ``thematic`` feature/label CLI commands).
It is deliberately NOT a per-table abstraction — three call sites sharing one
helper is the right amount of structure (design D-5 / task plan §4).

Idempotency: the table's primary key (declared in migration 0001) drives
``INSERT ... ON CONFLICT (pk) DO UPDATE``. A same-key re-run updates the
non-PK columns in place; it never duplicates a row. This is what makes a
same-``asof_date`` writer re-run a no-op on row count.

Batching: rows are chunked (default 1000) so a multi-thousand-row backfill
never builds one oversized INSERT statement.

ASCII flow:

    rows ──chunk(batch_size)──► pg_insert(table).values(chunk)
                                      │
                                      ▼
                       .on_conflict_do_update(
                           index_elements = pk_cols,
                           set_ = {non-pk cols -> EXCLUDED.col})
                                      │
                                      ▼
                          one transaction per call
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

DEFAULT_BATCH_SIZE = 1000


def market_upsert(
    engine: Engine,
    table: Table,
    rows: Sequence[Mapping[str, Any]],
    pk_cols: Sequence[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """UPSERT ``rows`` into ``table`` keyed on ``pk_cols``; return rows sent.

    Parameters
    ----------
    engine:
        SQLAlchemy Engine (from ``rainier.db.get_engine``).
    table:
        A ``rainier.db.schema`` Table object.
    rows:
        Row dicts. Keys must be column names of ``table``. Empty -> no-op.
    pk_cols:
        Conflict-target columns (the table's PK). Drives ``ON CONFLICT``.
    batch_size:
        Max rows per INSERT statement. Must be positive.

    All chunks run inside a single transaction so a mid-batch failure rolls
    back the whole call (no half-written asof_date).
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    rows = list(rows)
    if not rows:
        return 0

    pk_set = set(pk_cols)
    table_cols = set(table.columns.keys())
    update_cols = [c for c in table_cols if c not in pk_set]

    with engine.begin() as conn:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            stmt = pg_insert(table).values(chunk)
            if update_cols:
                stmt = stmt.on_conflict_do_update(
                    index_elements=list(pk_cols),
                    set_={c: stmt.excluded[c] for c in update_cols},
                )
            else:
                # PK-only table: nothing to update on conflict, just skip.
                stmt = stmt.on_conflict_do_nothing(index_elements=list(pk_cols))
            conn.execute(stmt)

    return len(rows)
