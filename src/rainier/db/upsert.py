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
                           set_ = {supplied non-pk cols
                                   minus immutable_cols -> EXCLUDED.col})
                                      │
                                      ▼
                          one transaction per call

``first_seen`` on the registries is passed as ``immutable_cols`` so it is
written once and never re-stamped on a same-key re-run.
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
    immutable_cols: Sequence[str] = (),
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
    immutable_cols:
        Columns that are insert-only — written on first INSERT but NEVER
        overwritten on a conflict. Use for first-seen provenance (e.g.
        ``first_seen`` on the ticker/sector registries) so a same-key re-run
        keeps the original value instead of stamping the re-run's date.

    The ``ON CONFLICT DO UPDATE`` set is the non-PK columns the caller
    actually supplied across ``rows``, minus ``immutable_cols``. We do NOT
    update every table column: a row dict that omits a column (e.g. a registry
    write that omits ``last_seen``) would otherwise reset that column to
    ``EXCLUDED``'s default (NULL), silently clobbering existing data on re-run.

    All chunks run inside a single transaction so a mid-batch failure rolls
    back the whole call (no half-written asof_date).
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    rows = list(rows)
    if not rows:
        return 0

    pk_set = set(pk_cols)
    immutable_set = set(immutable_cols)
    table_cols = set(table.columns.keys())
    # Only update columns the caller supplied (union of all row keys) AND that
    # exist on the table, excluding PK + immutable provenance columns. This is
    # what keeps a re-run from NULL-ing out columns it never meant to write.
    supplied_cols = {k for row in rows for k in row.keys()}
    update_cols = [
        c
        for c in table.columns.keys()  # preserve table column order, deterministic
        if c in table_cols
        and c in supplied_cols
        and c not in pk_set
        and c not in immutable_set
    ]

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
