"""Parquet <-> Postgres coverage parity check (``rainier db verify-coverage``).

Phase 3, task plan §2/§3. For each ``(asof_date, table)`` it compares parquet
vs PG on two axes:

  1. row count for that date;
  2. an order-independent content checksum over the canonical columns.

A per-table report is printed by the CLI; the module returns a ``VerifyReport``
with ``ok`` (all match) + a ``drift`` list naming each offending
``(table, asof_date)``. The CLI exits nonzero on any drift so it is
CI/cron-usable as the gate the operator runs after a backfill (and later after
daily dual-write runs) to confirm PG mirrors parquet.

Checksum determinism + float/NaN canonicalization (task plan §3, verify#8aab)
-----------------------------------------------------------------------------
The checksum must NOT depend on row order (PG and parquet return rows in
different orders), and must NOT false-positive on differences that are purely
about how a faithful value is *stored* on each side. Two such differences are
neutralized, uniformly for EVERY table (no per-column or per-table special-case
— ``_canon_cell`` is value-driven, it takes no schema input):

  1. Float width. The parquet stores some forward-return / feature columns as
     float32; the PG column may be ``REAL`` *or* ``DOUBLE PRECISION``. The live
     DDL can diverge from the static schema — observed on
     ``thematic_labels_daily`` (``DOUBLE`` in Neon, ``REAL`` in the schema
     source), which produced ~100 spurious mismatches in a live backfill while
     every other table was true parity. A naive repr/hash flags the
     float32-vs-float64 low-bit delta as drift. Casting EVERY float through
     ``numpy.float32`` on BOTH sides collapses a float32 value and its
     widened-to-float64 counterpart to the IDENTICAL 32-bit value, so they hash
     bit-for-bit equal regardless of which side stored which width. float32
     truncation is stable (no rounding-boundary instability) — significant-
     figure rounding was rejected because a value can straddle a boundary at the
     Nth digit. It never blinds the gate: two numbers that differ beyond float32
     precision still hash differently.

  2. NaN vs SQL NULL. Recent asof dates legitimately carry NaN forward returns
     (the forward window has not elapsed); ``pg_value`` writes NaN as PG NULL
     (Python ``None``). Both a float NaN and ``None`` canonicalize to the single
     ``("n",)`` token so this representation difference is not flagged — while a
     real float value stays a distinct ``("f", ...)`` token, so a
     NaN/NULL-vs-real-value difference is still caught.

Steps:

  * coerce every cell through ``pg_value`` (NaN/NaT -> None, numpy -> Python);
  * canonicalize via ``_canon_cell`` (every float -> float32; NaN/NULL -> one
    token; tz-aware datetimes -> UTC);
  * build a per-row tuple in fixed table-column order, sort the rows within a
    date group by primary key, then BLAKE2b-hash the canonical repr.

Identical (count, checksum) for a ``(date, table)`` => parity. The registries
(``date_col=None``) are checked as a single whole-table group keyed on the
sentinel date ``None``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from rainier.db.rows import TABLE_SPECS, TableSpec, frame_to_pg_rows


@dataclass(frozen=True)
class Drift:
    """One offending ``(table, asof_date)`` and what disagreed."""

    table: str
    asof_date: date | None
    reason: str  # human-readable: "row count 5 != 4" or "checksum mismatch"


@dataclass
class VerifyReport:
    """Result of a verify-coverage run."""

    drift: list[Drift] = field(default_factory=list)
    # Per (table, asof_date) -> (parquet_count, pg_count) for the printed report.
    rows: list[tuple[str, date | None, int, int, bool]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.drift


def _canon_cell(v) -> object:
    """Canonicalize a coerced cell for hashing — value-driven, no schema input.

    The checksum must report TRUE parity for data that is faithful but stored at
    a different float width or with a different missing-value representation on
    the two sides. Two such differences are neutralized here, uniformly for
    EVERY table (no per-column or per-table special-case):

    1. Float width. The parquet stores some forward-return / feature columns as
       float32 while the PG column may be REAL *or* DOUBLE PRECISION — the live
       DDL can diverge from the static schema (observed on
       ``thematic_labels_daily``: DOUBLE in Neon, REAL in the schema source).
       Casting EVERY float through float32 collapses a float32 parquet value and
       its widened-to-float64 PG counterpart to the identical 32-bit value, so
       they hash equal regardless of which side stored which width. float32
       truncation is stable (no rounding-boundary instability) and never blinds
       the gate: two numbers that differ beyond float32 precision still hash
       differently.

    2. NaN vs SQL NULL. Recent asof dates carry NaN forward returns (the forward
       window has not elapsed); ``pg_value`` writes NaN as PG NULL, which comes
       back as Python ``None``. Both a float NaN and ``None`` canonicalize to the
       single ``("n",)`` token so the representation difference is not flagged —
       a *real* float value stays a distinct ``("f", ...)`` token, so a
       NaN/NULL-vs-real-value difference is still caught as drift.

    Timezone-aware datetimes are normalized to UTC so the SAME instant hashes
    identically regardless of the session timezone PG rendered it in (TIMESTAMPTZ
    columns like ``fetched_at``/``computed_at`` come back in the connection's
    timezone, e.g. ``08:30-08:00``, while the parquet original is ``16:30+00:00``
    — both the same instant). Other non-floats are tagged + stringified.
    """
    if v is None:
        return ("n",)
    if isinstance(v, float):
        if v != v:  # NaN: collapse to the NULL token (faithful NaN<->NULL).
            return ("n",)
        return ("f", float(np.float32(v)))
    if isinstance(v, _dt.datetime) and v.tzinfo is not None:
        # Same instant, stable repr: convert to UTC before stringifying.
        return ("s", str(v.astimezone(_dt.timezone.utc)))
    return ("s", str(v))


def _checksum(
    rows: list[dict],
    columns: list[str],
    pk_cols: tuple[str, ...],
) -> str:
    """Order-independent content hash of ``rows`` projected to ``columns``.

    Rows are sorted by primary key (None sorts first via a (is_none, repr) key)
    so PG vs parquet row order is irrelevant, then each row is rendered as a
    fixed-column-order tuple of canonical cells (every float float32-normalized,
    NaN/NULL collapsed — see ``_canon_cell``) and BLAKE2b-hashed.
    """

    def sort_key(row: dict):
        return tuple((row.get(c) is None, str(row.get(c))) for c in pk_cols)

    h = hashlib.blake2b(digest_size=16)
    for row in sorted(rows, key=sort_key):
        cells = tuple(_canon_cell(row.get(c)) for c in columns)
        h.update(repr(cells).encode("utf-8"))
        h.update(b"\x1e")  # record separator
    return h.hexdigest()


def _normalize_dates(values: pd.Series) -> pd.Series:
    """Coerce a date-ish column to ``datetime.date`` for stable grouping."""
    return pd.to_datetime(values).dt.date


def _window_df(
    df: pd.DataFrame, spec: TableSpec, asof_start: date | None, asof_end: date | None
) -> pd.DataFrame:
    """Restrict ``df`` to [asof_start, asof_end] on the spec's date column.

    Registries (``date_col=None``) and an empty window are pass-through.
    Applied identically to the parquet and PG sides so they compare apples-to-
    apples.
    """
    if spec.date_col is None or (asof_start is None and asof_end is None) or df.empty:
        return df
    norm = _normalize_dates(df[spec.date_col])
    mask = pd.Series(True, index=df.index)
    if asof_start is not None:
        mask &= norm >= asof_start
    if asof_end is not None:
        mask &= norm <= asof_end
    return df[mask]


def _read_pg(engine: Engine, spec: TableSpec, columns: list[str]) -> pd.DataFrame:
    """Plain SELECT of all rows for ``spec`` (no ORM)."""
    # Column + table names come from the schema definition (TABLE_SPECS), never
    # from user input — no injection surface despite the f-string.
    col_list = ", ".join(columns)
    sql = f"SELECT {col_list} FROM market.{spec.name}"
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        records = [dict(r) for r in result.mappings()]
    return pd.DataFrame(records, columns=columns)


def _group_by_date(
    rows: list[dict], spec: TableSpec
) -> dict[date | None, list[dict]]:
    """Bucket coerced rows by their date column (single None bucket for registries)."""
    if spec.date_col is None:
        return {None: rows}
    groups: dict[date | None, list[dict]] = {}
    for row in rows:
        key = row.get(spec.date_col)
        groups.setdefault(key, []).append(row)
    return groups


def verify_coverage(
    engine: Engine,
    cache_dir: str | Path,
    asof_start: date | None = None,
    asof_end: date | None = None,
) -> VerifyReport:
    """Compare parquet caches in ``cache_dir`` against ``market.*`` per (date, table).

    Returns a ``VerifyReport``: ``ok`` when every ``(date, table)`` matches on
    both row count and checksum; ``drift`` lists each mismatch. ``asof_start`` /
    ``asof_end`` window the date-keyed tables (registries always compared whole).
    """
    cache_dir = Path(cache_dir)
    report = VerifyReport()

    for spec in TABLE_SPECS:
        columns = list(spec.table.columns.keys())
        path = cache_dir / f"{spec.parquet_name}.parquet"

        if path.exists():
            pq_df = pd.read_parquet(path)
        else:
            pq_df = pd.DataFrame(columns=columns)

        # Window both sides identically before comparing (registries pass through).
        pq_df = _window_df(pq_df, spec, asof_start, asof_end)
        pg_df = _window_df(_read_pg(engine, spec, columns), spec, asof_start, asof_end)

        # Coerce both sides through the SAME path so float/Decimal/Timestamp
        # representations are comparable.
        pq_rows = frame_to_pg_rows(_fill_columns(pq_df, columns), columns)
        pg_rows = frame_to_pg_rows(_fill_columns(pg_df, columns), columns)
        # Normalize the date column to datetime.date for stable bucketing.
        if spec.date_col is not None:
            for r in (*pq_rows, *pg_rows):
                r[spec.date_col] = _coerce_date(r.get(spec.date_col))

        pq_groups = _group_by_date(pq_rows, spec)
        pg_groups = _group_by_date(pg_rows, spec)

        for key in sorted(
            set(pq_groups) | set(pg_groups), key=lambda d: (d is None, str(d))
        ):
            pq_g = pq_groups.get(key, [])
            pg_g = pg_groups.get(key, [])
            match = True
            if len(pq_g) != len(pg_g):
                report.drift.append(
                    Drift(spec.name, key, f"row count parquet={len(pq_g)} pg={len(pg_g)}")
                )
                match = False
            else:
                pq_sum = _checksum(pq_g, columns, spec.pk_cols)
                pg_sum = _checksum(pg_g, columns, spec.pk_cols)
                if pq_sum != pg_sum:
                    report.drift.append(
                        Drift(spec.name, key, "checksum mismatch")
                    )
                    match = False
            report.rows.append((spec.name, key, len(pq_g), len(pg_g), match))

    return report


def _coerce_date(v):
    """Return a ``datetime.date`` for a date/datetime/Timestamp cell, else v."""
    if isinstance(v, pd.Timestamp):
        return v.date()
    if isinstance(v, _dt.datetime):
        return v.date()
    return v


def _fill_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ensure ``df`` has every column in ``columns`` (missing -> all-None).

    Both parquet and PG carry every canonical column here, but a missing-column
    cache (older snapshot) would otherwise get OMITTED by frame_to_pg_rows on
    one side only, skewing the checksum. Materializing the column on both sides
    keeps the row dicts shape-identical so only VALUE drift is reported.
    """
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = None
    return out


__all__ = ["Drift", "VerifyReport", "verify_coverage"]
