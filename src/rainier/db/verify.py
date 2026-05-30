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
neutralized:

  1. Float width — scoped to columns the PARQUET actually stores as float32.
     Those forward-return / feature columns carry only float32 precision; the PG
     mirror may be ``REAL`` *or* ``DOUBLE PRECISION`` (the live DDL can diverge
     from the static schema — observed on ``thematic_labels_daily``: ``DOUBLE``
     in Neon, ``REAL`` in the schema source — which produced ~100 spurious
     mismatches in a live backfill). A naive repr/hash flags the
     float32-vs-widened-float64 low-bit delta as drift. Casting both sides of a
     float32-origin column through ``numpy.float32`` collapses the stored
     float32 value and its widened-to-float64 PG counterpart to the IDENTICAL
     32-bit value, so they hash equal regardless of which side stored which
     width. The set of float32 columns is derived from the live parquet dtypes
     (``_float32_columns``), NOT the static schema — that is what fixed the
     ``thematic_labels_daily`` divergence the old schema-``REAL`` lookup missed.

     Crucially this is NOT applied to float64-origin columns. A column the
     parquet stores as float64 (OHLC prices, ``breadth_indicator_daily.value``)
     is hashed at full float64 precision on both sides, so genuine sub-float32
     drift in a DOUBLE column is still caught — the gate is not blinded for the
     data that legitimately needs full precision. float32 truncation is stable
     (no rounding-boundary instability) — significant-figure rounding was
     rejected because a value can straddle a boundary at the Nth digit.

  2. NaN vs SQL NULL (every column). Recent asof dates legitimately carry NaN
     forward returns (the forward window has not elapsed); ``pg_value`` writes
     NaN as PG NULL (Python ``None``). Both a float NaN and ``None`` canonicalize
     to the single ``("n",)`` token so this representation difference is not
     flagged — while a real float value stays a distinct ``("f", ...)`` token, so
     a NaN/NULL-vs-real-value difference is still caught.

Steps:

  * coerce every cell through ``pg_value`` (NaN/NaT -> None, numpy -> Python);
  * derive the float32-origin column set from the parquet frame's dtypes
    (``_float32_columns``); the PG side is hashed against the SAME set so a
    float32 parquet value and its widened PG mirror collapse identically;
  * canonicalize via ``_canon_cell`` (float in a float32 column -> float32;
    NaN/NULL -> one token; tz-aware datetimes -> UTC);
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


def _float32_columns(df: pd.DataFrame) -> frozenset[str]:
    """Column names the parquet frame stores at float32 precision.

    The parquet is the source-of-precision: a column physically stored as
    float32 carries only ~7 significant digits, so its PG mirror (REAL or a
    schema-diverged DOUBLE) can never hold more than the float32 round-trip.
    Those are exactly the columns whose float64-widened PG values must be cast
    back through float32 to compare equal. float64-stored columns (OHLC prices,
    ``breadth_indicator_daily.value``) are NOT in this set, so they stay full
    precision and real drift in them is still caught.

    Derived from the live parquet dtypes — NOT the static schema — because the
    live DDL can diverge from the schema (``thematic_labels_daily`` is REAL in
    the schema source but DOUBLE in Neon), and only the parquet dtype reliably
    tells us the actual precision the data was persisted at.
    """
    return frozenset(
        str(name)
        for name, dtype in df.dtypes.items()
        if dtype == np.float32
    )


def _canon_cell(v, is_float32: bool) -> object:
    """Canonicalize a coerced cell for hashing.

    ``is_float32`` is True only for columns the parquet stores at float32
    precision (see ``_float32_columns``). Two storage-only differences are
    neutralized so faithful data reports TRUE parity:

    1. Float width (float32 columns only). The parquet stores the column at
       float32 while the PG mirror may be REAL *or* DOUBLE PRECISION (the live
       DDL can diverge from the static schema — observed on
       ``thematic_labels_daily``: DOUBLE in Neon, REAL in the schema source).
       Casting both sides of a float32-origin column through float32 collapses a
       float32 parquet value and its widened-to-float64 PG counterpart to the
       identical 32-bit value, so they hash equal regardless of which side
       stored which width. float32 truncation is stable (no rounding-boundary
       instability). A float64-origin column (``is_float32`` False) is hashed at
       FULL float64 precision, so genuine sub-float32 drift in a DOUBLE column is
       still caught — the gate is not blinded for full-precision data.

    2. NaN vs SQL NULL (every column). Recent asof dates carry NaN forward
       returns (the forward window has not elapsed); ``pg_value`` writes NaN as
       PG NULL, which comes back as Python ``None``. Both a float NaN and
       ``None`` canonicalize to the single ``("n",)`` token so the
       representation difference is not flagged — a *real* float value stays a
       distinct ``("f", ...)`` token, so a NaN/NULL-vs-real-value difference is
       still caught as drift.

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
        if is_float32:
            return ("f", float(np.float32(v)))
        return ("f", v)
    if isinstance(v, _dt.datetime) and v.tzinfo is not None:
        # Same instant, stable repr: convert to UTC before stringifying.
        return ("s", str(v.astimezone(_dt.timezone.utc)))
    return ("s", str(v))


def _checksum(
    rows: list[dict],
    columns: list[str],
    pk_cols: tuple[str, ...],
    float32_cols: frozenset[str] = frozenset(),
) -> str:
    """Order-independent content hash of ``rows`` projected to ``columns``.

    Rows are sorted by primary key (None sorts first via a (is_none, repr) key)
    so PG vs parquet row order is irrelevant, then each row is rendered as a
    fixed-column-order tuple of canonical cells. Floats in ``float32_cols`` are
    float32-normalized (faithful float32<->widened-float64 parity); all other
    floats keep full float64 precision; NaN/NULL collapse to one token — see
    ``_canon_cell``. The result is BLAKE2b-hashed.
    """

    def sort_key(row: dict):
        return tuple((row.get(c) is None, str(row.get(c))) for c in pk_cols)

    h = hashlib.blake2b(digest_size=16)
    for row in sorted(rows, key=sort_key):
        cells = tuple(
            _canon_cell(row.get(c), c in float32_cols) for c in columns
        )
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

        # The parquet is the source-of-precision: only columns physically stored
        # as float32 get the float32 round-trip normalization (applied to BOTH
        # sides). float64 columns stay full precision so real DOUBLE drift is
        # still caught. Derived from live dtypes, not the static schema.
        float32_cols = _float32_columns(pq_df)

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
                pq_sum = _checksum(pq_g, columns, spec.pk_cols, float32_cols)
                pg_sum = _checksum(pg_g, columns, spec.pk_cols, float32_cols)
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
