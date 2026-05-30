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
about how a faithful value is *stored* on each side. Three such differences are
neutralized:

  1. Float width — scoped to columns that are float32-precision-limited on at
     least one side. A column is float32-normalized when EITHER the parquet
     stores it as float32 OR the PG schema declares it ``REAL`` (the union, see
     ``_float32_columns`` + ``_real_columns``). Both signals are needed:

       * Parquet float32 (live dtype): forward-return / feature columns the
         parquet persisted as float32. Their PG mirror may be ``REAL`` *or*
         ``DOUBLE PRECISION`` — the live DDL can diverge from the static schema
         (observed on ``thematic_labels_daily``: ``DOUBLE`` in Neon, ``REAL`` in
         the schema source), so the schema lookup alone missed it and produced
         ~100 spurious mismatches in a live backfill.
       * Schema ``REAL`` (static): when a parquet cache stores a column as
         float64 but the PG column is ``REAL``, PG rounds the value to float32
         on round-trip while the parquet float64 keeps full bits, so the parquet
         dtype alone misses it. The schema ``REAL`` membership catches it.

     For a column in the union, casting both sides through ``numpy.float32``
     collapses the float32 value (whichever side carries it) and its
     widened/un-rounded counterpart to the IDENTICAL 32-bit value, so faithful
     data hashes equal. float32 truncation is stable (no rounding-boundary
     instability) — significant-figure rounding was rejected because a value can
     straddle a boundary at the Nth digit.

     Crucially this is NOT applied to columns that are float64 on BOTH sides
     (parquet float64 + PG ``DOUBLE``: OHLC prices, ``breadth_indicator_daily``
     ``value``). Those are hashed at full float64 precision, so genuine
     sub-float32 drift in a DOUBLE column is still caught — the gate is not
     blinded for the data that legitimately needs full precision.

  2. NaN/NaT vs SQL NULL (every column). Recent asof dates legitimately carry
     NaN forward returns (the forward window has not elapsed); ``pg_value``
     writes NaN as PG NULL (Python ``None``). A numpy-aware scalar ``pd.isna``
     guard collapses Python-float NaN, numpy.float32 / numpy.float64 NaN, pandas
     NaT, and ``None`` to the single ``("n",)`` token (the float32-dtype
     forward-return columns arrive as numpy.float32 NaN, which is NOT a Python
     float, so a float-only guard leaked them — verify#0d00). A real float value
     stays a distinct ``("f", ...)`` token, so a NaN/NULL-vs-real-value
     difference is still caught.

  3. Signed zero (every float column). The parquet stores ``-0.0`` (e.g.
     ``thematic_labels_daily.fwd_10d_max_drawdown`` = "no drawdown") while PG
     normalizes it to ``+0.0`` on round-trip. ``-0.0 == +0.0`` numerically, but
     ``_checksum`` hashes ``repr(cells)`` and ``repr(-0.0) == '-0.0' !=
     repr(0.0) == '0.0'`` — so a value-equal pair hashed unequal (113 negative-
     zero cells across exactly 100 distinct asof_dates = the exact 100 drifted
     dates seen against live Neon, verify#0d00 signed-zero). The float branch
     adds ``0.0`` to the canonicalized float (``(-0.0)+0.0 == +0.0`` under
     IEEE-754; ``x + 0.0 == x`` for every other finite ``x``), so both signs of
     zero emit one identical token. A genuine near-zero drift (``0.0`` vs
     ``1e-9``) stays distinct.

Steps:

  * coerce every cell through ``pg_value`` (NaN/NaT -> None, numpy -> Python);
  * derive the float32 column set = parquet float32 dtypes (``_float32_columns``)
    UNION schema ``REAL`` columns (``_real_columns``); both sides are hashed
    against the SAME set so a float32 value and its widened/rounded counterpart
    collapse identically;
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
from sqlalchemy.dialects.postgresql import REAL
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


def _real_columns(spec: TableSpec) -> frozenset[str]:
    """Column names the static schema declares as PG ``REAL`` (float32) for ``spec``.

    Needed because a parquet cache can store a column as float64 while the PG
    column is ``REAL``: PG rounds it to float32 on round-trip, so the parquet
    dtype alone (``_float32_columns``) would miss it and false-positive drift on
    faithful data. The schema lookup catches that direction; the parquet-dtype
    lookup catches the live-DDL-diverged-DOUBLE direction. Their union is the
    float32-normalized set.
    """
    return frozenset(
        c.name for c in spec.table.columns if isinstance(c.type, REAL)
    )


def _float32_columns(df: pd.DataFrame) -> frozenset[str]:
    """Column names the parquet frame physically stores at float32 precision.

    A column persisted as float32 carries only ~7 significant digits, so its PG
    mirror (``REAL`` or a schema-diverged ``DOUBLE``) can never hold more than
    the float32 round-trip — those columns must be float32-normalized on both
    sides to compare faithfully. Derived from the LIVE parquet dtypes (not the
    static schema) so it catches ``thematic_labels_daily``, which is ``REAL`` in
    the schema source but ``DOUBLE`` in Neon. Unioned with ``_real_columns`` to
    also cover the float64-parquet-into-PG-``REAL`` direction.
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

    2. NaN/NaT vs SQL NULL (every column). Recent asof dates carry NaN forward
       returns (the forward window has not elapsed); ``pg_value`` writes NaN as
       PG NULL, which comes back as Python ``None``. The numpy-aware guard
       (``pd.isna`` on the scalar) collapses Python-float NaN, numpy.float32 /
       numpy.float64 NaN, pandas NaT, and ``None`` to the single ``("n",)``
       token, so the representation difference is not flagged. The float32
       forward-return columns arrive as numpy.float32 NaN, which is NOT a Python
       float — an ``isinstance(float)``-only guard missed those and leaked them
       to ``("s", "nan")`` vs the NULL side's ``("n",)`` (verify#0d00). A *real*
       float value stays a distinct ``("f", ...)`` token, so a
       NaN/NULL-vs-real-value difference is still caught as drift.

    Timezone-aware datetimes are normalized to UTC so the SAME instant hashes
    identically regardless of the session timezone PG rendered it in (TIMESTAMPTZ
    columns like ``fetched_at``/``computed_at`` come back in the connection's
    timezone, e.g. ``08:30-08:00``, while the parquet original is ``16:30+00:00``
    — both the same instant). Other non-floats are tagged + stringified.
    """
    if v is None:
        return ("n",)
    # Numpy-aware NULL guard, BEFORE any float32-cast / stringify branch. A
    # forward-return column is float32 dtype, so its NaN cells arrive as
    # numpy.float32 NaN — which is NOT a Python float, so the isinstance(float)
    # branch below missed it and leaked it to ("s", "nan") while the PG NULL side
    # hashed ("n",), causing ~100 spurious thematic_labels_daily mismatches
    # (verify#0d00; #110 only tested python-float NaN, which the float branch
    # already caught). pd.isna collapses python-float NaN, numpy.float32 /
    # numpy.float64 NaN, and pandas NaT to one NULL token. Guarded to SCALARS
    # only: pd.isna on an array/non-scalar returns an array (ambiguous truth) —
    # cells here are always scalars, but np.ndim==0 makes that explicit and safe.
    if np.ndim(v) == 0 and pd.isna(v):
        return ("n",)
    if isinstance(v, (float, np.floating)):
        # np.floating included so a raw (un-coerced) numpy float canonicalizes to
        # the SAME float token as its pg_value-widened Python-float counterpart
        # rather than stringifying. float() normalizes np.float32/64 -> Python
        # float for a stable repr; the is_float32 branch then collapses width.
        #
        # Signed-zero normalization (verify#0d00 signed-zero, the LOAD-BEARING
        # fix). The parquet stores -0.0 (e.g. fwd_10d_max_drawdown "no drawdown")
        # while PG normalizes it to +0.0 on round-trip. -0.0 == +0.0 numerically
        # (so every value-diff and both reviewers saw "no difference"), BUT
        # _checksum hashes repr(cells) and repr(-0.0) == '-0.0' != repr(0.0) ==
        # '0.0' -> a spurious mismatch. Adding 0.0 maps -0.0 -> +0.0 under
        # IEEE-754 ((-0.0)+0.0 == +0.0) while leaving every other value untouched
        # (x + 0.0 == x for finite x; NaN is already collapsed above). Applied to
        # BOTH the float32-normalized and full-precision branches so signed zero
        # in either kind of column produces one identical token.
        if is_float32:
            return ("f", float(np.float32(v)) + 0.0)
        return ("f", float(v) + 0.0)
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

        # float32-normalize a column when EITHER the parquet stores it as float32
        # (live-DDL-diverged DOUBLE case) OR the PG schema declares it REAL
        # (float64-parquet-into-REAL case). Both signals are needed; their union
        # is applied to BOTH sides. Columns that are float64 on both sides stay
        # full precision, so real DOUBLE drift is still caught.
        float32_cols = _float32_columns(pq_df) | _real_columns(spec)

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
