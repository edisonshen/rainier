"""Shared parquet-frame -> market.* row conversion + the table mapping.

Phase 2's dual-write writers (``cli.py``) and Phase 3's one-shot backfill
(``db/backfill.py``) + parity check (``db/verify.py``) all need the same two
primitives:

  * ``pg_value`` — coerce a pandas/numpy cell into a plain Python scalar
    psycopg can bind (NaN/NaT -> None, numpy ints/floats -> int/float,
    Timestamp -> datetime).
  * ``frame_to_pg_rows`` — project a DataFrame to a column list and coerce each
    cell, OMITTING columns absent from the frame (not emitting None — see the
    docstring for why that matters to ``market_upsert``'s omitted-column
    protection).

These lived as module-private helpers in cli.py through Phase 2. Phase 3 factors
them here (task plan §3 — "factor out the shared frame->rows logic rather than
duplicating") so the three call sites share one implementation. cli.py re-imports
them under their old private names to keep its call sites and existing tests
unchanged.

``TABLE_SPECS`` is the single source of truth for the parquet<->table mapping
used by backfill + verify: which parquet cache feeds which ``market.*`` table,
the conflict-key columns, insert-only (immutable) columns, and the per-table
date column used to group rows by ``asof_date`` for coverage reporting.
Registries have ``date_col=None`` (not date-keyed — they always load fully as
FK parents).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rainier.db import schema


def pg_value(v):
    """Coerce a pandas/numpy cell into a plain Python scalar psycopg can bind.

    NaN/NaT -> None; numpy ints/floats -> int/float; Timestamp -> datetime;
    date32 stays date. Keeps the conversion tolerant of the mixed-dtype frames
    the compute layer produces (float32 NaNs, int8 sentinels, etc.).
    """
    if v is None:
        return None
    # pandas NA / NaN / NaT
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if isinstance(v, _dt.datetime):
        return v
    if isinstance(v, _dt.date):
        return v
    return v


def frame_to_pg_rows(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    """Project ``df`` to ``columns`` and coerce each cell for psycopg binding.

    A column absent from ``df`` is OMITTED from the row dict, not emitted as
    None. Emitting None would make ``market_upsert`` treat the column as
    "supplied" and overwrite an existing PG value with NULL on conflict,
    defeating its omitted-column protection. Omitting it leaves any prior PG
    value intact (e.g. a ``trading_day_ordinal`` a fuller run already wrote).
    """
    if df.empty:
        return []
    rows: list[dict] = []
    present = set(df.columns)
    records = df.to_dict(orient="records")
    for rec in records:
        rows.append({col: pg_value(rec.get(col)) for col in columns if col in present})
    return rows


@dataclass(frozen=True)
class TableSpec:
    """One parquet cache <-> market.* table mapping.

    ``parquet_name``  base filename in the cache dir (no .parquet suffix).
    ``table``         the SQLAlchemy Core Table object.
    ``pk_cols``       conflict-target columns (ON CONFLICT key).
    ``immutable_cols`` insert-only columns (first_seen / registry identity).
    ``date_col``      column to group by for per-asof_date coverage; None for
                      the registries (FK parents, always loaded fully).
    """

    parquet_name: str
    table: object
    pk_cols: tuple[str, ...]
    immutable_cols: tuple[str, ...] = field(default_factory=tuple)
    date_col: str | None = None

    @property
    def name(self) -> str:
        """Short table name (matches ``market.<name>``)."""
        return self.table.name


# Load order matters: registries first (FK parents), then the date-keyed tables.
# A list, not a dict, so backfill iterates in FK-safe order deterministically.
TABLE_SPECS: list[TableSpec] = [
    TableSpec(
        parquet_name="ticker_registry",
        table=schema.tickers,
        pk_cols=("ticker_id",),
        immutable_cols=("symbol", "first_seen"),
        date_col=None,
    ),
    TableSpec(
        parquet_name="sector_registry",
        table=schema.sectors,
        pk_cols=("sector_id",),
        immutable_cols=("sector_name", "first_seen"),
        date_col=None,
    ),
    TableSpec(
        parquet_name="thematic_universe",
        table=schema.thematic_ohlcv,
        pk_cols=("symbol", "date"),
        date_col="date",
    ),
    TableSpec(
        parquet_name="thematic_features_daily",
        table=schema.thematic_features_daily,
        pk_cols=("asof_date", "symbol"),
        date_col="asof_date",
    ),
    TableSpec(
        parquet_name="thematic_labels_daily",
        table=schema.thematic_labels_daily,
        pk_cols=("asof_date", "symbol"),
        date_col="asof_date",
    ),
    # Market-breadth (breadth-pg-write-path). LONG == parquet, so verify's
    # one-parquet->table projection works directly (design D-1/D-5).
    TableSpec(
        parquet_name="sp500_breadth_daily",
        table=schema.breadth_indicator_daily,
        pk_cols=("asof_date", "indicator"),
        date_col="asof_date",
    ),
    # SPY/benchmark OHLCV. No immutable_cols — mirrors thematic_ohlcv (which
    # also pins none): the parquet _upsert is latest-write-wins, so an
    # incremental backfill-spy re-stamps fetched_at/yfinance_version on
    # overlapping dates. PG must adopt the same values to stay byte-for-parity
    # under verify-coverage (design D-5); pinning them immutable would drift PG
    # from the parquet and fail the checksum.
    TableSpec(
        parquet_name="spy_history",
        table=schema.benchmark_ohlcv,
        pk_cols=("symbol", "date"),
        date_col="date",
    ),
]


__all__ = ["TABLE_SPECS", "TableSpec", "frame_to_pg_rows", "pg_value"]
