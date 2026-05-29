"""One-shot historical backfill of the parquet caches into ``market.*``.

Phase 3, task plan §2/§3. Reads each parquet cache and UPSERTs its full history
into the matching canonical table via the Phase 2 ``market_upsert`` helper. This
is the explicit "seed the canonical store from existing parquet" step the
operator runs once before Phase 4 reads from Postgres.

Contrast with Phase 2 dual-write (cli.py): dual-write is additive + non-fatal
(parquet is load-bearing; a broken mirror DB never aborts the pipeline). The
backfill is the OPPOSITE posture — it is an explicit DB op, so the CLI wrapper
fails loud (ClickException) when DATABASE_URL is unset, and a SQLAlchemyError
here propagates rather than being swallowed.

FK ordering (registries are parents of features):

    ticker_registry ─┐
    sector_registry ─┴─► thematic_features_daily (FK ticker_id, sector_id)
    thematic_universe   (ohlcv, no FK)
    thematic_labels_daily (no FK)

``TABLE_SPECS`` lists the tables in FK-safe load order; we iterate it directly.

Idempotency comes from ``market_upsert``'s ON CONFLICT (pk) DO UPDATE — a
re-run leaves row counts unchanged.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from rainier.db.rows import TABLE_SPECS, TableSpec, frame_to_pg_rows
from rainier.db.upsert import market_upsert


def _window_frame(
    df: pd.DataFrame, spec: TableSpec, asof_start: date | None, asof_end: date | None
) -> pd.DataFrame:
    """Restrict ``df`` to [asof_start, asof_end] on the spec's date column.

    Registries (``date_col=None``) are never windowed — they are FK parents and
    must always load fully so feature FKs resolve. For date-keyed tables we
    compare on a normalized ``date`` (parquet stores ``datetime.date``; we
    coerce defensively in case a cache was written with Timestamps).
    """
    if spec.date_col is None or (asof_start is None and asof_end is None):
        return df
    col = df[spec.date_col]
    # Normalize to datetime.date for a clean comparison regardless of dtype.
    norm = pd.to_datetime(col).dt.date
    mask = pd.Series(True, index=df.index)
    if asof_start is not None:
        mask &= norm >= asof_start
    if asof_end is not None:
        mask &= norm <= asof_end
    return df[mask]


def backfill_from_parquet(
    engine: Engine,
    cache_dir: str | Path,
    asof_start: date | None = None,
    asof_end: date | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Backfill the parquet caches in ``cache_dir`` into ``market.*``.

    Returns ``{table_name: rows_written}`` (rows that WOULD be written when
    ``dry_run``). Tables are loaded in ``TABLE_SPECS`` order (registries first
    for FK safety). A missing parquet cache is skipped (count 0) rather than
    fatal — a partial cache is a valid state to seed from.

    ``asof_start`` / ``asof_end`` window only the date-keyed tables; registries
    always load fully (FK parents). ``dry_run`` reports counts without mutating.
    """
    cache_dir = Path(cache_dir)
    counts: dict[str, int] = {}

    for spec in TABLE_SPECS:
        path = cache_dir / f"{spec.parquet_name}.parquet"
        if not path.exists():
            counts[spec.name] = 0
            continue

        df = pd.read_parquet(path)
        df = _window_frame(df, spec, asof_start, asof_end)

        table_cols = list(spec.table.columns.keys())
        rows = frame_to_pg_rows(df, table_cols)
        counts[spec.name] = len(rows)

        if dry_run or not rows:
            continue

        market_upsert(
            engine,
            spec.table,
            rows,
            list(spec.pk_cols),
            immutable_cols=list(spec.immutable_cols),
        )

    return counts


__all__ = ["backfill_from_parquet"]
