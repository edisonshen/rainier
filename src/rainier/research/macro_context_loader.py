"""Read-only loader for ``data/cache/macro_context.parquet``.

Pure function of ``(symbol, date_range)`` -> ``pd.DataFrame``. Deterministic
sort key ``(symbol, date)`` ascending; identical inputs return byte-identical
DataFrames so the L3 evaluator's ``compute_input_hash`` stays stable.

Design references:
    docs/DESIGN-qu100-llm-backtest.md [D-026] (no live yfinance in backtest)
    docs/DESIGN-qu100-llm-backtest.md [D-014] (look-ahead control via ledger)
    docs/TASK-PLAN-vix-sector-backfill-d05f.md §6 (acceptance)

Look-ahead control:
    By default the loader returns ONLY the ledger-registered as-of fields
    (open/high/low/close/volume) plus the identity columns (symbol/date).
    yfinance's ``adjusted_close`` is back-adjusted for future splits and
    dividends — including it under an end_of_trading_day rule would leak
    look-ahead into the L3 evaluator. Callers that explicitly need the
    look-ahead-leaky columns must opt in via ``include_unregistered=True``;
    those callers must NOT participate in L3 input-hash computation.
    Per codex iter-7.

Missing-data policy:
    Symbols not present in the cache return an *empty* DataFrame with the
    production schema columns. Gaps inside a symbol's coverage are left
    absent — the ledger is the source of truth for which (symbol, date)
    pairs are observable.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data/cache/macro_context.parquet")

# Schema mirrored from scripts/backfill_macro_context.py. Kept in lockstep
# with the parquet writer; the test suite enforces both via parquet round-trip.
_SCHEMA_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
    "fetched_at",
    "yfinance_version",
)

# Columns the L3 evaluator may read AS-OF end_of_trading_day. Mirrors the
# ledger entries' `fields:` list at src/rainier/research/data_availability.yaml.
# `adjusted_close` is intentionally excluded (back-adjusted for future actions).
# `fetched_at` / `yfinance_version` are provenance, not market data; also
# excluded by default so they don't enter compute_input_hash.
_LEDGER_REGISTERED_FIELDS: tuple[str, ...] = (
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def _empty_frame(
    include_unregistered: bool = False,
) -> pd.DataFrame:
    cols = _SCHEMA_COLUMNS if include_unregistered else _LEDGER_REGISTERED_FIELDS
    return pd.DataFrame({col: pd.Series(dtype="object") for col in cols})


def _project_ledger_columns(
    df: pd.DataFrame, include_unregistered: bool
) -> pd.DataFrame:
    """Restrict to ledger-registered columns unless explicitly opted out."""
    if include_unregistered:
        return df
    keep = [c for c in _LEDGER_REGISTERED_FIELDS if c in df.columns]
    return df[keep]


def _coerce_date(value: str | date | datetime | pd.Timestamp) -> date:
    # ``datetime`` is a subclass of ``date``; treat it as a Timestamp-like and
    # truncate to date (see codex iter-5). Pure ``date`` passes through.
    if isinstance(value, datetime) or isinstance(value, pd.Timestamp):
        return pd.to_datetime(value).date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def read_all(
    path: str | Path | None = None,
    include_unregistered: bool = False,
) -> pd.DataFrame:
    """Load every row from the cache, sorted by ``(symbol, date)`` ascending.

    By default returns only ledger-registered fields (see module docstring on
    look-ahead control). Pass ``include_unregistered=True`` to receive the
    full parquet schema including ``adjusted_close``, ``fetched_at``, and
    ``yfinance_version``. Such callers MUST NOT feed the result into L3
    look-ahead-controlled hashing — those columns are not as-of observable.
    """
    cache = Path(path) if path is not None else DEFAULT_PATH
    if not cache.exists():
        raise FileNotFoundError(cache)
    df = pd.read_parquet(cache)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return _project_ledger_columns(df, include_unregistered)


def read_range(
    symbol: str,
    start: str | date | datetime | pd.Timestamp,
    end: str | date | datetime | pd.Timestamp,
    path: str | Path | None = None,
    include_unregistered: bool = False,
) -> pd.DataFrame:
    """Rows for ``symbol`` with ``start <= date <= end``.

    Bounds are **inclusive on both sides**. Missing symbol yields an empty
    DataFrame (with the production schema) rather than raising — gaps are
    explicit in the ledger, not in the loader's exception surface.

    By default returns only ledger-registered fields. See ``read_all`` for
    the ``include_unregistered`` opt-in semantics.
    """
    cache = Path(path) if path is not None else DEFAULT_PATH
    if not cache.exists():
        raise FileNotFoundError(cache)

    start_d = _coerce_date(start)
    end_d = _coerce_date(end)

    df = pd.read_parquet(cache)
    sub = df.loc[df["symbol"] == symbol]
    if sub.empty:
        return _empty_frame(include_unregistered=include_unregistered)

    sub = sub.loc[(sub["date"] >= start_d) & (sub["date"] <= end_d)]
    sub = sub.sort_values(["symbol", "date"]).reset_index(drop=True)
    return _project_ledger_columns(sub, include_unregistered)
