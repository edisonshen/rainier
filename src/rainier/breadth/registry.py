"""Stable-ID registries for tickers and sectors.

Per DESIGN-thematic-ranks-dashboard.md §5.4 + [D-015]: integer IDs are
assigned on first observation and **never reused or remapped**. The model
that learned ``ticker_id=42 → SMH`` keeps that mapping even if the YAML
universe shrinks — embeddings stay valid.

Each registry is an append-only parquet with three columns
``(<id_col>, <name_col>, first_seen)``. The next-available ID is
``max(existing) + 1`` (start at 1, never 0 — keeps 0 reserved as a
padding/unknown sentinel for embedding layers).

Implementation notes
--------------------
* Reads are cheap (whole file, < 1MB even at full universe).
* Writes are atomic (pyarrow → tmp → ``os.replace``) so a process kill mid-write
  cannot leave a half-written parquet visible.
* Functions accept explicit ``registry_path`` to keep tests isolated from
  ``data/cache/`` and to let the operator point at a non-default location.

ASCII flow:

    get_or_assign_ticker_id("XLK", asof, path)
            |
            v
        load_ticker_registry(path)   <- {} if file missing
            |
            v
        is symbol present?
           /        \\
          yes        no
           |          \\
       return id    next_id = max(ids)+1 (or 1 if empty)
                         |
                         v
                  append (id, symbol, first_seen)
                         |
                         v
                  atomic parquet write
                         |
                         v
                     return next_id
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


_TICKER_SCHEMA = pa.schema(
    [
        ("ticker_id", pa.int16()),
        ("symbol", pa.string()),
        ("first_seen", pa.date32()),
    ]
)

_SECTOR_SCHEMA = pa.schema(
    [
        ("sector_id", pa.int16()),
        ("sector_name", pa.string()),
        ("first_seen", pa.date32()),
    ]
)


def _empty_df(id_col: str, name_col: str) -> pd.DataFrame:
    return pd.DataFrame(columns=[id_col, name_col, "first_seen"])


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _read_registry(path: Path, id_col: str, name_col: str) -> pd.DataFrame:
    if not path.exists():
        return _empty_df(id_col, name_col)
    df = pd.read_parquet(path)
    # Defensive: enforce expected columns even on user-edited files.
    for col in (id_col, name_col, "first_seen"):
        if col not in df.columns:
            raise ValueError(
                f"registry parquet at {path} is missing column {col!r}; "
                f"refusing to mutate."
            )
    return df


def _write_registry_atomic(df: pd.DataFrame, path: Path, schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    try:
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _get_or_assign_id(
    name: str,
    asof: date,
    registry_path: Path,
    id_col: str,
    name_col: str,
    schema: pa.Schema,
) -> int:
    """Shared core for ticker + sector registries."""
    df = _read_registry(registry_path, id_col, name_col)
    if not df.empty:
        hits = df.loc[df[name_col] == name, id_col]
        if not hits.empty:
            return int(hits.iloc[0])

    # Next available id; start at 1 so 0 stays as embedding-padding sentinel.
    next_id = int(df[id_col].max()) + 1 if not df.empty else 1
    new_row = pd.DataFrame(
        [{id_col: next_id, name_col: name, "first_seen": asof}]
    )
    out = pd.concat([df, new_row], ignore_index=True)
    _write_registry_atomic(out, registry_path, schema)
    return next_id


# ---------------------------------------------------------------------------
# Ticker registry
# ---------------------------------------------------------------------------


def get_or_assign_ticker_id(
    symbol: str,
    asof: date,
    registry_path: Path,
) -> int:
    """Return the stable ``ticker_id`` for ``symbol``; assign if first-seen."""
    return _get_or_assign_id(
        name=symbol,
        asof=asof,
        registry_path=Path(registry_path),
        id_col="ticker_id",
        name_col="symbol",
        schema=_TICKER_SCHEMA,
    )


def load_ticker_registry(registry_path: Path) -> dict[str, int]:
    """Return the full ``{symbol: ticker_id}`` mapping (empty if file missing)."""
    df = _read_registry(Path(registry_path), "ticker_id", "symbol")
    if df.empty:
        return {}
    return {row.symbol: int(row.ticker_id) for row in df.itertuples(index=False)}


def load_ticker_first_seen(registry_path: Path) -> dict[str, date]:
    """Return ``{symbol: first_seen}`` from the ticker registry (empty if missing).

    Lets a downstream mirror (e.g. the PG dual-write) stamp the registry's
    stored first-seen date instead of the current as-of date — so enabling the
    mirror after the registry already exists preserves true provenance.
    """
    df = _read_registry(Path(registry_path), "ticker_id", "symbol")
    if df.empty:
        return {}
    return {row.symbol: row.first_seen for row in df.itertuples(index=False)}


# ---------------------------------------------------------------------------
# Sector registry
# ---------------------------------------------------------------------------


def get_or_assign_sector_id(
    sector_name: str,
    asof: date,
    registry_path: Path,
) -> int:
    """Return the stable ``sector_id`` for ``sector_name``; assign if first-seen."""
    return _get_or_assign_id(
        name=sector_name,
        asof=asof,
        registry_path=Path(registry_path),
        id_col="sector_id",
        name_col="sector_name",
        schema=_SECTOR_SCHEMA,
    )


def load_sector_registry(registry_path: Path) -> dict[str, int]:
    """Return the full ``{sector_name: sector_id}`` mapping (empty if missing)."""
    df = _read_registry(Path(registry_path), "sector_id", "sector_name")
    if df.empty:
        return {}
    return {
        row.sector_name: int(row.sector_id) for row in df.itertuples(index=False)
    }


def load_sector_first_seen(registry_path: Path) -> dict[str, date]:
    """Return ``{sector_name: first_seen}`` from the sector registry (empty if missing)."""
    df = _read_registry(Path(registry_path), "sector_id", "sector_name")
    if df.empty:
        return {}
    return {row.sector_name: row.first_seen for row in df.itertuples(index=False)}


# ---------------------------------------------------------------------------
# Bulk seeding from a universe dict
# ---------------------------------------------------------------------------


def seed_registries_from_universe(
    universe: dict[str, list[str]],
    asof: date,
    ticker_registry_path: Path,
    sector_registry_path: Path,
) -> None:
    """Register every (sector_name, symbol) pair from ``universe``.

    Idempotent: re-running with the same universe is a no-op. Order of
    assignment follows iteration order of the dict (Python 3.7+ guarantees
    insertion order), so two runs with identically-ordered YAML produce
    identical ID assignments.
    """
    for sector_name, tickers in universe.items():
        get_or_assign_sector_id(sector_name, asof, sector_registry_path)
        for sym in tickers:
            get_or_assign_ticker_id(sym, asof, ticker_registry_path)
