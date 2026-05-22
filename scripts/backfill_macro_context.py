"""One-off macro_context cache backfill.

Fetches daily OHLCV for the Phase 0 macro panel (36 tickers spanning the
volatility complex, size-bucket benchmarks, sector SPDRs, themes,
international, commodities, fixed-income, currency, crypto, and innovation)
into ``data/cache/macro_context.parquet`` — the substrate the LLM-backtest
L3 evaluator reads via ``rainier.research.macro_context_loader``.

The script is **operator-run, not CI-run**. It hits ``yfinance`` once when
invoked from the CLI. Tests inject a stub ``fetch_fn`` to keep CI offline.

Design references:
    docs/DESIGN-qu100-llm-backtest.md [D-026] [D-014] [D-011]
    docs/TASK-PLAN-vix-sector-backfill-d05f.md (canonical scope)

Usage:
    python scripts/backfill_macro_context.py \\
        --start 2024-10-01 --end 2026-05-01 \\
        --out data/cache/macro_context.parquet
    python scripts/backfill_macro_context.py --dry-run        # plan only
    python scripts/backfill_macro_context.py --force          # new cohort

ASCII layout — atomic write:
    out_path.tmp  <- pyarrow writes here first
        |  os.replace() (POSIX atomic on same filesystem)
        v
    out_path      <- visible to readers; .tmp never lingers
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Phase 0 final symbol list (resolved by coord 431c2abf on 2026-05-22).
#
# Decisions baked in:
#   - Ambiguous tickers (NASA/BAI/LABX): skipped — operator can add later as
#     a 1-line SYMBOLS edit + ledger entry.
#   - Proposed additions (UUP/IEF/TIP/ARKK): included.
#   - etf_persistent_gainer signal + broader universe: deferred to Slice 1.
#
# Adding/removing a symbol here REQUIRES a matching update to
# src/rainier/research/data_availability.yaml in the same commit, or the
# tests/test_data_availability_yaml.py schema test will fail. Lockstep is
# the look-ahead control surface.
# ---------------------------------------------------------------------------
SYMBOLS: tuple[str, ...] = (
    # Volatility / risk regime (4)
    "VIX",
    "VIX9D",
    "VIX3M",
    "UVXY",
    # Benchmarks / size buckets (4)
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    # Sector SPDR ETFs (11)
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",
    # Theme / sub-sector (3)
    "SMH",
    "XBI",
    "KRE",
    # International (3)
    "EEM",
    "FXI",
    "KWEB",
    # Commodities (3)
    "USO",
    "GDX",
    "SLV",
    # Fixed income / rates (4)
    "TLT",
    "HYG",
    "IEF",
    "TIP",
    # Currency (1)
    "UUP",
    # Crypto (2)
    "IBIT",
    "IREN",
    # Innovation (1)
    "ARKK",
)

# yfinance ticker map: index symbols need a ^-prefix on the wire, but the
# stored `symbol` column normalizes them off. Anything not in this map maps
# to itself.
_YF_TICKER_MAP: dict[str, str] = {
    "VIX": "^VIX",
    "VIX9D": "^VIX9D",
    "VIX3M": "^VIX3M",
}

DEFAULT_OUT = Path("data/cache/macro_context.parquet")
DEFAULT_START = "2024-10-01"
DEFAULT_END = "2026-05-01"

# Parquet column order — kept stable so the on-disk schema is deterministic.
_COLUMN_ORDER: tuple[str, ...] = (
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


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _yfinance_fetch(
    symbols: Iterable[str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Production fetcher. Hits the network. Not used in tests."""
    import yfinance as yf  # local import — keeps test collection offline-safe

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        wire = _YF_TICKER_MAP.get(sym, sym)
        # auto_adjust=False keeps `Close` and `Adj Close` distinct.
        df = yf.download(
            wire,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            actions=False,
        )
        if df is None or df.empty:
            out[sym] = pd.DataFrame(
                columns=[
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "adjusted_close",
                ]
            )
            continue

        # yfinance returns a MultiIndex on columns when group_by isn't set on
        # a single ticker call; flatten it.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Adj Close": "adjusted_close",
            }
        )
        # Indices have no volume; fill 0 (int64).
        if "volume" not in df.columns:
            df["volume"] = 0
        # adjusted_close falls back to close for indices when Adj Close is absent.
        if "adjusted_close" not in df.columns:
            df["adjusted_close"] = df["close"]
        df["date"] = pd.to_datetime(df["date"]).dt.date
        out[sym] = df[
            ["date", "open", "high", "low", "close", "volume", "adjusted_close"]
        ]
    return out


# ---------------------------------------------------------------------------
# Backfill core
# ---------------------------------------------------------------------------


def _to_normalized_frame(
    fetched: dict[str, pd.DataFrame], yfinance_version: str, fetched_at: datetime
) -> pd.DataFrame:
    """Stack per-symbol frames into the canonical macro_context schema.

    Symbol column is normalized (no ^-prefix). Sort key is (symbol, date)
    ascending. Volume is int64, even for indices.
    """
    rows: list[pd.DataFrame] = []
    for sym, df in fetched.items():
        if df.empty:
            continue
        # Strip ^-prefix defensively — even if upstream sent ^VIX, store VIX.
        clean_sym = sym.lstrip("^")
        sub = df.copy()
        sub["symbol"] = clean_sym
        sub["volume"] = sub["volume"].astype("int64")
        sub["fetched_at"] = fetched_at
        sub["yfinance_version"] = yfinance_version
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=_COLUMN_ORDER)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    # Enforce column order — pyarrow round-trip otherwise reorders by dict iter.
    out = out[list(_COLUMN_ORDER)]
    return out


def _write_parquet_atomic(df: pd.DataFrame, out_path: Path) -> None:
    """pyarrow write + os.replace; never leaves a half-written file visible."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    # Force a stable schema so byte-identical inputs produce byte-identical
    # parquet (modulo embedded fetched_at).
    schema = pa.schema(
        [
            ("symbol", pa.string()),
            ("date", pa.date32()),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
            ("volume", pa.int64()),
            ("adjusted_close", pa.float64()),
            ("fetched_at", pa.timestamp("ns", tz="UTC")),
            ("yfinance_version", pa.string()),
        ]
    )
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    try:
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, out_path)
    finally:
        # If write_table raised after creating tmp, scrub it.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _cohort_path(out_path: Path, fetched_at: datetime) -> Path:
    """`--force` writes a sibling cohort file, not in-place."""
    stamp = fetched_at.strftime("%Y%m%d_%H%M%S")
    return out_path.with_name(f"{out_path.stem}_{stamp}{out_path.suffix}")


def backfill(
    symbols: Iterable[str],
    start: str,
    end: str,
    out_path: Path,
    force: bool = False,
    dry_run: bool = False,
    fetch_fn: Callable[[Iterable[str], str, str], dict[str, pd.DataFrame]]
    | None = None,
) -> Path | dict[str, object]:
    """Run the backfill.

    Returns the parquet path actually written (a sibling cohort path when
    ``force=True`` and the destination already exists). In ``dry_run`` mode
    returns the planned ticker × date matrix without touching the network or
    the filesystem.
    """
    symbols = list(symbols)
    fetch_fn = fetch_fn or _yfinance_fetch
    out_path = Path(out_path)

    if dry_run:
        return {
            "symbols": symbols,
            "start": start,
            "end": end,
            "planned_out": str(out_path),
        }

    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} already exists. Pass --force to write a new cohort."
        )

    fetched_at = datetime.now(tz=timezone.utc)
    # Import lazily so tests with fetch_fn injection never need yfinance.
    try:
        import yfinance as yf

        yfinance_version = yf.__version__
    except Exception:
        yfinance_version = "stub"

    fetched = fetch_fn(symbols, start, end)
    df = _to_normalized_frame(fetched, yfinance_version, fetched_at)

    target = _cohort_path(out_path, fetched_at) if (out_path.exists() and force) else out_path
    _write_parquet_atomic(df, target)
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Backfill the macro_context parquet cache (36-ticker Phase 0 panel)."
        )
    )
    p.add_argument(
        "--symbols",
        default=",".join(SYMBOLS),
        help="Comma-separated tickers (defaults to the Phase 0 list of 36).",
    )
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument(
        "--force",
        action="store_true",
        help="When the cache exists, write a timestamped sibling cohort.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned ticker × date matrix; do not fetch or write.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    symbols = [s.strip() for s in ns.symbols.split(",") if s.strip()]
    out_path = Path(ns.out)
    result = backfill(
        symbols=symbols,
        start=ns.start,
        end=ns.end,
        out_path=out_path,
        force=ns.force,
        dry_run=ns.dry_run,
    )
    if ns.dry_run:
        plan = result  # type: ignore[assignment]
        print(
            f"DRY-RUN: would fetch {len(symbols)} symbols "
            f"{ns.start}..{ns.end} -> {plan['planned_out']}"  # type: ignore[index]
        )
        for sym in symbols:
            print(f"  {sym}")
    else:
        path = result  # type: ignore[assignment]
        df = pd.read_parquet(path)
        per_sym = df.groupby("symbol").size().to_dict()
        print(f"wrote {len(df)} rows -> {path}")
        for sym in sorted(per_sym):
            print(f"  {sym:8s}  {per_sym[sym]:4d} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
