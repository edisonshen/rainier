"""One-off thematic-universe OHLCV cache backfill.

Fetches daily OHLCV for the ~94-ticker Phase A universe loaded from
``config/thematic_universe.yaml`` into
``data/cache/thematic_universe.parquet`` — the substrate the Phase B
ranking layer reads.

The script is **operator-run, not CI-run**. It hits ``yfinance`` when
invoked from the CLI. Tests inject a stub ``fetch_fn`` to keep CI offline.

Mirrors ``scripts/backfill_macro_context.py`` (vix worker). Key differences:

  * SYMBOLS are loaded from YAML at run time, not hard-coded.
  * ``adjusted_close`` is intentionally NOT stored — yfinance back-adjusts
    it for future splits/dividends. The thematic universe is ranked on
    ``close`` only (look-ahead-clean), so we skip the column entirely.

Design refs:
    docs/DESIGN-thematic-ranks-dashboard.md §5.0 [D-005] [D-011]
    docs/TASK-PLAN-thematic-backfill-fd88.md §3

Usage:
    python scripts/backfill_thematic_universe.py \\
        --start 2024-10-01 --end 2026-05-01 \\
        --out data/cache/thematic_universe.parquet
    python scripts/backfill_thematic_universe.py --dry-run  # plan only
    python scripts/backfill_thematic_universe.py --force    # new cohort
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = ROOT / "config" / "thematic_universe.yaml"
DEFAULT_OUT = Path("data/cache/thematic_universe.parquet")
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
    "fetched_at",
    "yfinance_version",
)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_symbols_from_yaml(yaml_path: Path) -> list[str]:
    """Return the flat ticker list from ``thematic_universe.yaml``.

    Preserves YAML iteration order (sector buckets, then within-bucket order).
    Section bucket names themselves are display-only and not returned.
    """
    with Path(yaml_path).open("r") as fh:
        parsed = yaml.safe_load(fh)
    out: list[str] = []
    for tickers in (parsed.get("universe") or {}).values():
        if not isinstance(tickers, list):
            raise ValueError(
                f"universe block in {yaml_path} must be sector -> list[ticker]; "
                f"got {type(tickers).__name__}"
            )
        out.extend(tickers)
    return out


# ---------------------------------------------------------------------------
# Fetch (production)
# ---------------------------------------------------------------------------


def _yfinance_fetch(
    symbols: Iterable[str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Production fetcher. Hits the network. Not used in tests.

    NOTE on end-bound semantics:
        ``yfinance.download(..., end=X)`` treats ``end`` as EXCLUSIVE. To
        match a single-sided operator contract (``--end 2026-05-01`` should
        include 2026-05-01 if it's a trading day), we bump the wire ``end``
        by +1 calendar day before the call. Same convention vix worker uses.
    """
    import yfinance as yf  # local import — keeps test collection offline-safe

    end_wire = (pd.to_datetime(end).date() + timedelta(days=1)).isoformat()

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = yf.download(
            sym,
            start=start,
            end=end_wire,
            progress=False,
            auto_adjust=False,
            actions=False,
        )
        if df is None or df.empty:
            out[sym] = pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            )
            continue

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
                # Adj Close intentionally dropped per DESIGN §5.0 — look-ahead-leaky.
            }
        )
        if "volume" not in df.columns:
            df["volume"] = 0
        df["date"] = pd.to_datetime(df["date"]).dt.date
        out[sym] = df[["date", "open", "high", "low", "close", "volume"]]
    return out


# ---------------------------------------------------------------------------
# Normalize + atomic write
# ---------------------------------------------------------------------------


def _to_normalized_frame(
    fetched: dict[str, pd.DataFrame],
    yfinance_version: str,
    fetched_at: datetime,
) -> pd.DataFrame:
    """Stack per-symbol frames into the canonical thematic_universe schema."""
    rows: list[pd.DataFrame] = []
    for sym, df in fetched.items():
        if df.empty:
            continue
        sub = df.copy()
        sub["symbol"] = sym
        sub["volume"] = sub["volume"].astype("int64")
        sub["fetched_at"] = fetched_at
        sub["yfinance_version"] = yfinance_version
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=_COLUMN_ORDER)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out = out[list(_COLUMN_ORDER)]
    return out


def _write_parquet_atomic(df: pd.DataFrame, out_path: Path) -> None:
    """pyarrow write + os.replace; never leaves a half-written file visible."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    schema = pa.schema(
        [
            ("symbol", pa.string()),
            ("date", pa.date32()),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
            ("volume", pa.int64()),
            ("fetched_at", pa.timestamp("ns", tz="UTC")),
            ("yfinance_version", pa.string()),
        ]
    )
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    try:
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _cohort_path(out_path: Path, fetched_at: datetime) -> Path:
    """``--force`` writes a sibling cohort, never overwriting in place.

    Microsecond resolution + collision-avoidance suffix so two ``--force``
    runs in the same second cannot silently overwrite each other and violate
    the ledger's ``revision_immutability: true`` claim. Same shape vix
    worker landed (codex iter-2).
    """
    stamp = fetched_at.strftime("%Y%m%d_%H%M%S_%f")
    candidate = out_path.with_name(f"{out_path.stem}_{stamp}{out_path.suffix}")
    if not candidate.exists():
        return candidate
    for n in range(1, 1000):
        bumped = out_path.with_name(
            f"{out_path.stem}_{stamp}_{n}{out_path.suffix}"
        )
        if not bumped.exists():
            return bumped
    raise RuntimeError(
        f"could not find an unused cohort path next to {out_path} after 1000 "
        "attempts; refusing to overwrite existing cohort (revision_immutability)."
    )


# ---------------------------------------------------------------------------
# Validation (same tiers vix worker landed)
# ---------------------------------------------------------------------------


DEFAULT_MIN_COVERAGE_RATIO = 0.90
"""Per-symbol minimum row count relative to ``pd.bdate_range(start, end)``.

Same heuristic vix worker landed: 0.90 absorbs ~9 NYSE holidays/year on a
~252-day calendar with margin. The thematic universe has more newer ETFs
(DXYZ listed 2024; some ARK variants thin) → operator may need
``--allow-gaps`` for those.
"""


def _validate_coverage(
    fetched: dict[str, pd.DataFrame],
    symbols: list[str],
    start: str,
    end: str,
    allow_empty: set[str],
    allow_gaps: set[str],
    min_coverage: float,
) -> None:
    """Two-tier validation.

    Simpler than the macro_context (vix) validator because the thematic
    universe has no single anchor symbol — every member is a sector/theme
    ETF. The 4-tier anchor-symbol calendar-exactness gate vix landed is
    overkill here; per-symbol row-count ratio + empty-symbol fail-fast
    catches the same outages without needing an anchor.

    Tier 1: any requested symbol with **zero** rows → ``ValueError`` unless
    in ``allow_empty``. Catches outright outages / bad tickers.

    Tier 2: any symbol with ``row_count / bdays < min_coverage`` →
    ``ValueError`` unless in ``allow_gaps``. Catches catastrophic provider
    partials (rate-limit / outage).
    """
    # Tier 1: empty symbols.
    empty_missing = [
        sym
        for sym in symbols
        if sym not in allow_empty and (sym not in fetched or fetched[sym].empty)
    ]
    if empty_missing:
        raise ValueError(
            f"backfill returned zero rows for {len(empty_missing)} symbol(s): "
            f"{sorted(empty_missing)}. Re-run when the provider is healthy or "
            f"pass --allow-empty SYMBOL[,SYMBOL...] to permit known gaps."
        )

    # Tier 2: row-count ratio.
    bdays = len(pd.bdate_range(start=start, end=end))
    if bdays == 0:
        return

    partial: list[tuple[str, int, float]] = []
    for sym in symbols:
        if sym in allow_gaps:
            continue
        df = fetched.get(sym)
        if df is None or df.empty:
            continue
        ratio = len(df) / bdays
        if ratio < min_coverage:
            partial.append((sym, len(df), ratio))

    if partial:
        details = ", ".join(
            f"{sym} ({n_rows} rows, {ratio:.0%} coverage)"
            for sym, n_rows, ratio in sorted(partial)
        )
        raise ValueError(
            f"backfill coverage below threshold {min_coverage:.0%} for "
            f"{len(partial)} symbol(s) in window {start}..{end} "
            f"({bdays} business days): {details}. Re-run when the provider "
            f"is healthy or pass --allow-gaps SYMBOL[,SYMBOL...] to permit "
            f"known-sparse tickers."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def backfill(
    symbols: Iterable[str],
    start: str,
    end: str,
    out_path: Path,
    force: bool = False,
    dry_run: bool = False,
    fetch_fn: Callable[[Iterable[str], str, str], dict[str, pd.DataFrame]]
    | None = None,
    allow_empty: Iterable[str] | None = None,
    allow_gaps: Iterable[str] | None = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE_RATIO,
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
    allow_empty_set: set[str] = set(allow_empty or ())
    allow_gaps_set: set[str] = set(allow_gaps or ())

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
    try:
        import yfinance as yf

        yfinance_version = yf.__version__
    except Exception:
        yfinance_version = "stub"

    fetched = fetch_fn(symbols, start, end)

    _validate_coverage(
        fetched=fetched,
        symbols=symbols,
        start=start,
        end=end,
        allow_empty=allow_empty_set,
        allow_gaps=allow_gaps_set,
        min_coverage=min_coverage,
    )

    df = _to_normalized_frame(fetched, yfinance_version, fetched_at)

    target = (
        _cohort_path(out_path, fetched_at)
        if (out_path.exists() and force)
        else out_path
    )
    _write_parquet_atomic(df, target)
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Backfill the thematic_universe parquet cache (~94-ticker universe "
            "loaded from config/thematic_universe.yaml)."
        )
    )
    p.add_argument(
        "--yaml",
        default=str(DEFAULT_YAML),
        help="Path to thematic universe yaml. Default: config/thematic_universe.yaml.",
    )
    p.add_argument(
        "--symbols",
        default=None,
        help=(
            "Comma-separated tickers. If omitted (default), read from --yaml. "
            "Operator only needs this for ad-hoc one-off subset backfills."
        ),
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
    p.add_argument("--allow-empty", default="")
    p.add_argument("--allow-gaps", default="")
    p.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE_RATIO,
        help=(
            "Per-symbol minimum row count as a fraction of business days in "
            "the window. Default %(default).2f absorbs ~9 NYSE holidays/year."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    if ns.symbols:
        symbols = [s.strip() for s in ns.symbols.split(",") if s.strip()]
    else:
        symbols = load_symbols_from_yaml(Path(ns.yaml))
    allow_empty = [s.strip() for s in ns.allow_empty.split(",") if s.strip()]
    allow_gaps = [s.strip() for s in ns.allow_gaps.split(",") if s.strip()]
    out_path = Path(ns.out)
    result = backfill(
        symbols=symbols,
        start=ns.start,
        end=ns.end,
        out_path=out_path,
        force=ns.force,
        dry_run=ns.dry_run,
        allow_empty=allow_empty,
        allow_gaps=allow_gaps,
        min_coverage=ns.min_coverage,
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
