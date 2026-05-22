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
from datetime import datetime, timedelta, timezone
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
    """Production fetcher. Hits the network. Not used in tests.

    NOTE on end-bound semantics:
        ``yfinance.download(..., end=X)`` treats ``end`` as EXCLUSIVE while the
        loader (``macro_context_loader.read_range``) treats it as INCLUSIVE on
        both sides. To keep the operator-facing contract single-sided
        (``--end 2026-05-01`` should include 2026-05-01 if it's a trading day),
        we bump the wire ``end`` by +1 calendar day before the call.
    """
    import yfinance as yf  # local import — keeps test collection offline-safe

    # Inclusive-end: yfinance is exclusive, so add 1 day to match loader semantics.
    end_wire = (pd.to_datetime(end).date() + timedelta(days=1)).isoformat()

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        wire = _YF_TICKER_MAP.get(sym, sym)
        # auto_adjust=False keeps `Close` and `Adj Close` distinct.
        df = yf.download(
            wire,
            start=start,
            end=end_wire,
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
    """`--force` writes a sibling cohort file, not in-place.

    Uses microsecond resolution + a collision-avoidance suffix so two
    ``--force`` runs in the same second (automation / fast operator retry)
    cannot silently overwrite each other and violate the ledger's
    ``revision_immutability: true`` claim. Per codex iter-2.
    """
    stamp = fetched_at.strftime("%Y%m%d_%H%M%S_%f")
    candidate = out_path.with_name(f"{out_path.stem}_{stamp}{out_path.suffix}")
    if not candidate.exists():
        return candidate
    # Microsecond collision (rare but possible in tight test loops). Append a
    # deterministic suffix until we land on a free name. We refuse to
    # overwrite an existing cohort; immutability is the load-bearing claim.
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


DEFAULT_MIN_COVERAGE_RATIO = 0.90
"""Per-symbol minimum row count relative to ``pd.bdate_range(start, end)``.

NYSE has ~9 holidays/year + early closes; 0.90 absorbs ~37 days/year of legal
gaps in a 252-day calendar with margin. Anything below this signals a
provider outage / rate-limit partial / publication gap that should NOT be
silently committed to a cache whose ledger claims observability. Tunable per
backfill via ``min_coverage`` kwarg or ``--min-coverage`` CLI flag.
"""

DEFAULT_ANCHOR_SYMBOL = "SPY"
"""Benchmark whose date-set defines the expected NYSE trading calendar.

Phase 0 always includes SPY. Using its actual fetched date-set as the
ground-truth calendar avoids depending on ``pandas_market_calendars`` /
``exchange_calendars`` while still detecting "92% partial" scenarios that
the row-count ratio (tier 2) lets through. Any non-anchor symbol missing
>``DEFAULT_MAX_ANCHOR_GAP`` of the anchor's dates is flagged.
"""

DEFAULT_MAX_ANCHOR_GAP_RATIO = 0.02
"""Max fraction of anchor dates a symbol may be missing before tier-3 fails.

2% of 400 trading days = 8 days. This catches the codex iter-5 scenario
(92% partial coverage = 8% missing = above the 2% bar) while tolerating
single-day index publication anomalies. Tune via ``--max-anchor-gap``.
"""

DEFAULT_MIN_ANCHOR_COVERAGE = 0.95
"""Minimum row_count / bdays for the anchor symbol itself.

Tighter than ``DEFAULT_MIN_COVERAGE_RATIO`` (0.90) because the anchor is the
ground truth for tier-3 calendar-exactness. If the anchor is itself only
92% complete (common-mode provider truncation — every symbol gets clipped at
the same date), tier-2 passes at 90% and tier-3 sees no per-symbol gaps
because all symbols share the same truncated calendar. Validating the anchor
against ``pd.bdate_range`` at 95% catches that without needing
``pandas_market_calendars``. NYSE has ~9 holidays/year = 3.6% of 252 trading
days; 0.95 gives ~5pp of margin. Per codex iter-6.
"""


def _validate_coverage(
    fetched: dict[str, pd.DataFrame],
    symbols: list[str],
    start: str,
    end: str,
    allow_empty: set[str],
    allow_gaps: set[str],
    min_coverage: float,
    anchor_symbol: str = DEFAULT_ANCHOR_SYMBOL,
    max_anchor_gap: float = DEFAULT_MAX_ANCHOR_GAP_RATIO,
    min_anchor_coverage: float = DEFAULT_MIN_ANCHOR_COVERAGE,
) -> None:
    """Fail-fast on empty or partial per-symbol coverage.

    Four-tier validation:
      1. Any requested symbol with **zero** rows → ``ValueError`` (unless in
         ``allow_empty``). Catches outright outages / bad tickers.
      2. Any symbol with row_count / bdays < ``min_coverage`` → ``ValueError``
         (unless in ``allow_gaps``). Catches catastrophic provider partials
         (rate-limit / outage). ``bdate_range`` is a conservative upper bound
         on NYSE trading days, so the threshold is biased toward
         false-negatives.
      2.5 (anchor sanity, codex iter-6): the anchor symbol's own coverage
         must clear ``min_anchor_coverage`` (default 0.95) vs ``bdate_range``.
         Without this, a common-mode truncation (every symbol clipped at the
         same date) lets tier-2 pass at 92% AND tier-3 see no per-symbol gaps
         because all symbols share the same broken calendar. The anchor
         carries the ground truth, so its own coverage must be tighter than
         everyone else's.
      3. **NYSE-calendar exactness** (codex iter-5): when the anchor symbol
         (default ``SPY``) is present, its fetched date-set defines the
         ground-truth trading calendar. Any non-anchor symbol missing more
         than ``max_anchor_gap`` of the anchor's dates → ``ValueError``
         (unless in ``allow_gaps``). This catches the "92% partial" scenario
         that tier-2's 90% ratio lets through.

    The ledger declares end_of_trading_day observability for the full window;
    partial coverage would silently mislead the L3 evaluator.
    """
    # Tier 1: empty symbols.
    empty_missing = [
        sym
        for sym in symbols
        if sym not in allow_empty
        and (sym not in fetched or fetched[sym].empty)
    ]
    if empty_missing:
        raise ValueError(
            f"backfill returned zero rows for {len(empty_missing)} symbol(s): "
            f"{sorted(empty_missing)}. Re-run when the provider is healthy or "
            f"pass --allow-empty SYMBOL[,SYMBOL...] to permit known gaps."
        )

    # Tier 2: partial-coverage symbols (row-count ratio).
    bdays = len(pd.bdate_range(start=start, end=end))
    if bdays == 0:
        # Degenerate window — coverage check is meaningless, skip.
        return

    partial: list[tuple[str, int, float]] = []
    for sym in symbols:
        # allow_empty exempts ONLY tier-1 (zero rows). Partial coverage still
        # requires explicit allow_gaps opt-in. Codex iter-5.
        if sym in allow_gaps:
            continue
        df = fetched.get(sym)
        if df is None or df.empty:
            # Symbol returned zero rows AND must be in allow_empty (else
            # tier 1 would have raised). Don't double-flag.
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

    # Tier 2.5: anchor sanity. The anchor's own coverage must clear a tighter
    # threshold than tier-2's 90% so a common-mode truncation can't make the
    # anchor's clipped date-set the new "ground truth." Skip if anchor not in
    # the requested symbols.
    anchor_df = fetched.get(anchor_symbol)
    if anchor_df is None or anchor_df.empty:
        return  # operator omitted the anchor; tier-3 also skips below
    anchor_ratio = len(anchor_df) / bdays
    if anchor_ratio < min_anchor_coverage and anchor_symbol not in allow_gaps:
        raise ValueError(
            f"backfill anchor {anchor_symbol!r} coverage "
            f"{anchor_ratio:.1%} below {min_anchor_coverage:.0%} threshold "
            f"({len(anchor_df)} rows / {bdays} business days in window "
            f"{start}..{end}). The anchor is the ground truth for tier-3 "
            f"calendar validation, so its own coverage must be tighter than "
            f"min_coverage. Re-run when the provider is healthy or lower "
            f"--min-anchor-coverage (NOT recommended without auditing the "
            f"resulting cache)."
        )

    # Tier 3: NYSE-calendar exactness via anchor-symbol date-set comparison.
    # Only runs when the anchor is among the fetched symbols (it always is
    # for Phase 0 since SPY is required). If the operator passes a custom
    # symbol list without SPY, tier 3 is skipped silently.
    anchor_dates: set = set(anchor_df["date"].tolist())
    if not anchor_dates:
        return

    calendar_gaps: list[tuple[str, int, float]] = []
    for sym in symbols:
        # allow_empty exempts ONLY tier-1 (zero rows); partial calendar gaps
        # still require explicit allow_gaps opt-in. Codex iter-5.
        if sym == anchor_symbol or sym in allow_gaps:
            continue
        df = fetched.get(sym)
        if df is None or df.empty:
            # Allowed-empty symbols (their tier-1 was exempted upstream).
            continue
        sym_dates: set = set(df["date"].tolist())
        missing_dates = anchor_dates - sym_dates
        gap_ratio = len(missing_dates) / len(anchor_dates)
        if gap_ratio > max_anchor_gap:
            calendar_gaps.append((sym, len(missing_dates), gap_ratio))

    if calendar_gaps:
        details = ", ".join(
            f"{sym} (missing {n_missing}/{len(anchor_dates)} = {ratio:.1%})"
            for sym, n_missing, ratio in sorted(calendar_gaps)
        )
        raise ValueError(
            f"backfill calendar-gap exceeds {max_anchor_gap:.1%} (vs "
            f"{anchor_symbol} anchor with {len(anchor_dates)} dates) for "
            f"{len(calendar_gaps)} symbol(s): {details}. Re-run when the "
            f"provider is healthy or pass --allow-gaps SYMBOL[,SYMBOL...] "
            f"to permit known-sparse tickers."
        )


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
    anchor_symbol: str = DEFAULT_ANCHOR_SYMBOL,
    max_anchor_gap: float = DEFAULT_MAX_ANCHOR_GAP_RATIO,
    min_anchor_coverage: float = DEFAULT_MIN_ANCHOR_COVERAGE,
) -> Path | dict[str, object]:
    """Run the backfill.

    Returns the parquet path actually written (a sibling cohort path when
    ``force=True`` and the destination already exists). In ``dry_run`` mode
    returns the planned ticker × date matrix without touching the network or
    the filesystem.

    Coverage validation (see ``_validate_coverage`` for the two-tier rule):
      - Empty symbols (zero rows) raise ``ValueError`` unless in ``allow_empty``.
      - Partial-coverage symbols (< ``min_coverage`` × business days) raise
        ``ValueError`` unless in ``allow_gaps``.

    Silent partial backfills produce a cache the L3 evaluator cannot
    distinguish from "symbol observable, no data in this window" — fail fast
    so the operator re-runs.
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
    # Import lazily so tests with fetch_fn injection never need yfinance.
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
        anchor_symbol=anchor_symbol,
        max_anchor_gap=max_anchor_gap,
        min_anchor_coverage=min_anchor_coverage,
    )

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
    p.add_argument(
        "--allow-empty",
        default="",
        help=(
            "Comma-separated allowlist of symbols permitted to return zero "
            "rows (e.g., known delistings). Without this, any empty symbol "
            "causes backfill to abort before writing."
        ),
    )
    p.add_argument(
        "--allow-gaps",
        default="",
        help=(
            "Comma-separated allowlist of symbols permitted to return partial "
            "coverage (below --min-coverage). Use for tickers with known "
            "publication gaps; the cache will silently include only the "
            "available rows."
        ),
    )
    p.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE_RATIO,
        help=(
            "Per-symbol minimum row count as a fraction of business days in "
            "the window. Default %(default).2f absorbs ~9 NYSE holidays/year. "
            "Lower this for short windows or noisy providers."
        ),
    )
    p.add_argument(
        "--anchor-symbol",
        default=DEFAULT_ANCHOR_SYMBOL,
        help=(
            "Symbol whose fetched date-set defines the NYSE trading calendar "
            "for tier-3 (calendar-exact) validation. Default %(default)s. Must "
            "be among --symbols or tier-3 is skipped silently."
        ),
    )
    p.add_argument(
        "--max-anchor-gap",
        type=float,
        default=DEFAULT_MAX_ANCHOR_GAP_RATIO,
        help=(
            "Max fraction of anchor dates a non-anchor symbol may be missing "
            "before tier-3 aborts the backfill. Default %(default).2f "
            "(~8 days/400). Use --allow-gaps to whitelist known-sparse tickers."
        ),
    )
    p.add_argument(
        "--min-anchor-coverage",
        type=float,
        default=DEFAULT_MIN_ANCHOR_COVERAGE,
        help=(
            "Tier-2.5: the anchor symbol's own row_count / business-days "
            "ratio must clear this threshold. Default %(default).2f catches "
            "common-mode truncations that would otherwise pass tier-2 and "
            "tier-3 simultaneously (both anchored on a clipped calendar). "
            "Tightening below 0.95 risks false-negatives on legit NYSE holidays."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    symbols = [s.strip() for s in ns.symbols.split(",") if s.strip()]
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
        anchor_symbol=ns.anchor_symbol,
        max_anchor_gap=ns.max_anchor_gap,
        min_anchor_coverage=ns.min_anchor_coverage,
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
