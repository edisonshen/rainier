"""S&P 500 OHLCV backfill via yfinance.

Two operating modes, both atomic-write:

  * **One-shot backfill** (``incremental=False``, default): fetch from
    ``since`` → today for every symbol, write the long-format parquet.
  * **Incremental refresh** (``incremental=True``): fetch the last 5
    calendar days for every symbol, upsert into the existing parquet on
    ``(symbol, date)``. The 5-day window is the self-healing margin in
    case yesterday's cron missed.

Chunked yfinance loop with one transient-failure retry per chunk:

      +-------------+
      |  symbols    |   chunk_size=25
      +------+------+   (operator-tunable; mid-range of plan's 20-50)
             |
             v                                      yes
        +----+----+    +-----------+    +---------+----+
        | chunk i |--->| fetch_fn  |--->| ok rows |    |
        +---------+    +-----+-----+    +---------+    |
                            | RateLimitError              merge into per-symbol dict
                            v                              accumulator
                       +----+----+     +---------+         |
                       | retry?  |---->| fetch_fn|---------+
                       +----+----+ no  +---------+
                            |
                            v raise (caller sees the original)

Symbol-name handling: Wikipedia spelling lives in the YAML and in the
output parquet (``BRK.B``, ``BF.B``). yfinance wants dashes
(``BRK-B``, ``BF-B``) — the translation happens ONLY inside the prod
fetcher (``_yfinance_fetch``); the cached parquet keeps the dotted form
so YAML and parquet share a join key.

Atomic write: write the merged frame to ``<out>.tmp``, then
``os.replace`` onto the destination. A mid-fetch crash leaves the
previous parquet (if any) intact and the ``.tmp`` cleaned up.

Design refs:
    docs/DESIGN-market-breadth-webpage.md §3.2
    docs/TASK-PLAN-sp500-ohlcv-backfill-47b8.md §Acceptance / §Tests
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Public constants / errors
# ---------------------------------------------------------------------------


DEFAULT_OUT = Path("data/cache/sp500_universe.parquet")
DEFAULT_CHUNK_SIZE = 25
DEFAULT_RETRIES = 1
INCREMENTAL_WINDOW_DAYS = 5

# Fail-loud threshold for the cron path: if fewer than this fraction of
# requested symbols returned ANY rows, abort the write so the operator
# gets a non-zero exit + Discord alert instead of a silently incomplete
# parquet. 0.95 means "tolerate up to ~25 of 503 S&P-500 tickers being
# missing in a single run" — well above the steady-state delisting churn
# but below the "yfinance is having a bad day" mass-failure case.
DEFAULT_MIN_COVERAGE = 0.95

# Canonical column order. Kept identical to thematic_universe.parquet so
# downstream breadth-indicator code can read both with the same loader.
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

_SCHEMA = pa.schema(
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


class RateLimitError(RuntimeError):
    """yfinance rate-limit (transient). Retried once by ``backfill``.

    The prod fetcher (``_yfinance_fetch``) does NOT inspect yfinance's
    exception hierarchy directly — different yfinance versions raise
    different types. Tests inject this synthetic error via ``fetch_fn`` to
    exercise the retry path. Operators can also raise it from a custom
    fetcher.
    """


class CoverageError(RuntimeError):
    """Raised when too few requested symbols returned data — the run is
    aborted so the operator gets a Discord alert via the cron-wrapper's
    non-zero-exit path instead of a silently incomplete parquet.

    Hold the missing-symbol list on the exception so the alert body
    surfaces exactly which tickers failed.
    """

    def __init__(self, missing: list[str], requested: int, threshold: float) -> None:
        self.missing = missing
        self.requested = requested
        self.threshold = threshold
        super().__init__(
            f"yfinance coverage below threshold: {len(missing)}/{requested} "
            f"symbols returned no rows (need ≥{threshold:.0%} coverage). "
            f"First 10 missing: {missing[:10]}"
        )


FetchFn = Callable[
    [list[str], str, str], dict[str, pd.DataFrame]
]
"""Signature: ``(symbols, start_iso, end_iso) -> {symbol: per_symbol_df}``.

Each value DataFrame has columns ``date, open, high, low, close, volume``.
Empty value means "yfinance returned no rows" (delisted, bad ticker, etc).
"""


# ---------------------------------------------------------------------------
# Symbol translation
# ---------------------------------------------------------------------------


def _to_yfinance_symbol(sym: str) -> str:
    """``BRK.B`` -> ``BRK-B``; everything else unchanged.

    yfinance encodes share-class suffixes with a dash. The YAML uses
    Wikipedia spelling (dot) so the breadth-webpage join keys can read
    cleanly from either side.
    """
    return sym.replace(".", "-")


# ---------------------------------------------------------------------------
# Prod fetcher (real network)
# ---------------------------------------------------------------------------


def _yfinance_fetch(
    symbols: list[str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """Hit yfinance.download in chunked, multi-symbol mode.

    End-bound semantics: yfinance's ``end=`` is EXCLUSIVE. The caller
    passes an INCLUSIVE end-date string (``2024-03-15`` means "include
    the 2024-03-15 bar"); this fetcher bumps it by +1 calendar day on
    the wire so today's bar is captured when the operator runs
    ``--since 2020-01-01`` on a trading day. Matches the convention in
    ``scripts/backfill_thematic_universe.py`` (line 110).

    Returns a per-symbol map keyed by the original (dotted) ticker so the
    caller doesn't have to undo the dash translation.
    """
    import yfinance as yf  # local import → tests stay offline-safe

    if not symbols:
        return {}

    yf_symbols = [_to_yfinance_symbol(s) for s in symbols]
    yf_to_orig = dict(zip(yf_symbols, symbols))

    # Inclusive→exclusive: bump end by +1 calendar day so the user-facing
    # ``end=2024-03-15`` includes the 2024-03-15 bar on the wire.
    end_wire = (pd.to_datetime(end).date() + timedelta(days=1)).isoformat()

    df = yf.download(
        yf_symbols,
        start=start,
        end=end_wire,
        progress=False,
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        threads=True,
    )

    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        # Heuristic: yfinance returns an empty/None frame for the whole
        # chunk on rate-limits, 429s, transient network blips, and weekend
        # runs against an empty window. The first three are RETRYABLE; the
        # fourth isn't (no data to return is correct). Distinguish by the
        # requested window: a weekday window of 2+ days that returns zero
        # rows for an entire S&P 500 chunk is almost certainly rate-
        # limited (yfinance records per-ticker YFRateLimitError but bulk
        # `download` swallows them into empty rows). Surface as
        # RateLimitError so the caller's single retry actually fires.
        # Single-day or pure-weekend windows are kept as legitimate empty.
        try:
            start_d = pd.to_datetime(start).date()
            end_d = pd.to_datetime(end_wire).date()
            window_days = (end_d - start_d).days
            window_has_weekday = any(
                (start_d + timedelta(days=i)).weekday() < 5
                for i in range(max(window_days, 0))
            )
        except Exception:  # noqa: BLE001 — fall through to empty-frame return
            window_has_weekday = False
        if window_has_weekday and window_days >= 2:
            raise RateLimitError(
                f"yfinance returned no rows for {len(symbols)} symbols across "
                f"{start}→{end} — treating as transient (likely rate-limit). "
                f"First 5 symbols: {symbols[:5]}"
            )
        return {sym: pd.DataFrame() for sym in symbols}

    # Multi-symbol mode returns columns = MultiIndex[(ticker, field)].
    # Single-symbol mode returns flat columns. Normalize both.
    if isinstance(df.columns, pd.MultiIndex):
        top_level = df.columns.get_level_values(0).unique()
        for yf_sym in top_level:
            orig = yf_to_orig.get(yf_sym, yf_sym)
            sub = df[yf_sym].dropna(how="all").reset_index().rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            if "date" in sub.columns:
                sub["date"] = pd.to_datetime(sub["date"]).dt.date
            cols = [c for c in ("date", "open", "high", "low", "close", "volume") if c in sub.columns]
            out[orig] = sub[cols] if cols else pd.DataFrame()
    else:
        # Single-symbol path.
        sub = df.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        if "date" in sub.columns:
            sub["date"] = pd.to_datetime(sub["date"]).dt.date
        sole = symbols[0]
        cols = [c for c in ("date", "open", "high", "low", "close", "volume") if c in sub.columns]
        out[sole] = sub[cols] if cols else pd.DataFrame()

    # Backfill missing symbols (yfinance silently drops bad tickers).
    for sym in symbols:
        out.setdefault(sym, pd.DataFrame())
    return out


# ---------------------------------------------------------------------------
# Stack + atomic write
# ---------------------------------------------------------------------------


def _to_long_frame(
    fetched: dict[str, pd.DataFrame],
    yfinance_version: str,
    fetched_at: datetime,
) -> pd.DataFrame:
    """Stack per-symbol frames into the canonical long-format schema.

    Empty per-symbol frames are dropped silently; the caller decides whether
    that's an error (the prod cron will warn via downstream coverage checks).
    """
    rows: list[pd.DataFrame] = []
    for sym, df in fetched.items():
        if df is None or df.empty:
            continue
        sub = df.copy()
        sub["symbol"] = sym
        sub["volume"] = sub["volume"].astype("int64")
        sub["fetched_at"] = fetched_at
        sub["yfinance_version"] = yfinance_version
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=list(_COLUMN_ORDER))

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out[list(_COLUMN_ORDER)]


def _write_parquet_atomic(df: pd.DataFrame, out_path: Path) -> None:
    """pyarrow write to ``<path>.tmp`` then ``os.replace`` — no half-written file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    table = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)
    try:
        pq.write_table(table, tmp, compression="snappy")
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Chunking + retry
# ---------------------------------------------------------------------------


def _chunked(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _fetch_with_retry(
    fetch_fn: FetchFn,
    chunk: list[str],
    start: str,
    end: str,
    retries: int,
    retry_sleep: Callable[[int], None],
) -> dict[str, pd.DataFrame]:
    """Call ``fetch_fn`` with up to ``retries`` retries on ``RateLimitError``.

    The retry decision is a deterministic counter — no wall-clock timing
    inside the test path (tests pass ``retry_sleep=lambda _: None`` to
    short-circuit the back-off).
    """
    attempt = 0
    while True:
        try:
            return fetch_fn(chunk, start, end)
        except RateLimitError:
            if attempt >= retries:
                raise
            retry_sleep(attempt)
            attempt += 1


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def backfill(
    symbols: Iterable[str],
    since: str,
    out_path: str | Path = DEFAULT_OUT,
    *,
    incremental: bool = False,
    fetch_fn: FetchFn | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    retries: int = DEFAULT_RETRIES,
    retry_sleep: Callable[[int], None] | None = None,
    today: date | None = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> Path:
    """Run the backfill.

    Parameters
    ----------
    symbols:
        Tickers in YAML (Wikipedia) spelling. The fetcher translates dots
        to dashes internally; callers don't need to pre-translate.
    since:
        ISO date string. The full-backfill anchor. Ignored when
        ``incremental=True`` (window is fixed to the last 5 days).
    out_path:
        Destination parquet. Atomic-written.
    incremental:
        When ``True``, fetch only the last 5 calendar days and upsert into
        the existing parquet on ``(symbol, date)``.
    fetch_fn:
        Override the yfinance fetcher (tests inject a stub). Production
        callers leave this at the default to hit the network.
    chunk_size:
        Symbols per yfinance call. Default 25.
    retries:
        Per-chunk retries on ``RateLimitError``. Default 1.
    retry_sleep:
        Callback invoked between retries; receives the attempt index (0-based).
        Default: ``time.sleep(2 ** attempt)`` exponential back-off. Tests
        pass ``lambda _: None`` for determinism.
    today:
        Override "today" (tests inject a fixed date). Defaults to UTC today.
    min_coverage:
        Minimum fraction of requested symbols that must return rows.
        Default 0.95 (per the cron fail-loud design). Below threshold →
        ``CoverageError`` is raised BEFORE the parquet write, so the
        prior good parquet (if any) is preserved. Pass ``0.0`` to disable
        the check (useful for ad-hoc operator runs against narrow
        symbol lists).

    Returns
    -------
    Path
        The path actually written (equal to ``out_path``).

    Raises
    ------
    CoverageError
        Too few symbols returned rows. Prior parquet preserved.
    """
    symbols = list(symbols)
    if not symbols:
        raise ValueError("backfill called with empty symbols list")

    fetch_fn = fetch_fn or _yfinance_fetch
    retry_sleep = retry_sleep or (lambda attempt: time.sleep(2**attempt))
    out_path = Path(out_path)
    today_d = today or date.today()

    # Decide the fetch window.
    if incremental:
        start_iso = (today_d - timedelta(days=INCREMENTAL_WINDOW_DAYS)).isoformat()
        end_iso = today_d.isoformat()
    else:
        start_iso = since
        end_iso = today_d.isoformat()

    fetched_at = datetime.now(tz=timezone.utc)
    try:
        import yfinance as yf

        yfinance_version = yf.__version__
    except Exception:  # noqa: BLE001 — best-effort version capture
        yfinance_version = "stub"

    # Fetch in chunks. We collect the entire batch before touching the
    # destination parquet — a mid-fetch crash leaves the prior parquet
    # bytes intact (test_partial_failure_does_not_corrupt_parquet).
    all_fetched: dict[str, pd.DataFrame] = {}
    for chunk in _chunked(symbols, chunk_size):
        batch = _fetch_with_retry(
            fetch_fn=fetch_fn,
            chunk=chunk,
            start=start_iso,
            end=end_iso,
            retries=retries,
            retry_sleep=retry_sleep,
        )
        all_fetched.update(batch)

    new_df = _to_long_frame(all_fetched, yfinance_version, fetched_at)

    # Coverage gate — fail loud BEFORE writing if too many symbols silently
    # returned zero rows (yfinance transient outages, mass-ticker delisting,
    # bad chunk). Cron-wrapper's discord_on_failure=true picks up the
    # non-zero exit; the previous good parquet stays intact since we abort
    # before _write_parquet_atomic.
    if min_coverage > 0.0:
        missing = [
            sym
            for sym in symbols
            if all_fetched.get(sym) is None or all_fetched[sym].empty
        ]
        present = len(symbols) - len(missing)
        if present < min_coverage * len(symbols):
            raise CoverageError(
                missing=missing,
                requested=len(symbols),
                threshold=min_coverage,
            )

    # Upsert into the existing parquet on (symbol, date).
    merged = _upsert(new_df, out_path)
    _write_parquet_atomic(merged, out_path)
    return out_path


def _upsert(new_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Merge ``new_df`` into the existing parquet (if any) on (symbol, date).

    Latest write wins for any overlap — rows in ``new_df`` replace prior
    rows with the same key. The full backfill case ends up rewriting every
    row; the incremental case rewrites only rows in the last 5-day window.

    Stale-row caveat (codex iter-1 [P2], deferred):
        If a symbol is dropped from ``config/sp500_universe.yaml`` between
        runs, its existing rows linger in the parquet. Same for rows
        outside a later ``--since`` window. The breadth pipeline joins
        OHLCV to the YAML at compute time so stale parquet rows are
        invisible to downstream indicators, but operators who want a
        clean replace can ``rm data/cache/sp500_universe.parquet`` before
        re-running the one-shot. A ``--prune-to-universe`` flag belongs
        to the cohort-management follow-up (out of scope for this PR).
    """
    if not out_path.exists():
        return new_df

    existing = pd.read_parquet(out_path)
    if existing.empty:
        return new_df

    # Date column dtype harmonization: parquet round-trip returns date32 as
    # python `date`; new_df builds the column as `date` already. Ensure
    # we compare apples to apples.
    existing["date"] = pd.to_datetime(existing["date"]).dt.date
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.date

    # Drop existing rows whose (symbol, date) appears in new_df.
    if not new_df.empty:
        key = new_df[["symbol", "date"]].apply(tuple, axis=1)
        existing_key = existing[["symbol", "date"]].apply(tuple, axis=1)
        keep = ~existing_key.isin(set(key))
        existing = existing.loc[keep]

    merged = pd.concat([existing, new_df], ignore_index=True)
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
    return merged[list(_COLUMN_ORDER)]
