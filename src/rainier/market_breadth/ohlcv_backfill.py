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

    End-bound semantics: yfinance's ``end=`` is EXCLUSIVE. To match a
    single-sided operator contract (``--since 2020-01-01`` → include
    today's bar if it's a trading day), the caller bumps ``end`` by +1
    calendar day before calling here. Same convention vix + thematic land.

    Returns a per-symbol map keyed by the original (dotted) ticker so the
    caller doesn't have to undo the dash translation.
    """
    import yfinance as yf  # local import → tests stay offline-safe

    if not symbols:
        return {}

    yf_symbols = [_to_yfinance_symbol(s) for s in symbols]
    yf_to_orig = dict(zip(yf_symbols, symbols))

    df = yf.download(
        yf_symbols,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        actions=False,
        group_by="ticker",
        threads=True,
    )

    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
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

    Returns
    -------
    Path
        The path actually written (equal to ``out_path``).
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

    # Upsert into the existing parquet on (symbol, date).
    merged = _upsert(new_df, out_path)
    _write_parquet_atomic(merged, out_path)
    return out_path


def _upsert(new_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Merge ``new_df`` into the existing parquet (if any) on (symbol, date).

    Latest write wins for any overlap — rows in ``new_df`` replace prior
    rows with the same key. The full backfill case ends up rewriting every
    row; the incremental case rewrites only rows in the last 5-day window.
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
