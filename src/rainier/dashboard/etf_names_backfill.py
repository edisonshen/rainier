"""ETF names cache backfill — yfinance ``longName`` / ``shortName`` → parquet.

The ETF dashboard renderer reads this cache on each invocation so the
Name column shows ``iShares MSCI EAFE ETF`` instead of just ``EFA``. We
keep names in a side-cache parquet (not the main features parquet) so:

  * a one-shot backfill seeds names for every ticker in the YAML universe
    without waiting on the daily ranks compute path,
  * a 30-day ``--refresh-stale`` cron step keeps the cache current with
    minimal yfinance load (names rarely change),
  * the renderer degrades gracefully — when the parquet is missing or a
    symbol is absent, the Name cell falls back to the symbol itself.

Parquet schema::

    symbol      str   primary key (Wikipedia spelling, BRK.B not BRK-B)
    long_name   str   yfinance info.longName (None → blank string)
    short_name  str   yfinance info.shortName (None → blank string)
    fetched_at  ts    UTC tz-aware, when the row was last refreshed

Upsert-on-symbol semantics: re-running ``backfill_names`` reads the
existing parquet (if any), merges new rows over old ones keyed by
``symbol``, then atomic-writes the merged frame back. Atomic write =
``<out>.tmp`` then ``os.replace`` — a mid-run crash leaves the previous
parquet intact and the tmp file cleaned up.

Retry pattern mirrors ``market_breadth/ohlcv_backfill.py``: tests inject
a synthetic ``RateLimitError`` via the ``fetch_fn`` kwarg; production
callers leave ``fetch_fn=None`` to hit the network via ``_yfinance_fetch``.

Design refs:
    docs/DESIGN-etf-dashboard-name-column.md §3.1 / §3.2
    docs/TASK-PLAN-etf-dashboard-name-column.md §Acceptance / §Tests

Flow (single chunk → fetch → upsert → atomic write)::

    +-----------+    +----------+    +---------+    +--------+
    | symbols   |--->| fetch_fn |--->| new_df  |--->| upsert |
    | (YAML)    |    | (yf info)|    | (sym,   |    | by sym |
    +-----------+    +----+-----+    |  ln,sn, |    +---+----+
                          |          |  ts)    |        |
                          v retry?   +---------+        v
                     +----+----+                   +----+----+
                     | retry x1|                   | atomic  |
                     +----+----+                   | write   |
                          | fail                   +---------+
                          v
                     RateLimitError
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "RateLimitError",
    "backfill_names",
    "DEFAULT_OUT",
]


DEFAULT_OUT = Path("data/cache/etf_names.parquet")
DEFAULT_RETRIES = 1
STALE_THRESHOLD_DAYS = 30

_COLUMN_ORDER: tuple[str, ...] = ("symbol", "long_name", "short_name", "fetched_at")

_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("long_name", pa.string()),
        ("short_name", pa.string()),
        ("fetched_at", pa.timestamp("ns", tz="UTC")),
    ]
)


class RateLimitError(RuntimeError):
    """yfinance rate-limit (transient). Retried once by ``backfill_names``.

    yfinance versions raise different exception types for rate-limit
    failures; this synthetic wrapper gives tests + the prod fetcher a
    single retryable signal.
    """


# Signature: ``(symbols) -> {symbol: {"long_name": str|None, "short_name": str|None}}``.
# A missing or None ``long_name`` triggers the renderer's fallback to the
# symbol itself.
FetchFn = Callable[[list[str]], dict[str, dict[str, str | None]]]


# ---------------------------------------------------------------------------
# Prod fetcher (real network)
# ---------------------------------------------------------------------------


def _yfinance_fetch(symbols: list[str]) -> dict[str, dict[str, str | None]]:
    """Fetch yfinance ``info`` for each symbol; return {sym: {long, short}}.

    yfinance's per-ticker ``info`` is sequential by design; for the ~100
    ETFs in the thematic universe this is fast enough and simpler than
    bulk pulling. Each Ticker construction is cheap; the network call
    happens lazily on the first ``.info`` access.

    Symbol translation (``BRK.B`` → ``BRK-B``) mirrors the OHLCV
    backfill: yfinance wants the dash form, our YAML + parquet keep the
    Wikipedia (dot) form.
    """
    import yfinance as yf  # local import → tests stay offline

    out: dict[str, dict[str, str | None]] = {}
    for sym in symbols:
        yf_sym = sym.replace(".", "-")
        try:
            info = yf.Ticker(yf_sym).info or {}
        except Exception as exc:  # noqa: BLE001 — surface as RateLimitError
            msg = str(exc).lower()
            if "rate" in msg or "limit" in msg or "429" in msg or "throttle" in msg:
                raise RateLimitError(str(exc)) from exc
            # Other failures (network reset, JSON decode, etc.) treat as
            # missing — name falls back to symbol in the renderer.
            info = {}
        out[sym] = {
            "long_name": _coerce_str(info.get("longName")),
            "short_name": _coerce_str(info.get("shortName")),
        }
    return out


def _coerce_str(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------


def _fetch_with_retry(
    fetch_fn: FetchFn,
    symbols: list[str],
    retries: int,
    retry_sleep: Callable[[int], None],
) -> dict[str, dict[str, str | None]]:
    """Call ``fetch_fn`` with up to ``retries`` retries on ``RateLimitError``.

    Mirrors ``ohlcv_backfill._fetch_with_retry``: deterministic counter,
    no wall-clock inside the test path. Tests pass
    ``retry_sleep=lambda _: None`` to short-circuit the back-off.
    """
    attempt = 0
    while True:
        try:
            return fetch_fn(symbols)
        except RateLimitError:
            if attempt >= retries:
                raise
            retry_sleep(attempt)
            attempt += 1


# ---------------------------------------------------------------------------
# Upsert + atomic write
# ---------------------------------------------------------------------------


def _new_frame(
    fetched: dict[str, dict[str, str | None]], fetched_at: datetime
) -> pd.DataFrame:
    """Convert the fetcher's dict payload to the canonical long-format frame."""
    rows = []
    for sym, payload in fetched.items():
        rows.append(
            {
                "symbol": sym,
                "long_name": payload.get("long_name") or "",
                "short_name": payload.get("short_name") or "",
                "fetched_at": fetched_at,
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(_COLUMN_ORDER))
    df = pd.DataFrame(rows)
    return df[list(_COLUMN_ORDER)]


def _upsert(new_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Merge ``new_df`` into the existing parquet (if any) on ``symbol``.

    Latest write wins for any overlap — rows in ``new_df`` replace prior
    rows with the same symbol. New symbols are appended.
    """
    if not out_path.exists():
        return new_df
    existing = pd.read_parquet(out_path)
    if existing.empty:
        return new_df
    # Filter out rows in `existing` that are being replaced, then concat.
    new_symbols = set(new_df["symbol"].astype(str))
    kept = existing.loc[~existing["symbol"].astype(str).isin(new_symbols)]
    merged = pd.concat([kept, new_df], ignore_index=True)
    return merged[list(_COLUMN_ORDER)]


def _write_parquet_atomic(df: pd.DataFrame, out_path: Path) -> None:
    """pyarrow write + os.replace — never leaves a half-written parquet visible."""
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
# Public entry point
# ---------------------------------------------------------------------------


def backfill_names(
    symbols: Iterable[str],
    out_path: str | Path = DEFAULT_OUT,
    *,
    fetch_fn: FetchFn | None = None,
    retries: int = DEFAULT_RETRIES,
    retry_sleep: Callable[[int], None] | None = None,
) -> Path:
    """Fetch ETF long/short names from yfinance and atomic-write the cache parquet.

    Parameters
    ----------
    symbols:
        Tickers in YAML (Wikipedia) spelling. The internal fetcher
        translates dots to dashes when talking to yfinance.
    out_path:
        Destination parquet. Atomic-written. Default: ``data/cache/etf_names.parquet``.
    fetch_fn:
        Override the yfinance fetcher (tests inject a stub). Production
        callers leave this ``None`` to hit the network.
    retries:
        Per-call retries on ``RateLimitError``. Default 1 (one retry, then surface).
    retry_sleep:
        Callback invoked between retries; receives the attempt index (0-based).
        Default ``time.sleep(2 ** attempt)``. Tests pass ``lambda _: None``.

    Returns
    -------
    Path
        The path actually written (equal to ``out_path``).

    Raises
    ------
    RateLimitError
        After a single retry, a persistent rate limit surfaces cleanly so
        the operator (or cron-wrapper Discord alert) sees a non-zero exit.
    """
    symbols = [str(s) for s in symbols]
    if not symbols:
        raise ValueError("backfill_names called with empty symbols list")

    import time

    fetcher = fetch_fn or _yfinance_fetch
    sleep_cb = retry_sleep or (lambda attempt: time.sleep(2**attempt))
    out = Path(out_path)

    fetched_at = datetime.now(tz=timezone.utc)
    fetched = _fetch_with_retry(
        fetch_fn=fetcher,
        symbols=symbols,
        retries=retries,
        retry_sleep=sleep_cb,
    )

    new_df = _new_frame(fetched, fetched_at)
    merged = _upsert(new_df, out)
    _write_parquet_atomic(merged, out)
    return out


# ---------------------------------------------------------------------------
# Stale check (cron --refresh-stale)
# ---------------------------------------------------------------------------


def is_stale(out_path: str | Path, threshold_days: int = STALE_THRESHOLD_DAYS) -> bool:
    """Return True iff the parquet is missing or older than ``threshold_days``.

    Used by the cron-wrapper to decide whether to invoke the backfill in
    ``--refresh-stale`` mode — keeps daily yfinance load near zero for
    cache that rarely changes (ETF long names).
    """
    out = Path(out_path)
    if not out.exists():
        return True
    try:
        df = pd.read_parquet(out)
    except Exception:  # noqa: BLE001 — corrupt parquet → refresh
        return True
    if df.empty or "fetched_at" not in df.columns:
        return True
    newest = pd.to_datetime(df["fetched_at"]).max()
    if pd.isna(newest):
        return True
    now = datetime.now(tz=timezone.utc)
    # Strip tz if absent (legacy parquet) for delta calc.
    if newest.tzinfo is None:
        newest = newest.tz_localize("UTC")
    delta = now - newest
    return delta.days >= threshold_days
