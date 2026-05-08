"""fundamentals signal — P/E, market cap, next earnings date, last surprise.

Per-day LRU cache: same (symbol, scan_date) re-fetches go to cache, not yfinance.
yfinance failures degrade to all-None — never crash the thesis pipeline.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Any

import yfinance as yf

from .base import SignalContext, SignalValue

log = logging.getLogger(__name__)


def _coerce_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        f = float(val)
        if f != f:  # NaN guard
            return None
        return f
    except (TypeError, ValueError):
        return None


def _coerce_int(val: Any) -> int | None:
    try:
        if val is None:
            return None
        return int(val)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=512)
def _fetch_fundamentals_cached(
    symbol: str, scan_date_iso: str,
) -> tuple[float | None, float | None, int | None, str | None, float | None]:
    """Fetch + cache by (symbol, scan_date_iso).

    The scan-date key is what makes the cache day-bounded: a new day forces a
    refetch even when the daemon has been running continuously.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
    except Exception:
        log.exception("fundamentals_fetch_error symbol=%s", symbol)
        return (None, None, None, None, None)

    pe_trailing = _coerce_float(info.get("trailingPE"))
    pe_forward = _coerce_float(info.get("forwardPE"))
    market_cap = _coerce_int(info.get("marketCap"))

    next_earnings: str | None = None
    raw_earnings = info.get("earningsTimestamp") or info.get("earningsDate")
    if isinstance(raw_earnings, (int, float)):
        try:
            next_earnings = (
                datetime.fromtimestamp(int(raw_earnings), tz=timezone.utc).date().isoformat()
            )
        except (OverflowError, ValueError, OSError):
            next_earnings = None
    elif isinstance(raw_earnings, str):
        next_earnings = raw_earnings
    elif isinstance(raw_earnings, (list, tuple)) and raw_earnings:
        first = raw_earnings[0]
        if isinstance(first, (int, float)):
            try:
                next_earnings = (
                    datetime.fromtimestamp(int(first), tz=timezone.utc).date().isoformat()
                )
            except (OverflowError, ValueError, OSError):
                next_earnings = None
        elif hasattr(first, "isoformat"):
            try:
                next_earnings = first.isoformat()
            except Exception:
                next_earnings = None

    surprise = _coerce_float(
        info.get("earningsQuarterlyGrowth") or info.get("lastEarningsSurprise")
    )

    return (pe_trailing, pe_forward, market_cap, next_earnings, surprise)


def _clear_cache_for_tests() -> None:
    """Test helper — wipe the LRU cache between unit tests."""
    _fetch_fundamentals_cached.cache_clear()


class FundamentalsSignal:
    name = "fundamentals"
    version = "v1"
    cost_estimate_ms = 1000

    def compute(self, ctx: SignalContext) -> SignalValue | None:
        scan_iso = ctx.scan_date.isoformat() if isinstance(ctx.scan_date, date) else str(
            ctx.scan_date
        )
        pe_t, pe_f, mc, ne, surp = _fetch_fundamentals_cached(ctx.symbol, scan_iso)
        return {
            "pe_trailing": pe_t,
            "pe_forward": pe_f,
            "market_cap": mc,
            "next_earnings_date": ne,
            "last_surprise_pct": surp,
        }

    def render_for_prompt(self, value: SignalValue) -> str:
        pe_t = value.get("pe_trailing")
        pe_f = value.get("pe_forward")
        mc = value.get("market_cap")
        ne = value.get("next_earnings_date")
        surp = value.get("last_surprise_pct")
        parts = []
        if pe_t is not None:
            parts.append(f"P/E trailing {pe_t:.1f}")
        if pe_f is not None:
            parts.append(f"P/E fwd {pe_f:.1f}")
        if mc is not None:
            parts.append(f"market cap {mc / 1e9:.1f}B")
        if ne:
            parts.append(f"next earnings {ne}")
        if surp is not None:
            parts.append(f"last surprise {surp:+.2%}")
        if not parts:
            return "Fundamentals: no data available."
        return "Fundamentals: " + ", ".join(parts) + "."
