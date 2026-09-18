"""openstock_feed signal — per-ticker quote / news / social-sentiment from a
self-hosted OpenStock export.

OpenStock (https://github.com/Open-Dev-Society/OpenStock) is hosted outside
rainier; it aggregates Finnhub (quote, profile, company news) and Adanos
(Reddit / X / news / Polymarket buzz + bullish %). Rainier only reads a daily
JSON feed, either from ``OPENSTOCK_FEED_URL`` (``GET <url>?tickers=A,B``,
optional ``Authorization: Bearer $OPENSTOCK_FEED_TOKEN``) or from a local file
``OPENSTOCK_FEED_PATH``.

Feed schema (one document, keyed by upper-case ticker)::

    {
      "as_of": "2026-09-18T13:00:00Z",
      "stocks": {
        "NVDA": {
          "quote":   {"price": 182.1, "change_pct": 1.8},
          "profile": {"name": "NVIDIA Corp", "industry": "Semiconductors",
                      "market_cap": 4.4e12},
          "news":    {"count": 12, "headlines": ["..."]},
          "sentiment": {
            "reddit":     {"buzz_score": 71, "bullish_pct": 64, "trend": "rising"},
            "x":          {...}, "news": {...}, "polymarket": {...}
          }
        }
      }
    }

The feed is fetched once per (source, scan_date) and cached; a ticker missing
from the feed, or an unconfigured / unreachable feed, yields ``None`` so the
thesis pipeline simply omits this signal.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from .base import SignalContext, SignalValue

log = logging.getLogger(__name__)

SENTIMENT_SOURCES = ("reddit", "x", "news", "polymarket")
_MAX_HEADLINES = 5
_FETCH_TIMEOUT_S = 10.0


def _coerce_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        f = float(val)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _coerce_int(val: Any) -> int | None:
    try:
        return None if val is None else int(val)
    except (TypeError, ValueError):
        return None


def _stocks_from_doc(doc: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(doc, dict):
        return {}
    stocks = doc.get("stocks", doc)
    if isinstance(stocks, list):
        stocks = {
            str(row.get("symbol") or row.get("ticker")): row
            for row in stocks
            if isinstance(row, dict) and (row.get("symbol") or row.get("ticker"))
        }
    if not isinstance(stocks, dict):
        return {}
    return {str(k).upper(): v for k, v in stocks.items() if isinstance(v, dict)}


@lru_cache(maxsize=32)
def _load_feed_cached(source: str, scan_date_iso: str) -> dict[str, dict[str, Any]]:
    """Load the whole feed once per (source, scan_date). Empty dict on failure."""
    try:
        if source.startswith(("http://", "https://")):
            headers: dict[str, str] = {}
            token = os.environ.get("OPENSTOCK_FEED_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = httpx.get(source, headers=headers, timeout=_FETCH_TIMEOUT_S)
            resp.raise_for_status()
            doc = resp.json()
        else:
            doc = json.loads(Path(source).read_text())
    except Exception:
        log.warning("openstock_feed_load_error source=%s", source, exc_info=True)
        return {}
    return _stocks_from_doc(doc)


def _clear_cache_for_tests() -> None:
    _load_feed_cached.cache_clear()


def _feed_source(params: dict[str, Any]) -> str | None:
    return (
        params.get("url")
        or params.get("path")
        or os.environ.get("OPENSTOCK_FEED_URL")
        or os.environ.get("OPENSTOCK_FEED_PATH")
        or None
    )


class OpenStockFeedSignal:
    name = "openstock_feed"
    version = "v1"
    cost_estimate_ms = 500

    def compute(self, ctx: SignalContext) -> SignalValue | None:
        source = _feed_source(ctx.params)
        if not source:
            return None
        scan_iso = (
            ctx.scan_date.isoformat() if isinstance(ctx.scan_date, date) else str(ctx.scan_date)
        )
        row = _load_feed_cached(source, scan_iso).get(ctx.symbol.upper())
        if row is None:
            return None

        quote = row.get("quote") or {}
        profile = row.get("profile") or {}
        news = row.get("news") or {}
        raw_sent = row.get("sentiment") or {}

        headlines = news.get("headlines") or []
        if not isinstance(headlines, list):
            headlines = []

        sentiment: dict[str, dict[str, Any]] = {}
        for src in SENTIMENT_SOURCES:
            s = raw_sent.get(src)
            if not isinstance(s, dict):
                continue
            sentiment[src] = {
                "buzz_score": _coerce_float(s.get("buzz_score")),
                "bullish_pct": _coerce_float(s.get("bullish_pct")),
                "trend": s.get("trend"),
                "mentions": _coerce_int(s.get("mentions") or s.get("trade_count")),
            }

        return {
            "price": _coerce_float(quote.get("price")),
            "change_pct": _coerce_float(quote.get("change_pct")),
            "industry": profile.get("industry"),
            "market_cap": _coerce_float(profile.get("market_cap")),
            "news_count": _coerce_int(news.get("count")),
            "headlines": [str(h) for h in headlines[:_MAX_HEADLINES]],
            "sentiment": sentiment,
        }

    def render_for_prompt(self, value: SignalValue) -> str:
        parts: list[str] = []
        price = value.get("price")
        chg = value.get("change_pct")
        if price is not None:
            parts.append(
                f"last {price:.2f}" + (f" ({chg:+.1f}%)" if chg is not None else "")
            )
        if value.get("industry"):
            parts.append(f"industry {value['industry']}")
        nc = value.get("news_count")
        if nc is not None:
            parts.append(f"{nc} news items")

        sent_bits: list[str] = []
        for src in SENTIMENT_SOURCES:
            s = (value.get("sentiment") or {}).get(src)
            if not s:
                continue
            bit = src
            if s.get("buzz_score") is not None:
                bit += f" buzz {s['buzz_score']:.0f}"
            if s.get("bullish_pct") is not None:
                bit += f" bull {s['bullish_pct']:.0f}%"
            if s.get("trend"):
                bit += f" {s['trend']}"
            sent_bits.append(bit)
        if sent_bits:
            parts.append("sentiment: " + ", ".join(sent_bits))

        line = "OpenStock: " + ("; ".join(parts) if parts else "no data")
        headlines = value.get("headlines") or []
        if headlines:
            line += "\n  headlines: " + " | ".join(headlines)
        return line
