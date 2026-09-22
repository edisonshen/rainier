"""openstock_feed signal — per-ticker quote / news / social-sentiment from a
self-hosted OpenStock export.

OpenStock (https://github.com/Open-Dev-Society/OpenStock) is hosted outside
rainier; it aggregates Finnhub (quote, profile, company news) and Adanos
(Reddit / X / news / Polymarket buzz + bullish %). Rainier only reads a daily
JSON feed, either from ``OPENSTOCK_FEED_URL`` (``GET <url>?tickers=<symbol>``,
optional ``Authorization: Bearer $OPENSTOCK_FEED_TOKEN``) or from a local file
``OPENSTOCK_FEED_PATH``. Both come from the environment only (never from
settings.yaml params) so the feed target cannot be redirected by config edits.

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

A local file is loaded once per scan_date; a URL is fetched once per
(symbol, scan_date). Documents whose ``as_of`` is older than
``params.max_age_days`` (default 3) before scan_date are rejected. A ticker
missing from the feed, a stale document, or an unconfigured / unreachable
feed yields ``None`` so the thesis pipeline simply omits this signal.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from .base import SignalContext, SignalValue

log = logging.getLogger(__name__)

SENTIMENT_SOURCES = ("reddit", "x", "news", "polymarket")
_MAX_HEADLINES = 5
_FETCH_TIMEOUT_S = 10.0
_DEFAULT_MAX_AGE_DAYS = 3


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


def _parse_as_of(raw: Any) -> date | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
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


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


@lru_cache(maxsize=512)
def _load_feed_cached(
    source: str, scan_date_iso: str, tickers: str,
) -> tuple[date | None, dict[str, dict[str, Any]]]:
    """Load the feed once per cache key. Returns (as_of, stocks); empty on failure.

    ``tickers`` is the ``?tickers=`` query for URL sources and "" for files
    (a local file is one document for every symbol).
    """
    try:
        if _is_url(source):
            headers: dict[str, str] = {}
            token = os.environ.get("OPENSTOCK_FEED_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = httpx.get(
                source, params={"tickers": tickers}, headers=headers, timeout=_FETCH_TIMEOUT_S,
            )
            resp.raise_for_status()
            doc = resp.json()
        else:
            doc = json.loads(Path(source).read_text())
    except Exception:
        log.warning("openstock_feed_load_error source=%s", source, exc_info=True)
        return (None, {})
    as_of = _parse_as_of(doc.get("as_of")) if isinstance(doc, dict) else None
    return (as_of, _stocks_from_doc(doc))


def _clear_cache_for_tests() -> None:
    _load_feed_cached.cache_clear()


def _feed_source() -> str | None:
    return os.environ.get("OPENSTOCK_FEED_URL") or os.environ.get("OPENSTOCK_FEED_PATH") or None


class OpenStockFeedSignal:
    name = "openstock_feed"
    version = "v1"
    cost_estimate_ms = 500

    def compute(self, ctx: SignalContext) -> SignalValue | None:
        source = _feed_source()
        if not source:
            return None
        scan_date = (
            ctx.scan_date if isinstance(ctx.scan_date, date) else date.fromisoformat(str(ctx.scan_date))
        )
        symbol = ctx.symbol.upper()
        as_of, stocks = _load_feed_cached(
            source, scan_date.isoformat(), symbol if _is_url(source) else "",
        )
        max_age = int(ctx.params.get("max_age_days", _DEFAULT_MAX_AGE_DAYS))
        if as_of is not None and as_of < scan_date - timedelta(days=max_age):
            log.warning(
                "openstock_feed_stale as_of=%s scan_date=%s max_age_days=%d",
                as_of, scan_date, max_age,
            )
            return None
        row = stocks.get(symbol)
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
