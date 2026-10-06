"""Tests for openstock_feed signal — file + URL sources, cache, graceful None."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

from rainier.core.types import StockCandidate
from rainier.llm_thesis.signals import openstock_feed as os_mod
from rainier.llm_thesis.signals.base import SignalContext
from rainier.llm_thesis.signals.openstock_feed import OpenStockFeedSignal

FEED = {
    "as_of": "2026-09-18T13:00:00Z",
    "stocks": {
        "NVDA": {
            "quote": {"price": 182.1, "change_pct": 1.8},
            "profile": {"name": "NVIDIA", "industry": "Semiconductors", "market_cap": 4.4e12},
            "news": {"count": 12, "headlines": [f"h{i}" for i in range(8)]},
            "sentiment": {
                "reddit": {"buzz_score": 71, "bullish_pct": 64, "trend": "rising", "mentions": 300},
                "polymarket": {"buzz_score": 10, "trade_count": 4},
                "bogus": {"buzz_score": 1},
            },
        }
    },
}


def _ctx(symbol: str = "NVDA", params: dict | None = None):
    cand = StockCandidate(
        symbol=symbol, rank=5, rank_change=0, long_short="Long in",
        capital_flow_direction="+", sector="Technology", signal_strength=0.8,
    )
    return SignalContext(
        symbol=symbol, scan_date=date(2026, 9, 18),
        session_name="afternoon", candidate=cand, params=params or {},
    )


def setup_function(_):
    os_mod._clear_cache_for_tests()


def test_unconfigured_returns_none(monkeypatch):
    monkeypatch.delenv("OPENSTOCK_FEED_URL", raising=False)
    monkeypatch.delenv("OPENSTOCK_FEED_PATH", raising=False)
    assert OpenStockFeedSignal().compute(_ctx()) is None


def _write_feed(tmp_path, monkeypatch, doc=FEED):
    p = tmp_path / "feed.json"
    p.write_text(json.dumps(doc))
    monkeypatch.delenv("OPENSTOCK_FEED_URL", raising=False)
    monkeypatch.setenv("OPENSTOCK_FEED_PATH", str(p))


def test_file_source_extracts_fields(tmp_path, monkeypatch):
    _write_feed(tmp_path, monkeypatch)
    v = OpenStockFeedSignal().compute(_ctx())
    assert v["price"] == 182.1
    assert v["change_pct"] == 1.8
    assert v["industry"] == "Semiconductors"
    assert v["news_count"] == 12
    assert len(v["headlines"]) == os_mod._MAX_HEADLINES
    assert v["sentiment"]["reddit"]["bullish_pct"] == 64.0
    assert v["sentiment"]["reddit"]["mentions"] == 300
    assert v["sentiment"]["polymarket"]["mentions"] == 4
    assert "bogus" not in v["sentiment"]


def test_missing_ticker_returns_none(tmp_path, monkeypatch):
    _write_feed(tmp_path, monkeypatch)
    assert OpenStockFeedSignal().compute(_ctx("ZZZZ")) is None


def test_params_cannot_override_source(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENSTOCK_FEED_URL", raising=False)
    monkeypatch.delenv("OPENSTOCK_FEED_PATH", raising=False)
    p = tmp_path / "feed.json"
    p.write_text(json.dumps(FEED))
    sig = OpenStockFeedSignal()
    assert sig.compute(_ctx(params={"path": str(p)})) is None
    assert sig.compute(_ctx(params={"url": "http://169.254.169.254/"})) is None


def test_url_source_sends_tickers_and_bearer_and_caches_per_symbol(monkeypatch):
    monkeypatch.setenv("OPENSTOCK_FEED_URL", "https://os.example/api/export")
    monkeypatch.setenv("OPENSTOCK_FEED_TOKEN", "sekret")
    resp = MagicMock()
    resp.json.return_value = FEED
    with patch.object(os_mod.httpx, "get", return_value=resp) as get:
        sig = OpenStockFeedSignal()
        v1 = sig.compute(_ctx())
        v2 = sig.compute(_ctx())
        sig.compute(_ctx("AMD"))
    assert v1 == v2 and v1["price"] == 182.1
    assert get.call_count == 2
    first = get.call_args_list[0].kwargs
    assert first["params"] == {"tickers": "NVDA"}
    assert first["headers"] == {"Authorization": "Bearer sekret"}
    assert get.call_args_list[1].kwargs["params"] == {"tickers": "AMD"}


def test_stale_as_of_returns_none(tmp_path, monkeypatch):
    stale = dict(FEED, as_of="2026-09-12T20:00:00Z")  # 6 days before scan_date
    _write_feed(tmp_path, monkeypatch, stale)
    assert OpenStockFeedSignal().compute(_ctx()) is None
    assert OpenStockFeedSignal().compute(_ctx(params={"max_age_days": 10})) is not None


def test_missing_as_of_is_accepted(tmp_path, monkeypatch):
    doc = {k: v for k, v in FEED.items() if k != "as_of"}
    _write_feed(tmp_path, monkeypatch, doc)
    assert OpenStockFeedSignal().compute(_ctx())["price"] == 182.1


def test_fetch_error_returns_none(monkeypatch):
    monkeypatch.setenv("OPENSTOCK_FEED_URL", "https://os.example/api/export")
    with patch.object(os_mod.httpx, "get", side_effect=RuntimeError("boom")):
        assert OpenStockFeedSignal().compute(_ctx()) is None


def test_list_shaped_feed_is_accepted(tmp_path, monkeypatch):
    _write_feed(tmp_path, monkeypatch, {"stocks": [{"symbol": "nvda", "quote": {"price": 5}}]})
    v = OpenStockFeedSignal().compute(_ctx())
    assert v["price"] == 5.0


def test_render_for_prompt():
    sig = OpenStockFeedSignal()
    v = {
        "price": 182.1, "change_pct": 1.8, "industry": "Semis", "news_count": 3,
        "headlines": ["a", "b"],
        "sentiment": {"reddit": {"buzz_score": 71, "bullish_pct": 64, "trend": "rising"}},
    }
    text = sig.render_for_prompt(v)
    assert "last 182.10 (+1.8%)" in text
    assert "reddit buzz 71 bull 64% rising" in text
    assert "headlines: a | b" in text
    assert sig.render_for_prompt({}) == "OpenStock: no data"
