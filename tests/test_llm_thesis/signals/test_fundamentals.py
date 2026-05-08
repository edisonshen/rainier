"""Tests for fundamentals signal — yfinance mocked, LRU cache verified."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from rainier.core.types import StockCandidate
from rainier.llm_thesis.signals import fundamentals as fund_mod
from rainier.llm_thesis.signals.base import SignalContext
from rainier.llm_thesis.signals.fundamentals import FundamentalsSignal


def _ctx(symbol: str = "NVDA"):
    cand = StockCandidate(
        symbol=symbol, rank=5, rank_change=0, long_short="Long in",
        capital_flow_direction="+", sector="Technology", signal_strength=0.8,
    )
    return SignalContext(
        symbol=symbol, scan_date=date(2026, 5, 7),
        session_name="afternoon", candidate=cand, params={},
    )


def setup_function(_):
    fund_mod._clear_cache_for_tests()


def test_happy_path_extracts_fields():
    ticker = MagicMock()
    ticker.info = {
        "trailingPE": 32.1,
        "forwardPE": 26.0,
        "marketCap": 1_500_000_000_000,
        "earningsTimestamp": 1_762_000_000,
        "earningsQuarterlyGrowth": 0.094,
    }
    sig = FundamentalsSignal()
    with patch.object(fund_mod.yf, "Ticker", return_value=ticker):
        v = sig.compute(_ctx())
    assert v["pe_trailing"] == 32.1
    assert v["pe_forward"] == 26.0
    assert v["market_cap"] == 1_500_000_000_000
    assert v["next_earnings_date"]  # ISO date string
    assert v["last_surprise_pct"] == 0.094


def test_yfinance_error_returns_all_none():
    sig = FundamentalsSignal()
    with patch.object(fund_mod.yf, "Ticker", side_effect=RuntimeError("network down")):
        v = sig.compute(_ctx())
    assert v == {
        "pe_trailing": None,
        "pe_forward": None,
        "market_cap": None,
        "next_earnings_date": None,
        "last_surprise_pct": None,
    }


def test_per_day_lru_cache_hits_on_second_call():
    ticker = MagicMock()
    ticker.info = {"trailingPE": 1.0}
    sig = FundamentalsSignal()
    with patch.object(fund_mod.yf, "Ticker", return_value=ticker) as mock_t:
        sig.compute(_ctx())
        sig.compute(_ctx())  # same (symbol, scan_date) — cache hit
        assert mock_t.call_count == 1


def test_per_day_lru_cache_misses_when_date_advances():
    ticker = MagicMock()
    ticker.info = {"trailingPE": 1.0}
    sig = FundamentalsSignal()
    cand = StockCandidate(
        symbol="NVDA", rank=5, rank_change=0, long_short="Long in",
        capital_flow_direction="+", sector="Technology", signal_strength=0.8,
    )
    today = SignalContext(
        symbol="NVDA", scan_date=date(2026, 5, 7),
        session_name="afternoon", candidate=cand, params={},
    )
    tomorrow = SignalContext(
        symbol="NVDA", scan_date=date(2026, 5, 8),
        session_name="afternoon", candidate=cand, params={},
    )
    with patch.object(fund_mod.yf, "Ticker", return_value=ticker) as mock_t:
        sig.compute(today)
        sig.compute(tomorrow)
        assert mock_t.call_count == 2


def test_render_with_data_and_without():
    sig = FundamentalsSignal()
    s = sig.render_for_prompt(
        {"pe_trailing": 32.1, "pe_forward": 26.0, "market_cap": 1_500_000_000_000,
         "next_earnings_date": "2026-05-21", "last_surprise_pct": 0.094}
    )
    assert "32.1" in s and "1500.0B" in s
    empty = sig.render_for_prompt(
        {"pe_trailing": None, "pe_forward": None, "market_cap": None,
         "next_earnings_date": None, "last_surprise_pct": None}
    )
    assert "no data" in empty.lower()
