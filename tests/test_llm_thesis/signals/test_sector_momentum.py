"""Tests for sector_momentum signal — uses analyze_sectors_at delta computation."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

from rainier.core.types import SectorTrend, StockCandidate
from rainier.llm_thesis.signals.base import SignalContext
from rainier.llm_thesis.signals.sector_momentum import SectorMomentumSignal


def _ctx(sector: str = "Technology") -> SignalContext:
    cand = StockCandidate(
        symbol="NVDA", rank=5, rank_change=0, long_short="Long in",
        capital_flow_direction="+", sector=sector, signal_strength=0.8,
    )
    return SignalContext(
        symbol="NVDA", scan_date=date(2026, 5, 7),
        session_name="afternoon", candidate=cand, params={"days": 10},
    )


def _trend(sector, sentiment):
    return SectorTrend(
        sector=sector, long_in_count=10, short_in_count=2,
        net_sentiment=sentiment, top_stocks=[], trend_direction="bullish",
        sector_rank=1,
    )


def _patch_session_with_ts(latest_ts, prior_ts):
    fake = MagicMock()
    # First scalar call returns latest_ts, second prior_ts
    fake.query.return_value.filter.return_value.scalar.side_effect = [latest_ts, prior_ts]
    cm = MagicMock()
    cm.__enter__.return_value = fake
    cm.__exit__.return_value = False
    return cm


def test_delta_positive_with_both_sentiments():
    latest = datetime(2026, 5, 7, tzinfo=timezone.utc)
    prior = datetime(2026, 4, 27, tzinfo=timezone.utc)
    sig = SectorMomentumSignal()
    with patch(
        "rainier.llm_thesis.signals.sector_momentum.get_session",
        return_value=_patch_session_with_ts(latest, prior),
    ), patch(
        "rainier.analysis.sector_analyzer.analyze_sectors_at",
        side_effect=[
            [_trend("Technology", 0.61)],
            [_trend("Technology", 0.43)],
        ],
    ):
        v = sig.compute(_ctx())
    assert v["sector"] == "Technology"
    assert v["sentiment_today"] == 0.61
    assert v["sentiment_prior"] == 0.43
    assert v["delta"] == 0.18


def test_no_prior_data_yields_none_delta():
    latest = datetime(2026, 5, 7, tzinfo=timezone.utc)
    sig = SectorMomentumSignal()
    with patch(
        "rainier.llm_thesis.signals.sector_momentum.get_session",
        return_value=_patch_session_with_ts(latest, None),
    ), patch(
        "rainier.analysis.sector_analyzer.analyze_sectors_at",
        side_effect=[[_trend("Technology", 0.61)], []],
    ):
        v = sig.compute(_ctx())
    assert v["delta"] is None
    assert v["sentiment_prior"] is None


def test_unknown_sector_returns_none_payload_keys():
    sig = SectorMomentumSignal()
    v = sig.compute(_ctx(sector="Unknown"))
    assert v["sector"] == "Unknown"
    assert v["sentiment_today"] is None


def test_render_for_prompt():
    sig = SectorMomentumSignal()
    out = sig.render_for_prompt(
        {"sector": "Technology", "sentiment_today": 0.61, "sentiment_prior": 0.43,
         "delta": 0.18, "shifted_at": None}
    )
    assert "Technology" in out
    assert "+0.61" in out
