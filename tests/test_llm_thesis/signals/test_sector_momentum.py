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


def test_date_probe_skips_later_non_qu100_board():
    """Codex P2 regression — the max(data_date) probes are scoped to the QU100
    books. A NEWER non-QU100 board (concept) on a later date must NOT advance
    ``latest_date`` past the last QU100 day, or analyze_sectors_at's
    "latest QU100 day <= as_of" fallback would serve a stale day's sentiment as
    current. The probe must resolve to the QU100 day, not the concept day."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    from rainier.core.models import MoneyFlowSnapshot

    ddl = """
    CREATE TABLE money_flow_snapshots (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      captured_at TIMESTAMP NOT NULL, capture_session VARCHAR(20) NOT NULL,
      data_date DATE NOT NULL, view_type VARCHAR(10) NOT NULL DEFAULT 'daily',
      ranking_type VARCHAR(10) NOT NULL, symbol VARCHAR(10) NOT NULL,
      rank INTEGER NOT NULL, daily_change INTEGER, sector VARCHAR(100),
      industry VARCHAR(200), long_short VARCHAR(50), raw_data JSON
    );
    """
    engine = create_engine("sqlite://", future=True)
    with engine.begin() as conn:
        conn.execute(text(ddl.strip()))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    cap = datetime(2026, 5, 7, 20, 0, tzinfo=timezone.utc)
    with factory() as s:
        # Last QU100 day is May 6; a NEWER 'concept' board lands May 7 (== scan_date).
        # WITHOUT the QU100 scope the probe would pick May 7 (concept) and
        # analyze_sectors_at would serve the stale May 6 book as "today". With the
        # scope the probe correctly resolves to May 6.
        s.add_all([
            MoneyFlowSnapshot(captured_at=cap, capture_session="afternoon",
                              data_date=date(2026, 5, 6), ranking_type="top100",
                              symbol="NVDA", rank=1, sector="Technology",
                              long_short="Long in"),
            MoneyFlowSnapshot(captured_at=cap, capture_session="afternoon",
                              data_date=date(2026, 5, 7), ranking_type="concept",
                              symbol="ZZZ", rank=1, sector="Materials",
                              long_short="Long in"),
        ])
        s.commit()

    captured_as_of: list = []

    def _spy_analyze(as_of, session=None):
        captured_as_of.append(as_of)
        return [_trend("Technology", 0.5)]

    cm = MagicMock()
    cm.__enter__.return_value = factory()
    cm.__exit__.return_value = False
    sig = SectorMomentumSignal()
    with patch(
        "rainier.llm_thesis.signals.sector_momentum.get_session", return_value=cm,
    ), patch(
        "rainier.analysis.sector_analyzer.analyze_sectors_at", side_effect=_spy_analyze,
    ):
        sig.compute(_ctx())

    # latest_date probe must resolve to the QU100 day (May 6), never the later
    # concept row (May 7) — a non-QU100 board can't drive the as-of date.
    assert captured_as_of, "analyze_sectors_at should have been called for today"
    assert captured_as_of[0] == date(2026, 5, 6)
    engine.dispose()
