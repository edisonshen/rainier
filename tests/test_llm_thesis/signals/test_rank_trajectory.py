"""Tests for rank_trajectory signal — happy + empty + multi-day delta."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from rainier.core.types import StockCandidate
from rainier.llm_thesis.signals.base import SignalContext
from rainier.llm_thesis.signals.rank_trajectory import RankTrajectorySignal


def _ctx(symbol: str = "NVDA", days: int = 10) -> SignalContext:
    cand = StockCandidate(
        symbol=symbol, rank=5, rank_change=0, long_short="Long in",
        capital_flow_direction="+", sector="Technology", signal_strength=0.8,
    )
    return SignalContext(
        symbol=symbol, scan_date=date(2026, 5, 7),
        session_name="afternoon", candidate=cand,
        params={"days": days},
    )


def _row(d: date, rank: int):
    r = MagicMock()
    r.data_date = d
    r.best_rank = rank
    return r


def _patch_session(rows):
    """Mock get_session() context manager + nested query chain."""
    fake_session = MagicMock()
    chain = (
        fake_session.query.return_value.filter.return_value
        .group_by.return_value.order_by.return_value
    )
    chain.all.return_value = rows
    cm = MagicMock()
    cm.__enter__.return_value = fake_session
    cm.__exit__.return_value = False
    return cm


def test_happy_path_returns_points_and_trend():
    rows = [_row(date(2026, 4, 30), 30), _row(date(2026, 5, 7), 5)]
    sig = RankTrajectorySignal()
    with patch(
        "rainier.llm_thesis.signals.rank_trajectory.get_session",
        return_value=_patch_session(rows),
    ):
        v = sig.compute(_ctx())
    assert v["points"][0] == ("2026-04-30", 30)
    assert v["points"][1] == ("2026-05-07", 5)
    assert v["delta_10d"] == 25
    assert v["trend"] == "rising"


def test_falling_trend():
    rows = [_row(date(2026, 4, 30), 5), _row(date(2026, 5, 7), 30)]
    sig = RankTrajectorySignal()
    with patch(
        "rainier.llm_thesis.signals.rank_trajectory.get_session",
        return_value=_patch_session(rows),
    ):
        v = sig.compute(_ctx())
    assert v["delta_10d"] == -25
    assert v["trend"] == "falling"


def test_empty_db_returns_empty_points():
    sig = RankTrajectorySignal()
    with patch(
        "rainier.llm_thesis.signals.rank_trajectory.get_session",
        return_value=_patch_session([]),
    ):
        v = sig.compute(_ctx())
    assert v == {"points": [], "delta_10d": None, "trend": "flat"}


def test_db_error_returns_none():
    sig = RankTrajectorySignal()
    cm = MagicMock()
    cm.__enter__.side_effect = RuntimeError("DB down")
    with patch(
        "rainier.llm_thesis.signals.rank_trajectory.get_session", return_value=cm,
    ):
        assert sig.compute(_ctx()) is None


def test_render_for_prompt_with_points():
    sig = RankTrajectorySignal()
    out = sig.render_for_prompt(
        {
            "points": [("2026-04-30", 30), ("2026-05-07", 5)],
            "delta_10d": 25,
            "trend": "rising",
        }
    )
    assert "rising" in out
    assert "2026-04-30" in out
    assert "+25" in out
