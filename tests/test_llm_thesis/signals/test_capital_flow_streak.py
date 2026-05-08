"""Tests for capital_flow_streak signal."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from rainier.core.types import StockCandidate
from rainier.llm_thesis.signals.base import SignalContext
from rainier.llm_thesis.signals.capital_flow_streak import CapitalFlowStreakSignal


def _ctx(symbol: str = "NVDA"):
    cand = StockCandidate(
        symbol=symbol, rank=5, rank_change=0, long_short="Long in",
        capital_flow_direction="+", sector="Technology", signal_strength=0.8,
    )
    return SignalContext(
        symbol=symbol, scan_date=date(2026, 5, 7),
        session_name="afternoon", candidate=cand, params={"days": 10},
    )


def _row(d: date, dirn: str):
    r = MagicMock()
    r.flow_date = d
    r.capital_flow_direction = dirn
    return r


def _patch_session(rows):
    fake = MagicMock()
    fake.query.return_value.filter.return_value.order_by.return_value.all.return_value = rows
    cm = MagicMock()
    cm.__enter__.return_value = fake
    cm.__exit__.return_value = False
    return cm


def test_streak_counts_only_consecutive_recent():
    rows = [
        _row(date(2026, 5, 7), "+"),
        _row(date(2026, 5, 6), "+"),
        _row(date(2026, 5, 5), "+"),
        _row(date(2026, 5, 4), "-"),
        _row(date(2026, 5, 3), "+"),
    ]
    sig = CapitalFlowStreakSignal()
    with patch(
        "rainier.llm_thesis.signals.capital_flow_streak.get_session",
        return_value=_patch_session(rows),
    ):
        v = sig.compute(_ctx())
    assert v["direction"] == "+"
    assert v["streak_days"] == 3
    assert len(v["history"]) == 5


def test_neutral_direction_means_zero_streak():
    rows = [_row(date(2026, 5, 7), "N"), _row(date(2026, 5, 6), "+")]
    sig = CapitalFlowStreakSignal()
    with patch(
        "rainier.llm_thesis.signals.capital_flow_streak.get_session",
        return_value=_patch_session(rows),
    ):
        v = sig.compute(_ctx())
    assert v["direction"] == "N"
    assert v["streak_days"] == 0


def test_empty_history_means_streak_zero():
    sig = CapitalFlowStreakSignal()
    with patch(
        "rainier.llm_thesis.signals.capital_flow_streak.get_session",
        return_value=_patch_session([]),
    ):
        v = sig.compute(_ctx())
    assert v == {"streak_days": 0, "direction": "N", "history": []}


def test_render_for_inflow():
    sig = CapitalFlowStreakSignal()
    out = sig.render_for_prompt(
        {"streak_days": 4, "direction": "+", "history": [("2026-05-07", "+")]}
    )
    assert "inflow" in out
    assert "4" in out


def test_render_for_no_data():
    sig = CapitalFlowStreakSignal()
    out = sig.render_for_prompt({"streak_days": 0, "direction": "N", "history": []})
    assert "no recent flow data" in out
