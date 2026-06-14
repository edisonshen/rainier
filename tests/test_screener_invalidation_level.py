"""WS B — `_to_candidate` captures the trap-high invalidation level.

Only an EXACT `false_breakout` (with a `false_high` key-point) yields a
`bearish_invalidation_level`. `false_breakout_hs` (no `false_high`) and every
bullish pattern yield None — the scope guard the reclaim path relies on.
"""

from __future__ import annotations

from rainier.analysis.stock_screener import _to_candidate
from rainier.core.types import MoneyFlowSignal, PatternSignal, StockScreenResult


def _result(pattern: PatternSignal | None) -> StockScreenResult:
    return StockScreenResult(
        symbol="CRDO",
        name="CRDO",
        sector="Tech",
        money_flow_score=0.5,
        long_short="Long in",
        qu100_rank=1,
        sector_trend="bullish",
        sector_boost=0.0,
        patterns=[pattern] if pattern else [],
        best_pattern=pattern,
        composite_score=0.5,
        recommendation="watch",
        entry_price=pattern.entry_price if pattern else None,
        stop_loss=pattern.stop_loss if pattern else None,
        target=pattern.target_wave1 if pattern else None,
    )


def _signal_map() -> dict[str, MoneyFlowSignal]:
    return {
        "CRDO": MoneyFlowSignal(
            symbol="CRDO", rank=1, rank_change=0, long_short="Long in",
            capital_flow_direction="+", days_in_top100=5, sector="Tech",
            industry="Semis", signal_strength=0.5,
        )
    }


def _false_breakout(pattern_type: str, *, with_false_high: bool) -> PatternSignal:
    key_points = {"prior_high": 95.0, "rejection_bar": 3}
    if with_false_high:
        key_points["false_high"] = 100.0
    return PatternSignal(
        symbol="CRDO",
        pattern_type=pattern_type,
        direction="bearish",
        status="confirmed",
        confidence=0.9,
        entry_price=95.0,
        stop_loss=102.0,
        target_wave1=85.0,
        key_points=key_points,
    )


def test_exact_false_breakout_captures_invalidation_level():
    pattern = _false_breakout("false_breakout", with_false_high=True)
    cand = _to_candidate(_result(pattern), _signal_map(), {"CRDO": 96.0})
    # The trap high is false_high (100.0), NOT stop_loss (102.0).
    assert cand.bearish_invalidation_level == 100.0


def test_false_breakout_hs_yields_no_invalidation_level():
    pattern = _false_breakout("false_breakout_hs", with_false_high=False)
    cand = _to_candidate(_result(pattern), _signal_map(), {"CRDO": 96.0})
    assert cand.bearish_invalidation_level is None


def test_bullish_pattern_yields_no_invalidation_level():
    pattern = PatternSignal(
        symbol="CRDO",
        pattern_type="w_bottom",
        direction="bullish",
        status="confirmed",
        confidence=0.8,
        entry_price=95.0,
        stop_loss=90.0,
        target_wave1=110.0,
        key_points={"first_low": 88.0},
    )
    cand = _to_candidate(_result(pattern), _signal_map(), {"CRDO": 96.0})
    assert cand.bearish_invalidation_level is None


def test_no_pattern_yields_no_invalidation_level():
    cand = _to_candidate(_result(None), _signal_map(), {"CRDO": 96.0})
    assert cand.bearish_invalidation_level is None
