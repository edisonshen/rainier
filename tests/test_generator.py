"""Tests for signal generation."""

from datetime import datetime

import pandas as pd

from rainier.core.config import SignalConfig
from rainier.core.types import (
    AnalysisResult,
    Candle,
    Direction,
    PinBar,
    SRLevel,
    SRRole,
    SRType,
    Timeframe,
)
from rainier.signals.generator import generate_signals


def _make_analysis_with_pin_bar() -> tuple[AnalysisResult, pd.DataFrame]:
    sr_support = SRLevel(
        price=100.0, sr_type=SRType.HORIZONTAL,
        role=SRRole.SUPPORT, strength=0.95, touches=5,
        source_tf=Timeframe.H1,
    )
    sr_support_htf = SRLevel(
        price=104.3, sr_type=SRType.HORIZONTAL,
        role=SRRole.SUPPORT, strength=0.85, touches=3,
        source_tf=Timeframe.H4,
    )
    sr_support_dtf = SRLevel(
        price=104.6, sr_type=SRType.HORIZONTAL,
        role=SRRole.SUPPORT, strength=0.80, touches=2,
        source_tf=Timeframe.D1,
    )
    sr_resistance = SRLevel(
        price=120.0, sr_type=SRType.HORIZONTAL,
        role=SRRole.RESISTANCE, strength=0.9, touches=4,
        source_tf=Timeframe.D1,
    )
    candle = Candle(
        timestamp=datetime(2025, 1, 1, 10, 0),
        open=104.0, high=105.0, low=99.5, close=104.5,
        volume=3000.0, symbol="NQ", timeframe=Timeframe.H1,
    )
    pin_bar = PinBar(
        candle=candle, index=10, direction=Direction.LONG,
        wick_ratio=4.5, nearest_sr=sr_support, sr_distance_pct=0.005,
    )

    result = AnalysisResult(
        symbol="NQ", timeframe=Timeframe.H1,
        sr_levels=[sr_support, sr_support_htf, sr_support_dtf, sr_resistance],
        pin_bars=[pin_bar],
        bias=Direction.LONG,
    )

    # DataFrame with lower avg volume so pin bar volume spike scores high
    rows = [
        {"timestamp": datetime(2025, 1, 1, i), "open": 104, "high": 106,
         "low": 103, "close": 105, "volume": 1000}
        for i in range(20)
    ]
    df = pd.DataFrame(rows)
    return result, df


class TestGenerateSignals:
    def test_generates_signal_from_pin_bar(self):
        result, df = _make_analysis_with_pin_bar()
        signals = generate_signals(result, df)
        assert len(signals) >= 1

    def test_signal_has_entry_sl_tp(self):
        result, df = _make_analysis_with_pin_bar()
        signals = generate_signals(result, df)
        for sig in signals:
            assert sig.entry_price > 0
            assert sig.stop_loss > 0
            assert sig.take_profit > 0

    def test_long_signal_sl_below_entry(self):
        result, df = _make_analysis_with_pin_bar()
        signals = generate_signals(result, df)
        for sig in signals:
            if sig.direction == Direction.LONG:
                assert sig.stop_loss < sig.entry_price
                assert sig.take_profit > sig.entry_price

    def test_rr_ratio_computed(self):
        result, df = _make_analysis_with_pin_bar()
        signals = generate_signals(result, df)
        for sig in signals:
            assert sig.rr_ratio > 0

    def test_no_signals_below_confidence_threshold(self):
        result, df = _make_analysis_with_pin_bar()
        config = SignalConfig()
        config.scorer.min_confidence = 0.99  # very high threshold
        signals = generate_signals(result, df, config)
        assert signals == []

    def test_no_pin_bars_no_signals(self):
        result = AnalysisResult(symbol="NQ", timeframe=Timeframe.H1)
        df = pd.DataFrame([
            {"timestamp": datetime(2025, 1, 1), "open": 100, "high": 101,
             "low": 99, "close": 100, "volume": 100}
        ] * 5)
        signals = generate_signals(result, df)
        assert signals == []

    def test_tp_uses_next_sr_when_available(self):
        """TP should target the nearest S/R level above entry for LONG."""
        result, df = _make_analysis_with_pin_bar()
        signals = generate_signals(result, df)
        long_signals = [s for s in signals if s.direction == Direction.LONG]
        if long_signals:
            # Nearest S/R above entry (100.0) is sr_support_htf at 104.3
            assert long_signals[0].take_profit == 104.3

    def test_rr_exactly_at_min_passes_just_below_filtered(self):
        """Boundary: rr == min_rr_ratio must pass (filter is strict <)."""
        import math

        result, df = _make_analysis_with_pin_bar()
        pb = result.pin_bars[0]
        entry = 100.0
        sl = pb.candle.low - pb.candle.range * 0.1
        rr = (104.3 - entry) / (entry - sl)  # TP = next S/R at 104.3

        config = SignalConfig(min_rr_ratio=rr)
        config.scorer.min_confidence = 0.0
        signals = generate_signals(result, df, config)
        assert len(signals) == 1

        config_above = SignalConfig(min_rr_ratio=math.nextafter(rr, math.inf))
        config_above.scorer.min_confidence = 0.0
        assert generate_signals(result, df, config_above) == []


def _pin_bar(direction, sr_price, o, h, low, c):
    sr = SRLevel(
        price=sr_price, sr_type=SRType.HORIZONTAL,
        role=SRRole.SUPPORT if direction == Direction.LONG else SRRole.RESISTANCE,
        strength=0.9, touches=4,
    )
    candle = Candle(
        timestamp=datetime(2025, 1, 1, 10), open=o, high=h, low=low, close=c,
        volume=1000.0, symbol="NQ", timeframe=Timeframe.H1,
    )
    return PinBar(candle=candle, index=10, direction=direction,
                  wick_ratio=3.0, nearest_sr=sr)


class TestComputeLevels:
    def test_long_levels_hand_computed(self):
        from pytest import approx

        from rainier.signals.generator import _compute_levels

        pb = _pin_bar(Direction.LONG, sr_price=100.0, o=104.0, h=105.0, low=99.5, c=104.5)
        entry, sl, tp = _compute_levels(pb, [], SignalConfig(default_rr_target=2.0))
        assert entry == 100.0
        assert sl == approx(99.5 - 5.5 * 0.1)  # low - range*0.1
        assert tp == approx(entry + (entry - sl) * 2.0)  # default R:R fallback

    def test_long_tp_snaps_to_next_resistance(self):
        from rainier.signals.generator import _compute_levels

        pb = _pin_bar(Direction.LONG, sr_price=100.0, o=104.0, h=105.0, low=99.5, c=104.5)
        res = SRLevel(price=103.0, sr_type=SRType.HORIZONTAL,
                      role=SRRole.RESISTANCE, strength=0.8, touches=3)
        far = SRLevel(price=108.0, sr_type=SRType.HORIZONTAL,
                      role=SRRole.RESISTANCE, strength=0.9, touches=5)
        _, _, tp = _compute_levels(pb, [far, res], SignalConfig())
        assert tp == 103.0  # nearest above entry, not strongest

    def test_short_levels_hand_computed(self):
        from pytest import approx

        from rainier.signals.generator import _compute_levels

        pb = _pin_bar(Direction.SHORT, sr_price=105.0, o=101.0, h=105.5, low=100.5, c=100.8)
        entry, sl, tp = _compute_levels(pb, [], SignalConfig(default_rr_target=2.0))
        assert entry == 105.0
        assert sl == approx(105.5 + 5.0 * 0.1)  # high + range*0.1
        assert tp == approx(entry - (sl - entry) * 2.0)

    def test_short_tp_snaps_to_next_support(self):
        from rainier.signals.generator import _compute_levels

        pb = _pin_bar(Direction.SHORT, sr_price=105.0, o=101.0, h=105.5, low=100.5, c=100.8)
        sup = SRLevel(price=101.0, sr_type=SRType.HORIZONTAL,
                      role=SRRole.SUPPORT, strength=0.8, touches=3)
        _, _, tp = _compute_levels(pb, [sup], SignalConfig())
        assert tp == 101.0

    def test_no_nearest_sr_returns_none(self):
        from rainier.signals.generator import _compute_levels

        candle = Candle(
            timestamp=datetime(2025, 1, 1, 10), open=104.0, high=105.0,
            low=99.5, close=104.5, volume=1000.0, symbol="NQ", timeframe=Timeframe.H1,
        )
        pb = PinBar(candle=candle, index=10, direction=Direction.LONG, wick_ratio=3.0)
        assert _compute_levels(pb, [], SignalConfig()) == (None, 0.0, 0.0)

    def test_long_with_non_positive_risk_invalid(self):
        from rainier.signals.generator import _compute_levels

        # SL (low - range*0.1 = 99.4) sits above the S/R entry at 99.0 → invalid
        pb = _pin_bar(Direction.LONG, sr_price=99.0, o=100.2, h=100.5, low=99.5, c=100.3)
        assert _compute_levels(pb, [], SignalConfig()) == (None, 0.0, 0.0)

    def test_short_with_non_positive_risk_invalid(self):
        from rainier.signals.generator import _compute_levels

        # SL (high + range*0.1 = 100.6) sits below the S/R entry at 101.0 → invalid
        pb = _pin_bar(Direction.SHORT, sr_price=101.0, o=100.2, h=100.5, low=99.5, c=100.3)
        assert _compute_levels(pb, [], SignalConfig()) == (None, 0.0, 0.0)
