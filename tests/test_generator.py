"""Tests for signal generation."""

import pandas as pd
from datetime import datetime

from quant.core.config import SignalConfig
from quant.core.types import (
    AnalysisResult, Candle, Direction, PinBar, SRLevel, SRRole, SRType, Timeframe,
)
from quant.signals.generator import generate_signals


def _make_analysis_with_pin_bar() -> tuple[AnalysisResult, pd.DataFrame]:
    sr_support = SRLevel(
        price=100.0, sr_type=SRType.HORIZONTAL,
        role=SRRole.SUPPORT, strength=0.95, touches=5,
    )
    sr_resistance = SRLevel(
        price=120.0, sr_type=SRType.HORIZONTAL,
        role=SRRole.RESISTANCE, strength=0.9, touches=4,
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
        sr_levels=[sr_support, sr_resistance],
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
        """TP should target the next resistance level for LONG."""
        result, df = _make_analysis_with_pin_bar()
        signals = generate_signals(result, df)
        long_signals = [s for s in signals if s.direction == Direction.LONG]
        if long_signals:
            # Next resistance is at 120.0
            assert long_signals[0].take_profit == 120.0
