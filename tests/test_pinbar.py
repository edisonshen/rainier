"""Tests for pin bar detection and pin bar line derivation."""

import pandas as pd
from datetime import datetime, timedelta

from quant.analysis.pinbar import (
    derive_pin_bar_lines,
    detect_pin_bars_raw,
    match_pin_bars_to_levels,
)
from quant.core.config import PinBarConfig
from quant.core.types import Direction, SRLevel, SRRole, SRType


class TestDetectPinBarsRaw:
    def test_detects_bullish_pin_bar(self, pin_bar_candles):
        """Pin bar at index 10 should be detected by shape alone."""
        pin_bars = detect_pin_bars_raw(pin_bar_candles)
        assert len(pin_bars) >= 1
        bullish = [pb for pb in pin_bars if pb.direction == Direction.LONG]
        assert len(bullish) >= 1

    def test_pin_bar_has_valid_wick_ratio(self, pin_bar_candles):
        pin_bars = detect_pin_bars_raw(pin_bar_candles)
        for pb in pin_bars:
            assert pb.wick_ratio >= 2.0

    def test_no_pin_bars_on_flat_data(self, flat_candles):
        pin_bars = detect_pin_bars_raw(flat_candles)
        assert pin_bars == []

    def test_raw_pin_bars_have_no_sr(self, pin_bar_candles):
        """Raw detection should not assign S/R levels."""
        pin_bars = detect_pin_bars_raw(pin_bar_candles)
        for pb in pin_bars:
            assert pb.nearest_sr is None

    def test_doji_does_not_crash(self, base_timestamp):
        rows = [
            {"timestamp": base_timestamp + timedelta(hours=i),
             "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000.0}
            for i in range(10)
        ]
        df = pd.DataFrame(rows)
        pin_bars = detect_pin_bars_raw(df)
        assert isinstance(pin_bars, list)


class TestDerivePinBarLines:
    def test_clusters_wick_tips(self, pin_bar_candles):
        pin_bars = detect_pin_bars_raw(pin_bar_candles)
        if not pin_bars:
            return  # skip if no pin bars in fixture
        levels = derive_pin_bar_lines(pin_bars, atr=5.0)
        assert len(levels) > 0
        for level in levels:
            assert level.sr_type == SRType.HORIZONTAL
            assert level.touches >= 1

    def test_empty_pin_bars_returns_empty(self):
        levels = derive_pin_bar_lines([], atr=5.0)
        assert levels == []

    def test_strength_increases_with_touches(self):
        """More pin bars at same price → higher strength."""
        from quant.core.types import Candle, PinBar, Timeframe

        pbs = []
        for i in range(5):
            candle = Candle(
                timestamp=datetime(2025, 1, 1, i),
                open=104.0, high=105.0, low=99.5, close=104.5,
                volume=1000, symbol="NQ", timeframe=Timeframe.H1,
            )
            pbs.append(PinBar(candle=candle, index=i + 5, direction=Direction.LONG, wick_ratio=3.0))

        levels = derive_pin_bar_lines(pbs, atr=5.0)
        # All 5 pin bars at same wick tip (~99.5) should cluster into 1 level
        assert len(levels) == 1
        assert levels[0].touches == 5
        assert levels[0].strength > 0.5


class TestMatchPinBarsToLevels:
    def test_matches_when_near_level(self, pin_bar_candles):
        pin_bars = detect_pin_bars_raw(pin_bar_candles)
        sr_levels = [
            SRLevel(price=99.5, sr_type=SRType.HORIZONTAL,
                    role=SRRole.SUPPORT, strength=0.8, touches=3)
        ]
        matched = match_pin_bars_to_levels(pin_bars, sr_levels, proximity_pct=0.01)
        assert len(matched) >= 1
        for pb in matched:
            assert pb.nearest_sr is not None

    def test_no_match_when_far(self, pin_bar_candles):
        pin_bars = detect_pin_bars_raw(pin_bar_candles)
        sr_levels = [
            SRLevel(price=200.0, sr_type=SRType.HORIZONTAL,
                    role=SRRole.SUPPORT, strength=0.8, touches=3)
        ]
        matched = match_pin_bars_to_levels(pin_bars, sr_levels, proximity_pct=0.005)
        assert matched == []
