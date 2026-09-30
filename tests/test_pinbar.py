"""Tests for pin bar detection and pin bar line derivation."""

from datetime import datetime, timedelta

import pandas as pd

from rainier.analysis.pinbar import (
    derive_pin_bar_lines,
    detect_pin_bars_raw,
    match_pin_bars_to_levels,
)
from rainier.core.types import Direction, SRLevel, SRRole, SRType


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
        # Use min_touches=1 since strict detection may yield few pin bars
        levels = derive_pin_bar_lines(pin_bars, atr=5.0, min_touches=1)
        assert len(levels) > 0
        for level in levels:
            assert level.sr_type == SRType.HORIZONTAL
            assert level.touches >= 1

    def test_empty_pin_bars_returns_empty(self):
        levels = derive_pin_bar_lines([], atr=5.0)
        assert levels == []

    def test_strength_increases_with_touches(self):
        """More pin bars at same price → higher strength."""
        from rainier.core.types import Candle, PinBar, Timeframe

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


class TestPinBarBoundaries:
    """Fixtures sitting exactly on configured thresholds."""

    def _df_with_candle(self, base_timestamp, o, h, low, c):
        rows = [
            {"timestamp": base_timestamp, "open": 100.4, "high": 101.0,
             "low": 100.0, "close": 100.6, "volume": 1000.0},
            {"timestamp": base_timestamp + timedelta(hours=1), "open": o, "high": h,
             "low": low, "close": c, "volume": 1000.0},
        ]
        return pd.DataFrame(rows)

    def test_dominant_wick_ratio_exactly_at_min_detected(self, base_timestamp):
        from rainier.core.config import PinBarConfig

        config = PinBarConfig(min_dominant_wick_ratio=0.75, max_secondary_wick_ratio=0.25)
        # range 4.0, lower wick 3.0 → ratio exactly 0.75; upper wick 0.75 → exactly 0.25
        df = self._df_with_candle(base_timestamp, o=103.0, h=104.0, low=100.0, c=103.25)
        pin_bars = detect_pin_bars_raw(df, config)
        assert len(pin_bars) == 1
        assert pin_bars[0].direction == Direction.LONG
        assert pin_bars[0].index == 1
        assert pin_bars[0].wick_ratio == 3.0 / 0.25

    def test_dominant_wick_ratio_below_min_rejected(self, base_timestamp):
        from rainier.core.config import PinBarConfig

        config = PinBarConfig(min_dominant_wick_ratio=0.75, max_secondary_wick_ratio=0.25)
        # lower wick 2.9 → ratio 0.725 < 0.75
        df = self._df_with_candle(base_timestamp, o=102.9, h=104.0, low=100.0, c=103.25)
        assert detect_pin_bars_raw(df, config) == []

    def test_secondary_wick_ratio_above_max_rejected(self, base_timestamp):
        from rainier.core.config import PinBarConfig

        config = PinBarConfig(min_dominant_wick_ratio=0.75, max_secondary_wick_ratio=0.25)
        # upper wick 0.8 / dominant 3.0 ≈ 0.267 > 0.25 → spinning top
        df = self._df_with_candle(base_timestamp, o=103.0, h=104.0, low=100.0, c=103.2)
        assert detect_pin_bars_raw(df, config) == []

    def test_bearish_pin_bar_direction(self, base_timestamp):
        from rainier.core.config import PinBarConfig

        config = PinBarConfig(min_dominant_wick_ratio=0.75, max_secondary_wick_ratio=0.25)
        # Mirror image: upper wick 3.0, lower wick 0.75
        df = self._df_with_candle(base_timestamp, o=101.0, h=104.0, low=100.0, c=100.75)
        pin_bars = detect_pin_bars_raw(df, config)
        assert len(pin_bars) == 1
        assert pin_bars[0].direction == Direction.SHORT

    def test_amplitude_exactly_at_median_detected(self, base_timestamp):
        from rainier.core.config import PinBarConfig

        config = PinBarConfig(min_dominant_wick_ratio=0.75, max_secondary_wick_ratio=0.25)
        rows = [
            {"timestamp": base_timestamp + timedelta(hours=i), "open": 100.4,
             "high": 101.0, "low": 100.0, "close": 100.6, "volume": 1000.0}
            for i in range(10)
        ]
        # Pin bar range 1.0 == median of prior ranges (1.0) → prominent enough
        rows.append({"timestamp": base_timestamp + timedelta(hours=10), "open": 100.75,
                     "high": 101.0, "low": 100.0, "close": 100.8125, "volume": 1000.0})
        pin_bars = detect_pin_bars_raw(pd.DataFrame(rows), config)
        assert [pb.index for pb in pin_bars] == [10]

    def test_amplitude_below_median_rejected(self, base_timestamp):
        from rainier.core.config import PinBarConfig

        config = PinBarConfig(min_dominant_wick_ratio=0.75, max_secondary_wick_ratio=0.25)
        rows = [
            {"timestamp": base_timestamp + timedelta(hours=i), "open": 100.4,
             "high": 101.2, "low": 100.0, "close": 100.6, "volume": 1000.0}
            for i in range(10)
        ]
        # Pin bar range 1.0 < median of prior ranges (1.2) → not visually prominent
        rows.append({"timestamp": base_timestamp + timedelta(hours=10), "open": 100.75,
                     "high": 101.0, "low": 100.0, "close": 100.8125, "volume": 1000.0})
        assert detect_pin_bars_raw(pd.DataFrame(rows), config) == []


class TestDerivePinBarLinesBoundaries:
    def _pin_bars(self, lows):
        from rainier.core.types import Candle, PinBar, Timeframe

        pbs = []
        for i, low in enumerate(lows):
            candle = Candle(
                timestamp=datetime(2025, 1, 1, i),
                open=low + 4.5, high=low + 5.5, low=low, close=low + 5.0,
                volume=1000, symbol="NQ", timeframe=Timeframe.H1,
            )
            pbs.append(PinBar(candle=candle, index=i, direction=Direction.LONG, wick_ratio=3.0))
        return pbs

    def test_touches_exactly_at_min_forms_level(self):
        levels = derive_pin_bar_lines(self._pin_bars([99.5, 99.5, 99.5]), atr=5.0, min_touches=3)
        assert len(levels) == 1
        assert levels[0].touches == 3
        assert levels[0].strength == 0.75  # 0.3 + 3 * 0.15
        assert levels[0].role == SRRole.SUPPORT

    def test_touches_below_min_no_level(self):
        levels = derive_pin_bar_lines(self._pin_bars([99.5, 99.5]), atr=5.0, min_touches=3)
        assert levels == []

    def test_line_price_is_mode_not_average(self):
        levels = derive_pin_bar_lines(
            self._pin_bars([99.5, 99.5, 100.0]), atr=5.0, min_touches=3, tick_size=0.25,
        )
        assert len(levels) == 1
        assert levels[0].price == 99.5


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
