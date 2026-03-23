"""Tests for horizontal S/R detection."""

import pandas as pd

from rainier.analysis.pivots import compute_atr, detect_pivots
from rainier.analysis.sr_horizontal import detect_horizontal_sr
from rainier.core.config import PivotConfig, SRHorizontalConfig
from rainier.core.types import SRRole, SRType


class TestDetectHorizontalSR:
    def test_no_levels_with_insufficient_pivots(self, flat_candles):
        """No pivots → no S/R levels."""
        pivots = detect_pivots(flat_candles)
        atr = compute_atr(flat_candles)
        levels = detect_horizontal_sr(pivots, flat_candles, atr)
        assert levels == []

    def test_finds_levels_in_swing_data(self, swing_candles):
        pivots = detect_pivots(swing_candles, PivotConfig(lookback=3))
        atr = compute_atr(swing_candles)
        levels = detect_horizontal_sr(pivots, swing_candles, atr)
        assert len(levels) > 0

    def test_levels_have_correct_type(self, swing_candles):
        pivots = detect_pivots(swing_candles, PivotConfig(lookback=3))
        atr = compute_atr(swing_candles)
        levels = detect_horizontal_sr(pivots, swing_candles, atr)
        for level in levels:
            assert level.sr_type == SRType.HORIZONTAL
            assert level.role in (SRRole.SUPPORT, SRRole.RESISTANCE)

    def test_strength_between_0_and_1(self, swing_candles):
        pivots = detect_pivots(swing_candles, PivotConfig(lookback=3))
        atr = compute_atr(swing_candles)
        levels = detect_horizontal_sr(pivots, swing_candles, atr)
        for level in levels:
            assert 0.0 <= level.strength <= 1.0

    def test_sorted_by_strength_descending(self, swing_candles):
        pivots = detect_pivots(swing_candles, PivotConfig(lookback=3))
        atr = compute_atr(swing_candles)
        levels = detect_horizontal_sr(pivots, swing_candles, atr)
        strengths = [l.strength for l in levels]
        assert strengths == sorted(strengths, reverse=True)

    def test_single_pivot_returns_empty(self, base_timestamp):
        """Single pivot — not enough touches (need 3+), returns empty."""
        from datetime import timedelta
        from rainier.core.types import Pivot

        pivots = [Pivot(index=5, price=100.0, timestamp=base_timestamp, is_high=True)]
        df = pd.DataFrame([
            {"timestamp": base_timestamp + timedelta(hours=i),
             "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100}
            for i in range(20)
        ])
        atr = compute_atr(df)
        levels = detect_horizontal_sr(pivots, df, atr)
        assert levels == []
