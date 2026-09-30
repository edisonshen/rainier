"""Tests for horizontal S/R detection."""

import pandas as pd

from rainier.analysis.pivots import compute_atr, detect_pivots
from rainier.analysis.sr_horizontal import detect_horizontal_sr
from rainier.core.config import PivotConfig
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

    def test_touches_exactly_at_min_forms_level(self, base_timestamp):
        """Boundary: exactly min_touches pivots in one cluster → one level."""
        from datetime import timedelta

        from rainier.core.config import SRHorizontalConfig
        from rainier.core.types import Pivot

        pivots = [
            Pivot(index=i, price=100.0, timestamp=base_timestamp + timedelta(hours=i),
                  is_high=True)
            for i in (2, 5, 8)
        ]
        df = pd.DataFrame([
            {"timestamp": base_timestamp + timedelta(hours=i),
             "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100}
            for i in range(20)
        ])
        atr = compute_atr(df)
        levels = detect_horizontal_sr(pivots, df, atr, SRHorizontalConfig(min_touches=3))
        assert len(levels) == 1
        assert levels[0].touches == 3
        assert levels[0].price == 100.0
        assert levels[0].role == SRRole.RESISTANCE

    def test_touches_one_below_min_returns_empty(self, base_timestamp):
        from datetime import timedelta

        from rainier.core.config import SRHorizontalConfig
        from rainier.core.types import Pivot

        pivots = [
            Pivot(index=i, price=100.0, timestamp=base_timestamp + timedelta(hours=i),
                  is_high=True)
            for i in (2, 5)
        ]
        df = pd.DataFrame([
            {"timestamp": base_timestamp + timedelta(hours=i),
             "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100}
            for i in range(20)
        ])
        atr = compute_atr(df)
        levels = detect_horizontal_sr(pivots, df, atr, SRHorizontalConfig(min_touches=3))
        assert levels == []

    def test_half_highs_half_lows_is_support(self, base_timestamp):
        """Role: n_highs must be strictly more than half for resistance."""
        from datetime import timedelta

        from rainier.core.config import SRHorizontalConfig
        from rainier.core.types import Pivot

        pivots = [
            Pivot(index=2, price=100.0, timestamp=base_timestamp, is_high=True),
            Pivot(index=5, price=100.0, timestamp=base_timestamp, is_high=True),
            Pivot(index=8, price=100.0, timestamp=base_timestamp, is_high=False),
            Pivot(index=11, price=100.0, timestamp=base_timestamp, is_high=False),
        ]
        df = pd.DataFrame([
            {"timestamp": base_timestamp + timedelta(hours=i),
             "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100}
            for i in range(20)
        ])
        atr = compute_atr(df)
        levels = detect_horizontal_sr(pivots, df, atr, SRHorizontalConfig(min_touches=3))
        assert len(levels) == 1
        assert levels[0].role == SRRole.SUPPORT

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
