"""Tests for diagonal S/R (trendline) detection."""

from rainier.analysis.pivots import compute_atr, detect_pivots
from rainier.analysis.sr_diagonal import detect_diagonal_sr
from rainier.core.config import PivotConfig
from rainier.core.types import SRType


class TestDetectDiagonalSR:
    def test_finds_trendlines_in_swing_data(self, swing_candles):
        """Swing data has clear zigzag pivots → trendlines should be found."""
        pivots = detect_pivots(swing_candles, PivotConfig(lookback=3))
        atr = compute_atr(swing_candles)
        levels = detect_diagonal_sr(pivots, swing_candles, atr)
        assert len(levels) > 0
        for level in levels:
            assert level.sr_type == SRType.DIAGONAL

    def test_trendlines_have_slope_attribute(self, swing_candles):
        """All diagonal levels should have a slope (may be 0 if swing highs are at same price)."""
        pivots = detect_pivots(swing_candles, PivotConfig(lookback=3))
        atr = compute_atr(swing_candles)
        levels = detect_diagonal_sr(pivots, swing_candles, atr)
        for level in levels:
            assert isinstance(level.slope, float)

    def test_no_trendlines_on_flat_data(self, flat_candles):
        pivots = detect_pivots(flat_candles, PivotConfig(lookback=3))
        atr = compute_atr(flat_candles)
        levels = detect_diagonal_sr(pivots, flat_candles, atr)
        assert levels == []

    def test_strength_between_0_and_1(self, swing_candles):
        pivots = detect_pivots(swing_candles, PivotConfig(lookback=3))
        atr = compute_atr(swing_candles)
        levels = detect_diagonal_sr(pivots, swing_candles, atr)
        for level in levels:
            assert 0.0 <= level.strength <= 1.0

    def test_touches_exactly_at_min_forms_line(self, base_timestamp):
        """Boundary: exactly min_touches collinear swing highs → trendline kept."""
        from datetime import timedelta

        import pandas as pd

        from rainier.core.config import SRDiagonalConfig
        from rainier.core.types import Pivot, SRRole

        pivots = [
            Pivot(index=i, price=100.0 + i, timestamp=base_timestamp + timedelta(hours=i),
                  is_high=True)
            for i in (0, 5, 10)
        ]
        df = pd.DataFrame([
            {"timestamp": base_timestamp + timedelta(hours=i),
             "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100}
            for i in range(20)
        ])
        atr = compute_atr(df)
        levels = detect_diagonal_sr(pivots, df, atr, SRDiagonalConfig(min_touches=3))
        assert len(levels) >= 1
        for level in levels:
            assert level.touches == 3
            assert level.slope == 1.0
            assert level.role == SRRole.RESISTANCE

    def test_touches_one_below_min_returns_empty(self, base_timestamp):
        from datetime import timedelta

        import pandas as pd

        from rainier.core.config import SRDiagonalConfig
        from rainier.core.types import Pivot

        # Only 2 collinear points — every candidate pair touches 2 < min_touches=3
        pivots = [
            Pivot(index=i, price=100.0 + i, timestamp=base_timestamp + timedelta(hours=i),
                  is_high=True)
            for i in (0, 5)
        ]
        df = pd.DataFrame([
            {"timestamp": base_timestamp + timedelta(hours=i),
             "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100}
            for i in range(20)
        ])
        atr = compute_atr(df)
        levels = detect_diagonal_sr(pivots, df, atr, SRDiagonalConfig(min_touches=3))
        assert levels == []

    def test_support_lines_from_swing_lows(self, base_timestamp):
        from datetime import timedelta

        import pandas as pd

        from rainier.core.types import Pivot, SRRole

        pivots = [
            Pivot(index=i, price=100.0 - i * 0.1, timestamp=base_timestamp + timedelta(hours=i),
                  is_high=False)
            for i in (0, 5, 10)
        ]
        df = pd.DataFrame([
            {"timestamp": base_timestamp + timedelta(hours=i),
             "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100}
            for i in range(20)
        ])
        atr = compute_atr(df)
        levels = detect_diagonal_sr(pivots, df, atr)
        assert len(levels) >= 1
        for level in levels:
            assert level.role == SRRole.SUPPORT
            assert level.slope < 0

    def test_only_one_swing_point_returns_empty(self, base_timestamp):
        """Need at least 2 swing points to fit a line."""
        from datetime import timedelta

        from rainier.core.types import Pivot

        pivots = [Pivot(index=5, price=100.0, timestamp=base_timestamp, is_high=True)]
        import pandas as pd

        df = pd.DataFrame([
            {"timestamp": base_timestamp + timedelta(hours=i),
             "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100}
            for i in range(20)
        ])
        atr = compute_atr(df)
        levels = detect_diagonal_sr(pivots, df, atr)
        assert levels == []
