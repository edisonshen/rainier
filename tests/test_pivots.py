"""Tests for swing high/low detection."""

from rainier.analysis.pivots import compute_atr, detect_pivots
from rainier.core.config import PivotConfig


class TestDetectPivots:
    def test_no_pivots_on_flat_data(self, flat_candles):
        pivots = detect_pivots(flat_candles, PivotConfig(lookback=5))
        assert len(pivots) == 0

    def test_finds_pivots_on_swing_data(self, swing_candles):
        pivots = detect_pivots(swing_candles, PivotConfig(lookback=3))
        assert len(pivots) > 0
        highs = [p for p in pivots if p.is_high]
        lows = [p for p in pivots if not p.is_high]
        assert len(highs) > 0
        assert len(lows) > 0

    def test_pivot_prices_match_extremes(self, swing_candles):
        pivots = detect_pivots(swing_candles, PivotConfig(lookback=3))
        for p in pivots:
            if p.is_high:
                assert p.price == swing_candles.iloc[p.index]["high"]
            else:
                assert p.price == swing_candles.iloc[p.index]["low"]

    def test_too_few_bars_returns_empty(self, base_timestamp):
        import pandas as pd
        from datetime import timedelta

        df = pd.DataFrame([
            {"timestamp": base_timestamp + timedelta(hours=i),
             "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100}
            for i in range(5)
        ])
        pivots = detect_pivots(df, PivotConfig(lookback=5))
        assert pivots == []

    def test_single_bar_returns_empty(self, base_timestamp):
        import pandas as pd

        df = pd.DataFrame([
            {"timestamp": base_timestamp, "open": 100, "high": 110,
             "low": 90, "close": 105, "volume": 100}
        ])
        assert detect_pivots(df) == []


class TestComputeATR:
    def test_atr_returns_series(self, swing_candles):
        atr = compute_atr(swing_candles)
        assert len(atr) == len(swing_candles)
        assert atr.iloc[-1] > 0

    def test_atr_flat_data(self, flat_candles):
        atr = compute_atr(flat_candles)
        # Flat data: all ranges are 1.0
        assert abs(atr.iloc[-1] - 1.0) < 0.1
