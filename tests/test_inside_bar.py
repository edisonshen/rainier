"""Tests for inside bar detection."""

import pandas as pd

from rainier.analysis.inside_bar import detect_inside_bars


class TestDetectInsideBars:
    def test_finds_inside_bars(self, inside_bar_candles):
        inside_bars = detect_inside_bars(inside_bar_candles)
        assert len(inside_bars) >= 1

    def test_inside_bar_range_within_mother(self, inside_bar_candles):
        inside_bars = detect_inside_bars(inside_bar_candles)
        for ib in inside_bars:
            assert ib.candle.high <= ib.mother_candle.high
            assert ib.candle.low >= ib.mother_candle.low

    def test_compression_ratio_less_than_1(self, inside_bar_candles):
        inside_bars = detect_inside_bars(inside_bar_candles)
        for ib in inside_bars:
            assert 0.0 < ib.compression_ratio <= 1.0

    def test_no_inside_bars_on_single_candle(self, base_timestamp):
        df = pd.DataFrame([{
            "timestamp": base_timestamp, "open": 100, "high": 110,
            "low": 90, "close": 105, "volume": 100,
        }])
        inside_bars = detect_inside_bars(df)
        assert inside_bars == []

    def test_equal_high_low_skipped(self, base_timestamp):
        """Mother bar with zero range should be skipped."""
        from datetime import timedelta

        df = pd.DataFrame([
            {"timestamp": base_timestamp, "open": 100, "high": 100,
             "low": 100, "close": 100, "volume": 100},
            {"timestamp": base_timestamp + timedelta(hours=1), "open": 100, "high": 100,
             "low": 100, "close": 100, "volume": 100},
        ])
        inside_bars = detect_inside_bars(df)
        assert inside_bars == []
