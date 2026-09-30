"""Tests for directional bias determination."""

from datetime import datetime, timedelta

import pandas as pd

from rainier.analysis.bias import _sma_bias, determine_bias
from rainier.core.types import Direction, Pivot

BASE = datetime(2025, 1, 1)


def _pivot(index, price, is_high):
    return Pivot(index=index, price=price, timestamp=BASE + timedelta(hours=index), is_high=is_high)


def _close_df(closes):
    return pd.DataFrame({
        "timestamp": [BASE + timedelta(hours=i) for i in range(len(closes))],
        "open": closes,
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1000.0] * len(closes),
    })


def _structure(high_prices, low_prices):
    pivots = []
    idx = 0
    for h, lo in zip(high_prices, low_prices):
        pivots.append(_pivot(idx, lo, is_high=False))
        pivots.append(_pivot(idx + 1, h, is_high=True))
        idx += 2
    return pivots


class TestDetermineBias:
    def test_fewer_than_4_pivots_returns_none(self):
        pivots = [_pivot(i, 100.0 + i, is_high=bool(i % 2)) for i in range(3)]
        assert determine_bias(_close_df([100.0] * 20), pivots) is None

    def test_missing_swing_lows_returns_none(self):
        pivots = [_pivot(i, 100.0 + i, is_high=True) for i in range(4)]
        pivots.append(_pivot(4, 95.0, is_high=False))
        assert determine_bias(_close_df([100.0] * 20), pivots) is None

    def test_higher_highs_higher_lows_long(self):
        pivots = _structure([101, 102, 103, 104], [95, 96, 97, 98])
        assert determine_bias(_close_df([100.0] * 20), pivots) == Direction.LONG

    def test_lower_highs_lower_lows_short(self):
        pivots = _structure([104, 103, 102, 101], [98, 97, 96, 95])
        assert determine_bias(_close_df([100.0] * 20), pivots) == Direction.SHORT

    def test_mixed_structure_falls_back_to_sma(self):
        # highs: up, down, up (hh=2, lh=1); lows: down, up, down (hl=1, ll=2) → tie
        pivots = _structure([100, 101, 100, 101], [90, 89, 90, 89])
        rising = _close_df([100.0 + i * 0.5 for i in range(70)])
        falling = _close_df([100.0 - i * 0.5 for i in range(70)])
        assert determine_bias(rising, pivots) == Direction.LONG
        assert determine_bias(falling, pivots) == Direction.SHORT

    def test_mixed_structure_short_df_returns_none(self):
        pivots = _structure([100, 101, 100, 101], [90, 89, 90, 89])
        assert determine_bias(_close_df([100.0] * 20), pivots) is None


class TestSmaBias:
    def test_rising_sma_long(self):
        df = _close_df([100.0 + i * 0.5 for i in range(70)])
        assert _sma_bias(df) == Direction.LONG

    def test_falling_sma_short(self):
        df = _close_df([100.0 - i * 0.5 for i in range(70)])
        assert _sma_bias(df) == Direction.SHORT

    def test_flat_sma_none(self):
        df = _close_df([100.0] * 70)
        assert _sma_bias(df) is None

    def test_insufficient_history_none(self):
        # Needs period + 10 = 60 bars; 59 is one short
        df = _close_df([100.0 + i for i in range(59)])
        assert _sma_bias(df) is None

    def test_exactly_minimum_history_computes(self):
        df = _close_df([100.0 + i for i in range(60)])
        assert _sma_bias(df) == Direction.LONG

    def test_custom_period(self):
        df = _close_df([100.0 + i for i in range(25)])
        assert _sma_bias(df, period=10) == Direction.LONG
