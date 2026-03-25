"""Tests for rainier.analysis.pattern_primitives — swing points, breakouts, volume-price."""

from __future__ import annotations

import pandas as pd
import pytest

from rainier.analysis.pattern_primitives import (
    Breakout,
    SwingPoint,
    analyze_volume_price,
    detect_breakout,
    find_neckline,
    find_swing_points,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ohlcv(prices: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame from a list of close prices.

    Opens = closes shifted by 1, highs = close * 1.01, lows = close * 0.99.
    """
    n = len(prices)
    if volumes is None:
        volumes = [1000.0] * n
    df = pd.DataFrame({
        "open": [prices[max(0, i - 1)] for i in range(n)],
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": volumes,
    })
    return df


# ---------------------------------------------------------------------------
# find_swing_points
# ---------------------------------------------------------------------------


class TestFindSwingPoints:
    def test_find_swing_points_basic(self):
        """W-shape (100→80→100→70→100) spread over enough bars yields correct swings."""
        # Build a smooth W shape over 25 bars with lookback=3
        # Segment 1: 100→80 (bars 0-6), Segment 2: 80→100 (bars 6-12),
        # Segment 3: 100→70 (bars 12-18), Segment 4: 70→100 (bars 18-24)
        prices = (
            [100 - (20 * i / 6) for i in range(7)]       # 100 → 80
            + [80 + (20 * i / 6) for i in range(1, 7)]   # 80 → 100
            + [100 - (30 * i / 6) for i in range(1, 7)]  # 100 → 70
            + [70 + (30 * i / 6) for i in range(1, 7)]   # 70 → 100
        )
        df = make_ohlcv(prices)
        points = find_swing_points(df, lookback=3)

        swing_highs = [sp for sp in points if sp.type == "high"]
        swing_lows = [sp for sp in points if sp.type == "low"]

        # Should find at least the two lows (near 80, near 70) and the high between them
        assert len(swing_lows) >= 2, f"Expected >=2 swing lows, got {len(swing_lows)}"
        assert len(swing_highs) >= 1, f"Expected >=1 swing high, got {len(swing_highs)}"

        # The lowest swing low should be near 70
        lowest = min(swing_lows, key=lambda sp: sp.price)
        assert lowest.price == pytest.approx(70 * 0.99, rel=0.05)

    def test_find_swing_points_needs_minimum_bars(self):
        """DataFrame shorter than 2*lookback+1 returns empty list."""
        prices = [100.0, 90.0, 100.0]  # 3 bars, lookback=5 needs 11
        df = make_ohlcv(prices)
        points = find_swing_points(df, lookback=5)
        assert points == []


# ---------------------------------------------------------------------------
# detect_breakout
# ---------------------------------------------------------------------------


class TestDetectBreakout:
    def test_detect_breakout_up(self):
        """Price crosses above a level → Breakout with direction='up'."""
        prices = [90.0] * 5 + [95.0, 98.0, 101.0, 103.0, 105.0]
        df = make_ohlcv(prices)
        result = detect_breakout(df, level=100.0, direction="up")

        assert result is not None
        assert isinstance(result, Breakout)
        assert result.direction == "up"
        assert result.level == 100.0
        # Bar 7 has close=101
        assert result.bar_index == 7

    def test_detect_breakout_with_volume(self):
        """Volume > 1.5x average → with_volume=True."""
        # 20 bars of low volume, then breakout bar with high volume
        prices = [90.0] * 20 + [101.0]
        volumes = [1000.0] * 20 + [3000.0]  # 3x average
        df = make_ohlcv(prices, volumes)
        result = detect_breakout(df, level=100.0, direction="up", vol_multiplier=1.5)

        assert result is not None
        assert result.with_volume is True

    def test_detect_breakout_false(self):
        """Price crosses then reverses within 3 bars → false_breakout=True."""
        prices = [90.0] * 5 + [101.0, 99.0, 98.0]  # breaks above 100 then drops back
        df = make_ohlcv(prices)
        result = detect_breakout(df, level=100.0, direction="up")

        assert result is not None
        assert result.false_breakout is True

    def test_detect_breakout_not_found(self):
        """Price never crosses → returns None."""
        prices = [90.0] * 10
        df = make_ohlcv(prices)
        result = detect_breakout(df, level=100.0, direction="up")

        assert result is None


# ---------------------------------------------------------------------------
# analyze_volume_price
# ---------------------------------------------------------------------------


class TestAnalyzeVolumePrice:
    def test_analyze_volume_price_bullish(self):
        """Price up + volume up → no divergence."""
        # 21 bars: price trending up, volume increasing
        prices = [100.0 + i for i in range(21)]
        volumes = [1000.0 + i * 50 for i in range(21)]
        df = make_ohlcv(prices, volumes)
        signal = analyze_volume_price(df, window=20)

        assert signal.type == "price_up_vol_up"
        assert signal.divergence is False

    def test_analyze_volume_price_divergence(self):
        """Price up + volume down → divergence=True."""
        # 21 bars: price trending up, volume decreasing
        prices = [100.0 + i for i in range(21)]
        volumes = [2000.0 - i * 80 for i in range(21)]
        df = make_ohlcv(prices, volumes)
        signal = analyze_volume_price(df, window=20)

        assert signal.type == "price_up_vol_down"
        assert signal.divergence is True


# ---------------------------------------------------------------------------
# find_neckline
# ---------------------------------------------------------------------------


class TestFindNeckline:
    def test_find_neckline(self):
        """Swing points at similar levels → neckline is found."""
        swing_points = [
            SwingPoint(index=5, price=100.0, type="low", strength=3),
            SwingPoint(index=15, price=100.5, type="low", strength=3),
            SwingPoint(index=25, price=100.2, type="low", strength=3),
        ]
        # Need a dummy DataFrame (find_neckline uses df for reference only)
        df = make_ohlcv([100.0] * 30)
        neckline = find_neckline(swing_points, sp_type="low", df=df)

        assert neckline is not None
        assert neckline.type == "support"
        assert len(neckline.touch_points) >= 2
        assert neckline.price == pytest.approx(100.23, rel=0.02)

    def test_find_neckline_too_few_points(self):
        """Less than 2 swing points → returns None."""
        swing_points = [
            SwingPoint(index=5, price=100.0, type="low", strength=3),
        ]
        df = make_ohlcv([100.0] * 10)
        neckline = find_neckline(swing_points, sp_type="low", df=df)

        assert neckline is None
