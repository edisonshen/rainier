"""Shared test fixtures: synthetic OHLCV data with known patterns."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest


@pytest.fixture
def base_timestamp():
    return datetime(2025, 1, 1, 9, 0, 0)


@pytest.fixture
def flat_candles(base_timestamp):
    """20 candles with flat price action around 100.0 — no pivots expected."""
    rows = []
    for i in range(20):
        rows.append({
            "timestamp": base_timestamp + timedelta(hours=i),
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 1000.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def trending_up_candles(base_timestamp):
    """50 candles with clear uptrend — higher highs and higher lows."""
    rows = []
    price = 100.0
    for i in range(50):
        o = price
        h = price + 2.0
        l = price - 0.5
        c = price + 1.5
        rows.append({
            "timestamp": base_timestamp + timedelta(hours=i),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": 1000.0 + i * 10,
        })
        price = c
    return pd.DataFrame(rows)


@pytest.fixture
def swing_candles(base_timestamp):
    """50 candles with clear swing highs and lows — zigzag pattern.

    Pattern: up 5 bars, down 5 bars, repeated. Creates obvious pivots.
    """
    rows = []
    price = 100.0
    direction = 1  # 1 = up, -1 = down
    for i in range(50):
        if i % 10 < 5:
            direction = 1
        else:
            direction = -1

        move = direction * 2.0
        o = price
        if direction == 1:
            h = price + 3.0
            l = price - 0.5
            c = price + move
        else:
            h = price + 0.5
            l = price - 3.0
            c = price + move

        rows.append({
            "timestamp": base_timestamp + timedelta(hours=i),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": 1000.0,
        })
        price = c
    return pd.DataFrame(rows)


@pytest.fixture
def pin_bar_candles(base_timestamp):
    """Candles with a clear bullish pin bar at index 10 near price 100 (support).

    Setup: ranging around 100-110, then a bullish pin bar with long lower wick
    touching 100 (the support level), followed by more ranging.
    """
    rows = []
    for i in range(20):
        if i == 10:
            # Bullish pin bar: long lower wick, small body at top
            rows.append({
                "timestamp": base_timestamp + timedelta(hours=i),
                "open": 104.0,
                "high": 105.0,
                "low": 99.5,  # long lower wick
                "close": 104.5,  # body near top
                "volume": 2000.0,
            })
        elif i < 5:
            # Establish support around 100
            rows.append({
                "timestamp": base_timestamp + timedelta(hours=i),
                "open": 100.5,
                "high": 101.0,
                "low": 99.8 + (i * 0.1),
                "close": 100.5 + (i * 0.1),
                "volume": 1000.0,
            })
        else:
            # Range between 103-107
            rows.append({
                "timestamp": base_timestamp + timedelta(hours=i),
                "open": 104.0 + (i % 3) * 0.5,
                "high": 106.0 + (i % 3) * 0.5,
                "low": 103.0 + (i % 3) * 0.5,
                "close": 105.0 + (i % 3) * 0.5,
                "volume": 1000.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def inside_bar_candles(base_timestamp):
    """Candles with a clear inside bar pattern at indices 1-2."""
    rows = [
        # Mother bar (wide range)
        {"timestamp": base_timestamp, "open": 100.0, "high": 110.0,
         "low": 90.0, "close": 105.0, "volume": 2000.0},
        # Inside bar 1
        {"timestamp": base_timestamp + timedelta(hours=1), "open": 102.0, "high": 108.0,
         "low": 92.0, "close": 106.0, "volume": 1000.0},
        # Inside bar 2 (tighter)
        {"timestamp": base_timestamp + timedelta(hours=2), "open": 104.0, "high": 107.0,
         "low": 93.0, "close": 105.0, "volume": 800.0},
        # Breakout bar
        {"timestamp": base_timestamp + timedelta(hours=3), "open": 106.0, "high": 115.0,
         "low": 105.0, "close": 114.0, "volume": 3000.0},
    ]
    return pd.DataFrame(rows)
