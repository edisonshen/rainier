"""Tests for rainier.analysis.stock_patterns and rainier.analysis.target_calculator."""

from __future__ import annotations

import pandas as pd
import pytest

from rainier.analysis.pattern_primitives import VolumePriceSignal
from rainier.analysis.stock_patterns import detect_patterns, score_pattern
from rainier.analysis.target_calculator import (
    compute_double_bottom_targets,
    compute_double_top_targets,
    compute_hs_targets,
)
from rainier.core.config import StockScreenerConfig
from rainier.core.types import PatternSignal

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


def _make_w_bottom_prices() -> list[float]:
    """Build a W-bottom price series over ~30 bars.

    Bars 0-5:   decline from 100 to 80 (first bottom)
    Bars 6-10:  rally to 95 (neckline)
    Bars 11-15: decline back to 81 (second bottom, ~same level)
    Bars 16-25: rally above 95 (breakout confirmed)
    Bars 26-29: continuation above neckline
    """
    prices = []
    # Bars 0-5: 100 → 80
    for i in range(6):
        prices.append(100.0 - (20.0 * i / 5))
    # Bars 6-10: 80 → 95
    for i in range(1, 6):
        prices.append(80.0 + (15.0 * i / 5))
    # Bars 11-15: 95 → 81
    for i in range(1, 6):
        prices.append(95.0 - (14.0 * i / 5))
    # Bars 16-25: 81 → 100 (breakout past neckline at ~95)
    for i in range(1, 11):
        prices.append(81.0 + (19.0 * i / 10))
    # Bars 26-29: hold above neckline
    for _i in range(4):
        prices.append(101.0)
    return prices


def _make_m_top_prices() -> list[float]:
    """Build an M-top price series over ~30 bars (mirror of W-bottom).

    Bars 0-5:   rally from 80 to 100 (first top)
    Bars 6-10:  decline to 85 (neckline)
    Bars 11-15: rally back to 99 (second top, ~same level)
    Bars 16-25: decline below 85 (breakout confirmed)
    Bars 26-29: continuation below neckline
    """
    prices = []
    # Bars 0-5: 80 → 100
    for i in range(6):
        prices.append(80.0 + (20.0 * i / 5))
    # Bars 6-10: 100 → 85
    for i in range(1, 6):
        prices.append(100.0 - (15.0 * i / 5))
    # Bars 11-15: 85 → 99
    for i in range(1, 6):
        prices.append(85.0 + (14.0 * i / 5))
    # Bars 16-25: 99 → 80 (breakdown past neckline at ~85)
    for i in range(1, 11):
        prices.append(99.0 - (19.0 * i / 10))
    # Bars 26-29: hold below neckline
    for _i in range(4):
        prices.append(79.0)
    return prices


def _make_false_breakdown_prices() -> list[float]:
    """Build a false-breakdown price series.

    Establish a support at ~90, break below it, then recover within 5 bars.
    """
    prices = []
    # Bars 0-9: establish support around 90 with swing lows
    prices.extend([95.0, 93.0, 91.0, 90.0, 91.0, 93.0, 95.0, 93.0, 91.0, 90.0])
    # Bars 10-12: range above support
    prices.extend([92.0, 93.0, 91.0])
    # Bars 13-15: break below support
    prices.extend([89.0, 88.0, 87.0])
    # Bars 16-19: recover above support quickly
    prices.extend([89.0, 91.0, 93.0, 95.0])
    # Bars 20-29: continuation above
    prices.extend([96.0, 97.0, 96.0, 97.0, 98.0, 97.0, 98.0, 99.0, 98.0, 99.0])
    return prices


# ---------------------------------------------------------------------------
# detect_patterns — stock_patterns.py
# ---------------------------------------------------------------------------


class TestDetectPatterns:
    def test_detect_w_bottom(self):
        """W-bottom shape yields PatternSignal with correct attributes."""
        prices = _make_w_bottom_prices()
        df = make_ohlcv(prices)
        config = StockScreenerConfig(
            swing_lookback=3,
            min_pattern_bars=5,
            max_pattern_bars=50,
            neckline_tolerance_pct=0.05,
        )
        patterns = detect_patterns("TEST", df, config)

        w_bottoms = [p for p in patterns if p.pattern_type == "w_bottom"]
        assert len(w_bottoms) >= 1, (
            f"Expected w_bottom, got types: {[p.pattern_type for p in patterns]}"
        )

        p = w_bottoms[0]
        assert p.direction == "bullish"
        assert p.status in ("confirmed", "forming")
        assert p.symbol == "TEST"

    def test_detect_m_top(self):
        """M-top shape yields PatternSignal with correct attributes."""
        prices = _make_m_top_prices()
        df = make_ohlcv(prices)
        config = StockScreenerConfig(
            swing_lookback=3,
            min_pattern_bars=5,
            max_pattern_bars=50,
            neckline_tolerance_pct=0.05,
        )
        patterns = detect_patterns("TEST", df, config)

        m_tops = [p for p in patterns if p.pattern_type == "m_top"]
        assert len(m_tops) >= 1, f"Expected m_top, got types: {[p.pattern_type for p in patterns]}"

        p = m_tops[0]
        assert p.direction == "bearish"
        assert p.symbol == "TEST"

    def test_detect_false_breakdown(self):
        """Price breaks below support then recovers → false_breakdown pattern."""
        prices = _make_false_breakdown_prices()
        df = make_ohlcv(prices)
        config = StockScreenerConfig(
            swing_lookback=3,
            min_pattern_bars=3,
            max_pattern_bars=50,
            neckline_tolerance_pct=0.05,
        )
        patterns = detect_patterns("TEST", df, config)

        fb = [p for p in patterns if p.pattern_type == "false_breakdown"]
        assert len(fb) >= 1, (
            f"Expected false_breakdown, got types: {[p.pattern_type for p in patterns]}"
        )
        p = fb[0]
        assert p.direction == "bullish"
        assert p.status == "confirmed"

    def test_no_patterns_on_flat_data(self):
        """Flat price data returns empty list."""
        prices = [100.0] * 50
        df = make_ohlcv(prices)
        config = StockScreenerConfig()
        patterns = detect_patterns("FLAT", df, config)
        assert patterns == []


# ---------------------------------------------------------------------------
# score_pattern — stock_patterns.py
# ---------------------------------------------------------------------------


class TestScorePattern:
    def test_score_pattern(self):
        """Score with known inputs is in the expected range."""
        config = StockScreenerConfig()
        pattern = PatternSignal(
            symbol="TEST",
            pattern_type="w_bottom",
            direction="bullish",
            status="confirmed",
            confidence=0.0,
            entry_price=100.0,
            stop_loss=98.0,
            target_wave1=110.0,
            target_wave2=120.0,
            risk_pct=0.02,
            reward_pct=0.10,
            rr_ratio=5.0,
            neckline=100.0,
            key_points={"left_bottom": 90.0, "right_bottom": 90.5},
            volume_confirmed=True,
        )
        vol_price = VolumePriceSignal(
            type="price_up_vol_up",
            divergence=False,
            vol_ratio=1.5,
        )
        score = score_pattern(pattern, vol_price, config.pattern_weights)

        # Breakdown: 0.35*0.85=0.2975 + 0.20 + 0.05 + 0.075+0.075 + 0.15 + 0.10 = 0.9475
        assert 0.5 <= score <= 1.0, f"Score {score} out of expected range"
        assert score == pytest.approx(0.9475, abs=0.01)


# ---------------------------------------------------------------------------
# target_calculator.py
# ---------------------------------------------------------------------------


class TestTargetCalculator:
    def test_compute_double_bottom_targets(self):
        """Neckline=100, bottoms at 90 and 90.5 → known target levels."""
        targets = compute_double_bottom_targets(
            neckline=100.0, bottom1=90.0, bottom2=90.5
        )
        distance = 100.0 - (90.0 + 90.5) / 2  # 9.75

        assert targets.entry == 100.0
        assert targets.target_wave1 == pytest.approx(100.0 + distance)  # 109.75
        assert targets.stop_loss == pytest.approx(100.0 * 0.98)  # 98.0
        assert targets.target_wave2 == pytest.approx(109.75 + distance)  # 119.5

    def test_compute_double_top_targets(self):
        """Mirror of double bottom — bearish targets."""
        targets = compute_double_top_targets(
            neckline=100.0, top1=110.0, top2=109.5
        )
        avg_top = (110.0 + 109.5) / 2  # 109.75
        distance = avg_top - 100.0  # 9.75

        assert targets.entry == 100.0
        assert targets.target_wave1 == pytest.approx(100.0 - distance)  # 90.25
        assert targets.stop_loss == pytest.approx(100.0 * 1.02)  # 102.0

    def test_compute_hs_targets_bullish(self):
        """Inverse H&S: neckline=100, head=80 → target_wave1=120."""
        targets = compute_hs_targets(
            neckline=100.0, head_price=80.0, direction="bullish"
        )
        distance = abs(100.0 - 80.0)  # 20

        assert targets.entry == 100.0
        assert targets.target_wave1 == pytest.approx(100.0 + distance)  # 120.0
        assert targets.stop_loss == pytest.approx(100.0 * 0.98)  # 98.0

    def test_rr_ratio_zero_risk(self):
        """If entry equals SL (zero risk), rr_ratio should be 0."""
        targets = compute_double_bottom_targets(
            neckline=100.0,
            bottom1=90.0,
            bottom2=90.5,
            stop_buffer_pct=0.0,  # SL = entry
        )
        assert targets.stop_loss == targets.entry
        assert targets.rr_ratio == 0.0
