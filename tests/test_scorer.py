"""Tests for confidence scoring."""

import pandas as pd

from rainier.core.config import ScorerConfig
from rainier.core.types import Candle, Direction, PinBar, SRLevel, SRRole, SRType, Timeframe
from rainier.signals.scorer import score_setup
from datetime import datetime


def _make_pin_bar(
    wick_ratio: float = 3.0,
    sr_strength: float = 0.8,
    volume: float = 1000.0,
) -> PinBar:
    candle = Candle(
        timestamp=datetime(2025, 1, 1),
        open=104.0, high=105.0, low=99.5, close=104.5,
        volume=volume, symbol="NQ", timeframe=Timeframe.H1,
    )
    sr = SRLevel(
        price=100.0, sr_type=SRType.HORIZONTAL,
        role=SRRole.SUPPORT, strength=sr_strength, touches=3,
    )
    return PinBar(
        candle=candle, index=10, direction=Direction.LONG,
        wick_ratio=wick_ratio, nearest_sr=sr, sr_distance_pct=0.005,
    )


def _make_df(avg_volume: float = 1000.0) -> pd.DataFrame:
    rows = [
        {"timestamp": datetime(2025, 1, 1), "open": 100, "high": 105,
         "low": 99, "close": 104, "volume": avg_volume}
    ] * 20
    return pd.DataFrame(rows)


class TestScorer:
    def test_score_between_0_and_1(self):
        pb = _make_pin_bar()
        df = _make_df()
        score = score_setup(pb, df, Direction.LONG)
        assert 0.0 <= score <= 1.0

    def test_stronger_sr_higher_score(self):
        df = _make_df()
        weak = score_setup(_make_pin_bar(sr_strength=0.2), df, None)
        strong = score_setup(_make_pin_bar(sr_strength=0.9), df, None)
        assert strong > weak

    def test_higher_wick_ratio_higher_score(self):
        df = _make_df()
        small = score_setup(_make_pin_bar(wick_ratio=2.0), df, None)
        large = score_setup(_make_pin_bar(wick_ratio=5.0), df, None)
        assert large > small

    def test_all_zero_subscore_still_works(self):
        """Pin bar with no S/R → sr_strength=0, should not crash."""
        candle = Candle(
            timestamp=datetime(2025, 1, 1),
            open=100, high=101, low=99, close=100,
            volume=0, symbol="NQ", timeframe=Timeframe.H1,
        )
        pb = PinBar(
            candle=candle, index=0, direction=Direction.LONG,
            wick_ratio=0.0, nearest_sr=None, sr_distance_pct=0.0,
        )
        df = _make_df(avg_volume=0)
        score = score_setup(pb, df, None)
        assert 0.0 <= score <= 1.0

    def test_trend_alignment_boosts_score(self):
        df = _make_df()
        aligned = score_setup(_make_pin_bar(), df, Direction.LONG)
        counter = score_setup(_make_pin_bar(), df, Direction.SHORT)
        assert aligned > counter

    def test_confluence_no_levels_fallback(self):
        """No sr_levels passed → fallback score 0.5 (backward-compat)."""
        pb = _make_pin_bar()
        df = _make_df()
        score_no_levels = score_setup(pb, df, None, sr_levels=None)
        assert 0.0 <= score_no_levels <= 1.0

    def test_confluence_single_tf(self):
        """One nearby TF → confluence 0.3."""
        pb = _make_pin_bar()
        df = _make_df()
        levels = [
            SRLevel(price=104.0, sr_type=SRType.HORIZONTAL, role=SRRole.SUPPORT,
                    strength=0.7, touches=3, source_tf=Timeframe.H1),
        ]
        score = score_setup(pb, df, None, sr_levels=levels)
        assert 0.0 <= score <= 1.0

    def test_confluence_two_tfs_higher_than_one(self):
        """Two distinct TFs nearby → higher confluence than one."""
        pb = _make_pin_bar()
        df = _make_df()
        one_tf = [
            SRLevel(price=104.0, sr_type=SRType.HORIZONTAL, role=SRRole.SUPPORT,
                    strength=0.7, touches=3, source_tf=Timeframe.H1),
        ]
        two_tfs = [
            SRLevel(price=104.0, sr_type=SRType.HORIZONTAL, role=SRRole.SUPPORT,
                    strength=0.7, touches=3, source_tf=Timeframe.H1),
            SRLevel(price=104.3, sr_type=SRType.HORIZONTAL, role=SRRole.SUPPORT,
                    strength=0.7, touches=3, source_tf=Timeframe.H4),
        ]
        s1 = score_setup(pb, df, None, sr_levels=one_tf)
        s2 = score_setup(pb, df, None, sr_levels=two_tfs)
        assert s2 > s1

    def test_confluence_three_tfs_highest(self):
        """Three distinct TFs nearby → max confluence (1.0)."""
        pb = _make_pin_bar()
        df = _make_df()
        three_tfs = [
            SRLevel(price=104.0, sr_type=SRType.HORIZONTAL, role=SRRole.SUPPORT,
                    strength=0.7, touches=3, source_tf=Timeframe.H1),
            SRLevel(price=104.3, sr_type=SRType.HORIZONTAL, role=SRRole.SUPPORT,
                    strength=0.7, touches=3, source_tf=Timeframe.H4),
            SRLevel(price=104.2, sr_type=SRType.HORIZONTAL, role=SRRole.SUPPORT,
                    strength=0.7, touches=3, source_tf=Timeframe.D1),
        ]
        s = score_setup(pb, df, None, sr_levels=three_tfs)
        assert s > score_setup(pb, df, None, sr_levels=three_tfs[:2])

    def test_confluence_far_levels_excluded(self):
        """S/R levels far from pin bar price should not count."""
        pb = _make_pin_bar()  # close=104.5
        df = _make_df()
        far_levels = [
            SRLevel(price=200.0, sr_type=SRType.HORIZONTAL, role=SRRole.RESISTANCE,
                    strength=0.9, touches=5, source_tf=Timeframe.D1),
            SRLevel(price=50.0, sr_type=SRType.HORIZONTAL, role=SRRole.SUPPORT,
                    strength=0.9, touches=5, source_tf=Timeframe.W1),
        ]
        near_levels = [
            SRLevel(price=104.0, sr_type=SRType.HORIZONTAL, role=SRRole.SUPPORT,
                    strength=0.7, touches=3, source_tf=Timeframe.H1),
            SRLevel(price=104.3, sr_type=SRType.HORIZONTAL, role=SRRole.SUPPORT,
                    strength=0.7, touches=3, source_tf=Timeframe.H4),
        ]
        s_far = score_setup(pb, df, None, sr_levels=far_levels)
        s_near = score_setup(pb, df, None, sr_levels=near_levels)
        assert s_near > s_far
