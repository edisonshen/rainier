"""Tests for ScoringStrategy implementations."""

import pandas as pd
import pytest

from rainier.core.protocols import ScoringStrategy
from rainier.core.types import PatternSignal
from rainier.ml.scorers import BookScorer


def _make_pattern(**overrides) -> PatternSignal:
    defaults = dict(
        symbol="AAPL",
        pattern_type="w_bottom",
        direction="bullish",
        status="confirmed",
        confidence=0.0,
        entry_price=150.0,
        stop_loss=145.0,
        target_wave1=160.0,
        rr_ratio=2.0,
        neckline=155.0,
        key_points={"left_bottom": 145.0, "right_bottom": 145.5, "neckline": 155.0},
        volume_confirmed=True,
    )
    defaults.update(overrides)
    return PatternSignal(**defaults)


def _make_features() -> pd.DataFrame:
    return pd.DataFrame({
        "volume_ratio": [1.5],
        "is_bullish": [1.0],
    })


class TestBookScorer:
    def test_implements_protocol(self):
        scorer = BookScorer()
        assert isinstance(scorer, ScoringStrategy)

    def test_score_in_range(self):
        scorer = BookScorer()
        pattern = _make_pattern()
        score = scorer.score(pattern, _make_features())
        assert 0.0 <= score <= 1.0

    def test_confirmed_higher_than_forming(self):
        scorer = BookScorer()
        features = _make_features()
        confirmed = scorer.score(_make_pattern(status="confirmed"), features)
        forming = scorer.score(_make_pattern(status="forming"), features)
        assert confirmed > forming

    def test_volume_confirmed_higher(self):
        scorer = BookScorer()
        features = _make_features()
        with_vol = scorer.score(_make_pattern(volume_confirmed=True), features)
        no_vol = scorer.score(_make_pattern(volume_confirmed=False), features)
        assert with_vol > no_vol

    def test_higher_rr_higher_score(self):
        scorer = BookScorer()
        features = _make_features()
        high_rr = scorer.score(_make_pattern(rr_ratio=3.5), features)
        low_rr = scorer.score(_make_pattern(rr_ratio=1.0), features)
        assert high_rr > low_rr

    def test_false_breakdown_weighted_higher(self):
        scorer = BookScorer()
        features = _make_features()
        fb = scorer.score(_make_pattern(pattern_type="false_breakdown"), features)
        tri = scorer.score(_make_pattern(pattern_type="sym_triangle_bottom"), features)
        assert fb > tri

    def test_no_neckline_lower_clarity(self):
        scorer = BookScorer()
        features = _make_features()
        with_neck = scorer.score(_make_pattern(neckline=155.0), features)
        no_neck = scorer.score(_make_pattern(neckline=0.0), features)
        assert with_neck > no_neck

    def test_custom_weights(self):
        scorer = BookScorer(pattern_weights={"w_bottom": 1.0})
        features = _make_features()
        score = scorer.score(_make_pattern(), features)
        assert score > 0.5  # max weight should yield high score
