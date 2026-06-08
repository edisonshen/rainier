"""ResonanceScorer tests: weighting math + category balance."""

from __future__ import annotations

import numpy as np
import pytest

from rainier.signals.panel import SignalPanel
from rainier.signals.resonance import ResonanceScorer, category_balanced_weights


def test_equal_weight_is_plain_fraction():
    panel = SignalPanel()
    scorer = ResonanceScorer(panel, mode="equal")
    # 4 members, half risk-on → score 0.5 everywhere
    series = {
        "price>SMA20": np.array([1.0, 1.0, 0.0]),
        "price>SMA50": np.array([1.0, 0.0, 0.0]),
        "RSI14>50": np.array([0.0, 1.0, 0.0]),
        "VIX<25": np.array([0.0, 0.0, 0.0]),
    }
    score = scorer.score(series)
    assert np.allclose(score, [0.5, 0.5, 0.0])


def test_category_balanced_weights_sum_per_category():
    panel = SignalPanel()
    names = [m.name for m in panel.members]
    w = category_balanced_weights(panel, names)
    # each category's member weights sum to 1.0
    by_cat: dict[str, float] = {}
    for name in names:
        by_cat.setdefault(panel.category_of(name), 0.0)
        by_cat[panel.category_of(name)] += w[name]
    for cat, total in by_cat.items():
        assert total == pytest.approx(1.0), cat


def test_category_balanced_score_is_uniform_over_categories():
    """3 trend members all on + 1 momentum member off (only 2 categories present)
    → score = (1.0 + 0.0)/2 = 0.5, regardless of how many trend members."""
    panel = SignalPanel()
    scorer = ResonanceScorer(panel, mode="category_balanced")
    series = {
        "price>SMA20": np.array([1.0]),
        "price>SMA50": np.array([1.0]),
        "price>SMA66": np.array([1.0]),  # 3 trend members all risk-on
        "RSI14>50": np.array([0.0]),     # 1 momentum member risk-off
    }
    score = scorer.score(series)
    # trend category contributes 1.0, momentum 0.0 → average 0.5
    assert score[0] == pytest.approx(0.5)


def test_score_in_unit_range():
    panel = SignalPanel()
    scorer = ResonanceScorer(panel, mode="category_balanced")
    rng = np.random.default_rng(3)
    series = {m.name: (rng.random(50) > 0.5).astype(float) for m in panel.members}
    score = scorer.score(series)
    assert (score >= 0).all() and (score <= 1).all()


def test_custom_weights_all_zero_raises():
    panel = SignalPanel()
    scorer = ResonanceScorer(panel, mode="custom", custom_weights={"nonexistent": 1.0})
    with pytest.raises(ValueError):
        scorer.score({"price>SMA20": np.array([1.0])})


def test_unknown_mode_raises():
    panel = SignalPanel()
    with pytest.raises(ValueError):
        ResonanceScorer(panel, mode="bogus")
