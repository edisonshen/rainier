"""Tests for ml_builders — focused regression coverage.

Phase B shipped ``build_panel_tensor`` + ``ThematicTradingEnv`` largely as
consumer-deliverables (training loop is downstream-owned). These tests cover
the constructor rejection paths that came out of codex review.

Design ref: docs/DESIGN-thematic-ranks-dashboard.md §5.6 ([D-016]).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from rainier.breadth.ml_builders import ThematicTradingEnv


def _stub_features_panel() -> pd.DataFrame:
    """Minimal wide panel — MultiIndex(date, symbol), feature columns."""
    dates = [date(2024, 10, 1), date(2024, 10, 2)]
    syms = ["AAA", "BBB"]
    rows = []
    for d in dates:
        for s in syms:
            rows.append({"asof_date": d, "symbol": s, "rank": 50})
    df = pd.DataFrame(rows).set_index(["asof_date", "symbol"])
    return df


def _stub_labels_frame() -> pd.DataFrame:
    dates = [date(2024, 10, 1), date(2024, 10, 2)]
    syms = ["AAA", "BBB"]
    rows = []
    for d in dates:
        for s in syms:
            rows.append(
                {
                    "asof_date": d,
                    "symbol": s,
                    "fwd_5d_excess_ret": 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_env_rejects_unknown_reward():
    """Unknown reward names raise ValueError."""
    pytest.importorskip("gymnasium")
    with pytest.raises(ValueError, match="unknown reward"):
        ThematicTradingEnv(
            features=_stub_features_panel(),
            labels=_stub_labels_frame(),
            reward="nonsense_reward",
        )


def test_env_rejects_after_cost_reward_until_implemented():
    """Regression — codex iter-3 / iter-5 [P2]: previously the constructor
    accepted ``fwd_5d_excess_ret_after_cost`` and silently stripped the
    suffix in step(), so the after-cost reward equalled the pre-cost
    reward (no cost ever applied). The env must reject these names with
    a NotImplementedError that names the follow-up requirement.
    """
    pytest.importorskip("gymnasium")
    for variant in (
        "fwd_5d_excess_ret_after_cost",
        "fwd_10d_excess_ret_after_cost",
        "fwd_20d_excess_ret_after_cost",
    ):
        with pytest.raises(NotImplementedError, match="cost-adjusted"):
            ThematicTradingEnv(
                features=_stub_features_panel(),
                labels=_stub_labels_frame(),
                reward=variant,
            )


def test_env_accepts_excess_ret_rewards():
    """Constructor accepts the supported excess-return reward names."""
    pytest.importorskip("gymnasium")
    for variant in (
        "fwd_5d_excess_ret",
        "fwd_10d_excess_ret",
        "fwd_20d_excess_ret",
    ):
        env = ThematicTradingEnv(
            features=_stub_features_panel(),
            labels=_stub_labels_frame(),
            reward=variant,
        )
        assert env.reward == variant


def test_env_rejects_unknown_action_space():
    """Action spaces other than 'long_short_topk' raise."""
    pytest.importorskip("gymnasium")
    with pytest.raises(ValueError, match="unknown action_space"):
        ThematicTradingEnv(
            features=_stub_features_panel(),
            labels=_stub_labels_frame(),
            action_space="continuous_weights",
        )


# ---------------------------------------------------------------------------
# build_panel_tensor — basic axis correctness
# ---------------------------------------------------------------------------


def test_build_panel_tensor_shape_and_axes():
    """``build_panel_tensor`` returns a (dates, tickers, features) tensor
    with sorted axes for byte-deterministic downstream consumption.
    """
    from rainier.breadth.ml_builders import build_panel_tensor

    dates = [date(2024, 10, 1) + timedelta(days=i) for i in range(3)]
    syms = ["BBB", "AAA"]  # intentionally unsorted input
    rows = []
    for d in dates:
        for s in syms:
            rows.append(
                {
                    "asof_date": d,
                    "symbol": s,
                    "rank": float(50),
                    "r_5": float(20),
                }
            )
    panel = pd.DataFrame(rows).set_index(["asof_date", "symbol"])[["rank", "r_5"]]

    arr, ax_dates, ax_tickers, ax_features = build_panel_tensor(panel)
    assert arr.shape == (3, 2, 2)
    # Axes are sorted for determinism.
    assert ax_tickers == ["AAA", "BBB"]
    assert ax_dates == sorted(dates)
    assert ax_features == ["rank", "r_5"]


def test_build_panel_tensor_empty_input():
    """Empty panel returns an empty tensor + empty axes (no crash)."""
    from rainier.breadth.ml_builders import build_panel_tensor

    panel = pd.DataFrame(
        {"asof_date": [], "symbol": [], "rank": []}
    ).set_index(["asof_date", "symbol"])
    arr, ax_dates, ax_tickers, ax_features = build_panel_tensor(panel)
    assert arr.shape == (0, 0, 0)
    assert ax_dates == [] and ax_tickers == [] and ax_features == []
