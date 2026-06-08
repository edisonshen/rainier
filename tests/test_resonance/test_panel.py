"""SignalPanel tests: lookback invariant, truncation, frozen warmup, leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rainier.signals.panel import (
    MAX_LOOKBACK,
    WARMUP_BARS,
    PanelInputs,
    PanelMember,
    SignalPanel,
    breadth_pct_above_sma,
    rsi,
)


def test_rsi_zero_loss_window_is_100_not_neutral():
    # codex P2 (r6): an uninterrupted advance (no down days) → RSI 100, not 50,
    # so RSI14>50 actually fires in the strongest uptrend.
    up_only = pd.Series([100.0 * (1.01 ** i) for i in range(40)])
    r = rsi(up_only, 14)
    assert r.iloc[-1] == pytest.approx(100.0)
    assert (r.iloc[20:] > 50).all()
    # a perfectly flat window stays neutral 50
    flat = pd.Series([100.0] * 40)
    assert rsi(flat, 14).iloc[-1] == pytest.approx(50.0)


def test_panel_size_and_categories():
    panel = SignalPanel()
    # frozen ≤66 (the count is well under the cap; assert both)
    assert len(panel.members) <= MAX_LOOKBACK
    cats = {m.category for m in panel.members}
    assert cats == {"trend", "momentum", "volatility", "structure", "cross_asset", "breadth"}


def test_lookback_invariant_all_members_le_66():
    panel = SignalPanel()
    for m in panel.members:
        assert m.composed_lookback <= MAX_LOOKBACK, m.name


def test_lookback_invariant_rejects_oversize_member():
    bad = PanelMember("price>SMA200", "trend", 200, True, lambda i: np.zeros(len(i.qqq)))
    try:
        SignalPanel(members=(bad,))
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_no_expanding_windows_rewritten_vol_members(trending_inputs):
    """The two prototype expanding-median members are rewritten to finite rolling.

    Truncation test (b): for every finite-window member, dropping history before
    t-66 must leave the value at t bit-identical. Expanding windows would fail.
    """
    panel = SignalPanel()
    full = panel.compute(trending_inputs)
    n = len(trending_inputs.qqq)
    t = n - 1  # evaluate the last bar
    cut = t - MAX_LOOKBACK
    # truncate every input to [cut, n)
    inp = trending_inputs
    trunc = PanelInputs(
        qqq=inp.qqq.iloc[cut:].reset_index(drop=True),
        vix=inp.vix.iloc[cut:].reset_index(drop=True),
        spy=inp.spy.iloc[cut:].reset_index(drop=True),
        breadth=inp.breadth.iloc[cut:].reset_index(drop=True),
    )
    trunc_series = panel.compute(trunc)
    finite = {m.name for m in panel.members if m.finite_window}
    for name, series in full.items():
        if name not in finite:
            continue
        # last bar of truncated == last bar of full for finite-window members
        assert series[t] == trunc_series[name][-1], name


def test_warmup_is_frozen_integer_constant():
    # The warmup MUST be a pre-registered int that does not depend on data length.
    assert isinstance(WARMUP_BARS, int)
    assert WARMUP_BARS > 0


def test_leakage_future_spike_does_not_change_past(trending_inputs):
    """Inject a future spike at t+1; the member values at t must be unchanged."""
    panel = SignalPanel()
    base = panel.compute(trending_inputs)
    t = 300  # well past warmup, with a future bar available
    spiked = trending_inputs.qqq.copy()
    # giant spike on bar t+1 only
    spiked.loc[t + 1, ["open", "high", "low", "close"]] = [1e6, 1e6, 1e6, 1e6]
    vix2 = trending_inputs.vix.copy()
    vix2.loc[t + 1] = 200.0
    spy2 = trending_inputs.spy.copy()
    spy2.loc[t + 1] = 1e6
    inp2 = PanelInputs(qqq=spiked, vix=vix2, spy=spy2, breadth=trending_inputs.breadth)
    after = panel.compute(inp2)
    for name in base:
        assert base[name][t] == after[name][t], f"{name} leaked future into t={t}"


def test_trending_inputs_fire_risk_on(trending_inputs):
    panel = SignalPanel()
    series = panel.compute(trending_inputs)
    last = len(trending_inputs.qqq) - 1
    # in a clean steady uptrend, trend members should be risk-on at the end
    for name in ("price>SMA20", "price>SMA50", "price>SMA66", "SMA22>SMA44"):
        assert series[name][last] == 1.0, name


def test_absent_spy_and_breadth_drop_members(trending_inputs):
    panel = SignalPanel()
    inp = PanelInputs(qqq=trending_inputs.qqq, vix=trending_inputs.vix)
    series = panel.compute(inp)
    assert not any(k.startswith("SPY") for k in series)
    assert not any("breadth" in k for k in series)
    # trend members still present
    assert "price>SMA20" in series


def test_breadth_pct_above_sma_range():
    rng = np.random.default_rng(1)
    cols = {f"S{i}": 100.0 + np.cumsum(rng.normal(0, 1, 200)) for i in range(10)}
    prices = pd.DataFrame(cols)
    b = breadth_pct_above_sma(prices, n=50)
    valid = b.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()
    # before 50 bars there is no SMA → NaN
    assert b.iloc[:49].isna().all()
