"""Daily-MTM sim tests: no-lookahead +1 shift, costs, cash, metrics."""

from __future__ import annotations

import numpy as np
import pytest

from rainier.backtest.daily_mtm import (
    max_drawdown,
    run_portfolio,
    shift_decision,
)


def test_shift_decision_is_plus_one_with_flat_first_bar():
    w = np.array([1.0, 1.0, 0.0, 1.0])
    assert list(shift_decision(w)) == [0.0, 1.0, 1.0, 0.0]


def test_no_lookahead_first_bar_return_ignored():
    """The big return on bar 0 must NOT be captured — decision applies t→t+1.

    Weight decided at close[0]=1.0 only acts on bar 1's return. Bar 0 is flat.
    """
    weights = {"X": np.array([1.0, 1.0, 1.0])}
    rets = {"X": np.array([10.0, 0.0, 0.0])}  # +1000% on bar 0 only
    r = run_portfolio("t", weights, rets, n_years=1.0, cash_apy=0.0, one_way_cost=0.0)
    # bar0 flat → equity stays 1.0 on bar0 (no lookahead capture of the +1000%)
    assert r.equity[0] == pytest.approx(1.0)


def test_cash_earns_rate_when_flat():
    weights = {"X": np.zeros(252)}
    rets = {"X": np.zeros(252)}
    r = run_portfolio("cash", weights, rets, n_years=1.0, cash_apy=0.04, one_way_cost=0.0)
    # ~4% over a year of daily compounding
    assert r.equity[-1] == pytest.approx(1.04, abs=1e-3)


def test_turnover_cost_charged_on_switch():
    weights = {"X": np.array([0.0, 1.0, 1.0])}  # shifted → [0,0,1]: one switch at t=2
    rets = {"X": np.array([0.0, 0.0, 0.0])}
    r = run_portfolio("c", weights, rets, n_years=1.0, cash_apy=0.0, one_way_cost=0.01)
    assert r.switches == 1
    # one 0→1 switch costs 1.0 * 0.01 → equity 0.99
    assert r.equity[-1] == pytest.approx(0.99)


def test_per_day_cash_series_accepted():
    weights = {"X": np.zeros(3)}
    rets = {"X": np.zeros(3)}
    r = run_portfolio("c", weights, rets, n_years=1.0,
                      cash_apy=np.array([0.0, 0.0, 0.0]), one_way_cost=0.0)
    assert r.equity[-1] == pytest.approx(1.0)


def test_max_drawdown_basic():
    curve = np.array([1.0, 1.2, 0.6, 0.9])
    # peak 1.2 → trough 0.6 = 50%
    assert max_drawdown(curve) == pytest.approx(0.5)


def test_gate_to_sim_integration_no_lookahead():
    """End-to-end: ResonanceGate decision → sim. The sim's +1 shift means a
    perfect future-knowing gate cannot capture the same-day return."""
    import pandas as pd

    from rainier.signals.panel import PanelInputs, SignalPanel
    from rainier.signals.resonance import ResonanceScorer
    from rainier.signals.resonance_gate import ResonanceGate

    n = 400
    t = np.arange(n)
    close = 100.0 * (1.0 + 0.001) ** t
    qqq = pd.DataFrame({
        "open": np.r_[close[0], close[:-1]], "high": close * 1.005,
        "low": close * 0.995, "close": close,
    })
    inp = PanelInputs(qqq=qqq, vix=pd.Series(np.full(n, 15.0)))
    gate = ResonanceGate(ResonanceScorer(SignalPanel(), mode="equal"), buy=0.6, sell=0.4)
    decision = gate.decide(inp)
    rets = {"TQQQ": qqq["close"].pct_change().fillna(0).to_numpy()}
    r = run_portfolio("res", {"TQQQ": decision}, rets, n_years=n / 252,
                      cash_apy=0.04, one_way_cost=0.0003)
    # in a clean uptrend the gate should be invested most of the time and profit
    assert r.total_return > 0
    assert 0.0 <= r.exposure <= 1.0
