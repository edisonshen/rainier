"""ResonanceGate tests: deterministic state machine, hysteresis, boot, leakage."""

from __future__ import annotations

import numpy as np
import pytest

from rainier.core.types import Timeframe
from rainier.signals.panel import PanelInputs, SignalPanel
from rainier.signals.resonance import ResonanceScorer
from rainier.signals.resonance_gate import ResonanceGate, run_state_machine


def test_state_machine_fixed_sequence():
    # BUY=0.6, SELL=0.4, warmup=0 → fully deterministic from the score sequence.
    score = np.array([0.5, 0.7, 0.5, 0.3, 0.5, 0.65, 0.41, 0.39])
    w = run_state_machine(score, buy=0.6, sell=0.4, warmup=0)
    # t0: 0.5<BUY → CASH; 0.7≥BUY → TQQQ; 0.5 hold TQQQ; 0.3≤SELL → CASH;
    # 0.5 hold CASH; 0.65≥BUY → TQQQ; 0.41 hold (between); 0.39≤SELL → CASH
    assert list(w) == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]


def test_hysteresis_band_holds_between_thresholds():
    # scores parked strictly between SELL and BUY never flip state.
    score = np.array([0.7, 0.5, 0.55, 0.45, 0.5])  # enter then stay
    w = run_state_machine(score, buy=0.6, sell=0.4, warmup=0)
    assert list(w) == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_buy_must_exceed_sell():
    with pytest.raises(ValueError):
        run_state_machine(np.array([0.5]), buy=0.4, sell=0.6, warmup=0)
    with pytest.raises(ValueError):
        run_state_machine(np.array([0.5]), buy=0.5, sell=0.5, warmup=0)


def test_warmup_forces_cash_then_boots():
    score = np.full(20, 0.9)  # always above BUY
    w = run_state_machine(score, buy=0.6, sell=0.4, warmup=5)
    assert list(w[:5]) == [0.0] * 5      # forced cash during warmup
    assert list(w[5:]) == [1.0] * 15     # boots TQQQ at t0=5


def test_boot_cash_when_below_buy():
    score = np.full(10, 0.5)  # below BUY at boot
    w = run_state_machine(score, buy=0.6, sell=0.4, warmup=3)
    assert (w == 0.0).all()


def test_warmup_independent_of_data_length():
    # t0 is data-independent: prepending bars shifts the warmup boundary by the
    # same amount, it does not move with the data values.
    short = run_state_machine(np.full(10, 0.9), buy=0.6, sell=0.4, warmup=4)
    longer = run_state_machine(np.full(30, 0.9), buy=0.6, sell=0.4, warmup=4)
    assert list(short[:4]) == [0.0] * 4
    assert list(longer[:4]) == [0.0] * 4
    assert short[4] == 1.0 and longer[4] == 1.0


def _inputs(n=400):
    t = np.arange(n)
    close = 100.0 * (1.0 + 0.001) ** t
    import pandas as pd
    qqq = pd.DataFrame({
        "open": np.r_[close[0], close[:-1]],
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
    })
    return PanelInputs(qqq=qqq, vix=pd.Series(np.full(n, 15.0)))


def test_gate_decide_uptrend_goes_long():
    panel = SignalPanel()
    gate = ResonanceGate(ResonanceScorer(panel, mode="category_balanced"), buy=0.6, sell=0.4)
    w = gate.decide(_inputs())
    # forced cash through warmup, then long in a clean uptrend
    assert (w[: gate.warmup] == 0.0).all()
    assert w[-1] == 1.0


def test_weight_strategy_protocol_shape():
    import pandas as pd
    panel = SignalPanel()
    gate = ResonanceGate(ResonanceScorer(panel, mode="equal"), buy=0.6, sell=0.4)
    n = 400
    t = np.arange(n)
    close = 100.0 * (1.0 + 0.001) ** t
    df = pd.DataFrame({
        "open": np.r_[close[0], close[:-1]], "high": close * 1.005,
        "low": close * 0.995, "close": close, "vix": np.full(n, 15.0),
    })
    out = gate.weights(df, "TQQQ", Timeframe.D1)
    assert set(out.keys()) == {"TQQQ"}
    assert out["TQQQ"].shape == (n,)
    assert set(np.unique(out["TQQQ"])).issubset({0.0, 1.0})


def test_gate_leakage_future_bar_does_not_change_decision():
    """A spike on bar t+1 must not change the (un-shifted) decision at bar t."""
    panel = SignalPanel()
    gate = ResonanceGate(ResonanceScorer(panel, mode="category_balanced"), buy=0.6, sell=0.4)
    inp = _inputs()
    base = gate.decide(inp)
    t = 300
    q2 = inp.qqq.copy()
    q2.loc[t + 1, ["open", "high", "low", "close"]] = [1e6, 1e6, 1e6, 1e6]
    v2 = inp.vix.copy()
    v2.loc[t + 1] = 1.0
    after = gate.decide(PanelInputs(qqq=q2, vix=v2))
    assert np.array_equal(base[: t + 1], after[: t + 1])
