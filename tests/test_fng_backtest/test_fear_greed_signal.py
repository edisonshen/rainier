"""Core-use-case tests for the Fear & Greed + QQQ-MA market-entry backtest.

Lean suite (operator standard: test the contracts users depend on, not a
matrix). The core user function is ``rainier fng-backtest``; the load-bearing
contracts are one test each:

1. No lookahead — a signal from ``F&G[D]`` + ``close[D]`` earns ``return[D+1]``
   and ZERO on day D (carry-then-apply, one-day lag).
2. Each combine logic's in/out mask matches its rule.
3. Metrics are correct — buy-and-hold primitives + annualized CAGR/Calmar/Sharpe
   against an inline numpy reference, plus exposure / switch count / per-transition
   slippage on a hand-specified switching mask.
4. Train/OOS split integrity — selection reads the train window only, per-combo
   OOS/train metrics equal ``compute_metrics`` on the raw slices (no leakage), and
   a degenerate split raises a clear ``ValueError``.
5. Auto-verdict — the 3-inequality rule returns eligible / context-only
   deterministically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rainier.backtest.fear_greed_signal import (
    ArmMetrics,
    Combo,
    ComboResult,
    Logic,
    buy_and_hold_mask,
    compute_metrics,
    equity_curve,
    logic_eligible,
    logic_mask,
    ma_mask,
    run_backtest,
    select_best_on_train,
)


def _arm(calmar: float, sharpe: float = 0.0, max_dd: float = 0.1) -> ArmMetrics:
    return ArmMetrics(
        final_value=1.0, total_return=0.0, cagr=0.0, max_dd=max_dd,
        calmar=calmar, sharpe=sharpe, exposure=1.0, switches=0,
        hit_rate=0.0, n_days=10,
    )


# --------------------------------------------------------------------------
# 1. No lookahead — the load-bearing correctness property
# --------------------------------------------------------------------------


@pytest.mark.parametrize("d_signal", [1, 2, 3])
def test_no_lookahead_signal_earns_next_day_return(d_signal: int) -> None:
    """A mask that is 'in' only on day ``D`` captures ``return[D+1]`` and 0 on D."""
    close = np.array([100.0, 110.0, 90.0, 95.0, 105.0])
    mask = np.zeros(len(close), dtype=bool)
    mask[d_signal] = True  # decision at close[D] gates the NEXT return

    _equity, strat_ret = equity_curve(close, mask, slippage_bp=0.0)

    ret = np.zeros(len(close))
    ret[1:] = close[1:] / close[:-1] - 1.0

    # Zero on day D (the same-day return period is NOT captured — no lookahead).
    assert strat_ret[d_signal] == pytest.approx(0.0)
    # return[D+1] IS captured by the position set at day D.
    assert strat_ret[d_signal + 1] == pytest.approx(ret[d_signal + 1])
    # Nothing else is captured.
    captured = {d_signal + 1}
    for i in range(1, len(close)):
        if i not in captured:
            assert strat_ret[i] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 2. Each combine logic's in/out matches its rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "logic,threshold,expected",
    [
        # trend (close>SMA2) = [F,T,T,T,T]; fng = [50,10,60,80,25]
        (Logic.CONTRARIAN, 30.0, [False, True, False, False, True]),   # fng<=30
        (Logic.MOMENTUM, 60.0, [False, False, True, True, False]),     # fng>=60
        (Logic.GREED_AVOID, 75.0, [False, True, True, False, True]),   # fng<75
    ],
)
def test_logic_mask_matches_rule(
    logic: Logic, threshold: float, expected: list[bool]
) -> None:
    close = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    fng = np.array([50.0, 10.0, 60.0, 80.0, 25.0])
    # Sanity: the shared trend filter is close>SMA(2), valid from index 1.
    assert list(ma_mask(close, 2)) == [False, True, True, True, True]

    mask = logic_mask(close, fng, logic, threshold, window=2)
    assert list(mask) == expected


# --------------------------------------------------------------------------
# 3. Metrics are correct — primitives + annualized ref + slippage/switches
# --------------------------------------------------------------------------

# Named toy series (task-plan oracle): buy-and-hold, always in.
_BH_CLOSE = np.array([100.0, 101.0, 100.0, 103.0, 102.0, 105.0])


def test_metrics_match_toy_oracle_and_reference() -> None:
    """Every metric on the named toy series: buy-and-hold primitives, the equity
    curve, the annualized CAGR/Calmar/Sharpe pinned to an inline numpy reference,
    and — on a second hand-specified switching mask — exposure, switch count, and
    that slippage is charged once per transition."""
    # --- buy-and-hold primitives on the named oracle (slippage charged 0 here) ---
    m = compute_metrics(_BH_CLOSE, buy_and_hold_mask(len(_BH_CLOSE)), slippage_bp=5.0)
    assert m.total_return == pytest.approx(0.05)          # 105/100 - 1
    assert m.final_value == pytest.approx(1.05)
    assert m.max_dd == pytest.approx(1.0 / 101.0)         # peak 1.01 -> 1.00
    assert m.exposure == pytest.approx(1.0)
    assert m.switches == 0
    assert m.hit_rate == pytest.approx(3.0 / 5.0)         # 3 up of 5 return periods

    equity, _ = equity_curve(_BH_CLOSE, buy_and_hold_mask(len(_BH_CLOSE)), 5.0)
    np.testing.assert_allclose(equity, [1.0, 1.01, 1.0, 1.03, 1.02, 1.05])

    # --- annualized metrics vs the reference formula (252, rf=0), no slippage ---
    m_ann = compute_metrics(_BH_CLOSE, buy_and_hold_mask(len(_BH_CLOSE)), slippage_bp=0.0)
    ref_equity = np.array([1.0, 1.01, 1.0, 1.03, 1.02, 1.05])
    n = len(ref_equity)
    ref_strat_ret = np.zeros(n)
    ref_strat_ret[1:] = ref_equity[1:] / ref_equity[:-1] - 1.0
    years = n / 252.0
    ref_cagr = ref_equity[-1] ** (1.0 / years) - 1.0
    peak = np.maximum.accumulate(ref_equity)
    ref_max_dd = float((1.0 - ref_equity / peak).max())
    ref_calmar = ref_cagr / ref_max_dd
    ref_sharpe = ref_strat_ret.mean() / ref_strat_ret.std(ddof=1) * np.sqrt(252.0)
    assert m_ann.cagr == pytest.approx(ref_cagr)
    assert m_ann.calmar == pytest.approx(ref_calmar)
    assert m_ann.sharpe == pytest.approx(ref_sharpe)

    # --- exposure / switch count / per-transition slippage (switching mask) ---
    close = np.array([100.0, 101.0, 102.0, 101.0, 102.0, 103.0])
    mask = np.array([True, True, False, False, True, True])

    m0 = compute_metrics(close, mask, slippage_bp=0.0)
    assert m0.exposure == pytest.approx(4.0 / 6.0)
    assert m0.switches == 2  # in->out at idx2, out->in at idx4

    # Strategy return off the mask (slippage=0): captures ret[1], ret[2], ret[5].
    ret = np.zeros(len(close))
    ret[1:] = close[1:] / close[:-1] - 1.0
    expected = (1 + ret[1]) * (1 + ret[2]) * (1 + ret[5])
    assert m0.final_value == pytest.approx(expected)

    # Two transitions -> equity scaled by (1 - slip)^2.
    slip = 10.0 * 1e-4
    m1 = compute_metrics(close, mask, slippage_bp=10.0)
    assert m1.final_value == pytest.approx(m0.final_value * (1 - slip) ** 2)


# --------------------------------------------------------------------------
# 4. Train/OOS split integrity — train-only selection, no leakage, fail-loud
# --------------------------------------------------------------------------


def test_train_oos_split_integrity() -> None:
    """Selection reads the TRAIN window only; per-combo OOS/train metrics equal
    ``compute_metrics`` on the raw slices (masks from full history, no seam warmup
    loss, no leakage); a degenerate split (empty train or OOS) fails loudly."""
    logic = Logic.CONTRARIAN

    # --- selection maximizes TRAIN Calmar, ignoring OOS; gate metric reads OOS ---
    # row_a is best on TRAIN but worst on OOS; row_b is the reverse.
    row_a = ComboResult(
        combo=Combo(logic, 20, 20.0), train=_arm(2.0), oos=_arm(0.1), full=_arm(1.0)
    )
    row_b = ComboResult(
        combo=Combo(logic, 20, 30.0), train=_arm(1.0), oos=_arm(5.0), full=_arm(1.0)
    )
    picked = select_best_on_train([row_a, row_b], logic)
    assert picked is row_a                          # selection ignored OOS
    assert picked.oos.calmar == pytest.approx(0.1)  # gate reads OOS, not train

    # --- run_backtest slices train/OOS off full-history masks without leakage ---
    rng = np.random.default_rng(0)
    n = 80
    dates = pd.bdate_range("2021-01-01", periods=n)
    qqq = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    fng = np.clip(50 + rng.normal(0, 15, n), 0, 100)
    combo = Combo(Logic.CONTRARIAN, 5, 30.0)
    split_date = dates[50]

    result = run_backtest(
        signal_close=qqq, trade_close=qqq, fng=fng, dates=dates,
        split_date=split_date, slippage_bp=5.0, combos=[combo], ma_windows=[5],
    )
    split = int(np.searchsorted(dates.values, np.datetime64(split_date)))
    mask = logic_mask(qqq, fng, combo.logic, combo.threshold, combo.ma)

    cr = result.combo_results[0]
    assert cr.oos == compute_metrics(qqq[split:], mask[split:], 5.0)
    assert cr.train == compute_metrics(qqq[:split], mask[:split], 5.0)
    # Benchmarks present for the report.
    assert 5 in result.ma_only
    assert result.buy_and_hold.full.final_value > 0

    # --- degenerate split (outside the aligned span) raises a clear ValueError ---
    bad_n = 40
    bad_dates = pd.bdate_range("2021-01-01", periods=bad_n)
    bad_qqq = 100.0 * np.cumprod(1 + np.full(bad_n, 0.001))
    bad_fng = np.full(bad_n, 50.0)
    for bad_split in ("2000-01-01", "2099-01-01"):  # empty train, then empty OOS
        with pytest.raises(ValueError, match="degenerate"):
            run_backtest(
                signal_close=bad_qqq, trade_close=bad_qqq, fng=bad_fng, dates=bad_dates,
                split_date=pd.Timestamp(bad_split),
                combos=[Combo(Logic.CONTRARIAN, 5, 30.0)], ma_windows=[5],
            )


# --------------------------------------------------------------------------
# 5. Auto-verdict — 3-inequality rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "logic_oos,ma_oos,expected",
    [
        # all three clear: Calmar >= 1.15x, Sharpe >=, MaxDD <=
        (_arm(1.20, sharpe=1.1, max_dd=0.20), _arm(1.00, sharpe=1.0, max_dd=0.20), True),
        # Calmar edge too small (1.10 < 1.15x)
        (_arm(1.10, sharpe=1.1, max_dd=0.20), _arm(1.00, sharpe=1.0, max_dd=0.20), False),
        # Sharpe worse
        (_arm(1.50, sharpe=0.9, max_dd=0.20), _arm(1.00, sharpe=1.0, max_dd=0.20), False),
        # MaxDD worse (deeper)
        (_arm(1.50, sharpe=1.1, max_dd=0.30), _arm(1.00, sharpe=1.0, max_dd=0.20), False),
    ],
)
def test_logic_eligible_three_inequalities(
    logic_oos: ArmMetrics, ma_oos: ArmMetrics, expected: bool
) -> None:
    assert logic_eligible(logic_oos, ma_oos) is expected
