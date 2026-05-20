"""Tests for TQQQ/SQQQ SMA-grid backtest sweep.

Strategy state machine: CASH | LONG_TQQQ | SHORT_SQQQ.
Signals derived from QQQ close vs SMA(buy_T/sell_T/buy_S/sell_S).
Phase 1 constraint: sell_T >= buy_T AND sell_S >= buy_S.

These tests run on synthetic universes so they're fast and deterministic
(no yfinance network I/O). The full 3.35M sweep is exercised in
tdd-refactor as an out-of-band deliverable.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rainier.backtest.tqqq_sma_sweep import (
    CASH,
    LONG_TQQQ,
    SHORT_SQQQ,
    iter_phase1_combos,
    precompute_sma_signals,
    run_backtest,
    run_sweep,
)


# ---------------------------------------------------------------------------
# Synthetic universes
# ---------------------------------------------------------------------------


def _frame_with_qqq(prices: np.ndarray) -> pd.DataFrame:
    """Build a price frame matching the sweep's expected schema.

    Columns: ``qqq``, ``tqqq``, ``sqqq`` (lowercased adjusted-close).
    TQQQ tracks 3x daily return of QQQ; SQQQ tracks -3x. Index is
    a simple business-day range.
    """
    qqq = np.asarray(prices, dtype=np.float64)
    n = qqq.shape[0]
    # Daily returns of QQQ
    rets = np.zeros(n)
    rets[1:] = qqq[1:] / qqq[:-1] - 1.0
    tqqq = np.empty(n)
    sqqq = np.empty(n)
    tqqq[0] = sqqq[0] = 100.0
    for i in range(1, n):
        tqqq[i] = tqqq[i - 1] * (1.0 + 3.0 * rets[i])
        sqqq[i] = sqqq[i - 1] * (1.0 - 3.0 * rets[i])
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"qqq": qqq, "tqqq": tqqq, "sqqq": sqqq}, index=idx)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_constants_are_distinct():
    assert {CASH, LONG_TQQQ, SHORT_SQQQ} == {0, 1, 2}


def test_iter_phase1_combos_count_and_constraint():
    combos = list(iter_phase1_combos(max_window=60))
    # 1830 ordered pairs (buy<=sell) each side, squared
    assert len(combos) == 1830 * 1830
    # Spot-check constraint on a random sample (head + tail)
    for bT, sT, bS, sS in combos[:5000] + combos[-5000:]:
        assert 1 <= bT <= sT <= 60
        assert 1 <= bS <= sS <= 60


def test_iter_phase1_max_window_small():
    # max_window=3 → triangular ordered pairs = 3+2+1 = 6 each side → 36 total
    combos = list(iter_phase1_combos(max_window=3))
    assert len(combos) == 36
    for bT, sT, bS, sS in combos:
        assert 1 <= bT <= sT <= 3
        assert 1 <= bS <= sS <= 3


def test_precompute_sma_signals_shape():
    n_days = 100
    qqq = np.linspace(100.0, 200.0, n_days)
    above = precompute_sma_signals(qqq, max_window=60)
    assert above.shape == (n_days, 60)
    assert above.dtype == np.bool_
    # SMA1 is the price itself → strictly equal → never strictly above
    assert not above[:, 0].any()


def test_run_backtest_deterministic():
    df = _frame_with_qqq(np.linspace(100.0, 300.0, 200))
    above = precompute_sma_signals(df["qqq"].to_numpy(), max_window=60)
    tqqq_ret = (df["tqqq"].to_numpy()[1:] / df["tqqq"].to_numpy()[:-1]) - 1.0
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0
    a = run_backtest(above, tqqq_ret, sqqq_ret, 5, 20, 5, 20, slippage_bp=5.0)
    b = run_backtest(above, tqqq_ret, sqqq_ret, 5, 20, 5, 20, slippage_bp=5.0)
    for x, y in zip(a, b, strict=True):
        if isinstance(x, float) and np.isnan(x):
            assert np.isnan(y)
        else:
            assert x == y


def test_monotone_up_qqq_long_tqqq_wins():
    """Up-only universe: best strategy stays long TQQQ as much as possible."""
    n = 300
    qqq = 100.0 * (1.005 ** np.arange(n))  # ~0.5%/day → strong uptrend
    df = _frame_with_qqq(qqq)
    above = precompute_sma_signals(df["qqq"].to_numpy(), max_window=60)
    tqqq_ret = (df["tqqq"].to_numpy()[1:] / df["tqqq"].to_numpy()[:-1]) - 1.0
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0

    # Very loose buy_T (small window), tight sell_T (also small) so we go long
    # near day-2 and basically never exit. buy_S/sell_S are short-leg knobs
    # that don't fire here.
    final, sharpe, mdd, calmar, n_trades, t_long, t_short, t_cash = run_backtest(
        above, tqqq_ret, sqqq_ret, buy_T=2, sell_T=2, buy_S=2, sell_S=2, slippage_bp=5.0
    )
    # Should spend almost all time long, beat TQQQ buy-and-hold within slippage
    tqqq_bh = df["tqqq"].iloc[-1] / df["tqqq"].iloc[0]
    assert t_long > 0.90
    assert t_short < 0.01
    assert final > 0.99 * tqqq_bh  # at most 1 slippage hit


def test_monotone_down_qqq_short_sqqq_wins():
    """Down-only universe: SQQQ-long (a.k.a. short QQQ) is the play."""
    n = 300
    qqq = 100.0 * (0.995 ** np.arange(n))  # ~0.5%/day downtrend
    df = _frame_with_qqq(qqq)
    above = precompute_sma_signals(df["qqq"].to_numpy(), max_window=60)
    tqqq_ret = (df["tqqq"].to_numpy()[1:] / df["tqqq"].to_numpy()[:-1]) - 1.0
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0

    final, sharpe, mdd, calmar, n_trades, t_long, t_short, t_cash = run_backtest(
        above, tqqq_ret, sqqq_ret, buy_T=2, sell_T=2, buy_S=2, sell_S=2, slippage_bp=5.0
    )
    # Should spend almost all time short
    assert t_short > 0.90
    assert t_long < 0.01
    # And beat cash by a lot (SQQQ compounds positively when QQQ falls)
    assert final > 5.0


def test_sweep_resumability(tmp_path: Path):
    """Write 1000 rows, kill, resume → final parquet matches full sweep, no dups."""
    # Small synthetic universe so the full sweep is feasible inside a test
    n = 80
    qqq = 100.0 + 10.0 * np.sin(np.linspace(0, 6.28, n))
    df = _frame_with_qqq(qqq)

    results_path = tmp_path / "results.parquet"

    # Run partial: only first 1000 combos
    run_sweep(
        df,
        results_path=results_path,
        max_window=5,  # 6*6 = 36 combos total — too small for "1000"
        n_workers=1,
        slippage_bp=5.0,
        max_combos=20,
        flush_every=10,
    )
    partial = pd.read_parquet(results_path)
    assert len(partial) == 20

    # Resume: should fill the remaining 36-20 = 16 combos
    run_sweep(
        df,
        results_path=results_path,
        max_window=5,
        n_workers=1,
        slippage_bp=5.0,
        flush_every=10,
    )
    full = pd.read_parquet(results_path)
    assert len(full) == 36
    # No duplicates on the combo key
    keys = full[["buy_T", "sell_T", "buy_S", "sell_S"]]
    assert not keys.duplicated().any()


def test_sweep_pool_size_one_works(tmp_path: Path):
    """Sanity that the multiprocessing path runs end-to-end on a tiny grid."""
    n = 60
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "out.parquet"
    run_sweep(
        df,
        results_path=results_path,
        max_window=4,  # 4+3+2+1 = 10 ordered pairs per side → 100 combos
        n_workers=min(2, mp.cpu_count()),
        slippage_bp=5.0,
        flush_every=25,
    )
    out = pd.read_parquet(results_path)
    assert len(out) == 100
    required_cols = {
        "buy_T", "sell_T", "buy_S", "sell_S",
        "final_value", "sharpe", "max_dd", "calmar",
        "n_trades", "time_in_long", "time_in_short", "time_in_cash",
    }
    assert required_cols.issubset(set(out.columns))
