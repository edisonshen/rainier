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

from rainier.backtest.tqqq_sma_sweep import (
    CASH,
    LONG_TQQQ,
    SHORT_SQQQ,
    SweepInputMismatchError,
    clean_cache,
    compute_strategy_id,
    dedup_by_strategy_id,
    iter_phase1_combos,
    iter_phase2_combos,
    precompute_sma_signals,
    results_cache_companions,
    run_backtest,
    run_sweep,
    walk_forward_top_n,
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
    above, valid = precompute_sma_signals(qqq, max_window=60)
    assert above.shape == (n_days, 60)
    assert valid.shape == (n_days, 60)
    assert above.dtype == np.bool_
    assert valid.dtype == np.bool_
    # SMA1 is the price itself → not a meaningful signal → valid=False always
    assert not above[:, 0].any()
    assert not valid[:, 0].any()
    # SMA(w) validity: days 0..w-2 invalid, w-1..end valid
    for w in range(2, 8):
        assert not valid[: w - 1, w - 1].any(), f"warmup days for w={w} should be invalid"
        assert valid[w - 1:, w - 1].all(), f"post-warmup days for w={w} should be valid"


def test_run_backtest_deterministic():
    df = _frame_with_qqq(np.linspace(100.0, 300.0, 200))
    above, valid = precompute_sma_signals(df["qqq"].to_numpy(), max_window=60)
    tqqq_ret = (df["tqqq"].to_numpy()[1:] / df["tqqq"].to_numpy()[:-1]) - 1.0
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0
    a = run_backtest(above, valid, tqqq_ret, sqqq_ret, 5, 20, 5, 20, slippage_bp=5.0)
    b = run_backtest(above, valid, tqqq_ret, sqqq_ret, 5, 20, 5, 20, slippage_bp=5.0)
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
    above, valid = precompute_sma_signals(df["qqq"].to_numpy(), max_window=60)
    tqqq_ret = (df["tqqq"].to_numpy()[1:] / df["tqqq"].to_numpy()[:-1]) - 1.0
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0

    # Very loose buy_T (small window), tight sell_T (also small) so we go long
    # near day-2 and basically never exit. buy_S/sell_S are short-leg knobs
    # that don't fire here.
    final, sharpe, mdd, calmar, n_trades, t_long, t_short, t_cash, _curve_hash = run_backtest(
        above, valid, tqqq_ret, sqqq_ret, buy_T=2, sell_T=2, buy_S=2, sell_S=2, slippage_bp=5.0
    )
    # Should spend almost all time long, retain most of TQQQ buy-and-hold
    # (some drag from waiting for SMA(2) to warm up + entry slippage).
    tqqq_bh = df["tqqq"].iloc[-1] / df["tqqq"].iloc[0]
    assert t_long > 0.90
    assert t_short < 0.01
    # Strategy must beat unleveraged QQQ B&H (the leveraged signal should
    # capture most of the trend) but realistically lags TQQQ B&H by the
    # 1-day warmup + slippage drag.
    qqq_bh = df["qqq"].iloc[-1] / df["qqq"].iloc[0]
    assert final > qqq_bh
    assert final > 0.95 * tqqq_bh


def test_monotone_down_qqq_short_sqqq_wins():
    """Down-only universe: SQQQ-long (a.k.a. short QQQ) is the play."""
    n = 300
    qqq = 100.0 * (0.995 ** np.arange(n))  # ~0.5%/day downtrend
    df = _frame_with_qqq(qqq)
    above, valid = precompute_sma_signals(df["qqq"].to_numpy(), max_window=60)
    tqqq_ret = (df["tqqq"].to_numpy()[1:] / df["tqqq"].to_numpy()[:-1]) - 1.0
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0

    final, sharpe, mdd, calmar, n_trades, t_long, t_short, t_cash, _curve_hash = run_backtest(
        above, valid, tqqq_ret, sqqq_ret, buy_T=2, sell_T=2, buy_S=2, sell_S=2, slippage_bp=5.0
    )
    # Should spend almost all time short
    assert t_short > 0.90
    assert t_long < 0.01
    # And beat cash by a lot (SQQQ compounds positively when QQQ falls)
    assert final > 5.0


def test_no_spurious_short_during_sma_warmup():
    """Regression for codex iter-1: invalid SMA must not fire the SHORT entry.

    Previously `not above[d, w-1]` evaluated True for days 0..w-2 (SMA window
    not yet filled), so every combo entered SHORT_SQQQ on day 0 and during
    the warmup period regardless of price action — a systematic bias that
    poisoned the entire 3.35M-combo sweep.

    Set up a strong uptrend with a long buy_S window: day-0 to day buy_S-2
    have no valid short-leg SMA, so the strategy must stay CASH (or enter
    LONG once buy_T warms up) rather than entering SHORT_SQQQ.
    """
    n = 80
    qqq = 100.0 * (1.01 ** np.arange(n))  # strong uptrend → SQQQ would lose
    df = _frame_with_qqq(qqq)
    above, valid = precompute_sma_signals(df["qqq"].to_numpy(), max_window=60)
    tqqq_ret = (df["tqqq"].to_numpy()[1:] / df["tqqq"].to_numpy()[:-1]) - 1.0
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0

    # buy_T = sell_T = 60: long-leg warm-up takes 59 days. buy_S = sell_S = 50:
    # short-leg warm-up takes 49 days. During days 0..48, BOTH legs are
    # invalid — the strategy must stay CASH. The bug pre-fix would short
    # SQQQ on day 0 (col_bS[0]=False, validity ignored).
    final, _, _, _, n_trades, t_long, t_short, t_cash, _curve_hash = run_backtest(
        above, valid, tqqq_ret, sqqq_ret,
        buy_T=60, sell_T=60, buy_S=50, sell_S=50, slippage_bp=5.0,
    )
    # All 80 days are within the warmup of one or both legs; the strategy
    # must never enter SHORT_SQQQ in an uptrending universe with no valid
    # short signal.
    assert t_short == 0.0, f"expected no short days during warmup, got {t_short}"
    # And since the long leg only warms up at day 59 (and uptrend never
    # crosses below SMA(60)), strategy should be in LONG_TQQQ for days
    # 59..79 (~26% of the run) and CASH before then.
    assert t_long > 0.20
    assert t_cash > 0.65


def test_sweep_refuses_to_mix_rows_when_slippage_changes(tmp_path: Path):
    """Regression for codex iter-2: cache invalidation on input drift.

    Pre-fix, rerunning with a different slippage would silently reuse rows
    computed at the prior slippage (skipping by combo key), then merge them
    with rows at the new slippage. The fix: stamp the parquet with a
    fingerprint of (prices, slippage_bp, max_window) and refuse to extend
    a mismatched parquet.
    """
    import pytest

    n = 60
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "out.parquet"

    # First sweep: slippage 5 bp
    run_sweep(
        df, results_path=results_path, max_window=3, n_workers=1,
        slippage_bp=5.0, flush_every=10,
    )
    assert results_path.exists()
    assert results_path.with_suffix(".fingerprint.txt").exists()

    # Second sweep with different slippage must refuse to extend
    with pytest.raises(SweepInputMismatchError, match="different inputs"):
        run_sweep(
            df, results_path=results_path, max_window=3, n_workers=1,
            slippage_bp=10.0, flush_every=10,
        )

    # Second sweep with different prices must refuse to extend
    qqq2 = 100.0 + 2.0 * np.linspace(0, 50, n)
    df2 = _frame_with_qqq(qqq2)
    with pytest.raises(SweepInputMismatchError, match="different inputs"):
        run_sweep(
            df2, results_path=results_path, max_window=3, n_workers=1,
            slippage_bp=5.0, flush_every=10,
        )

    # Same inputs as the original → should succeed as no-op resume
    run_sweep(
        df, results_path=results_path, max_window=3, n_workers=1,
        slippage_bp=5.0, flush_every=10,
    )


def test_sweep_refuses_legacy_parquet_without_fingerprint(tmp_path: Path):
    """A pre-existing parquet without a fingerprint file is treated as
    suspect — the user must explicitly delete it to start fresh."""
    import pytest

    n = 60
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "out.parquet"

    # Run once, then delete just the fingerprint to simulate a pre-v0.x parquet
    run_sweep(
        df, results_path=results_path, max_window=3, n_workers=1,
        slippage_bp=5.0, flush_every=10,
    )
    results_path.with_suffix(".fingerprint.txt").unlink()

    with pytest.raises(SweepInputMismatchError, match="no fingerprint"):
        run_sweep(
            df, results_path=results_path, max_window=3, n_workers=1,
            slippage_bp=5.0, flush_every=10,
        )


def test_sma1_column_is_never_tradable():
    """Regression for codex iter-1: SMA(1) is structurally False / invalid.

    The cumsum path skips w=1 and column 0 stays all-False. Pre-fix, the
    backtest treated `not above[d, 0]` as a valid SHORT_SQQQ trigger, so a
    combo with buy_S=1 would short on every day where the strategy was in
    CASH — equivalent to a "permanent short" strategy. Post-fix, valid[:, 0]
    is all False, so SMA(1) windows produce zero trades.
    """
    n = 100
    qqq = 100.0 + np.linspace(0, 50, n)  # uptrend, can't matter what the trend is
    df = _frame_with_qqq(qqq)
    above, valid = precompute_sma_signals(df["qqq"].to_numpy(), max_window=10)
    tqqq_ret = (df["tqqq"].to_numpy()[1:] / df["tqqq"].to_numpy()[:-1]) - 1.0
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0

    # All four legs at window=1 → all four signal columns are valid=False.
    # Strategy must stay 100% CASH the whole run (no trades, equity=1.0).
    final, _, _, _, n_trades, t_long, t_short, t_cash, _curve_hash = run_backtest(
        above, valid, tqqq_ret, sqqq_ret,
        buy_T=1, sell_T=1, buy_S=1, sell_S=1, slippage_bp=5.0,
    )
    assert n_trades == 0
    assert t_cash == 1.0
    assert t_long == 0.0
    assert t_short == 0.0
    assert final == 1.0


def test_sweep_resumability(tmp_path: Path):
    """Write 1000 rows, kill, resume → final parquet matches full sweep, no dups."""
    # Small synthetic universe so the full sweep is feasible inside a test
    n = 80
    qqq = 100.0 + 10.0 * np.sin(np.linspace(0, 6.28, n))
    df = _frame_with_qqq(qqq)

    results_path = tmp_path / "results.parquet"

    # max_window=4 → ordered pairs per side = 4+3+2+1 = 10 → 100 combos total
    # Run partial: only first 30 combos
    run_sweep(
        df,
        results_path=results_path,
        max_window=4,
        n_workers=1,
        slippage_bp=5.0,
        max_combos=30,
        flush_every=10,
    )
    partial = pd.read_parquet(results_path)
    assert len(partial) == 30

    # Resume: should fill the remaining 100-30 = 70 combos
    run_sweep(
        df,
        results_path=results_path,
        max_window=4,
        n_workers=1,
        slippage_bp=5.0,
        flush_every=10,
    )
    full = pd.read_parquet(results_path)
    assert len(full) == 100
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


def test_render_report_handles_missing_walkforward(tmp_path: Path):
    """Regression: report should render even if walkforward parquet doesn't exist.

    The docstring on `render_report` and the `pd.DataFrame()` fallback at the
    top of the function both promise graceful degradation when the walkforward
    file is missing. Previously `_top50_table` would crash with KeyError on
    `set_index(['buy_T', ...])` because the empty fallback DataFrame has no
    columns.
    """
    from rainier.backtest.tqqq_sma_report import render_report

    # Build a real (tiny) sweep so results.parquet has the expected schema
    n = 60
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"
    run_sweep(
        df,
        results_path=results_path,
        max_window=4,
        n_workers=1,
        slippage_bp=5.0,
        flush_every=25,
    )

    # Intentionally point at a nonexistent walkforward file
    missing_wf = tmp_path / "does_not_exist.parquet"
    out_html = tmp_path / "report.html"

    render_report(
        prices=df,
        results_path=results_path,
        walkforward_path=missing_wf,
        output_path=out_html,
        sweep_wall_seconds=1.0,
        slippage_bp=5.0,
        max_window=4,
    )
    assert out_html.exists()
    html = out_html.read_text(encoding="utf-8")
    # Should still render the top-50 table (with empty train/test/delta cells)
    assert "top-50 winners" in html
    # Section 7 should show the missing-walkforward fallback
    assert "Walk-forward parquet not found" in html


# ---------------------------------------------------------------------------
# Phase-4 polish: strategy_id fingerprint + leaderboard dedup
# ---------------------------------------------------------------------------


def test_compute_strategy_id_deterministic_and_uint64():
    """compute_strategy_id is a passthrough on the path-identity curve_hash
    emitted by run_backtest. Identical hash → identical id; distinct hash
    → distinct id; result fits in uint64.

    Post codex iter-3: the old summary-metric heuristic was replaced with
    a true path-equivalence signal (FNV-1a over the daily state sequence).
    This test now exercises the new contract.
    """
    a = compute_strategy_id(0x123456789ABCDEF0)
    b = compute_strategy_id(0x123456789ABCDEF0)
    assert a == b
    assert 0 <= a < 2**64
    # Distinct hash → distinct id
    assert compute_strategy_id(0x123456789ABCDEF1) != a
    # Zero is a valid (though unlikely) id
    assert compute_strategy_id(0) == 0
    # Top of range: uint64 max wraps cleanly
    assert compute_strategy_id(2**64 - 1) == 2**64 - 1


def test_curve_hash_distinguishes_different_paths():
    """codex iter-3 regression: the strategy_id MUST be path-sensitive, not a
    rounded-summary heuristic.

    Pre-fix, compute_strategy_id hashed (final_value, n_trades, time_in_long,
    time_in_short) rounded to 2-4 dp. Two combos with genuinely-different
    equity curves could land on the same rounded tuple and get collapsed.

    Post-fix, the strategy_id is derived from a path-identity hash (FNV-1a
    over the daily end-of-day state sequence) computed inside run_backtest.
    Distinct state sequences MUST produce distinct curve_hashes.

    Here we construct two combos in a varied-price universe where the long
    and short legs are both active. Different SMA windows produce different
    entry/exit timings → different state sequences → must produce different
    curve_hashes. The pre-fix heuristic would not reliably catch this
    distinction if the summary stats happened to round the same way.
    """
    # Oscillating universe so both legs are active and different SMA windows
    # produce genuinely different state sequences.
    n = 300
    t = np.arange(n)
    qqq = 100.0 + 30.0 * np.sin(t / 15.0) + 0.05 * t  # oscillating + drift
    df = _frame_with_qqq(qqq)
    above, valid = precompute_sma_signals(df["qqq"].to_numpy(), max_window=60)
    tqqq_ret = (df["tqqq"].to_numpy()[1:] / df["tqqq"].to_numpy()[:-1]) - 1.0
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0

    out_a = run_backtest(above, valid, tqqq_ret, sqqq_ret, 5, 10, 5, 10, slippage_bp=5.0)
    out_b = run_backtest(above, valid, tqqq_ret, sqqq_ret, 7, 12, 7, 12, slippage_bp=5.0)

    # 9-tuple now: last field is the curve_hash.
    curve_hash_a = int(out_a[-1])
    curve_hash_b = int(out_b[-1])
    assert curve_hash_a != curve_hash_b, (
        f"different SMA windows produced same curve_hash — fingerprint isn't path-sensitive\n"
        f"  a (5,10,5,10): {curve_hash_a}\n  b (7,12,7,12): {curve_hash_b}"
    )
    # compute_strategy_id passthrough must preserve the distinction.
    assert compute_strategy_id(curve_hash_a) != compute_strategy_id(curve_hash_b)


def test_curve_hash_distinguishes_same_bar_churn():
    """codex iter-2 [P1] regression: end-of-day state alone is NOT sufficient
    to distinguish equity curves when a combo exits and re-enters the same
    side on the same bar.

    Two combos in this test are constructed to produce different n_trades
    even when the daily end-of-day state sequence is similar. If the hash
    mixed only state and not n_trades, churning combos could collide with
    quieter ones. With n_trades mixed in, the curve_hashes diverge.

    We use a price universe that crosses an SMA threshold frequently so that
    short sell_T windows produce same-day exit/re-entry churn relative to
    longer sell_T windows.
    """
    n = 400
    t = np.arange(n)
    # Whipsaw universe: rapid oscillation crosses SMA(3) and SMA(20)
    # at very different rates → very different trade counts even though
    # both end up long for most of the run.
    qqq = 100.0 + 8.0 * np.sin(t / 3.0) + 0.02 * t
    df = _frame_with_qqq(qqq)
    above, valid = precompute_sma_signals(df["qqq"].to_numpy(), max_window=60)
    tqqq_ret = (df["tqqq"].to_numpy()[1:] / df["tqqq"].to_numpy()[:-1]) - 1.0
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0

    # Short sell_T (3) → many quick exits & re-entries
    out_churn = run_backtest(above, valid, tqqq_ret, sqqq_ret, 3, 3, 3, 3, slippage_bp=5.0)
    # Long sell_T (30) → patient strategy, fewer trades
    out_patient = run_backtest(above, valid, tqqq_ret, sqqq_ret, 3, 30, 3, 30, slippage_bp=5.0)

    n_trades_churn = int(out_churn[4])
    n_trades_patient = int(out_patient[4])
    # Sanity: churn strategy must trade more than patient one
    assert n_trades_churn > n_trades_patient, (
        f"expected churn > patient trades, got churn={n_trades_churn}, patient={n_trades_patient}"
    )
    # Different trade counts → different curve_hashes (n_trades mixed in)
    assert int(out_churn[-1]) != int(out_patient[-1])


def test_curve_hash_identical_combos_match():
    """Same combo, same inputs → same curve_hash (deterministic FNV-1a)."""
    n = 200
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    above, valid = precompute_sma_signals(df["qqq"].to_numpy(), max_window=60)
    tqqq_ret = (df["tqqq"].to_numpy()[1:] / df["tqqq"].to_numpy()[:-1]) - 1.0
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0
    out_a = run_backtest(above, valid, tqqq_ret, sqqq_ret, 4, 8, 4, 8, slippage_bp=5.0)
    out_b = run_backtest(above, valid, tqqq_ret, sqqq_ret, 4, 8, 4, 8, slippage_bp=5.0)
    assert int(out_a[-1]) == int(out_b[-1])


def test_dormant_short_legs_share_strategy_id(tmp_path: Path):
    """Two combos with dormant short legs (buy_S=1 makes SQQQ entry condition
    structurally False) must collapse to the same strategy_id when buy_T/sell_T
    are held fixed and only sell_S varies.

    With buy_S=1, the short-leg signal is ``QQQ_close < SMA(1) = QQQ_close``
    — always False — so the short leg never fires regardless of sell_S. Two
    combos that differ only in sell_S must produce identical equity curves
    and therefore the same strategy_id.
    """
    n = 80
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"

    # max_window=5 so combos (22,44,1,1) and (22,44,1,5) are out of reach;
    # use max_window=10 with combos (3,5,1,1) and (3,5,1,5) instead. The
    # spec's example used (22,44,1,1) and (22,44,1,5); same principle.
    run_sweep(
        df, results_path=results_path, max_window=10, n_workers=1,
        slippage_bp=5.0, flush_every=50,
    )
    out = pd.read_parquet(results_path)
    assert "strategy_id" in out.columns
    a = out[(out.buy_T == 3) & (out.sell_T == 5) & (out.buy_S == 1) & (out.sell_S == 1)]
    b = out[(out.buy_T == 3) & (out.sell_T == 5) & (out.buy_S == 1) & (out.sell_S == 5)]
    assert len(a) == 1 and len(b) == 1
    assert int(a.iloc[0]["strategy_id"]) == int(b.iloc[0]["strategy_id"])


def test_dedup_by_strategy_id_returns_one_per_group(tmp_path: Path):
    """Dedup returns exactly one row per unique strategy_id, and
    n_equivalent sums back to the input row count."""
    n = 80
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"
    run_sweep(
        df, results_path=results_path, max_window=10, n_workers=1,
        slippage_bp=5.0, flush_every=50,
    )
    raw = pd.read_parquet(results_path)
    deduped = dedup_by_strategy_id(raw)
    # One row per unique strategy_id
    assert len(deduped) == raw["strategy_id"].nunique()
    assert not deduped["strategy_id"].duplicated().any()
    # n_equivalent is the size of each equivalence class
    assert int(deduped["n_equivalent"].sum()) == len(raw)
    # Representative tuple is the lexicographic min within each group
    for sid, grp in raw.groupby("strategy_id"):
        rep = deduped[deduped["strategy_id"] == sid].iloc[0]
        keys = grp[["buy_T", "sell_T", "buy_S", "sell_S"]].apply(tuple, axis=1).tolist()
        expected_min = min(keys)
        actual = (int(rep.buy_T), int(rep.sell_T), int(rep.buy_S), int(rep.sell_S))
        assert actual == expected_min, f"sid={sid}: rep {actual} != lex-min {expected_min}"


def test_dedup_raises_when_strategy_id_missing():
    """A v1-schema parquet (no strategy_id column) must be rejected with a
    clear KeyError pointing at the migration path."""
    import pytest

    df = pd.DataFrame({
        "buy_T": [1, 2], "sell_T": [1, 2], "buy_S": [1, 2], "sell_S": [1, 2],
        "final_value": [1.0, 2.0], "n_trades": [0, 5],
        "time_in_long": [0.0, 0.5], "time_in_short": [0.0, 0.3],
    })
    with pytest.raises(KeyError, match="strategy_id"):
        dedup_by_strategy_id(df)


def test_strategy_id_schema_version_invalidates_legacy_parquet(tmp_path: Path):
    """A v1 results.parquet (no strategy_id column) must be rejected by the
    sweep fingerprint even when prices/slippage/max_window are identical —
    the schema version is part of the fingerprint.
    """
    import pytest

    from rainier.backtest.tqqq_sma_sweep import _RESULTS_SCHEMA_VERSION

    n = 60
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"

    # Simulate a legacy v1-schema parquet on disk: write the file by hand
    # with the OLD fingerprint string that did NOT include schema=...
    import hashlib as _hashlib

    legacy_fp = _hashlib.sha256()
    legacy_fp.update(df[["qqq", "tqqq", "sqqq"]].sort_index().to_parquet())
    legacy_fp.update(b"|slippage_bp=5.000000|max_window=3")
    fp_str = legacy_fp.hexdigest()
    # Write a minimal v1 parquet (no strategy_id) and the legacy fingerprint
    legacy_df = pd.DataFrame({
        "buy_T": [1], "sell_T": [1], "buy_S": [1], "sell_S": [1],
        "final_value": [1.0], "sharpe": [0.0], "max_dd": [0.0], "calmar": [0.0],
        "n_trades": [0], "time_in_long": [0.0], "time_in_short": [0.0],
        "time_in_cash": [1.0],
    })
    legacy_df.to_parquet(results_path, index=False)
    results_path.with_suffix(".fingerprint.txt").write_text(fp_str)

    # New sweep with same inputs must refuse because the schema version flipped.
    with pytest.raises(SweepInputMismatchError, match="different inputs"):
        run_sweep(
            df, results_path=results_path, max_window=3, n_workers=1,
            slippage_bp=5.0, flush_every=10,
        )
    # Sanity: the schema version constant is non-empty (so a future change
    # to it requires a deliberate bump, not an accidental empty string).
    assert _RESULTS_SCHEMA_VERSION
    assert "strategy_id" in _RESULTS_SCHEMA_VERSION


def test_report_dormant_cells_render_as_dash(tmp_path: Path):
    """For top-50 rows where ``time_in_short == 0``, the buy_S/sell_S cells
    must render as the literal string ``-`` in the HTML output.

    The synthetic universe in this test is a monotone uptrend, so every
    combo's short leg should be dormant (time_in_short == 0). The resulting
    HTML must contain rows with `-` in the buy_S/sell_S positions and the
    `LONG-only` type label.
    """
    from rainier.backtest.tqqq_sma_report import render_report

    n = 200
    qqq = 100.0 * (1.005 ** np.arange(n))  # strong uptrend → short leg dormant
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"
    run_sweep(
        df, results_path=results_path, max_window=4, n_workers=1,
        slippage_bp=5.0, flush_every=50,
    )

    raw = pd.read_parquet(results_path)
    # In this universe most combos should be LONG-only (time_in_short == 0)
    long_only = raw[(raw["time_in_short"] == 0.0) & (raw["time_in_long"] > 0.0)]
    assert len(long_only) > 0, "synthetic uptrend should produce LONG-only combos"

    out_html = tmp_path / "report.html"
    render_report(
        prices=df,
        results_path=results_path,
        walkforward_path=tmp_path / "no_wf.parquet",  # missing → graceful
        output_path=out_html,
        sweep_wall_seconds=1.0,
        slippage_bp=5.0,
        max_window=4,
    )
    html = out_html.read_text(encoding="utf-8")
    # Dormant-cell marker should appear in the table for buy_S/sell_S
    assert "LONG-only" in html, "type column should label dormant-short rows"
    # The literal "-" cell (right-aligned num class) should appear for the
    # dormant short-leg columns; check at least one such row exists.
    assert ">-</td>" in html or ">-<" in html


def test_report_strategy_type_classifier():
    """Direct unit test of the type classifier helper — keeps the
    LONG-only / SHORT-only / BOTH-legs convention pinned even if the
    rendering changes later."""
    from rainier.backtest.tqqq_sma_report import _strategy_type

    assert _strategy_type(0.0, 0.5) == "SHORT-only"
    assert _strategy_type(0.5, 0.0) == "LONG-only"
    assert _strategy_type(0.4, 0.4) == "BOTH-legs"
    # Pure-cash (neither leg fired) is BOTH-legs, NOT LONG-only or SHORT-only
    # — neither leg is structurally dormant, just no signal fired.
    assert _strategy_type(0.0, 0.0) == "BOTH-legs"


# ---------------------------------------------------------------------------
# Phase 2: unconstrained 12.96M grid (adds anti-trend combos to Phase 1)
# ---------------------------------------------------------------------------


def test_iter_phase2_combos_count_full_grid():
    """Phase 2 enumerates ALL (buy_T, sell_T, buy_S, sell_S) with each in
    {1..max_window}, no constraint. Count = max_window**4.

    For max_window=60 this is 60^4 = 12,960,000. The set is exactly Phase 1
    (1830*1830 = 3,348,900 trend-following) ∪ Phase 2-only anti-trend
    (9,611,100 combos where sell_T < buy_T OR sell_S < buy_S).

    We count via the generator instead of materializing a 12.96M-tuple list
    (codex review iter-1 [P2]) — sum(1 for _ in ...) is O(1) memory and
    keeps pytest from chewing ~1GB on small CI runners.
    """
    count = sum(1 for _ in iter_phase2_combos(max_window=60))
    assert count == 60 * 60 * 60 * 60 == 12_960_000


def test_iter_phase2_combos_no_constraint_small_window():
    """max_window=3 → 3^4 = 81 combos vs Phase 1's 36 (trend-following only)."""
    combos = list(iter_phase2_combos(max_window=3))
    assert len(combos) == 81
    # Every tuple is in range
    for bT, sT, bS, sS in combos:
        assert 1 <= bT <= 3
        assert 1 <= sT <= 3
        assert 1 <= bS <= 3
        assert 1 <= sS <= 3


def test_iter_phase2_combos_are_unique():
    """No duplicates — the generator emits each (bT, sT, bS, sS) exactly once."""
    combos = list(iter_phase2_combos(max_window=5))
    assert len(combos) == 5 ** 4
    assert len(set(combos)) == len(combos)


def test_iter_phase2_superset_of_phase1():
    """Phase 2 strictly contains Phase 1; the difference is the anti-trend set.

    Anti-trend set = combos where sell_T < buy_T OR sell_S < buy_S.
    For max_window=60: |Phase 2| - |Phase 1| = 12_960_000 - 3_348_900 = 9_611_100.
    """
    mw = 6  # small enough to enumerate both sets
    p1 = set(iter_phase1_combos(max_window=mw))
    p2 = set(iter_phase2_combos(max_window=mw))
    # Phase 1 ⊆ Phase 2
    assert p1.issubset(p2)
    # The difference is anti-trend (sell < buy on at least one leg)
    anti = p2 - p1
    for bT, sT, bS, sS in anti:
        assert (sT < bT) or (sS < bS), (
            f"({bT},{sT},{bS},{sS}) is in p2 - p1 but is trend-following"
        )
    # Cardinality check for max_window=60 case (analytic count)
    p1_60 = 1830 * 1830  # 3,348,900
    p2_60 = 60 ** 4       # 12,960,000
    assert p2_60 - p1_60 == 9_611_100


def test_iter_phase2_emits_anti_trend_examples():
    """Spot-check that classic anti-trend combos (sell < buy) are emitted."""
    combos = set(iter_phase2_combos(max_window=10))
    # Anti-trend long leg only
    assert (5, 3, 5, 5) in combos
    # Anti-trend short leg only
    assert (5, 5, 5, 3) in combos
    # Anti-trend both legs
    assert (8, 2, 8, 2) in combos
    # And the trend-following examples are still in there
    assert (3, 5, 3, 5) in combos
    assert (1, 10, 1, 10) in combos


def test_sweep_phase2_emits_full_grid(tmp_path: Path):
    """run_sweep with phase=2 must emit max_window**4 rows on a small grid.

    Uses max_window=4 → 4^4 = 256 combos (vs Phase 1's 100), entirely
    enumerable in-process. Verifies the resulting parquet contains every
    (bT, sT, bS, sS) tuple in {1..4}^4 exactly once.
    """
    n = 80
    qqq = 100.0 + 10.0 * np.sin(np.linspace(0, 6.28, n))
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "phase2.parquet"

    run_sweep(
        df,
        results_path=results_path,
        max_window=4,
        n_workers=1,
        slippage_bp=5.0,
        flush_every=64,
        phase=2,
    )
    out = pd.read_parquet(results_path)
    assert len(out) == 4 ** 4 == 256
    keys = {(int(r.buy_T), int(r.sell_T), int(r.buy_S), int(r.sell_S))
            for r in out.itertuples()}
    assert keys == {(bT, sT, bS, sS)
                    for bT in range(1, 5)
                    for sT in range(1, 5)
                    for bS in range(1, 5)
                    for sS in range(1, 5)}


def test_phase2_report_includes_comparison_section(tmp_path: Path):
    """Phase 2 reports must contain the Phase 1 vs Phase 2 comparison section
    with a clear YES/NO headline answering 'does any anti-trend combo beat
    the trend-following winner?'.
    """
    from rainier.backtest.tqqq_sma_report import render_report

    n = 200
    # Mixed regime: an up-then-down-then-up oscillation so both trend-following
    # and anti-trend combos can have non-trivial behavior.
    t = np.arange(n)
    qqq = 100.0 + 20.0 * np.sin(t / 25.0) + 0.05 * t
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"

    # Phase 2 sweep on a small grid so the test is fast (max_window=4 → 256 combos)
    run_sweep(
        df, results_path=results_path, max_window=4, n_workers=1,
        slippage_bp=5.0, flush_every=64, phase=2,
    )

    out_html = tmp_path / "phase2_report.html"
    render_report(
        prices=df,
        results_path=results_path,
        walkforward_path=tmp_path / "no_wf.parquet",
        output_path=out_html,
        sweep_wall_seconds=1.0,
        slippage_bp=5.0,
        max_window=4,
        phase=2,
    )
    assert out_html.exists()
    html = out_html.read_text(encoding="utf-8")
    # Headline section anchor + title present
    assert 'id="compare"' in html
    assert "Phase 1 vs Phase 2 comparison" in html
    # The headline must be one of YES / NO / DEGENERATE (see
    # _phase1_vs_phase2_section — DEGENERATE fires when the anti-trend
    # winner has n_trades=1, i.e. is effective buy-and-hold).
    assert ("headline" in html.lower())
    assert (
        "YES — best anti-trend combo" in html
        or "NO — trend-following winner" in html
        or "DEGENERATE — best anti-trend combo" in html
    )
    # The comparison table must show all three rows
    assert "Phase 1 (trend-following" in html
    assert "Phase 2 only (anti-trend" in html
    assert "Phase 2 (full grid)" in html
    # TOC must list 10 entries (extra Phase-2 section)
    assert "10. reproducibility" in html


def test_phase2_report_uses_max_window_derived_counts(tmp_path: Path):
    """Codex review iter-1 [P2]: hard-coded 60-derived combo counts
    (12,960,000 / 3,348,900 / 9,611,100) in the framing and comparison
    sections would be wrong for any --max-window != 60. The text must derive
    counts from the actual ``max_window`` argument.
    """
    from rainier.backtest.tqqq_sma_report import render_report

    n = 60
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"

    # max_window=4 → Phase-2 grid is 4^4 = 256, Phase-1 subset is T(4)^2 = 100
    run_sweep(
        df, results_path=results_path, max_window=4, n_workers=1,
        slippage_bp=5.0, flush_every=64, phase=2,
    )
    out_html = tmp_path / "phase2_report.html"
    render_report(
        prices=df,
        results_path=results_path,
        walkforward_path=tmp_path / "no_wf.parquet",
        output_path=out_html,
        sweep_wall_seconds=1.0,
        slippage_bp=5.0,
        max_window=4,
        phase=2,
    )
    html = out_html.read_text(encoding="utf-8")
    # Correct counts must appear (256 total, 100 trend-following, 156 anti)
    assert "256" in html
    assert "100" in html
    assert "{1..4}^4" in html
    assert "{1..4}" in html
    # WRONG counts must NOT appear
    assert "12,960,000" not in html
    assert "3,348,900" not in html
    assert "9,611,100" not in html
    assert "{1..60}" not in html


def test_walk_forward_top_n_filters_to_phase1_when_phase_arg_is_1(tmp_path: Path):
    """Codex review iter-2 [P2]: when the shared parquet has been extended
    by a Phase-2 sweep, ``walk_forward_top_n(..., phase=1)`` must restrict
    to the trend-following subset before dedup/top-N selection. Otherwise
    the Phase-1 walk-forward parquet would contain anti-trend Phase-2
    winners labeled as Phase-1 validation.
    """
    n = 80
    qqq = 100.0 + 10.0 * np.sin(np.linspace(0, 6.28, n))
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"

    # Populate the parquet with the full Phase-2 grid (256 combos at mw=4)
    run_sweep(
        df, results_path=results_path, max_window=4, n_workers=1,
        slippage_bp=5.0, flush_every=64, phase=2,
    )

    wf_p1 = walk_forward_top_n(
        df, results_path=results_path, top_n=20,
        split_date="2020-02-15",  # mid-range of synthetic data
        slippage_bp=5.0, max_window=4, phase=1,
    )
    # Every row in the Phase-1 walk-forward must satisfy the trend-following
    # constraint. A bug here would surface anti-trend (sell < buy) combos.
    assert (wf_p1["sell_T"] >= wf_p1["buy_T"]).all()
    assert (wf_p1["sell_S"] >= wf_p1["buy_S"]).all()

    # Phase 2 walk-forward (default behavior) must see the full grid —
    # i.e. it is NOT filtered, so it can include anti-trend rows.
    wf_p2 = walk_forward_top_n(
        df, results_path=results_path, top_n=20,
        split_date="2020-02-15",
        slippage_bp=5.0, max_window=4, phase=2,
    )
    # Sanity: with the full grid available, p2 walk-forward sees at least as
    # many DISTINCT (post-dedup) curves as p1 — the anti-trend rows expand
    # the dedup'd pool. We don't assert strict >, since synthetic data can
    # collapse heavily; just that the filter argument matters.
    assert len(wf_p2) >= 1


def test_phase1_report_ignores_phase2_rows_on_disk(tmp_path: Path):
    """Codex review iter-1 [P2] (cache contamination): the same parquet is
    shared across phases (Phase 2 is a superset of Phase 1 by design). If a
    Phase-2 sweep has populated the parquet and the user then re-renders
    --phase 1, the resulting report MUST contain only the trend-following
    subset — not the full 4^4 grid masquerading as Phase 1.

    We run Phase 2 first (256 combos) and then render a Phase-1 report. The
    Phase-1 report's subtitle must claim ``T(4)^2 = 100 combos``, not 256.
    """
    from rainier.backtest.tqqq_sma_report import render_report

    n = 60
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"

    # Populate the parquet with the full Phase-2 grid first.
    run_sweep(
        df, results_path=results_path, max_window=4, n_workers=1,
        slippage_bp=5.0, flush_every=64, phase=2,
    )
    assert len(pd.read_parquet(results_path)) == 256

    out_html = tmp_path / "phase1_report.html"
    render_report(
        prices=df,
        results_path=results_path,
        walkforward_path=tmp_path / "no_wf.parquet",
        output_path=out_html,
        sweep_wall_seconds=1.0,
        slippage_bp=5.0,
        max_window=4,
        phase=1,
    )
    html = out_html.read_text(encoding="utf-8")
    # Subtitle must report 100 combos (Phase-1 subset), not 256
    assert "100 combos" in html
    assert "256 combos" not in html
    # No Phase-2 comparison section in a Phase-1 report
    assert 'id="compare"' not in html


def test_phase1_report_unchanged_no_comparison_section(tmp_path: Path):
    """Phase 1 reports (default) must NOT contain the comparison section —
    we don't want to clobber the Phase 1 deliverable.
    """
    from rainier.backtest.tqqq_sma_report import render_report

    n = 60
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"
    run_sweep(
        df, results_path=results_path, max_window=4, n_workers=1,
        slippage_bp=5.0, flush_every=64, phase=1,
    )

    out_html = tmp_path / "phase1_report.html"
    render_report(
        prices=df,
        results_path=results_path,
        walkforward_path=tmp_path / "no_wf.parquet",
        output_path=out_html,
        sweep_wall_seconds=1.0,
        slippage_bp=5.0,
        max_window=4,
        # phase defaults to 1
    )
    html = out_html.read_text(encoding="utf-8")
    assert 'id="compare"' not in html
    assert "Phase 1 vs Phase 2 comparison" not in html
    # Phase 1 TOC ends at 9. reproducibility
    assert "9. reproducibility" in html
    assert "10. reproducibility" not in html


def test_phase2_headline_flags_degenerate_buy_and_hold_winner(tmp_path: Path):
    """Regression for the suspicious (2,1,1,1) headline finding.

    The unconstrained Phase-2 grid surfaces combos like (2,1,1,1) where
    ``sell_T=1`` makes the long-exit signal structurally dormant (the SMA(1)
    validity mask in :func:`precompute_sma_signals` prevents the exit firing),
    so the strategy enters LONG_TQQQ once on the first ``QQQ > SMA2`` bar and
    never exits. The resulting equity curve is essentially leveraged
    buy-and-hold TQQQ with one slippage hit — NOT a discovered anti-trend
    strategy.

    The headline must disclose this rather than dressing it up as a winning
    anti-trend combo. The disclosure is gated on ``n_trades == 1`` for the
    anti-trend winner row. We construct a monotonically rising price series
    so the (2,1,1,1)-like combo enters early and rides up forever, producing
    n_trades=1.
    """
    from rainier.backtest.tqqq_sma_report import render_report

    n = 120
    # Monotonic upward drift → QQQ > SMA always after warmup, never exits.
    # TQQQ rides up; SQQQ never used since short leg is dormant (buy_S=1).
    qqq = 100.0 + np.arange(n, dtype=np.float64) * 0.5
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"

    run_sweep(
        df, results_path=results_path, max_window=4, n_workers=1,
        slippage_bp=5.0, flush_every=64, phase=2,
    )

    # Verify the suspect combo (2,1,1,1) really did produce n_trades=1
    out = pd.read_parquet(results_path)
    row = out[
        (out["buy_T"] == 2)
        & (out["sell_T"] == 1)
        & (out["buy_S"] == 1)
        & (out["sell_S"] == 1)
    ].iloc[0]
    assert int(row.n_trades) == 1, (
        "(2,1,1,1) on monotonic-up data must enter once and never exit — "
        "this is the buy-and-hold-equivalent scenario the headline must flag."
    )

    out_html = tmp_path / "phase2_report.html"
    render_report(
        prices=df,
        results_path=results_path,
        walkforward_path=tmp_path / "no_wf.parquet",
        output_path=out_html,
        sweep_wall_seconds=1.0,
        slippage_bp=5.0,
        max_window=4,
        phase=2,
    )
    html = out_html.read_text(encoding="utf-8")
    # The DEGENERATE headline must fire (the anti-trend winner has n_trades=1)
    assert "DEGENERATE — best anti-trend combo" in html
    # The supporting explanatory note must call out SMA(1) validity gating
    assert "SMA(1) validity mask" in html
    # The buy-and-hold flag must appear in the trades column
    assert "(buy-and-hold)" in html
    # The qbox must use the warn (not accent) color since this is NOT a win
    assert "border-left-color: var(--warn)" in html


def test_phase2_headline_celebrates_meaningful_anti_trend_winner(tmp_path: Path):
    """When the anti-trend winner has n_trades > 1, it's a real comparator
    and the headline should say YES (positive framing). Sanity check that we
    only suppress to DEGENERATE for the n_trades=1 case.
    """
    from rainier.backtest.tqqq_sma_report import _phase1_vs_phase2_section

    # Hand-build a tiny frame with one anti-trend row that has n_trades > 1
    # and beats the trend-following winner. We bypass the sweep entirely so
    # the test is independent of the inner backtest's stochasticity on
    # synthetic data.
    rows = [
        # Trend-following winner (sell >= buy on both legs), n_trades=10
        {"buy_T": 2, "sell_T": 5, "buy_S": 2, "sell_S": 5,
         "final_value": 100.0, "sharpe": 0.8, "max_dd": 0.3, "calmar": 1.5,
         "n_trades": 10, "time_in_long": 0.7, "time_in_short": 0.0,
         "time_in_cash": 0.3, "strategy_id": np.uint64(1)},
        # Anti-trend winner (sell_T < buy_T), n_trades=20 — meaningful churn
        {"buy_T": 5, "sell_T": 2, "buy_S": 2, "sell_S": 5,
         "final_value": 200.0, "sharpe": 1.0, "max_dd": 0.4, "calmar": 1.2,
         "n_trades": 20, "time_in_long": 0.5, "time_in_short": 0.3,
         "time_in_cash": 0.2, "strategy_id": np.uint64(2)},
    ]
    df = pd.DataFrame(rows)
    html = _phase1_vs_phase2_section(df)
    assert "YES — best anti-trend combo" in html
    assert "DEGENERATE" not in html
    # qbox uses accent (positive) color
    assert "border-left-color: var(--accent)" in html


def test_sweep_phase2_extends_phase1_parquet(tmp_path: Path):
    """A Phase-1 parquet on disk + a Phase-2 rerun on the same inputs must
    extend (not rebuild) the parquet to the full grid. Resumability across
    phases is the key property — we don't want to re-run the 3.35M Phase-1
    combos when the user asks for Phase 2 on top.
    """
    n = 80
    qqq = 100.0 + 10.0 * np.sin(np.linspace(0, 6.28, n))
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"

    # Phase 1 first: 10*10 = 100 combos at max_window=4
    run_sweep(
        df, results_path=results_path, max_window=4, n_workers=1,
        slippage_bp=5.0, flush_every=64, phase=1,
    )
    phase1 = pd.read_parquet(results_path)
    assert len(phase1) == 100

    # Phase 2 extends with the remaining 156 anti-trend combos → total 256
    run_sweep(
        df, results_path=results_path, max_window=4, n_workers=1,
        slippage_bp=5.0, flush_every=64, phase=2,
    )
    phase2 = pd.read_parquet(results_path)
    assert len(phase2) == 256
    # No duplicate combo keys
    keys = phase2[["buy_T", "sell_T", "buy_S", "sell_S"]]
    assert not keys.duplicated().any()


# ---------------------------------------------------------------------------
# Cache hygiene — clean_cache + results_cache_companions (slim audit R3)
# ---------------------------------------------------------------------------


def test_results_cache_companions_lists_existing_parquet_and_fingerprint(tmp_path: Path):
    """A completed sweep leaves results.parquet + a .fingerprint.txt; the
    companion helper must return exactly those two (and nothing absent)."""
    n = 60
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"
    run_sweep(
        df, results_path=results_path, max_window=3, n_workers=1,
        slippage_bp=5.0, flush_every=10,
    )

    companions = results_cache_companions(results_path)
    names = {p.name for p in companions}
    assert names == {"results.parquet", "results.fingerprint.txt"}
    for p in companions:
        assert p.exists()


def test_results_cache_companions_omits_missing(tmp_path: Path):
    """When neither file exists the helper returns an empty list (no crash)."""
    assert results_cache_companions(tmp_path / "nope.parquet") == []


def test_clean_cache_removes_directory_and_is_idempotent(tmp_path: Path):
    """clean_cache deletes the whole cache dir, returns the removed entries,
    and is a no-op (empty list) when the dir is already gone."""
    cache_dir = tmp_path / "tqqq_sma"
    cache_dir.mkdir()
    # Populate with a realistic-looking set of cached artifacts
    (cache_dir / "results.parquet").write_bytes(b"x" * 1024)
    (cache_dir / "results.fingerprint.txt").write_text("deadbeef")
    (cache_dir / "prices.parquet").write_bytes(b"y" * 512)

    removed = clean_cache(cache_dir)
    assert {p.name for p in removed} == {
        "results.parquet", "results.fingerprint.txt", "prices.parquet",
    }
    assert not cache_dir.exists()

    # Idempotent: second call on the now-absent dir is a clean no-op
    assert clean_cache(cache_dir) == []


def test_no_results_cache_flow_drops_parquet_but_keeps_walkforward(tmp_path: Path):
    """Simulate the --no-results-cache path: after the walk-forward parquet is
    derived, deleting the results companions must leave the walk-forward output
    intact (the run's deliverable) while reclaiming the giant cache."""
    n = 80
    qqq = 100.0 + 10.0 * np.sin(np.linspace(0, 6.28, n))
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"
    run_sweep(
        df, results_path=results_path, max_window=4, n_workers=1,
        slippage_bp=5.0, flush_every=64,
    )
    # Derive the walk-forward set the way the CLI does, to a sibling parquet
    top_wf = walk_forward_top_n(
        df, results_path=results_path, top_n=5, slippage_bp=5.0, max_window=4,
    )
    wf_path = results_path.parent / "top_walkforward.parquet"
    top_wf.to_parquet(wf_path, index=False)

    # --no-results-cache: drop results.parquet + fingerprint
    for path in results_cache_companions(results_path):
        path.unlink()

    assert not results_path.exists()
    assert not results_path.with_suffix(".fingerprint.txt").exists()
    # The walk-forward deliverable survives and is still readable
    assert wf_path.exists()
    reloaded = pd.read_parquet(wf_path)
    assert len(reloaded) == 5
