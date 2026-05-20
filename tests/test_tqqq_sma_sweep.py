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
    compute_strategy_id,
    dedup_by_strategy_id,
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
    final, sharpe, mdd, calmar, n_trades, t_long, t_short, t_cash = run_backtest(
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

    final, sharpe, mdd, calmar, n_trades, t_long, t_short, t_cash = run_backtest(
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
    final, _, _, _, n_trades, t_long, t_short, t_cash = run_backtest(
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
    final, _, _, _, n_trades, t_long, t_short, t_cash = run_backtest(
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
    """Identical inputs → identical id; small perturbations → different id."""
    a = compute_strategy_id(4.02, 312, 0.5123, 0.2456)
    b = compute_strategy_id(4.02, 312, 0.5123, 0.2456)
    assert a == b
    assert 0 <= a < 2**64
    # 2-dp precision on final_value: 4.024 rounds to 4.02 → same id
    assert compute_strategy_id(4.024, 312, 0.5123, 0.2456) == a
    # Distinct trade count → distinct id
    assert compute_strategy_id(4.02, 313, 0.5123, 0.2456) != a


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
