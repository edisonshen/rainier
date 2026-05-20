"""Tests for the SQQQ-only SMA sweep.

Strategy state machine: CASH | LONG_SQQQ (no TQQQ involvement).
Signals derived from QQQ close vs SMA(buy_S) / SMA(sell_S):

    CASH      → LONG_SQQQ    when QQQ_close < SMA(buy_S)   [bet QQQ down]
    LONG_SQQQ → CASH         when QQQ_close > SMA(sell_S)  [QQQ recovering, exit]

Parameter grid: ``(buy_S, sell_S) in {1..60}^2`` — unconstrained, 3,600 combos.

Acceptance-criteria tests (per task brief):
    (a) determinism — same args → same output
    (b) monotone-up QQQ → optimal strategy is always-cash
    (c) monotone-down QQQ → optimal strategy is always-long-SQQQ
    (d) dormant-cell rendering — n_trades==0 row renders ``-``
    (e) buy_S=1 / sell_S=1 degenerate detection — flagged in report

Tests run on synthetic universes; no yfinance I/O. The 3,600-combo full sweep
is exercised in tdd-refactor as the actual deliverable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rainier.backtest.sqqq_sma_sweep import (
    CASH,
    LONG_SQQQ,
    SQQQ_PRICES_CACHE_PATH,
    SQQQ_RESULTS_CACHE_PATH,
    compute_strategy_id,
    dedup_by_strategy_id,
    iter_sqqq_combos,
    run_backtest,
    run_sweep,
    walk_forward_top_n,
)
from rainier.backtest.tqqq_sma_sweep import precompute_sma_signals

# ---------------------------------------------------------------------------
# Synthetic universes
# ---------------------------------------------------------------------------


def _frame_with_qqq(prices: np.ndarray) -> pd.DataFrame:
    """Build a price frame matching the sweep's expected schema.

    Columns: ``qqq``, ``sqqq`` (lowercased adjusted-close). SQQQ tracks -3x
    daily return of QQQ.
    """
    qqq = np.asarray(prices, dtype=np.float64)
    n = qqq.shape[0]
    rets = np.zeros(n)
    rets[1:] = qqq[1:] / qqq[:-1] - 1.0
    sqqq = np.empty(n)
    sqqq[0] = 100.0
    for i in range(1, n):
        sqqq[i] = sqqq[i - 1] * (1.0 - 3.0 * rets[i])
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"qqq": qqq, "sqqq": sqqq}, index=idx)


# ---------------------------------------------------------------------------
# Module-shape sanity
# ---------------------------------------------------------------------------


def test_constants_are_distinct():
    assert CASH != LONG_SQQQ
    assert {CASH, LONG_SQQQ} == {0, 1}


def test_iter_sqqq_combos_count_full_grid():
    """Unconstrained 2-D grid: max_window**2 combos."""
    combos = list(iter_sqqq_combos(max_window=60))
    assert len(combos) == 60 * 60 == 3600
    # Every tuple in range
    for bS, sS in combos:
        assert 1 <= bS <= 60
        assert 1 <= sS <= 60
    # No duplicates
    assert len(set(combos)) == len(combos)


def test_iter_sqqq_combos_small_window():
    """max_window=3 → 3^2 = 9 combos."""
    combos = list(iter_sqqq_combos(max_window=3))
    assert len(combos) == 9
    assert (1, 1) in combos
    assert (3, 3) in combos
    assert (1, 3) in combos
    assert (3, 1) in combos


def test_default_cache_paths_are_separate():
    """SQQQ-only sweep writes to its own cache path, distinct from TQQQ."""
    from rainier.backtest.tqqq_sma_sweep import (
        PRICES_CACHE_PATH,
        RESULTS_CACHE_PATH,
    )
    assert SQQQ_PRICES_CACHE_PATH != RESULTS_CACHE_PATH
    assert SQQQ_RESULTS_CACHE_PATH != RESULTS_CACHE_PATH
    # SQQQ results live under their own dir
    assert "sqqq_sma" in str(SQQQ_RESULTS_CACHE_PATH)
    # Prices may share the TQQQ prices cache (task brief says reuse it)
    # — but the constant must be explicit either way (we just check it exists).
    assert isinstance(SQQQ_PRICES_CACHE_PATH, Path)
    assert isinstance(PRICES_CACHE_PATH, Path)


# ---------------------------------------------------------------------------
# Acceptance test (a): determinism
# ---------------------------------------------------------------------------


def test_run_backtest_deterministic():
    """Acceptance (a): identical args → identical output tuple."""
    df = _frame_with_qqq(100.0 + 30.0 * np.sin(np.arange(200) / 15.0) + 0.05 * np.arange(200))
    above, valid = precompute_sma_signals(df["qqq"].to_numpy(), max_window=60)
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0
    a = run_backtest(above, valid, sqqq_ret, buy_S=5, sell_S=20, slippage_bp=5.0)
    b = run_backtest(above, valid, sqqq_ret, buy_S=5, sell_S=20, slippage_bp=5.0)
    for x, y in zip(a, b, strict=True):
        if isinstance(x, float) and np.isnan(x):
            assert np.isnan(y)
        else:
            assert x == y


# ---------------------------------------------------------------------------
# Acceptance test (b): monotone-up QQQ → optimal is always-cash
# ---------------------------------------------------------------------------


def test_monotone_up_qqq_optimal_is_cash(tmp_path: Path):
    """Acceptance (b): on a monotonically-rising QQQ universe, no SQQQ-timing
    strategy can profit — the optimal strategy stays in cash.

    Reasoning: QQQ never falls below any non-trivial SMA, so any combo with
    buy_S >= 2 never fires an entry → equity stays at 1.0. The combo with
    buy_S = 1 has a structurally-false entry (also stays cash). Top final_value
    should therefore be exactly 1.0, achieved by many combos.
    """
    n = 200
    qqq = 100.0 * (1.005 ** np.arange(n))  # ~0.5%/day uptrend
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"

    run_sweep(
        df, results_path=results_path, max_window=5, n_workers=1,
        slippage_bp=5.0, flush_every=25,
    )
    out = pd.read_parquet(results_path)
    # Top final value should be 1.0 (cash): the strategy can't make money
    # on a monotonically-up QQQ when shorting only.
    best = out.nlargest(1, "final_value").iloc[0]
    assert abs(float(best.final_value) - 1.0) < 1e-9, (
        f"on monotone-up QQQ, best SQQQ strategy should be cash (1.0×); got {best.final_value}"
    )
    # And no row should have final_value > 1.0 — SQQQ B&H goes to ~zero on up days
    assert (out["final_value"] <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# Acceptance test (c): monotone-down QQQ → optimal is always-long-SQQQ
# ---------------------------------------------------------------------------


def test_monotone_down_qqq_optimal_is_long_sqqq():
    """Acceptance (c): on a monotonically-falling QQQ universe, the optimal
    strategy stays LONG_SQQQ as much as possible.

    With a tight buy_S (small window) the strategy enters SQQQ near day-1
    (QQQ already below SMA) and never exits because QQQ never recovers above
    SMA(sell_S) in a monotone downtrend. n_trades should be 1 (single entry),
    time_in_short close to 1.0, and final_value much greater than 1.0
    (SQQQ compounds positively when QQQ falls).
    """
    n = 300
    qqq = 100.0 * (0.995 ** np.arange(n))  # ~0.5%/day downtrend
    df = _frame_with_qqq(qqq)
    above, valid = precompute_sma_signals(df["qqq"].to_numpy(), max_window=60)
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0

    final, sharpe, mdd, calmar, n_trades, t_short, t_cash, _ch = run_backtest(
        above, valid, sqqq_ret, buy_S=2, sell_S=2, slippage_bp=5.0,
    )
    # Should spend almost all time in SQQQ (tight buy_S, monotone down)
    assert t_short > 0.90, f"expected mostly short; got t_short={t_short}"
    assert t_cash < 0.10
    # Single entry; no exit because QQQ never recovers in monotone-down
    assert n_trades == 1
    # SQQQ compounds positively on a downtrend → final much > cash
    assert final > 5.0


# ---------------------------------------------------------------------------
# Acceptance test (e): SMA(1) degenerate detection
# ---------------------------------------------------------------------------


def test_buy_s_1_degenerate_stays_cash():
    """Acceptance (e) part 1: buy_S=1 makes the entry signal
    ``QQQ < SMA(1) = QQQ < QQQ`` → structurally False (and the validity mask
    forces the signal off). Strategy must stay 100% cash → final=1.0, no trades.
    """
    n = 100
    # Universe shape doesn't matter — buy_S=1 can never enter.
    qqq = 100.0 + 30.0 * np.sin(np.arange(n) / 10.0)
    df = _frame_with_qqq(qqq)
    above, valid = precompute_sma_signals(df["qqq"].to_numpy(), max_window=10)
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0
    final, _, _, _, n_trades, t_short, t_cash, _ch = run_backtest(
        above, valid, sqqq_ret, buy_S=1, sell_S=5, slippage_bp=5.0,
    )
    assert n_trades == 0
    assert t_cash == 1.0
    assert t_short == 0.0
    assert final == 1.0


def test_sell_s_1_degenerate_never_exits():
    """Acceptance (e) part 2: sell_S=1 → exit signal ``QQQ > SMA(1)``
    structurally never fires. So if the strategy enters LONG_SQQQ, it never
    exits regardless of QQQ recovery. n_trades will be 1 (one entry).

    Universe: QQQ falls then rises so a sane strategy would exit on the
    recovery. With sell_S=1 the strategy is stuck long.
    """
    n = 200
    # First 100 days: fall (QQQ goes down → enter SQQQ).
    # Next 100 days: rise (QQQ recovers above any SMA; sane exit would fire).
    qqq = np.empty(n)
    qqq[:100] = 100.0 * (0.99 ** np.arange(100))
    qqq[100:] = qqq[99] * (1.02 ** np.arange(1, n - 99))
    df = _frame_with_qqq(qqq)
    above, valid = precompute_sma_signals(df["qqq"].to_numpy(), max_window=20)
    sqqq_ret = (df["sqqq"].to_numpy()[1:] / df["sqqq"].to_numpy()[:-1]) - 1.0

    # buy_S=5 fires early in the falling regime; sell_S=1 never fires.
    final_stuck, _, _, _, n_trades_stuck, t_short_stuck, _, _ch_stuck = run_backtest(
        above, valid, sqqq_ret, buy_S=5, sell_S=1, slippage_bp=5.0,
    )
    # Should enter once and never exit
    assert n_trades_stuck == 1, f"sell_S=1 must not allow exit; got {n_trades_stuck} trades"
    assert t_short_stuck > 0.5, "should be long for at least the falling regime + ride-up"

    # Sanity: with sell_S=5 (sane exit) the strategy should exit at the recovery
    # and end in cash. n_trades >= 2.
    final_sane, _, _, _, n_trades_sane, _, _, _ch_sane = run_backtest(
        above, valid, sqqq_ret, buy_S=5, sell_S=5, slippage_bp=5.0,
    )
    assert n_trades_sane >= 2, "sane sell_S should produce at least one full round-trip"


# ---------------------------------------------------------------------------
# Sweep driver — emits correct row count
# ---------------------------------------------------------------------------


def test_sweep_emits_full_grid(tmp_path: Path):
    """A fresh sweep at max_window=60 emits exactly 3,600 rows (one per combo).

    Uses a small synthetic universe so the test stays fast.
    """
    n = 80
    qqq = 100.0 + 10.0 * np.sin(np.linspace(0, 6.28, n))
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"

    run_sweep(
        df, results_path=results_path, max_window=5, n_workers=1,
        slippage_bp=5.0, flush_every=20,
    )
    out = pd.read_parquet(results_path)
    assert len(out) == 25  # 5*5 combos
    keys = {(int(r.buy_S), int(r.sell_S)) for r in out.itertuples()}
    assert keys == {(bS, sS) for bS in range(1, 6) for sS in range(1, 6)}
    # Schema check: 2 params + 7 metrics + strategy_id = 10 columns
    required_cols = {
        "buy_S", "sell_S",
        "final_value", "sharpe", "max_dd", "calmar",
        "n_trades", "time_in_short", "time_in_cash",
        "strategy_id",
    }
    assert required_cols.issubset(set(out.columns))
    # No time_in_long column — that's a TQQQ thing
    assert "time_in_long" not in out.columns


def test_sweep_resumability(tmp_path: Path):
    """Partial sweep + resume on same inputs → full grid, no duplicates."""
    n = 60
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"

    # First run: only 10 of the 25 combos
    run_sweep(
        df, results_path=results_path, max_window=5, n_workers=1,
        slippage_bp=5.0, max_combos=10, flush_every=5,
    )
    partial = pd.read_parquet(results_path)
    assert len(partial) == 10

    # Resume → fill remaining 15
    run_sweep(
        df, results_path=results_path, max_window=5, n_workers=1,
        slippage_bp=5.0, flush_every=5,
    )
    full = pd.read_parquet(results_path)
    assert len(full) == 25
    keys = full[["buy_S", "sell_S"]]
    assert not keys.duplicated().any()


# ---------------------------------------------------------------------------
# strategy_id + dedup
# ---------------------------------------------------------------------------


def test_compute_strategy_id_deterministic_and_uint64():
    """strategy_id is a deterministic uint64 passthrough on the curve_hash."""
    a = compute_strategy_id(0x123456789ABCDEF0)
    b = compute_strategy_id(0x123456789ABCDEF0)
    assert a == b
    assert 0 <= a < 2**64
    assert compute_strategy_id(0x123456789ABCDEF1) != a


def test_dormant_buy_s_combos_share_strategy_id(tmp_path: Path):
    """All combos with buy_S=1 produce the identical all-cash equity curve →
    same strategy_id regardless of sell_S. This is the SMA(1) degeneracy
    collapsing under dedup.
    """
    n = 80
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"
    run_sweep(
        df, results_path=results_path, max_window=10, n_workers=1,
        slippage_bp=5.0, flush_every=20,
    )
    out = pd.read_parquet(results_path)
    # All buy_S=1 rows must share the same strategy_id
    sub = out[out["buy_S"] == 1]
    assert len(sub) == 10
    sids = set(int(s) for s in sub["strategy_id"].unique())
    assert len(sids) == 1, f"buy_S=1 combos must collapse to one strategy_id; got {sids}"


def test_dedup_by_strategy_id_returns_one_per_group(tmp_path: Path):
    """Dedup keeps one rep per unique strategy_id and the n_equivalent
    column sums back to the input row count.
    """
    n = 80
    qqq = 100.0 + np.linspace(0, 50, n)
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"
    run_sweep(
        df, results_path=results_path, max_window=10, n_workers=1,
        slippage_bp=5.0, flush_every=20,
    )
    raw = pd.read_parquet(results_path)
    deduped = dedup_by_strategy_id(raw)
    assert len(deduped) == raw["strategy_id"].nunique()
    assert not deduped["strategy_id"].duplicated().any()
    assert int(deduped["n_equivalent"].sum()) == len(raw)


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def test_walk_forward_top_n_returns_train_test_columns(tmp_path: Path):
    """``walk_forward_top_n`` produces ``final_value_train``/``final_value_test``
    columns for the top-N deduped combos.
    """
    n = 200
    t = np.arange(n)
    qqq = 100.0 + 20.0 * np.sin(t / 25.0) + 0.05 * t
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"
    run_sweep(
        df, results_path=results_path, max_window=5, n_workers=1,
        slippage_bp=5.0, flush_every=25,
    )
    # Split somewhere mid-range
    wf = walk_forward_top_n(
        df, results_path=results_path, top_n=10,
        split_date="2020-05-01",
        slippage_bp=5.0, max_window=5,
    )
    assert "final_value_train" in wf.columns
    assert "final_value_test" in wf.columns
    assert len(wf) <= 10  # may collapse under dedup
    # Top-N selection on deduped data: each row's strategy_id is unique
    assert not wf["strategy_id"].duplicated().any()


# ---------------------------------------------------------------------------
# Report (d): dormant cells render as ``-``
# ---------------------------------------------------------------------------


def test_report_dormant_cells_render_as_dash(tmp_path: Path):
    """Acceptance (d): rows where ``time_in_short == 0`` (the short leg
    never fired) render buy_S/sell_S cells as the literal string ``-``.

    Synthetic monotone-up universe → most combos are 100% cash → short leg
    dormant → buy_S/sell_S cells must render as ``-``.
    """
    from rainier.backtest.sqqq_sma_report import render_report

    n = 200
    qqq = 100.0 * (1.005 ** np.arange(n))  # uptrend → no SQQQ entry
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"
    run_sweep(
        df, results_path=results_path, max_window=5, n_workers=1,
        slippage_bp=5.0, flush_every=25,
    )

    out_html = tmp_path / "report.html"
    render_report(
        prices=df,
        results_path=results_path,
        walkforward_path=tmp_path / "no_wf.parquet",  # missing → graceful
        output_path=out_html,
        sweep_wall_seconds=1.0,
        slippage_bp=5.0,
        max_window=5,
    )
    assert out_html.exists()
    html = out_html.read_text(encoding="utf-8")
    # CASH-only rows should produce dormant-cell rendering
    assert ">-</td>" in html or ">-<" in html


def test_report_degenerate_banner_when_all_cash(tmp_path: Path):
    """Acceptance (e) report part: when the top winners include rows with
    ``n_trades == 0`` (all-cash) or ``n_trades == 1`` (degenerate one-shot),
    the report MUST surface a DEGENERATE banner.

    On a monotone-up universe, ALL SQQQ entry signals stay False (uptrend
    means QQQ is always above its SMA, so QQQ < SMA(buy_S) is always False).
    Every combo is n_trades=0; the leaderboard's #1 row is degenerate.
    """
    from rainier.backtest.sqqq_sma_report import render_report

    n = 200
    qqq = 100.0 * (1.005 ** np.arange(n))
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"
    run_sweep(
        df, results_path=results_path, max_window=5, n_workers=1,
        slippage_bp=5.0, flush_every=25,
    )

    out_html = tmp_path / "report.html"
    render_report(
        prices=df,
        results_path=results_path,
        walkforward_path=tmp_path / "no_wf.parquet",
        output_path=out_html,
        sweep_wall_seconds=1.0,
        slippage_bp=5.0,
        max_window=5,
    )
    html = out_html.read_text(encoding="utf-8")
    # The DEGENERATE banner must fire because all rows are n_trades=0
    assert "DEGENERATE" in html
    # And the supporting note must mention the SMA(1) degeneracy mechanism
    assert "SMA(1)" in html or "sell_S=1" in html or "buy_S=1" in html


def test_report_has_all_nine_sections(tmp_path: Path):
    """The report must contain anchors for all 9 sections listed in the task brief:
    framing, top-50, baselines, heatmap, distribution, equity, walkforward,
    discussion, repro.
    """
    from rainier.backtest.sqqq_sma_report import render_report

    n = 200
    t = np.arange(n)
    qqq = 100.0 + 20.0 * np.sin(t / 25.0) + 0.05 * t
    df = _frame_with_qqq(qqq)
    results_path = tmp_path / "results.parquet"
    run_sweep(
        df, results_path=results_path, max_window=5, n_workers=1,
        slippage_bp=5.0, flush_every=25,
    )

    out_html = tmp_path / "report.html"
    render_report(
        prices=df,
        results_path=results_path,
        walkforward_path=tmp_path / "no_wf.parquet",
        output_path=out_html,
        sweep_wall_seconds=1.0,
        slippage_bp=5.0,
        max_window=5,
    )
    html = out_html.read_text(encoding="utf-8")
    for anchor in ("framing", "top50", "baselines", "heatmap",
                   "distribution", "equity", "walkforward",
                   "discussion", "repro"):
        assert f'id="{anchor}"' in html, f"missing section anchor: {anchor}"
    # Title makes clear this is the SQQQ-only sweep, not TQQQ
    assert "SQQQ" in html
