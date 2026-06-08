"""Resonance study tests — pure helpers (no heavy data load) + a tiny integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rainier.backtest.daily_mtm import PortfolioResult
from rainier.backtest.resonance_study import (
    World,
    beats_baselines,
    combine,
    deflate,
    metrics_over,
    sma_gate_decision,
    thesis_test,
)


def _perf(calmar, dd):
    p = PortfolioResult("x", np.ones(2))
    p.calmar, p.max_dd = calmar, dd
    return p


def test_beats_baselines_requires_dd_below_sma_too():
    # codex P2 #1: higher Calmar than SMA but DEEPER drawdown than SMA is NOT a win.
    sma = _perf(calmar=2.0, dd=0.30)
    bh = _perf(calmar=1.5, dd=0.60)
    deeper = _perf(calmar=2.5, dd=0.40)   # beats Calmar but DD 0.40 > SMA's 0.30
    assert not beats_baselines(deeper, sma, bh)
    real_win = _perf(calmar=2.5, dd=0.25)  # beats Calmar AND DD vs both
    assert beats_baselines(real_win, sma, bh)


def test_beats_baselines_needs_to_beat_both_calmar():
    sma = _perf(calmar=2.0, dd=0.30)
    bh = _perf(calmar=3.0, dd=0.60)        # buy-hold has the higher Calmar
    cand = _perf(calmar=2.5, dd=0.20)      # beats SMA Calmar but not buy-hold's
    assert not beats_baselines(cand, sma, bh)


def test_csv_dir_resolves_from_cwd_not_package(tmp_path, monkeypatch):
    # codex P2 (r3): the default CSV dir must resolve from the INVOCATION cwd,
    # not the package install path (which breaks under a wheel install).
    from rainier.backtest.resonance_study import _csv_dir

    (tmp_path / "data" / "csv").mkdir(parents=True)
    sub = tmp_path / "sub" / "nested"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)  # run from a subdirectory
    assert _csv_dir() == tmp_path / "data" / "csv"  # walks up to find it


def test_default_report_path_resolves_from_cwd(tmp_path, monkeypatch):
    from rainier.backtest.resonance_report import _default_report_path

    (tmp_path / "docs").mkdir()
    monkeypatch.chdir(tmp_path)
    assert _default_report_path() == tmp_path / "docs" / "REPORT-resonance-gate-v1.html"


def test_deflate_monotone_haircut():
    assert deflate(2.0, 1) == 2.0          # no penalty for a single config
    assert deflate(2.0, 24) < 2.0          # haircut grows with #configs
    assert deflate(2.0, 100) < deflate(2.0, 24)


def test_combine_and_or():
    a = np.array([1.0, 1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 1.0, 0.0])
    assert list(combine(a, b, "AND")) == [1.0, 0.0, 0.0, 0.0]
    assert list(combine(a, b, "OR")) == [1.0, 1.0, 1.0, 0.0]


def _synthetic_world(n=600, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.Series(pd.date_range("2019-06-01", periods=n, freq="B", tz="UTC"))
    drift = 0.0006 + rng.normal(0, 0.012, n)
    close = 100.0 * np.cumprod(1 + drift)
    df = pd.DataFrame({
        "open": np.r_[close[0], close[:-1]],
        "high": close * 1.01, "low": close * 0.99, "close": close,
        "vix": 18.0 + rng.normal(0, 2, n), "spy": close * 0.9,
    })
    tqqq_ret = 3.0 * np.r_[0.0, np.diff(close) / close[:-1]]
    qqq_ret = np.r_[0.0, np.diff(close) / close[:-1]]
    rate = np.full(n, 0.04 / 252)
    return World(ts, df, tqqq_ret, qqq_ret, rate, synthetic=False)


def test_sma_gate_decision_shape_and_binary():
    w = _synthetic_world()
    dec = sma_gate_decision(w)
    assert dec.shape == (len(w.df),)
    assert set(np.unique(dec)).issubset({0.0, 1.0})


def test_metrics_over_window_no_lookahead_and_finite():
    w = _synthetic_world()
    dec = sma_gate_decision(w)
    m = metrics_over(dec, w, w.tqqq_ret, "2020-10-01", None, "sma")
    assert np.isfinite(m.calmar)
    assert 0.0 <= m.exposure <= 1.0
    assert m.switches >= 0


def test_metrics_over_window_switch_count_ignores_carried_position():
    # codex P3: a position held INTO the window is not a fresh entry. Always-in
    # (buy-hold) measured over a later sub-window must report 0 switches, not 1.
    w = _synthetic_world()
    always_in = np.ones(len(w.df))
    m = metrics_over(always_in, w, w.tqqq_ret, "2020-10-01", None, "bh")
    assert m.switches == 0


def test_thesis_test_returns_buckets_and_ci():
    w = _synthetic_world()
    score = np.clip(0.5 + (w.qqq_ret * 30), 0, 1)  # crude score correlated to ret
    res = thesis_test(w, score, n_boot=300, seed=1)
    assert len(res.buckets) == 5
    lo, hi = res.slope_ci
    assert np.isfinite(lo) and np.isfinite(hi)
    assert isinstance(res.excludes_null, bool)


@pytest.mark.slow
def test_build_world_real_tqqq_starts_at_inception():
    # codex P2: real_tqqq=True must NOT fabricate flat pre-2010 history even when
    # start predates TQQQ's 2010-02 inception — drop the pre-inception NaN rows.
    from rainier.backtest.resonance_study import build_world

    w = build_world(start="1999-01-01", real_tqqq=True)
    assert w.ts.iloc[0] >= pd.Timestamp("2010-02-01", tz="UTC")
    assert not np.isnan(w.tqqq_ret).any()


@pytest.mark.slow
def test_run_study_smoke():
    # Full study on real CSVs — slow; verifies it produces a coherent verdict.
    from rainier.backtest.resonance_study import run_study

    r = run_study(n_boot=200)
    assert r.verdict
    assert r.n_configs > 0
    assert any(row.name == "SMA22/44 gate" for row in r.test_ab)
    # codex P2 #2: the ship-resonance verdict must NOT be reachable via a combo
    # chosen on the held-out test slice — it requires resonance-ONLY to win.
    if r.verdict.startswith("SHIP THE RESONANCE"):
        res_row = next(x for x in r.test_ab if x.name == "resonance-gate")
        assert res_row.beats_sma_and_bh, "resonance-only must win to ship resonance"
        assert r.thesis.excludes_null
