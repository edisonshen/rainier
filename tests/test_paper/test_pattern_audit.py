"""WS B — pattern forward-return audit.

Acceptance (task plan):
- forward-return: exact value for a known price path; near-window-end → null,
  not 0.
- aggregation groups by (pattern, regime, horizon) with per-pattern n surfaced
  and the `unknown`-regime cell reported (not dropped); directional-correctness
  column correct.
"""

from __future__ import annotations

import pandas as pd

from rainier.core.config import StockScreenerConfig
from rainier.paper.pattern_audit import (
    HORIZONS,
    aggregate,
    aggregate_to_frame,
    build_corpus,
    corpus_path,
    directional_correct,
    forward_return,
    render_report_markdown,
    write_corpus,
)


def _make_ohlcv(prices: list[float]) -> pd.DataFrame:
    n = len(prices)
    df = pd.DataFrame(
        {
            "open": [prices[max(0, i - 1)] for i in range(n)],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }
    )
    df.index = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    return df


# ---------------------------------------------------------------------------
# forward_return — exact value; NULL (not 0) near the window end.
# ---------------------------------------------------------------------------


def test_forward_return_exact_value():
    df = _make_ohlcv([100.0, 101.0, 102.0, 103.0, 110.0, 120.0])
    # H=5 from t=0: close[5]/close[0]-1 = 120/100 - 1 = 0.20
    assert abs(forward_return(df, 0, 5) - 0.20) < 1e-12
    # H=1 from t=0: 101/100 - 1 = 0.01
    assert abs(forward_return(df, 0, 1) - 0.01) < 1e-12


def test_forward_return_null_near_window_end_not_zero():
    df = _make_ohlcv([100.0] * 10)
    # t=8, H=5 → t+H=13 >= len(10) → None (NEVER 0.0).
    out = forward_return(df, 8, 5)
    assert out is None
    # A genuine no-move at a valid horizon IS 0.0 (distinguishable from null).
    assert forward_return(df, 0, 5) == 0.0


def test_forward_return_target_exactly_at_last_bar():
    df = _make_ohlcv([100.0, 1, 2, 3, 4, 130.0])
    # t=0, H=5 → t+H=5 == last index (valid). 130/100 - 1 = 0.30
    assert abs(forward_return(df, 0, 5) - 0.30) < 1e-12
    # t=1, H=5 → t+H=6 >= len → None
    assert forward_return(df, 1, 5) is None


# ---------------------------------------------------------------------------
# directional_correct — bullish/bearish vs realized move.
# ---------------------------------------------------------------------------


def test_directional_correctness():
    assert directional_correct("bullish", 0.05) is True
    assert directional_correct("bullish", -0.05) is False
    assert directional_correct("bearish", -0.05) is True
    assert directional_correct("bearish", 0.05) is False
    # flat / null → no read
    assert directional_correct("bullish", 0.0) is None
    assert directional_correct("bearish", None) is None


# ---------------------------------------------------------------------------
# build_corpus — emissions tagged with fwd returns + regime; deterministic.
# ---------------------------------------------------------------------------


def _false_breakdown_block() -> list[float]:
    p = [95.0, 93.0, 91.0, 90.0, 91.0, 93.0, 95.0, 93.0, 91.0, 90.0]
    p += [92.0, 93.0, 91.0]
    p += [89.0, 88.0, 87.0]
    p += [89.0, 91.0, 92.0]
    p += [92.5, 92.0, 92.5, 92.0, 92.5]
    return p


_CONFIG = StockScreenerConfig(
    swing_lookback=3, min_pattern_bars=3, max_pattern_bars=50,
    neckline_tolerance_pct=0.05,
)


def test_build_corpus_attaches_regime_and_fwd_returns():
    # 60+ bars of benign history clears the min-history floor; then a
    # false-breakdown block that becomes actionable; then forward bars so the
    # emission has 5/10/20d forward returns to attach.
    prices = [100.0] * 60
    prices += _false_breakdown_block()  # actionable near its end
    prices += [92.5, 93.0, 92.5, 93.0, 92.5, 93.0, 92.5, 93.0,
               92.5, 93.0, 92.5, 93.0, 92.5, 93.0, 92.5, 93.0,
               92.5, 93.0, 92.5, 93.0, 92.5]  # 21 forward bars
    df = _make_ohlcv(prices)

    corpus = build_corpus(
        {"AAA": df},
        config=_CONFIG,
        regime_fn=lambda as_of: "bull",
        min_history_bars=60,
    )
    assert not corpus.empty
    assert set(corpus["regime"]) == {"bull"}
    for h in HORIZONS:
        assert f"fwd_return_{h}d" in corpus.columns
    # At least one emission has a non-null 5d forward return (forward bars exist).
    assert corpus["fwd_return_5d"].notna().any()
    # Deterministic: a second build yields identical bytes.
    corpus2 = build_corpus(
        {"AAA": df}, config=_CONFIG,
        regime_fn=lambda as_of: "bull", min_history_bars=60,
    )
    pd.testing.assert_frame_equal(corpus, corpus2)


def test_build_corpus_memoizes_regime_per_date():
    """regime_fn is invoked at most once per distinct as_of date — a 1-year ×
    N-symbol audit must not re-query SPY per emission (the default regime_fn
    opens a DB session each call)."""
    prices = [100.0] * 60
    prices += _false_breakdown_block()
    prices += [92.5, 93.0] * 11  # forward bars
    # Two symbols sharing the SAME date index → same dates emitted twice.
    df = _make_ohlcv(prices)
    calls: list = []

    def counting_regime(as_of):
        calls.append(as_of)
        return "bull"

    corpus = build_corpus(
        {"AAA": df, "BBB": df},
        config=_CONFIG,
        regime_fn=counting_regime,
        min_history_bars=60,
    )
    assert not corpus.empty
    # Both symbols emit on overlapping dates, but regime_fn fires once per
    # distinct date — so call count == number of distinct dates, strictly less
    # than the number of corpus rows when any date repeats across symbols.
    assert len(calls) == len(set(calls))  # no duplicate-date calls
    assert len(calls) <= len(corpus)


def test_build_corpus_skips_detector_exception(monkeypatch):
    """A detector exception on one window is logged + skipped (mirrors the live
    screener), not propagated to abort the whole audit. Guards codex iter-1 P1."""
    import rainier.paper.pattern_audit as pa

    prices = [100.0] * 60 + _false_breakdown_block() + [92.5, 93.0] * 11
    df = _make_ohlcv(prices)

    real_emission_at = pa.emission_at
    calls = {"n": 0}

    def flaky_emission_at(symbol, window, config):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("synthetic detector blowup")
        return real_emission_at(symbol, window, config)

    monkeypatch.setattr(pa, "emission_at", flaky_emission_at)
    # Must NOT raise — the bad window is dropped, the rest still build.
    corpus = build_corpus(
        {"AAA": df}, config=_CONFIG, regime_fn=lambda a: "bull",
        min_history_bars=60,
    )
    assert calls["n"] > 1  # kept going past the failing window
    assert isinstance(corpus, pd.DataFrame)


def test_run_pattern_audit_defaults_min_history_to_config(monkeypatch, tmp_path):
    """When min_history_bars is not passed, run_pattern_audit derives it from
    config.min_daily_bars (the live data gate). Guards codex iter-1 P2."""
    import rainier.paper.pattern_audit as pa

    captured = {}

    def fake_build_corpus(prices, *, config, min_history_bars, **kw):
        captured["min_history_bars"] = min_history_bars
        return pd.DataFrame(columns=list(pa.CORPUS_COLUMNS))

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(pa, "build_corpus", fake_build_corpus)
    monkeypatch.setattr("rainier.core.database.get_session", lambda: _FakeSession())
    monkeypatch.setattr(
        "rainier.paper.pattern_replay.load_prices",
        lambda s, syms, start_date=None: {},
    )

    cfg = StockScreenerConfig(min_daily_bars=80)
    # window_days=None skips the max-date query (no real session here).
    pa.run_pattern_audit(
        config=cfg, symbols=["AAA"], corpus_dir=tmp_path, window_days=None
    )
    assert captured["min_history_bars"] == 80


def test_run_pattern_audit_window_days_is_exactly_n_dates(monkeypatch, tmp_path):
    """An N-day window spans exactly N calendar dates ending at the latest bar:
    window_start == latest-(N-1), not latest-N. Guards codex iter-5 P3."""
    import datetime as _dt

    import rainier.paper.pattern_audit as pa

    captured = {}

    def fake_build_corpus(prices, *, config, min_history_bars, window_start, **kw):
        captured["window_start"] = window_start
        return pd.DataFrame(columns=list(pa.CORPUS_COLUMNS))

    latest = _dt.date(2026, 6, 16)

    class _Res:
        def scalar(self):
            return _dt.datetime(latest.year, latest.month, latest.day)

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, stmt):
            return _Res()

    monkeypatch.setattr(pa, "build_corpus", fake_build_corpus)
    monkeypatch.setattr("rainier.core.database.get_session", lambda: _FakeSession())
    monkeypatch.setattr(
        "rainier.paper.pattern_replay.load_prices",
        lambda s, syms, start_date=None: {},
    )

    pa.run_pattern_audit(
        config=StockScreenerConfig(), symbols=["AAA"],
        corpus_dir=tmp_path, window_days=365,
    )
    # exactly 365 dates: [latest-364 .. latest]
    assert captured["window_start"] == latest - _dt.timedelta(days=364)


def test_build_corpus_window_start_bounds_as_of(monkeypatch):
    """window_start bounds the AS-OF date: emissions before the cutoff are
    dropped, earlier bars still loaded for lookback/fwd edges. Guards codex
    iter-2 P1 (audit must not silently span all history)."""
    import datetime as _dt

    # Emit on EVERY window (stub emission) so we can count by date cleanly.
    import rainier.paper.pattern_audit as pa
    from rainier.paper.pattern_replay import PatternEmission

    prices = [100.0 + i * 0.01 for i in range(120)]
    df = _make_ohlcv(prices)  # date index starts 2025-01-01

    def stub_emission(symbol, window, config):
        ts = window.index[-1]
        return PatternEmission(
            symbol=symbol, as_of=ts, pattern_type="bull_flag",
            direction="bullish", status="confirmed", confidence=0.7,
            pattern_contribution=0.45, entry_price=100.0, stop_loss=95.0,
            target_wave1=110.0, close_at_t=float(window["close"].iloc[-1]),
        )

    monkeypatch.setattr(pa, "emission_at", stub_emission)
    cutoff = _dt.date(2025, 4, 1)
    corpus = build_corpus(
        {"AAA": df}, config=_CONFIG, regime_fn=lambda a: "bull",
        min_history_bars=30, window_start=cutoff,
    )
    assert not corpus.empty
    assert corpus["as_of"].min() >= cutoff  # nothing before the cutoff
    # Without a cutoff the corpus would start at the min-history floor date.
    full = build_corpus(
        {"AAA": df}, config=_CONFIG, regime_fn=lambda a: "bull",
        min_history_bars=30,
    )
    assert full["as_of"].min() < cutoff  # full history reaches earlier
    assert len(corpus) < len(full)


def test_build_corpus_window_gate_rejects_when_min_exceeds_lookback(monkeypatch):
    """A min_history_bars above the lookback emits NOTHING — mirrors live
    `_fetch_stock_data` rejecting a 6-month frame shorter than min_bars. Guards
    codex iter-2 P2."""
    import rainier.paper.pattern_audit as pa
    from rainier.paper.pattern_replay import PatternEmission

    df = _make_ohlcv([100.0] * 200)  # 200 bars lifetime

    def stub_emission(symbol, window, config):
        ts = window.index[-1]
        return PatternEmission(
            symbol=symbol, as_of=ts, pattern_type="bull_flag",
            direction="bullish", status="confirmed", confidence=0.7,
            pattern_contribution=0.45, entry_price=100.0, stop_loss=95.0,
            target_wave1=110.0, close_at_t=100.0,
        )

    monkeypatch.setattr(pa, "emission_at", stub_emission)
    # lookback trims the window to 50 bars; a 60-bar gate rejects every window.
    corpus = build_corpus(
        {"AAA": df}, config=_CONFIG, regime_fn=lambda a: "bull",
        min_history_bars=60, lookback_bars=50,
    )
    assert corpus.empty


def test_build_corpus_skips_thin_history():
    df = _make_ohlcv([100.0] * 30)  # below the 60-bar floor
    corpus = build_corpus(
        {"AAA": df}, config=_CONFIG, regime_fn=lambda a: "bull",
        min_history_bars=60,
    )
    assert corpus.empty


# ---------------------------------------------------------------------------
# aggregate — groups by (pattern, regime, horizon); unknown cell reported;
# directional-correctness correct; n surfaced; thin flag.
# ---------------------------------------------------------------------------


def _corpus_row(pattern_type, direction, regime, **fwd):
    row = {
        "symbol": "X", "as_of": pd.Timestamp("2025-01-01").date(),
        "pattern_type": pattern_type, "direction": direction,
        "status": "confirmed", "confidence": 0.7,
        "pattern_contribution": 0.45, "regime": regime, "close_at_t": 100.0,
        "fwd_return_5d": None, "fwd_return_10d": None, "fwd_return_20d": None,
    }
    row.update(fwd)
    return row


def test_aggregate_groups_and_reports_unknown_cell():
    corpus = pd.DataFrame([
        _corpus_row("false_breakdown", "bullish", "bull", fwd_return_5d=0.04),
        _corpus_row("false_breakdown", "bullish", "bull", fwd_return_5d=-0.02),
        _corpus_row("false_breakdown", "bullish", "bull", fwd_return_5d=0.06),
        # an unknown-regime emission must surface as its OWN cell, not dropped
        _corpus_row("false_breakdown", "bullish", "unknown", fwd_return_5d=0.01),
        # a bearish pattern that correctly precedes a decline
        _corpus_row("m_top", "bearish", "bear", fwd_return_5d=-0.03),
    ])
    cells = aggregate(corpus)
    by_key = {(c.pattern_type, c.regime, c.horizon): c for c in cells}

    fb_bull_5 = by_key[("false_breakdown", "bull", 5)]
    assert fb_bull_5.n == 3
    assert abs(fb_bull_5.win_rate - 2 / 3) < 1e-9  # 2 of 3 positive
    assert abs(fb_bull_5.mean_fwd_return - (0.04 - 0.02 + 0.06) / 3) < 1e-9
    # bullish: directional-correct == win-rate here (up move == correct)
    assert abs(fb_bull_5.directional_correctness - 2 / 3) < 1e-9
    assert fb_bull_5.thin is True  # n < 30

    # unknown regime is reported, not dropped
    assert ("false_breakdown", "unknown", 5) in by_key
    assert by_key[("false_breakdown", "unknown", 5)].n == 1

    # bearish pattern preceding a decline → directionally correct
    m_top_5 = by_key[("m_top", "bear", 5)]
    assert m_top_5.directional_correctness == 1.0
    assert m_top_5.win_rate == 0.0  # the move was DOWN (loss for a long read)


def test_aggregate_horizon_with_all_null_returns_zero_n_cell():
    corpus = pd.DataFrame([
        _corpus_row("bull_flag", "bullish", "bull", fwd_return_5d=0.02),
        # 10d/20d are null for this row → those cells exist with n=0
    ])
    cells = aggregate(corpus)
    by_key = {(c.pattern_type, c.regime, c.horizon): c for c in cells}
    assert by_key[("bull_flag", "bull", 5)].n == 1
    assert by_key[("bull_flag", "bull", 10)].n == 0
    assert by_key[("bull_flag", "bull", 10)].win_rate is None


# ---------------------------------------------------------------------------
# corpus persistence + report rendering.
# ---------------------------------------------------------------------------


def test_write_corpus_roundtrips(tmp_path):
    corpus = pd.DataFrame([
        _corpus_row("false_breakdown", "bullish", "bull", fwd_return_5d=0.04),
    ])
    out = write_corpus(corpus, corpus_dir=tmp_path)
    assert out == corpus_path(tmp_path)
    assert out.exists()
    back = pd.read_parquet(out)
    assert len(back) == 1
    assert back.iloc[0]["pattern_type"] == "false_breakdown"


def test_render_report_includes_unknown_and_thin_flag():
    corpus = pd.DataFrame([
        _corpus_row("false_breakdown", "bullish", "bull", fwd_return_5d=0.04),
        _corpus_row("false_breakdown", "bullish", "unknown", fwd_return_5d=0.01),
    ])
    agg = aggregate_to_frame(corpus)
    md = render_report_markdown(corpus, agg, window_label="2025-06..2026-06")
    assert "QU100 pattern forward-return audit" in md
    assert "false_breakdown" in md
    assert "unknown" in md  # the unknown-regime cell is rendered
    assert "⚠" in md  # thin cells (n<30) flagged
    assert "survivorship" in md.lower()


def test_render_report_empty_corpus():
    from rainier.paper.pattern_audit import CORPUS_COLUMNS

    empty = pd.DataFrame(columns=list(CORPUS_COLUMNS))
    agg = aggregate_to_frame(empty)
    md = render_report_markdown(empty, agg, window_label="n/a")
    assert "no emissions" in md.lower()
