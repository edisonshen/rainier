"""Unit tests for scripts/mes_5m_sweetspot_study.py.

Deterministic synthetic OHLCV fixtures (no network, no CSV, no sleep). Each signal
function is verified against a hand-constructed series where the answer is known by
construction, plus the no-lookahead timing invariant of the simulator.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "mes_5m_sweetspot_study.py"
_MOD_NAME = "mes_5m_sweetspot_study"


def _load_module():
    spec = importlib.util.spec_from_file_location(_MOD_NAME, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: `from __future__ import annotations` makes dataclass
    # field annotations strings that dataclasses resolves via sys.modules[__module__].
    sys.modules[_MOD_NAME] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _df(rows: list[dict]) -> pd.DataFrame:
    base = pd.Timestamp("2026-02-02 14:00:00+00:00")  # a Monday, mid-RTH
    for i, r in enumerate(rows):
        r.setdefault("timestamp", base + pd.Timedelta(minutes=5 * i))
        r.setdefault("volume", 1000.0)
    return pd.DataFrame(rows)


def _bar(o, h, l, c, v=1000.0):
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}


# ---------------------------------------------------------------------------
# load_clean_5m
# ---------------------------------------------------------------------------

def test_load_clean_drops_zero_range_and_zero_volume(mod, tmp_path):
    p = tmp_path / "MES_5m.csv"
    df = pd.DataFrame(
        [
            # good bar
            {"timestamp": "2026-01-01 14:00:00+00:00", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 500},
            # zero-range junk (open==high==low==close) — must drop
            {"timestamp": "2026-01-01 14:05:00+00:00", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 500},
            # zero-volume edge bar — must drop
            {"timestamp": "2026-01-01 14:10:00+00:00", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 0},
            # duplicate timestamp of the good bar — keep last only
            {"timestamp": "2026-01-01 14:00:00+00:00", "open": 100, "high": 102, "low": 98, "close": 101, "volume": 700},
            {"timestamp": "2026-01-01 14:15:00+00:00", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 800},
        ]
    )
    df.to_csv(p, index=False)
    out = mod.load_clean_5m(p)
    assert (out["high"] - out["low"] > 0).all()
    assert (out["volume"] > 0).all()
    assert out["timestamp"].is_unique
    assert out["timestamp"].is_monotonic_increasing
    # 2 surviving timestamps (14:00 dedup-kept, 14:15); the 14:00 kept row is the LAST (close 101)
    assert len(out) == 2
    assert out.iloc[0]["close"] == 101


# ---------------------------------------------------------------------------
# vwma — rolling, volume-weighted, not session-reset
# ---------------------------------------------------------------------------

def test_vwma_equals_hand_computation(mod):
    rows = [_bar(10, 12, 8, 10, v=100) for _ in range(3)]
    rows += [_bar(20, 22, 18, 20, v=300)]
    df = _df(rows)
    v = mod.vwma(df, n=4).to_numpy()
    # first 3 are NaN (window=4)
    assert np.isnan(v[:3]).all()
    # bar3: hlc3 each = price (h+l+c)/3; bars 0-2 hlc3=10 vol100, bar3 hlc3=20 vol300
    expected = (10 * 100 * 3 + 20 * 300) / (100 * 3 + 300)
    assert v[3] == pytest.approx(expected)


def test_vwma_volume_weights_dominant_bar(mod):
    # one huge-volume bar pulls the VWMA toward its price
    rows = [_bar(10, 11, 9, 10, v=1)] * 3 + [_bar(50, 51, 49, 50, v=1_000_000)]
    df = _df(rows)
    v = mod.vwma(df, n=4).to_numpy()
    assert v[3] > 45  # dominated by the heavy bar near 50


# ---------------------------------------------------------------------------
# td_setup_buy / td_buy_context — canonical DeMark setup count
# ---------------------------------------------------------------------------

def test_td_setup_buy_completes_at_nine(mod):
    # strictly descending closes => every bar t>=4 has close<close[4] => count climbs
    closes = list(np.arange(100, 70, -1.0))  # 30 descending bars
    rows = [_bar(c, c + 1, c - 1, c) for c in closes]
    df = _df(rows)
    out = mod.td_setup_buy(df, completed=9)
    # bar index where run first reaches 9: run increments from t=4; run==9 at t=4+8=12
    first_true = int(np.argmax(out))
    assert out[first_true]
    assert first_true == 12


def test_td_setup_buy_resets_on_up_close(mod):
    # 8 descending then a jump up (resets), then descending again
    closes = list(np.arange(100, 92, -1.0)) + [200.0] + list(np.arange(100, 70, -1.0))
    rows = [_bar(c, c + 1, c - 1, c) for c in closes]
    df = _df(rows)
    out = mod.td_setup_buy(df, completed=9)
    # the up-close at index 8 breaks the run; a 9 can only complete after the reset
    assert not out[:9].any()


def test_td_buy_context_window(mod):
    closes = list(np.arange(100, 88, -1.0)) + list(np.arange(89, 110, 1.0))
    rows = [_bar(c, c + 1, c - 1, c) for c in closes]
    df = _df(rows)
    ctx = mod.td_buy_context(df, lookback=6, min_count=7)
    # context True near the exhaustion bottom, False once we've drifted far above it
    assert ctx[11] or ctx[12]  # near the bottom
    assert not ctx[-1]         # far past the bottom


# ---------------------------------------------------------------------------
# ribbon filters
# ---------------------------------------------------------------------------

def test_ribbon_bullish_requires_above_and_rising(mod):
    # steadily rising market: close above sma44 and sma44 rising late in the series
    closes = list(np.arange(100, 200, 1.0))  # 100 rising bars
    rows = [_bar(c, c + 0.5, c - 0.5, c) for c in closes]
    df = _df(rows)
    out = mod.ribbon_bullish(df)
    assert out[-1]            # rising uptrend => bullish at the end
    assert not out[:44].any()  # sma44 undefined early => not bullish


def test_ribbon_bullish_false_in_downtrend(mod):
    closes = list(np.arange(200, 100, -1.0))
    rows = [_bar(c, c + 0.5, c - 0.5, c) for c in closes]
    df = _df(rows)
    out = mod.ribbon_bullish(df)
    assert not out.any()  # falling sma44 + price below => never bullish


def test_ribbon_stacked_true_only_when_ordered(mod):
    closes = list(np.arange(100, 260, 1.0))  # long rise => sma22>sma44>sma120 eventually
    rows = [_bar(c, c + 0.5, c - 0.5, c) for c in closes]
    df = _df(rows)
    out = mod.ribbon_stacked(df)
    assert out[-1]
    assert not out[:119].any()  # sma120 undefined before index 119


def test_above_vwma(mod):
    rows = [_bar(10, 11, 9, 10, v=100) for _ in range(5)]
    rows += [_bar(50, 51, 49, 50, v=100)]   # jump well above the VWMA
    df = _df(rows)
    out = mod.above_vwma(df, n=4)
    assert out[-1]      # last close 50 >> rolling vwma
    assert not out[0]   # NaN window => treated False


# ---------------------------------------------------------------------------
# fractal_up_trend — the BUY triangle
# ---------------------------------------------------------------------------

def _fractal_positive_series() -> list[dict]:
    """Hand-built series that satisfies fractalsDown + strongFractal at the final bar.

    Indices (t = last bar): need a down-fractal shape in lows over t-5..t-1 and a
    3-bar bullish thrust with body expansion + new high. We construct a V-bottom:
    declining lows into a trough then a strong green reversal, with a prior EMA(5)<EMA(11)
    (guaranteed by a long downtrend preceding the V).
    """
    # long downtrend so ema5<ema11 holds at t-3
    pre = [_bar(c, c + 0.5, c - 0.5, c) for c in np.arange(140, 110, -1.0)]
    # the fractal window (last 6 bars). Build lows: low[5]>low[4]>low[3]<low[2]<low[1] (V-bottom)
    # bars are appended in chronological order; the LAST appended is bar t.
    # t-5 .. t indices map to the last 6 appended bars.
    window = [
        # t-5: high low (start of V)  open<close not required here
        _bar(110, 111, 108, 109),
        # t-4: lower low
        _bar(109, 110, 106, 107),
        # t-3: trough (lowest low). small red body (for body-expansion contrast).
        _bar(107, 107.5, 104, 106.5),
        # t-2: green, low rises
        _bar(106.5, 109, 105, 108.8),
        # t-1: big green body (close>open[1]... we need close[t-1]>open[t-1]); body large
        _bar(108.0, 113, 107.5, 112.5),
        # t: green close>open[t-1]; new high>high[t-3]; close>open[t-1]
        _bar(112.5, 116, 112, 115.5, v=5000),  # high volume so volume gate can pass
    ]
    return pre + window


def test_fractal_up_trend_fires_on_constructed_bottom(mod):
    df = _df(_fractal_positive_series())
    sig = mod.fractal_up_trend(df)
    # the signal must fire at the final bar (the constructed reversal)
    assert sig[-1], "fractalUpTrend should fire on the hand-built V-bottom reversal"


def test_fractal_up_trend_silent_on_flat_market(mod):
    rows = [_bar(100, 100.5, 99.5, 100.0) for _ in range(60)]
    df = _df(rows)
    sig = mod.fractal_up_trend(df)
    assert not sig.any(), "no fractal on a flat tape"


def test_fractal_up_trend_silent_on_pure_uptrend(mod):
    # monotonic up: lows never form the required down-fractal shape
    closes = list(np.arange(100, 160, 1.0))
    rows = [_bar(c, c + 1, c - 0.2, c + 0.5) for c in closes]
    df = _df(rows)
    sig = mod.fractal_up_trend(df)
    assert not sig.any()


def test_fractal_indexing_is_causal(mod):
    """A signal at bar t must depend ONLY on bars <= t (no lookahead).

    Mutating any bar AFTER the first signal must not change that signal.
    """
    df = _df(_fractal_positive_series())
    sig = mod.fractal_up_trend(df)
    first = int(np.argmax(sig))
    assert sig[first]
    # append a wildly different future bar; the earlier signal is unchanged
    df2 = pd.concat([df, _df([_bar(999, 1000, 998, 999)])], ignore_index=True)
    sig2 = mod.fractal_up_trend(df2)
    assert sig2[first] == sig[first]


# ---------------------------------------------------------------------------
# confluence assembly
# ---------------------------------------------------------------------------

def test_compute_entries_is_subset_of_base_fractal(mod):
    df = _df(_fractal_positive_series())
    base = mod.EntryConfig(False, False, False, False, False)
    strict = mod.EntryConfig(True, True, False, True, True)
    e_base = mod.compute_entries(df, base)
    e_strict = mod.compute_entries(df, strict)
    # every confluence entry is also a base-fractal entry (AND of filters)
    assert (e_strict <= e_base).all()


# ---------------------------------------------------------------------------
# simulate — no-lookahead timing + cost
# ---------------------------------------------------------------------------

def test_simulate_enters_next_bar_open(mod):
    # signal at bar 0; verify entry price == open[1], not close[0]
    rows = [_bar(100, 101, 99, 100.5), _bar(102, 103, 101, 102.5)]
    rows += [_bar(102, 110, 102, 109) for _ in range(5)]  # rising, no exit until time/EOD
    df = _df(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    atr_s = mod.atr(df, 14).to_numpy()
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("time", bars=3)
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last, bars_per_yr=1e5, n_years=0.01)
    assert res.n == 1
    assert res.trades[0].entry_bar == 1
    assert res.trades[0].entry_price == df["open"].iat[1]


def test_simulate_cost_is_charged(mod):
    # flat round-trip: with cost, net return must be negative by ~ROUND_TRIP_PTS/entry
    rows = [_bar(100, 101, 99, 100.0)]
    rows += [_bar(100, 101, 99, 100.0) for _ in range(10)]
    df = _df(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    atr_s = mod.atr(df, 14).to_numpy()
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("time", bars=3)
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last, bars_per_yr=1e5, n_years=0.01)
    assert res.n == 1
    assert res.trades[0].ret < 0  # flat price, only cost => loss
    assert res.trades[0].ret == pytest.approx(-mod.ROUND_TRIP_PTS / 100.0, abs=1e-9)


def test_simulate_flattens_at_session_end(mod):
    # two sessions; an entry near session 1 end must exit at session_end, not hold over
    base = pd.Timestamp("2026-02-02 19:50:00+00:00")  # near UTC day end
    rows = []
    for i in range(6):
        rows.append({"timestamp": base + pd.Timedelta(minutes=5 * i),
                     **_bar(100 + i, 101 + i, 99 + i, 100.5 + i)})
    df = pd.DataFrame(rows)
    # force a day rollover after bar 2
    df.loc[3:, "timestamp"] = pd.Timestamp("2026-02-03 14:00:00+00:00") + \
        pd.to_timedelta(np.arange(3) * 5, unit="m")
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    atr_s = mod.atr(df, 14).to_numpy()
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("time", bars=100)  # would otherwise hold forever
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last, bars_per_yr=1e5, n_years=0.01)
    assert res.n == 1
    assert res.trades[0].reason == "session_end"
    # exit bar is the last bar of session 1 (index 2)
    assert res.trades[0].exit_bar == 2


def test_rth_last_bar_marks_cash_close(mod):
    # bars at 19:55 (in RTH), 20:00 (out), and a fresh day in RTH — the 19:55 bar is rth_last
    ts = ["2026-02-02 19:55:00+00:00", "2026-02-02 20:00:00+00:00",
          "2026-02-03 15:00:00+00:00"]
    df = pd.DataFrame([{"timestamp": pd.Timestamp(t), **_bar(100, 101, 99, 100.0)} for t in ts])
    out = mod.rth_last_bar(df)
    assert out[0]       # last RTH bar before the 20:00 close
    assert not out[1]   # 20:00 is outside RTH
    assert out[2]       # the lone RTH bar of the next day is also its last RTH bar


def test_in_rth_filter(mod):
    rows = []
    # 03:00 UTC (overnight) and 15:00 UTC (RTH)
    for hh in (3, 15):
        rows.append({"timestamp": pd.Timestamp(f"2026-02-02 {hh:02d}:00:00+00:00"),
                     **_bar(100, 101, 99, 100.5)})
    df = pd.DataFrame(rows)
    out = mod.in_rth(df)
    assert not out[0]  # 03:00 UTC outside RTH
    assert out[1]      # 15:00 UTC inside RTH


# ---------------------------------------------------------------------------
# regression: HVN add-on is actually wired into compute_entries + the grid
# ---------------------------------------------------------------------------

def test_hvn_filter_is_applied(mod):
    df = _df(_fractal_positive_series())
    base = mod.EntryConfig(False, False, False, False, False, require_hvn=False)
    with_hvn = mod.EntryConfig(False, False, False, False, False, require_hvn=True)
    e_base = mod.compute_entries(df, base)
    e_hvn = mod.compute_entries(df, with_hvn)
    # HVN is an additional AND-filter: its entries are a subset of the base fractal entries.
    assert (e_hvn <= e_base).all()


def test_hvn_in_entry_grid(mod):
    grid = mod.entry_grid()
    # the VRVP/HVN add-on must actually appear in the swept grid (codex P2 regression)
    assert any(ec.require_hvn for ec in grid)


def test_hvn_rejects_breakout_outside_range(mod):
    """A close ABOVE the prior visible range is a breakout, not an HVN (codex P3 regression)."""
    window = 20
    # build a flat-ish base inside [99, 101], then a breakout close at 200
    rows = [_bar(100, 101, 99, 100.0, v=1000) for _ in range(window)]
    rows.append(_bar(150, 205, 150, 200.0, v=1000))  # breakout far above the range
    df = _df(rows)
    out = mod.hvn_proximity(df, window=window, bins=8, top_frac=0.3)
    assert not out[-1]  # breakout close outside [99,101] => not classified as an HVN


# ---------------------------------------------------------------------------
# regression: intrabar ATR stop/target takes precedence over session-end flatten
# ---------------------------------------------------------------------------

def test_atr_stop_beats_session_end_on_last_bar(mod):
    """A stop touched on a session's last bar must fill at the stop, not at the close."""
    # entry at open[1]; the session's LAST bar (index 2) trades through the stop intrabar.
    base = pd.Timestamp("2026-02-02 19:50:00+00:00")
    rows = [
        {"timestamp": base, **_bar(100, 101, 99, 100.5)},                    # signal bar
        {"timestamp": base + pd.Timedelta(minutes=5), **_bar(100, 101, 99, 100.0)},  # entry @ open=100
        # session-last bar: low spikes well below the stop AND closes higher than the stop.
        {"timestamp": base + pd.Timedelta(minutes=10), **_bar(100, 101, 80, 99.0)},
        # next session (so bar 2 is the session-last bar)
        {"timestamp": pd.Timestamp("2026-02-03 14:00:00+00:00"), **_bar(99, 100, 98, 99.0)},
    ]
    df = pd.DataFrame(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    # large ATR so the stop sits ~ entry-? ; we instead pin ATR via a controlled series:
    # use a tiny atr so stop is just below entry and the bar-2 low (80) blows through it.
    atr_s = np.full(len(df), 5.0)  # k will be applied by the config
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    # stop = entry - 1.0*ATR = 100 - 5 = 95; bar2 low=80 <= 95 => stop fills.
    xc = mod.ExitConfig("atr", atr_k=1.0, rr=2.0)
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last, bars_per_yr=1e5, n_years=0.01)
    assert res.n == 1
    assert res.trades[0].reason == "stop"           # NOT "session_end"
    assert res.trades[0].exit_bar == 2
    # filled at min(open=100, stop=95) = 95, not the bar close (99)
    assert res.trades[0].exit_price == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# regression: time-stop holds EXACTLY `bars` 5-minute bars (entry bar = bar #1)
# ---------------------------------------------------------------------------

def test_time_stop_holds_exactly_n_bars(mod):
    # signal at bar 0 → entry at open[1]. time3 must exit at the close of bar 1+3-1 = 3.
    rows = [_bar(100, 101, 99, 100.0) for _ in range(10)]
    df = _df(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    atr_s = mod.atr(df, 14).to_numpy()
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("time", bars=3)
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last, bars_per_yr=1e5, n_years=0.01)
    assert res.n == 1
    assert res.trades[0].entry_bar == 1
    # held bars = entry_bar(1), 2, 3 = exactly 3 bars; exit at bar 3
    assert res.trades[0].exit_bar == 3
    assert res.trades[0].reason == "time"


# ---------------------------------------------------------------------------
# regression: in-trade drawdown is marked on the equity curve (codex P2)
# ---------------------------------------------------------------------------

def test_equity_marks_in_trade_drawdown(mod):
    """A trade that dips deep mid-hold then recovers to a flat exit must still show the
    in-trade drawdown in the equity curve (max-DD can't ignore adverse excursion)."""
    # entry @ open[1]=100; price plunges to ~70 at bar 2, recovers to 100 by the time exit.
    rows = [
        _bar(100, 101, 99, 100.0),                 # 0 signal
        _bar(100, 101, 99, 100.0),                 # 1 entry @ open 100
        _bar(100, 100, 70, 71.0),                  # 2 deep dip (close 71)
        _bar(72, 101, 72, 100.0),                  # 3 recover to 100
        _bar(100, 101, 99, 100.0),                 # 4 time-stop exit @ close 100
        _bar(100, 101, 99, 100.0),
    ]
    df = _df(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    atr_s = mod.atr(df, 14).to_numpy()
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("time", bars=4)  # hold bars 1..4, exit @ bar 4 close (~100)
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last, bars_per_yr=1e5, n_years=0.01)
    assert res.n == 1
    # final return is ~flat (only cost), BUT the equity curve must have dipped at bar 2.
    assert res.equity[2] < 0.9   # ~71/100 marked → ≥10% in-trade drawdown
    assert res.max_dd > 0.2      # max-DD reflects the deep mid-trade excursion
