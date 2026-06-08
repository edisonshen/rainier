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


def test_load_clean_keeps_valid_row_when_later_dup_is_phantom(mod, tmp_path):
    """If a duplicate timestamp's LATER row is a phantom, the valid earlier row must survive —
    phantom filtering happens BEFORE dedup (codex P2 regression)."""
    p = tmp_path / "MES_5m.csv"
    df = pd.DataFrame(
        [
            # valid bar at 14:00
            {"timestamp": "2026-01-01 14:00:00+00:00", "open": 100, "high": 102, "low": 98, "close": 101, "volume": 500},
            # DUPLICATE 14:00 whose later row is a zero-range phantom — must NOT win dedup
            {"timestamp": "2026-01-01 14:00:00+00:00", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 500},
            {"timestamp": "2026-01-01 14:05:00+00:00", "open": 101, "high": 103, "low": 100, "close": 102, "volume": 600},
        ]
    )
    df.to_csv(p, index=False)
    out = mod.load_clean_5m(p)
    # 14:00 must survive (the valid row), NOT be dropped because the later dup was a phantom
    assert (out["timestamp"] == pd.Timestamp("2026-01-01 14:00:00+00:00")).any()
    row = out[out["timestamp"] == pd.Timestamp("2026-01-01 14:00:00+00:00")].iloc[0]
    assert row["close"] == 101  # the valid bar, not the phantom
    assert len(out) == 2


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
    # fires ONCE on the crossing bar, not on every later bar of the continuing selloff
    assert out.sum() == 1
    assert not out[13:].any()


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


def test_td_buy_context_uses_completion_event_not_running_count(mod):
    """In a LONG selloff the count climbs past min_count; context must NOT stay true for every
    later bar — only within `lookback` of the completion event (codex P2 regression)."""
    # 30 strictly-descending bars: buy-setup completes (count hits 7) at index 4+6 = 10,
    # then keeps climbing to 9, 10, ... Context must be False well after the completion window.
    closes = list(np.arange(200, 170, -1.0))
    rows = [_bar(c, c + 0.5, c - 0.5, c) for c in closes]
    df = _df(rows)
    ctx = mod.td_buy_context(df, lookback=6, min_count=7)
    assert ctx[10]        # completion event (count reaches 7) → within window
    assert ctx[15]        # still within lookback (16th-ish bar)
    assert not ctx[-1]    # many bars after completion, deep in the selloff → NOT true anymore


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


# ===========================================================================
# SHORT side — the DERIVED mirror of the long signal (not in the operator's Pine)
# ===========================================================================

def _fractal_down_positive_series() -> list[dict]:
    """Hand-built series that satisfies fractalsUp + strongBearFractal at the final bar.

    Exact axis-flip of `_fractal_positive_series`: a long uptrend (so ema5>ema11 at t-3), then
    an up-fractal in the HIGHS (high[3] is the local PEAK) followed by a strong red reversal that
    makes a new low. Lows/highs/closes mirrored about the price level.
    """
    # long uptrend so ema5>ema11 holds at t-3
    pre = [_bar(c, c + 0.5, c - 0.5, c) for c in np.arange(110, 140, 1.0)]
    # the fractal window (last 6 bars). Build highs: high[5]<high[4]<high[3]>high[2]>high[1] (peak)
    window = [
        # t-5: low high (start of the peak)
        _bar(140, 141, 139, 140.5),
        # t-4: higher high
        _bar(140.5, 143, 139, 142),
        # t-3: peak (highest high). small green body (for body-expansion contrast).
        _bar(142, 145, 141.5, 142.5),
        # t-2: red, high falls
        _bar(142.5, 143, 140, 140.2),
        # t-1: big red body (close[t-1]<open[t-1]); body large
        _bar(141.0, 141.5, 136, 136.5),
        # t: red close<open[t-1]; new low<low[t-3]; close[t]<open[t-1]
        _bar(136.5, 137, 133, 133.5, v=5000),  # high volume so volume gate can pass
    ]
    return pre + window


def test_fractal_down_trend_fires_on_constructed_top(mod):
    df = _df(_fractal_down_positive_series())
    sig = mod.fractal_down_trend(df)
    assert sig[-1], "fractalDownTrend should fire on the hand-built peak reversal"


def test_fractal_down_trend_silent_on_flat_market(mod):
    rows = [_bar(100, 100.5, 99.5, 100.0) for _ in range(60)]
    df = _df(rows)
    sig = mod.fractal_down_trend(df)
    assert not sig.any(), "no short fractal on a flat tape"


def test_fractal_down_trend_silent_on_pure_downtrend(mod):
    # monotonic down: highs never form the required up-fractal shape
    closes = list(np.arange(160, 100, -1.0))
    rows = [_bar(c, c + 0.2, c - 1, c - 0.5) for c in closes]
    df = _df(rows)
    sig = mod.fractal_down_trend(df)
    assert not sig.any()


def test_fractal_down_trend_is_causal(mod):
    """A short signal at bar t must depend ONLY on bars <= t (no lookahead)."""
    df = _df(_fractal_down_positive_series())
    sig = mod.fractal_down_trend(df)
    first = int(np.argmax(sig))
    assert sig[first]
    df2 = pd.concat([df, _df([_bar(1, 2, 0.5, 1)])], ignore_index=True)
    sig2 = mod.fractal_down_trend(df2)
    assert sig2[first] == sig[first]


def test_fractal_down_is_mirror_not_identical_to_long(mod):
    """On the LONG-positive fixture the short fractal must NOT fire (and vice-versa) — they are
    opposite-direction signals, not the same array."""
    long_df = _df(_fractal_positive_series())
    short_df = _df(_fractal_down_positive_series())
    assert mod.fractal_up_trend(long_df)[-1]
    assert not mod.fractal_down_trend(long_df)[-1]   # long bottom ≠ short top
    assert mod.fractal_down_trend(short_df)[-1]
    assert not mod.fractal_up_trend(short_df)[-1]


def test_td_setup_sell_completes_at_nine(mod):
    # strictly ascending closes => every bar t>=4 has close>close[4] => count climbs
    closes = list(np.arange(70, 100, 1.0))  # 30 ascending bars
    rows = [_bar(c, c + 1, c - 1, c) for c in closes]
    df = _df(rows)
    out = mod.td_setup_sell(df, completed=9)
    first_true = int(np.argmax(out))
    assert out[first_true]
    assert first_true == 12        # run hits 9 at t = 4 + 8 = 12 (mirror of the buy test)
    assert out.sum() == 1
    assert not out[13:].any()


def test_td_setup_sell_resets_on_down_close(mod):
    closes = list(np.arange(92, 100, 1.0)) + [10.0] + list(np.arange(70, 100, 1.0))
    rows = [_bar(c, c + 1, c - 1, c) for c in closes]
    df = _df(rows)
    out = mod.td_setup_sell(df, completed=9)
    assert not out[:9].any()


def test_td_sell_context_uses_completion_event(mod):
    """In a long rally the count climbs past min_count; sell context must NOT stay true for every
    later bar — only within lookback of the completion event (mirror of the buy-context test)."""
    closes = list(np.arange(170, 200, 1.0))  # 30 strictly ascending
    rows = [_bar(c, c + 0.5, c - 0.5, c) for c in closes]
    df = _df(rows)
    ctx = mod.td_sell_context(df, lookback=6, min_count=7)
    assert ctx[10]        # completion event (count reaches 7)
    assert ctx[15]        # still within lookback
    assert not ctx[-1]    # deep in the rally, far past the completion → no longer true


def test_ribbon_bearish_requires_below_and_falling(mod):
    closes = list(np.arange(200, 100, -1.0))  # falling
    rows = [_bar(c, c + 0.5, c - 0.5, c) for c in closes]
    df = _df(rows)
    out = mod.ribbon_bearish(df)
    assert out[-1]             # falling downtrend => bearish at the end
    assert not out[:44].any()  # sma44 undefined early


def test_ribbon_bearish_false_in_uptrend(mod):
    closes = list(np.arange(100, 200, 1.0))
    rows = [_bar(c, c + 0.5, c - 0.5, c) for c in closes]
    df = _df(rows)
    assert not mod.ribbon_bearish(df).any()


def test_ribbon_stacked_bear_true_only_when_ordered(mod):
    closes = list(np.arange(260, 100, -1.0))  # long fall => sma22<sma44<sma120
    rows = [_bar(c, c + 0.5, c - 0.5, c) for c in closes]
    df = _df(rows)
    out = mod.ribbon_stacked_bear(df)
    assert out[-1]
    assert not out[:119].any()


def test_below_vwma(mod):
    rows = [_bar(50, 51, 49, 50, v=100) for _ in range(5)]
    rows += [_bar(10, 11, 9, 10, v=100)]   # drop well below the VWMA
    df = _df(rows)
    out = mod.below_vwma(df, n=4)
    assert out[-1]
    assert not out[0]   # NaN window => treated False


def test_short_confluence_is_subset_of_base_short_fractal(mod):
    df = _df(_fractal_down_positive_series())
    base = mod.EntryConfig(False, False, False, False, False, direction="short")
    strict = mod.EntryConfig(True, True, False, True, False, direction="short")
    e_base = mod.compute_entries(df, base)
    e_strict = mod.compute_entries(df, strict)
    assert (e_strict <= e_base).all()


def test_compute_entries_dispatches_on_direction(mod):
    """A short EntryConfig must use the short base signal, not the long one (cache isolation)."""
    df = _df(_fractal_down_positive_series())
    cache = {}
    long_e = mod.compute_entries(df, mod.EntryConfig(False, False, False, False, False,
                                                     direction="long"), cache)
    short_e = mod.compute_entries(df, mod.EntryConfig(False, False, False, False, False,
                                                      direction="short"), cache)
    # the short fixture fires the short fractal at the last bar, not the long fractal
    assert short_e[-1]
    assert not long_e[-1]


# ---------------------------------------------------------------------------
# SHORT simulation — direction-signed P&L + mirrored exits
# ---------------------------------------------------------------------------

def test_simulate_short_profits_when_price_falls(mod):
    # falling price => a short is profitable (entry/exit - 1 > 0)
    rows = [_bar(100 - i, 101 - i, 99 - i, 100 - i) for i in range(8)]
    df = _df(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    atr_s = mod.atr(df, 14).to_numpy()
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("time", bars=3)
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last,
                       bars_per_yr=1e5, n_years=0.01, direction="short")
    assert res.n == 1
    assert res.trades[0].direction == "short"
    assert res.trades[0].ret > 0          # short into a decline → gain
    assert res.total_pts > 0


def test_simulate_short_loses_when_price_rises(mod):
    rows = [_bar(100 + i, 101 + i, 99 + i, 100 + i) for i in range(8)]
    df = _df(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    atr_s = mod.atr(df, 14).to_numpy()
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("time", bars=3)
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last,
                       bars_per_yr=1e5, n_years=0.01, direction="short")
    assert res.trades[0].ret < 0          # short into a rally → loss


def test_book_trade_short_pnl_is_linear_relative_to_entry(mod):
    """Regression (codex P2): short P&L must be LINEAR relative to the ENTRY price, not a
    price ratio. A short 100→90 earns +10% gross (NOT +11.1% from entry/exit-1); 100→110
    loses -10% gross (NOT -9.1%). Cost is ROUND_TRIP_PTS/entry on top."""
    cost = mod.ROUND_TRIP_PTS / 100.0
    # winning short: 100 -> 90  => gross +10%
    net_win, eq_win = mod._book_trade(1.0, 100.0, 90.0, "short")
    assert net_win == pytest.approx(0.10 - cost)
    assert eq_win == pytest.approx(1.0 * (1.0 + 0.10 - cost))
    # losing short: 100 -> 110 => gross -10%
    net_loss, _ = mod._book_trade(1.0, 100.0, 110.0, "short")
    assert net_loss == pytest.approx(-0.10 - cost)


def test_book_trade_long_short_symmetric_about_entry(mod):
    """A long 100→110 and a short 100→90 are equal-and-opposite moves about the entry, so their
    gross returns have identical magnitude (+10%) — the entry-price denominator makes them symmetric."""
    cost = mod.ROUND_TRIP_PTS / 100.0
    net_long, _ = mod._book_trade(1.0, 100.0, 110.0, "long")
    net_short, _ = mod._book_trade(1.0, 100.0, 90.0, "short")
    assert net_long == pytest.approx(net_short)        # both +10% gross, same cost
    assert net_long == pytest.approx(0.10 - cost)


def test_mtm_factor_matches_book_trade_basis(mod):
    """The mark-to-market factor used for in-trade equity must use the SAME linear-relative-to-entry
    basis as _book_trade, so the equity curve and the booked return agree at the exit price."""
    # short held to 90 from 100: mtm factor = (2*100-90)/100 = 1.10 (the gross +10%)
    assert mod._mtm_factor(100.0, 90.0, short=True) == pytest.approx(1.10)
    # adverse short to 110: factor = (2*100-110)/100 = 0.90 (the gross -10%)
    assert mod._mtm_factor(100.0, 110.0, short=True) == pytest.approx(0.90)
    # long to 110: factor = 110/100 = 1.10
    assert mod._mtm_factor(100.0, 110.0, short=False) == pytest.approx(1.10)
    # at the exit price the mtm factor equals 1 + the booked GROSS return (cost-free check)
    net, _ = mod._book_trade(1.0, 100.0, 90.0, "short")
    assert mod._mtm_factor(100.0, 90.0, short=True) == pytest.approx(1.0 + net + mod.ROUND_TRIP_PTS / 100.0)


def test_simulate_short_atr_stop_is_above_entry(mod):
    """Short ATR stop sits ABOVE the entry; an upside spike fills at the stop (mirror of long)."""
    rows = [_bar(100, 101, 99, 100), _bar(100, 101, 99, 100),
            _bar(100, 110, 100, 108), _bar(100, 101, 99, 100)]
    df = _df(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    atr_s = np.full(len(df), 5.0)         # stop = entry + 1*5 = 105
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("atr", atr_k=1.0, rr=2.0)
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last,
                       bars_per_yr=1e5, n_years=0.01, direction="short")
    assert res.trades[0].reason == "stop"
    assert res.trades[0].exit_price == pytest.approx(105.0)  # filled at the stop ABOVE entry


def test_simulate_short_atr_target_is_below_entry(mod):
    rows = [_bar(100, 101, 99, 100), _bar(100, 101, 99, 100),
            _bar(95, 96, 88, 90), _bar(100, 101, 99, 100)]
    df = _df(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    atr_s = np.full(len(df), 5.0)         # stop=105, target = 100 - 2*(105-100) = 90
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("atr", atr_k=1.0, rr=2.0)
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last,
                       bars_per_yr=1e5, n_years=0.01, direction="short")
    assert res.trades[0].reason == "target"
    assert res.trades[0].exit_price == pytest.approx(90.0)   # target BELOW entry


def test_simulate_short_vwma_exit_on_reclaim(mod):
    """A short's vwmaCross exit triggers when price RECLAIMS the VWMA (close > VWMA), the mirror
    of the long exit (close < VWMA)."""
    # 5 low bars set a low VWMA, then a bar pops well above it → short should exit on reclaim.
    rows = [_bar(10, 11, 9, 10, v=100) for _ in range(5)]
    rows += [_bar(10, 11, 9, 10, v=100)]            # signal bar (close ~ VWMA)
    rows += [_bar(50, 51, 49, 50, v=100)]           # entry bar opens at 50, far above VWMA
    rows += [_bar(50, 51, 49, 50, v=100) for _ in range(3)]
    df = _df(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[5] = True
    atr_s = mod.atr(df, 14).to_numpy()
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("vwma")
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last,
                       bars_per_yr=1e5, n_years=0.01, direction="short")
    assert res.n == 1
    assert res.trades[0].reason == "vwmaCross"


def test_simulate_short_marks_in_trade_drawdown(mod):
    """A short whose price spikes UP mid-hold (adverse) must mark a drawdown on the equity curve."""
    rows = [
        _bar(100, 101, 99, 100.0),   # 0 signal
        _bar(100, 101, 99, 100.0),   # 1 entry @ open 100 (short)
        _bar(100, 130, 100, 129.0),  # 2 price spikes UP (adverse for a short)
        _bar(128, 129, 99, 100.0),   # 3 back to 100
        _bar(100, 101, 99, 100.0),   # 4 time-stop exit @ ~100
        _bar(100, 101, 99, 100.0),
    ]
    df = _df(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    atr_s = mod.atr(df, 14).to_numpy()
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("time", bars=4)
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last,
                       bars_per_yr=1e5, n_years=0.01, direction="short")
    assert res.equity[2] < 0.9   # short marked-to-market at entry/close = 100/129 ≈ 0.78 → drawdown
    assert res.max_dd > 0.2


# ---------------------------------------------------------------------------
# COMBINED simulation — one position at a time, either direction
# ---------------------------------------------------------------------------

def test_combined_one_position_at_a_time(mod):
    """A long entry while a long is already open is ignored; the next trade can only start after
    the prior exit (no pyramiding, no double-booking)."""
    rows = [_bar(100, 101, 99, 100.0) for _ in range(12)]
    df = _df(rows)
    long_e = np.zeros(len(df), dtype=bool)
    short_e = np.zeros(len(df), dtype=bool)
    long_e[0] = True
    long_e[2] = True   # fires WHILE the first trade is open (entry@1, time3 holds 1..3) → ignored
    short_e[4] = True  # fires after the first exit → a new SHORT trade
    atr_s = mod.atr(df, 14).to_numpy()
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("time", bars=3)
    res = mod.simulate_combined(df, long_e, short_e, xc, atr_s, vwma_s, sess_last,
                                bars_per_yr=1e5, n_years=0.01)
    assert res.n == 2                       # the in-trade signal at bar 2 was skipped
    assert res.trades[0].direction == "long"
    assert res.trades[0].entry_bar == 1
    assert res.trades[1].direction == "short"
    assert res.trades[1].entry_bar == 5     # after the long exited at bar 3


def test_combined_long_wins_tie_break(mod):
    """On a bar where BOTH a long and a short signal fire, the documented tie-break takes LONG."""
    rows = [_bar(100, 101, 99, 100.0) for _ in range(8)]
    df = _df(rows)
    long_e = np.zeros(len(df), dtype=bool)
    short_e = np.zeros(len(df), dtype=bool)
    long_e[0] = True
    short_e[0] = True   # same bar → long must win
    atr_s = mod.atr(df, 14).to_numpy()
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("time", bars=3)
    res = mod.simulate_combined(df, long_e, short_e, xc, atr_s, vwma_s, sess_last,
                                bars_per_yr=1e5, n_years=0.01)
    assert res.n == 1
    assert res.trades[0].direction == "long"


def test_run_combined_grid_produces_combined_rows(mod):
    """The combined sweep must emit Rows tagged mode='combined' (so the report can label L/S)."""
    df = _df(_fractal_positive_series() + _fractal_down_positive_series())
    rows = mod.run_combined_grid(df, bars_per_yr=1e5, n_years=0.01)
    # at least the bare L/S config should trade on a fixture carrying both a long and short signal
    assert any(r.mode == "combined" for r in rows)


def test_entry_grid_short_mirrors_long_shape(mod):
    """The short grid is the exact mirror of the long grid: same flag combos, direction='short'."""
    long_grid = mod.entry_grid("long")
    short_grid = mod.entry_grid("short")
    assert len(long_grid) == len(short_grid)
    for lc, sc in zip(long_grid, short_grid):
        assert lc.direction == "long" and sc.direction == "short"
        assert (lc.require_vwma, lc.require_ribbon, lc.require_stacked,
                lc.require_td, lc.require_rth, lc.require_hvn) == \
               (sc.require_vwma, sc.require_ribbon, sc.require_stacked,
                sc.require_td, sc.require_rth, sc.require_hvn)


def _stub_row(mod, total_return, mar, mode):
    """Minimal Row carrying just the fields the long/short verdict reads (label + res metrics)."""
    res = mod.Result(trades=[], equity=np.array([1.0]))
    res.n = 50
    res.total_return = total_return
    res.mar = mar
    res.win_rate = 0.5
    res.profit_factor = 1.2
    res.max_dd = 0.05
    res.exposure = 0.3
    ec = mod.EntryConfig(False, False, False, False, False, direction=mode if mode != "combined" else "long")
    return mod.Row(ec, mod.ExitConfig("time", bars=48), res, mode=mode)


def _stub_wf(mod, oos_return, oos_mar, n=40):
    """Build a (in_sample, oos_result, oos_best) walk-forward tuple; only oos ([1]) is read here."""
    oos = mod.Result(trades=[], equity=np.array([1.0]))
    oos.n = n
    oos.total_return = oos_return
    oos.mar = oos_mar
    oos.sharpe = 1.0
    oos.max_dd = 0.05
    is_res = mod.Result(trades=[], equity=np.array([1.0]))
    is_res.n = 60
    is_res.total_return = oos_return
    is_res.mar = oos_mar
    return (("stub-config", is_res), oos, ("oos-best", oos))


def test_combined_verdict_does_not_claim_beat_when_oos_worse_than_long(mod):
    """Regression (codex P2): a POSITIVE combined OOS that is still WORSE than the long-only OOS
    must NOT be reported as beating long-only. Combined OOS Ret/DD 1.6 < long OOS 2.7 ⇒ 'does NOT beat'."""
    long_best = _stub_row(mod, 0.06, 3.7, "long")
    short_best = _stub_row(mod, 0.03, 2.5, "short")
    combined_best = _stub_row(mod, 0.065, 4.8, "combined")  # in-sample mar > long (cherry-picked)
    bh = mod.Result(trades=[], equity=np.array([1.0]))
    bh.total_return = 0.065
    bh.mar = 0.7
    bh.sharpe = 1.0
    bh.max_dd = 0.097
    wf_long = _stub_wf(mod, 0.051, 2.72)       # long OOS Ret/DD 2.72
    wf_combined = _stub_wf(mod, 0.016, 1.61)   # combined OOS positive but 1.61 < 2.72
    wf_short = _stub_wf(mod, -0.014, -0.62)
    html = mod.build_long_short_section(long_best, short_best, combined_best, bh,
                                        wf_combined, wf_short, wf_long=wf_long)
    assert "does NOT beat" in html
    assert "barely beats" not in html


def test_combined_verdict_claims_beat_only_when_oos_exceeds_long(mod):
    """The 'barely beats' phrasing fires ONLY when combined OOS Ret/DD truly exceeds long OOS."""
    long_best = _stub_row(mod, 0.06, 3.7, "long")
    short_best = _stub_row(mod, 0.03, 2.5, "short")
    combined_best = _stub_row(mod, 0.08, 4.8, "combined")
    bh = mod.Result(trades=[], equity=np.array([1.0]))
    bh.total_return = 0.065
    bh.mar = 0.7
    bh.max_dd = 0.097
    wf_long = _stub_wf(mod, 0.051, 2.72)
    wf_combined = _stub_wf(mod, 0.07, 3.10)    # combined OOS 3.10 > long OOS 2.72
    # short still DRAGS (fails OOS) so the verdict stays in the short_drags branch, but the
    # combined book legitimately beats long-only OOS → the phrasing is 'barely beats', not 'does NOT'.
    wf_short = _stub_wf(mod, -0.01, -0.4)
    html = mod.build_long_short_section(long_best, short_best, combined_best, bh,
                                        wf_combined, wf_short, wf_long=wf_long)
    assert "barely beats" in html
    assert "does NOT beat" not in html


def test_short_drags_when_combined_underperforms_long_even_if_short_leg_ok(mod):
    """Regression (codex iter-2 P2): when the SHORT-only leg looks fine OOS but the COMBINED book's
    OOS Ret/DD is still below long-only OOS, shorts did NOT add value — the verdict must stay in the
    'drags / does NOT beat' branch, never the 'adds value / matches' branch."""
    long_best = _stub_row(mod, 0.06, 3.7, "long")
    short_best = _stub_row(mod, 0.04, 2.0, "short")     # short-only looks healthy in-sample
    combined_best = _stub_row(mod, 0.07, 4.0, "combined")
    bh = mod.Result(trades=[], equity=np.array([1.0]))
    bh.total_return = 0.065
    bh.mar = 0.7
    bh.max_dd = 0.097
    wf_long = _stub_wf(mod, 0.051, 2.72)
    wf_short = _stub_wf(mod, 0.02, 1.50, n=30)          # short-only OOS positive (would pass alone)
    wf_combined = _stub_wf(mod, 0.03, 1.80)             # but combined OOS 1.80 < long OOS 2.72
    html = mod.build_long_short_section(long_best, short_best, combined_best, bh,
                                        wf_combined, wf_short, wf_long=wf_long)
    assert "does NOT beat" in html
    assert "SHORT side <b>adds</b>" not in html
    assert "matches" not in html


def _row_with_n(mod, n, mar, mode="short"):
    res = mod.Result(trades=[], equity=np.array([1.0]))
    res.n = n
    res.mar = mar
    res.total_return = 0.02
    ec = mod.EntryConfig(False, False, False, False, False,
                         direction=mode if mode != "combined" else "long")
    return mod.Row(ec, mod.ExitConfig("time", bars=24), res, mode=mode)


def test_capped_best_returns_none_when_no_config_is_under_the_cap(mod):
    """The short/combined pick honors the MAX_TRADES selectivity CAP as a HARD gate — if EVERY
    config fires >= MAX_TRADES (too busy to be a rare high-conviction setup), return None
    (→ the 'none under the cap' report path), rather than crowning a busy config."""
    busy = [_row_with_n(mod, mod.MAX_TRADES, 9.9),       # exactly at the cap → ineligible
            _row_with_n(mod, mod.MAX_TRADES + 20, 5.0)]  # 50-trade config → ineligible
    assert mod.capped_best(busy) is None
    assert mod.capped_best([]) is None


def test_capped_best_crowns_a_rare_config_under_the_cap(mod):
    """When at least one config is under MAX_TRADES, capped_best returns the selective sweet-spot
    pick and EXCLUDES the busy high-frequency configs (even if their Ret/DD looks better)."""
    rows = [_row_with_n(mod, mod.MAX_TRADES + 20, 99.0),  # 50-trade busy config — must NOT win
            _row_with_n(mod, 10, 2.0)]                    # rare 10-trade setup — eligible
    best = mod.capped_best(rows)
    assert best is not None
    assert best.res.n < mod.MAX_TRADES   # the busy 99.0-mar config is excluded by the cap
    assert best.res.n == 10


def test_pick_sweet_spot_excludes_busy_configs(mod):
    """The headline pick is the best Ret/DD among RARE (n < cap) configs; a 50-trade config with a
    higher Ret/DD must lose to a 10-trade config under the cap."""
    rows = [_row_with_n(mod, 50, 99.0, mode="long"),   # busy, great ratio — excluded by cap
            _row_with_n(mod, 10, 3.0, mode="long")]    # rare — the eligible winner
    best = mod.pick_sweet_spot(rows)
    assert best.res.n == 10


def test_capped_best_does_not_change_max_trades(mod):
    """Guard: the operator-fixed selectivity cap is exactly 30 and capped_best keys off it."""
    assert mod.MAX_TRADES == 30


def test_verdict_uses_no_confluence_note_when_bare_fractal_wins(mod):
    """Regression (codex iter-3 P3): if the sweet spot IS the bare fractal (no confluence filters),
    the 'vs bare fractal' line must use the no-filter base_note, NOT the generic 'filters discarded
    winners' note — the winner has no filters to have discarded anything."""
    bare = mod.EntryConfig(False, False, False, False, False)   # no confluence filters
    best = _stub_row(mod, 0.094, 3.0, "long")
    best = mod.Row(bare, mod.ExitConfig("time", bars=48), best.res, mode="long")
    baseline = best                                            # winner == the bare baseline
    bh = mod.Result(trades=[], equity=np.array([1.0]))
    bh.total_return = 0.065
    bh.mar = 0.7
    bh.max_dd = 0.097
    wf = _stub_wf(mod, 0.05, 2.7, n=74)
    html = mod.build_verdict(best, baseline, bh, wf)
    assert "uses NO confluence filter" in html
    assert "filters\n                     discarded" not in html and "discarded winning dip-buys" not in html


def test_combined_verdict_conservative_without_long_wf(mod):
    """With no long walk-forward to compare against, the section must NOT claim combined beats long."""
    long_best = _stub_row(mod, 0.06, 3.7, "long")
    combined_best = _stub_row(mod, 0.08, 4.8, "combined")
    bh = mod.Result(trades=[], equity=np.array([1.0]))
    bh.total_return = 0.065
    bh.mar = 0.7
    bh.max_dd = 0.097
    wf_combined = _stub_wf(mod, 0.07, 3.10)
    html = mod.build_long_short_section(long_best, None, combined_best, bh,
                                        wf_combined, None, wf_long=None)
    assert "barely beats" not in html


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


def test_session_last_bar_rolls_at_17et_not_utc_midnight(mod):
    """The CME equity session rolls at 17:00 ET, not 00:00 UTC (codex P2 regression).

    EST: 17:00 ET = 22:00 UTC. Bars at 21:55 ET-eve and 22:00 ET-eve straddle the close; the
    21:55 (= last bar of the trading day) must be the session_last bar, and 00:00 UTC must NOT.
    """
    ts = [
        "2026-02-02 21:55:00+00:00",  # 16:55 ET — last bar before the 17:00 ET close
        "2026-02-02 22:00:00+00:00",  # 17:00 ET — opens the next trading day
        "2026-02-03 00:00:00+00:00",  # 19:00 ET — same trading day, mid-session (NOT boundary)
        "2026-02-03 02:00:00+00:00",  # 21:00 ET — still same trading day (trailing bar)
    ]
    df = pd.DataFrame([{"timestamp": pd.Timestamp(t), **_bar(100, 101, 99, 100.0)} for t in ts])
    out = mod.session_last_bar(df)
    assert out[0]       # 16:55 ET is the last bar before the 17:00 ET close
    assert not out[1]   # 17:00 ET opens the next trading day
    assert not out[2]   # 00:00 UTC is mid-session of the new trading day, NOT a boundary


def test_rth_last_bar_marks_cash_close_edt(mod):
    # EDT (April): cash close = 16:00 ET = 20:00 UTC. 19:55 UTC = 15:55 ET (last RTH bar).
    ts = ["2026-04-06 19:55:00+00:00", "2026-04-06 20:00:00+00:00",
          "2026-04-07 15:00:00+00:00"]
    df = pd.DataFrame([{"timestamp": pd.Timestamp(t), **_bar(100, 101, 99, 100.0)} for t in ts])
    out = mod.rth_last_bar(df)
    assert out[0]       # 15:55 ET — last RTH bar before the close
    assert not out[1]   # 16:00 ET is outside RTH (exclusive close)
    assert out[2]       # the lone RTH bar of the next day is also its last RTH bar


def test_in_rth_is_timezone_aware(mod):
    # 15:00 UTC is RTH under BOTH EST (10:00 ET) and EDT (11:00 ET). 14:00 UTC is RTH only
    # under EDT (10:00 ET); under EST it is 09:00 ET — BEFORE the 09:30 open → NOT RTH.
    rows = [
        {"timestamp": pd.Timestamp("2026-02-02 14:00:00+00:00"), **_bar(1, 2, 0.5, 1)},  # EST 09:00 → out
        {"timestamp": pd.Timestamp("2026-02-02 15:00:00+00:00"), **_bar(1, 2, 0.5, 1)},  # EST 10:00 → in
        {"timestamp": pd.Timestamp("2026-04-06 14:00:00+00:00"), **_bar(1, 2, 0.5, 1)},  # EDT 10:00 → in
        {"timestamp": pd.Timestamp("2026-02-02 03:00:00+00:00"), **_bar(1, 2, 0.5, 1)},  # overnight → out
    ]
    df = pd.DataFrame(rows)
    out = mod.in_rth(df)
    assert not out[0]  # EST premarket
    assert out[1]      # EST cash hours
    assert out[2]      # EDT cash hours (same UTC hour, different ET → still RTH)
    assert not out[3]  # overnight


def test_rth_entry_never_fills_outside_rth(mod):
    """An RTH-gated signal on the LAST RTH bar (entry would fill at the cash close, outside
    RTH) must be suppressed (codex P2 regression)."""
    df = _df(_fractal_positive_series())
    base = mod.fractal_up_trend(df)
    sig_bar = int(np.argmax(base))
    assert base[sig_bar]
    # EDT date: stamp the signal bar at 19:55 UTC (15:55 ET, last RTH bar). Its entry bar
    # (t+1) is 20:00 UTC = 16:00 ET, OUTSIDE RTH.
    anchor = pd.Timestamp("2026-04-06 19:55:00+00:00")
    df["timestamp"] = [anchor + pd.Timedelta(minutes=5 * (i - sig_bar)) for i in range(len(df))]
    rth_cfg = mod.EntryConfig(False, False, False, False, True)  # require_rth
    e = mod.compute_entries(df, rth_cfg)
    # signal bar is in RTH but its entry bar (16:00 ET) is OUT of RTH → suppressed
    assert not e[sig_bar]


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


def test_long_gap_open_through_target_books_target_not_stop(mod):
    """Regression (codex iter-3 P2): a long bar that OPENS above the target is a target fill at the
    open, even if the same bar's later range also dips below the stop. The open is known to occur
    first, so booking it as a stop would mis-report a winner as a loser."""
    base = pd.Timestamp("2026-02-02 14:00:00+00:00")
    rows = [
        {"timestamp": base, **_bar(100, 101, 99, 100.5)},                              # signal bar
        {"timestamp": base + pd.Timedelta(minutes=5), **_bar(100, 101, 99, 100.0)},    # entry @ open=100
        # next bar GAPS open to 112 (above tp=110), then its range also crosses stop=95 (low=90).
        {"timestamp": base + pd.Timedelta(minutes=10), **_bar(112, 113, 90, 95.0)},
        {"timestamp": base + pd.Timedelta(minutes=15), **_bar(95, 96, 94, 95.0)},
    ]
    df = pd.DataFrame(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    atr_s = np.full(len(df), 5.0)   # stop=100-5=95, tp=100+2*5=110
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("atr", atr_k=1.0, rr=2.0)
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last, bars_per_yr=1e5, n_years=0.01)
    assert res.n == 1
    assert res.trades[0].reason == "target"            # NOT "stop"
    assert res.trades[0].exit_price == pytest.approx(112.0)   # filled at the gap-open, above tp
    assert res.trades[0].ret > 0


def test_short_gap_open_through_target_books_target_not_stop(mod):
    """Mirror of the long gap case: a short bar that OPENS below its target is a target fill at the
    open, even if the bar's later range also spikes above the stop."""
    base = pd.Timestamp("2026-02-02 14:00:00+00:00")
    rows = [
        {"timestamp": base, **_bar(100, 101, 99, 100.0)},
        {"timestamp": base + pd.Timedelta(minutes=5), **_bar(100, 101, 99, 100.0)},    # entry @ open=100
        # short: stop=105 (above), tp=90 (below). bar gaps open to 88 (below tp), high 110 (above stop).
        {"timestamp": base + pd.Timedelta(minutes=10), **_bar(88, 110, 87, 105.0)},
        {"timestamp": base + pd.Timedelta(minutes=15), **_bar(88, 89, 87, 88.0)},
    ]
    df = pd.DataFrame(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    atr_s = np.full(len(df), 5.0)   # stop=100+5=105, tp=100-2*5=90
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("atr", atr_k=1.0, rr=2.0)
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last,
                       bars_per_yr=1e5, n_years=0.01, direction="short")
    assert res.n == 1
    assert res.trades[0].reason == "target"
    assert res.trades[0].exit_price == pytest.approx(88.0)   # gap-open below tp
    assert res.trades[0].ret > 0


def test_long_gap_open_through_stop_fills_at_open(mod):
    """A long bar that gaps OPEN below the stop fills at the (worse) open, not the stop level."""
    base = pd.Timestamp("2026-02-02 14:00:00+00:00")
    rows = [
        {"timestamp": base, **_bar(100, 101, 99, 100.5)},
        {"timestamp": base + pd.Timedelta(minutes=5), **_bar(100, 101, 99, 100.0)},    # entry @ 100
        {"timestamp": base + pd.Timedelta(minutes=10), **_bar(90, 91, 89, 90.0)},      # gaps below stop=95
        {"timestamp": base + pd.Timedelta(minutes=15), **_bar(90, 91, 89, 90.0)},
    ]
    df = pd.DataFrame(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    atr_s = np.full(len(df), 5.0)   # stop=95
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("atr", atr_k=1.0, rr=2.0)
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last, bars_per_yr=1e5, n_years=0.01)
    assert res.trades[0].reason == "stop"
    assert res.trades[0].exit_price == pytest.approx(90.0)   # filled at the gap-open, below the stop


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


def test_equity_breakeven_mark_not_clobbered(mod):
    """A real equity value of exactly 1.0 (breakeven mark) must NOT be treated as an unfilled
    slot and overwritten by the forward-fill (codex P2 regression)."""
    # entry @ open[1]=100; bar 2 returns exactly to 100 (close==entry → marked equity == 1.0),
    # bar 3 dips to 90, exit @ bar 3 (time-stop). The bar-2 mark must stay 1.0, not become the
    # later dip value or vice-versa.
    rows = [
        _bar(100, 101, 99, 100.0),   # 0 signal
        _bar(100, 101, 99, 100.0),   # 1 entry @ open 100
        _bar(100, 101, 99, 100.0),   # 2 close 100 == entry → marked equity exactly 1.0
        _bar(100, 101, 90, 90.0),    # 3 dip; time-stop exit here
        _bar(90, 91, 89, 90.0),
        _bar(90, 91, 89, 90.0),
    ]
    df = _df(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[0] = True
    atr_s = mod.atr(df, 14).to_numpy()
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("time", bars=3)  # exit @ bar 3
    res = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last, bars_per_yr=1e5, n_years=0.01)
    assert res.n == 1
    # bar 2 marked at close/entry = 100/100 = 1.0, MINUS no cost yet (cost applied at exit) → 1.0
    assert res.equity[2] == pytest.approx(1.0)
    # the curve is NaN-free and monotonic in fill (no clobbered breakeven)
    assert not np.isnan(res.equity).any()


def test_score_start_excludes_warmup_from_metrics(mod):
    """score_start drops leading warm-up bars from exposure + equity risk metrics (codex P3)."""
    # 10 flat warm-up bars (no trade), then one winning trade in the scored region.
    rows = [_bar(100, 101, 99, 100.0) for _ in range(10)]
    rows += [_bar(100, 101, 99, 100.0), _bar(100, 101, 99, 100.0)]  # signal, entry
    rows += [_bar(100, 110, 100, 109.0) for _ in range(4)]          # run up
    df = _df(rows)
    entries = np.zeros(len(df), dtype=bool)
    entries[10] = True  # signal in the scored region; entry at bar 11
    atr_s = mod.atr(df, 14).to_numpy()
    vwma_s = mod.vwma(df, 4).to_numpy()
    sess_last = mod.session_last_bar(df)
    xc = mod.ExitConfig("time", bars=3)
    full = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last,
                        bars_per_yr=1e5, n_years=0.01, score_start=0)
    scored = mod.simulate(df, entries, xc, atr_s, vwma_s, sess_last,
                          bars_per_yr=1e5, n_years=0.01, score_start=10)
    # excluding 10 flat warm-up bars raises exposure (same held bars / smaller denominator)
    assert scored.exposure > full.exposure
    # total_return (trade-based) is unchanged by score_start
    assert scored.total_return == pytest.approx(full.total_return)
