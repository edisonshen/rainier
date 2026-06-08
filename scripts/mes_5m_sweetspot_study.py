"""MES 5-minute "sweet spot" study — decode the operator's discretionary chart
stack into backtestable signals, then find the risk-adjusted-best confluence config.

Scope: EXPLORATORY (one PR for the artifact, no production wiring). Self-contained —
pandas / numpy / plotly only, reads the local MES 5-minute CSV, writes an HTML report.
The signal logic is structured (one clean function per indicator) so it can be promoted
into a real FractalSignalEmitter later — but this script does NOT build that emitter.

The operator trades MES on a 5-minute chart with a discretionary indicator stack
(decoded from their actual Pine source + chart screenshots):

  (1) Fractal long  — bullish bottom-reversal triangle (Pine v4 `fractalUpTrend`).
  (2) MA ribbon     — sma5/ema25/sma22/sma44/sma120/sma200/sma233 + BB(20,2): trend context.
  (3) VWAP 55       — a 55-bar ROLLING VWMA (NOT session VWAP): dynamic S/R.
  (4) 九轉序列【老貓】 — DeMark TD Sequential SETUP count (the 7/9 numbers): exhaustion timing.
  (5) VRVP          — visible-range volume profile: discretionary S/R. Approximated with a
                      rolling volume-by-price high-volume-node proximity flag and wired as an
                      OPTIONAL add-on on the anchor configs (its marginal effect is measured,
                      but it is not crossed with every combo — it is the lowest-priority signal).

What it does:
  1. Load + clean MES_5m (drop zero-range / zero-volume phantom bars, dedup, sort).
  2. Faithfully reproduce each signal as a pure function (unit-tested separately).
  3. Backtest the fractal entry ALONE (baseline) and in CONFLUENCE combinations
     (fractal + price>vma55 + ribbon-bullish + TD9-buy-context), reusing the exit
     families from fractal_backtest_study.py (ATR×R:R, time-stop, VWMA-cross).
  4. Sweep the grid, rank by RISK-ADJUSTED performance (Ret/DD, Sharpe, PF) with a
     trade-count floor, walk-forward validate the top config, benchmark vs buy-and-hold.
  5. Emit docs/RESEARCH-mes-5m-sweetspot.html companion render (the .md is the
     committed source-of-truth; this HTML report is the live render of the numbers).

Position model (single-instrument timing, directly comparable to buy-and-hold):
  - One position at a time (no pyramiding); signals while in a trade are skipped.
  - Long-only (the fractal is a long-only bottom-reversal signal).
  - Each trade deploys full equity (100% in/out). Trade return = (exit/entry - 1)
    minus round-trip cost.
  - ENTER on the NEXT bar's open after the signal-bar close (no same-close fills —
    honoring the repo's r2 timing fix). EXITS that trigger intrabar fill at the
    level; end-of-session and end-of-data flatten at close.

Honest caveats baked into the report:
  - 5-month sample → limited regime diversity → high overfitting risk.
  - 24h futures data; RTH-session-awareness applied to the time-of-day filter and to
    end-of-session flattening, but the Pine TD-count and VWMA are ROLLING (not
    session-anchored) so they are NOT reset per session — preserved as written.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "csv" / "MES_5m.csv"
OUT_HTML = ROOT / "docs" / "RESEARCH-mes-5m-sweetspot.html"

# MES is liquid; round-trip slippage+commission ~0.5-1.0 index point. At ~7000 index
# level that is ~0.7-1.4 bps. We use 1.0 point round-trip as the assumption.
ROUND_TRIP_PTS = 1.0
POINT_VALUE = 5.0  # MES = $5 per 1.00 index point per contract

# Robust-sample floor: below this, win-rate / PF / return are noise on a 5-month window.
MIN_TRADES = 30
# Headline sweet-spot bar: a config must clear this trade count to be the *featured* pick,
# so the single highest Ret/DD config (which can be a thin ~30-trade fluke) is not crowned.
ROBUST_TRADES = 60

# Regular Trading Hours for the S&P = 09:30-16:00 ET. The CSV is UTC; `in_rth` converts to
# US/Eastern and gates on ET clock time, so it is correct across the sample's EST→EDT change
# (no fixed UTC offset, which would be wrong by an hour for the EST portion).


# ---------------------------------------------------------------------------
# Formatters + shared HTML chrome (house aesthetic — matches the docs/REPORT-* renders)
# ---------------------------------------------------------------------------

def pct(x: float) -> str:
    if x == float("inf"):
        return "∞"
    if not np.isfinite(x):
        return "—"
    return f"{x * 100:+.1f}%"


def num(x: float) -> str:
    if x == float("inf"):
        return "∞"
    if not np.isfinite(x):
        return "—"
    return f"{x:.2f}"


PAGE_HEAD = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MES 5-minute — Sweet-Spot Study</title>
<style>
:root{--bg:#0b0e14;--panel:#11151f;--panel2:#161b27;--ink:#d7dce5;--muted:#8b93a7;
--line:#222a3a;--green:#3ddc97;--red:#e0566f;--amber:#e3b341;--blue:#5aa9ff;--accent:#7c9cff;
--mono:"SF Mono",ui-monospace,Menlo,Consolas,monospace;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6;font-size:16px}
main{max-width:1000px;margin:0 auto;padding:40px 32px}
h1{font-size:30px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:14px;margin:0 0 26px}
h2.sec{font-size:21px;margin:42px 0 12px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:24px 0 6px}
p{margin:11px 0}.muted{color:var(--muted);font-size:13.5px}
code{font-family:var(--mono);background:#0e1320;padding:1.5px 6px;border-radius:5px;font-size:13px;color:#cdd6f4}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left}
th{background:var(--panel2);color:#eef1f6;font-weight:600}
td.r,th.r{text-align:right;font-family:var(--mono)}
.pos{color:var(--green)}.neg{color:var(--red)}
.box{border:1px solid var(--line);border-radius:12px;padding:18px 22px;margin:18px 0;background:var(--panel)}
.box.key{border-left:3px solid var(--green);background:#0e1714}
.box.warn{border-left:3px solid var(--amber);background:#1a1710}
.box h4{margin:0 0 8px;font-size:13px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}
.stat{border:1px solid var(--line);border-radius:10px;padding:14px 16px;background:var(--panel)}
.stat .v{font-family:var(--mono);font-size:24px;font-weight:600}
.stat .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em;margin-top:3px}
svg{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin-top:10px}
.legend{font-size:12px;color:var(--muted);margin-top:6px}.legend b{font-family:var(--mono)}
ul{margin:11px 0}
.foot{color:var(--muted);font-size:12px;margin-top:40px;border-top:1px solid var(--line);padding-top:16px}
</style></head><body><main>"""


# ---------------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------------

def load_clean_5m(path: Path) -> pd.DataFrame:
    """Load MES 5-minute bars; drop zero-range and zero-volume phantom bars, dedup, sort.

    Per project memory ([[project_mes_csv_dirty]]) the MES CSVs carry zero-range junk
    (open==high==low==close) and (here) zero-volume edge bars. Both inject false
    highs/lows and break volume-dependent signals (the fractal's volume gate, the VWMA).
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.drop_duplicates("timestamp", keep="last")
    df = df[(df["high"] - df["low"]) > 0]  # drop zero-range junk
    df = df[df["volume"] > 0]              # drop zero-volume edge bars (VWMA needs volume)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# Indicators (pure functions — unit-tested)
# ---------------------------------------------------------------------------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def vwma(df: pd.DataFrame, n: int = 55) -> pd.Series:
    """55-bar ROLLING volume-weighted moving average of HLC3 — the operator's "VWAP 55".

    NOT a session-anchored VWAP: it is a rolling window, so it does NOT reset per
    session. Preserved as the Pine v5 writes it: sum(hlc3*vol, n) / sum(vol, n).
    """
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    num_ = (hlc3 * df["volume"]).rolling(n).sum()
    den = df["volume"].rolling(n).sum()
    return num_ / den


def fractal_up_trend(df: pd.DataFrame) -> np.ndarray:
    """Reproduce the Pine v4 `fractalUpTrend` — the bullish bottom-reversal BUY triangle.

    Pine source (decoded), evaluated at the close of the current bar t. `[k]` = k bars ago:

        sma6Volume = sma(volume,6)
        fractalVolumeChange = (volume - sma6Volume)/sma6Volume*100
        fractalsDown = low[3]<low[4] and low[4]<low[5] and low[2]>low[3] and low[1]>low[2]
                       and sma6Volume[2] > fractalVolumeChange
        ema12 = ema(close,5); ema26 = ema(close,11)        // NOTE 5/11, not 12/26
        strongFractal = close>open[1] and close[1]>open[1] and close[2]>open[2]
                        and abs(close[1]-open[1])>abs(close[3]-open[3])
                        and high>high[3] and ema12[3]<ema26[3]
        fractalUpTrend = fractalsDown and strongFractal

    Returns a bool array, True at the SIGNAL bar t (caller enters next bar's open).
    """
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    vol = df["volume"].to_numpy()
    n = len(df)

    sma6vol = sma(df["volume"], 6).to_numpy()
    # (volume - sma6Volume)/sma6Volume*100 ; guard div-by-zero (cleaned data has vol>0,
    # but the 6-bar SMA can still be 0 only if a window is all-zero — impossible post-clean;
    # guard anyway for safety / for synthetic test fixtures).
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_vol_change = np.where(sma6vol > 0, (vol - sma6vol) / sma6vol * 100.0, np.nan)

    e_fast = ema(df["close"], 5).to_numpy()   # Pine "ema12"
    e_slow = ema(df["close"], 11).to_numpy()  # Pine "ema26"

    sig = np.zeros(n, dtype=bool)
    for t in range(5, n):
        # fractalsDown — a down-fractal shape in the lows over t-5..t-1 + volume gate.
        fractals_down = (
            low[t - 3] < low[t - 4]
            and low[t - 4] < low[t - 5]
            and low[t - 2] > low[t - 3]
            and low[t - 1] > low[t - 2]
            and np.isfinite(sma6vol[t - 2])
            and np.isfinite(frac_vol_change[t])
            and sma6vol[t - 2] > frac_vol_change[t]
        )
        if not fractals_down:
            continue
        # strongFractal — three-bar bullish thrust + body expansion + new high + prior-downtrend.
        strong = (
            c[t] > o[t - 1]
            and c[t - 1] > o[t - 1]
            and c[t - 2] > o[t - 2]
            and abs(c[t - 1] - o[t - 1]) > abs(c[t - 3] - o[t - 3])
            and h[t] > h[t - 3]
            and np.isfinite(e_fast[t - 3])
            and np.isfinite(e_slow[t - 3])
            and e_fast[t - 3] < e_slow[t - 3]
        )
        if strong:
            sig[t] = True
    return sig


def td_setup_buy(df: pd.DataFrame, completed: int = 9) -> np.ndarray:
    """DeMark TD Sequential bullish SETUP count — the 九轉序列 7/9 numbers.

    Canonical DeMark spec (Thomas DeMark, *The New Science of Technical Analysis*, 1994):
      - A bullish (buy) setup increments while close < close[4]; resets to 0 otherwise.
      - A completed setup is a run of 9 consecutive bars each closing below close 4 bars
        earlier. The completed "9" flags DOWNSIDE EXHAUSTION → long-reversal timing.

    Returns a bool array True ONLY on the bar where the buy-setup count first REACHES
    `completed` (the crossing bar), not on every later bar of a continuing selloff — a "9"
    fires once per setup, per DeMark semantics (default 9; pass 7 for the "7" context). Count
    is ROLLING, NOT reset per session (mirrors the chart's rolling indicator).
    """
    c = df["close"].to_numpy()
    n = len(df)
    out = np.zeros(n, dtype=bool)
    run = 0
    for t in range(n):
        if t >= 4 and c[t] < c[t - 4]:
            run += 1
        else:
            run = 0
        # fire once, on the bar where the run crosses up into `completed`
        if run == completed:
            out[t] = True
    return out


def td_buy_context(df: pd.DataFrame, lookback: int = 6, min_count: int = 7) -> np.ndarray:
    """True at bar t if a buy-setup COMPLETION (the count crossing up through `min_count`)
    occurred within the last `lookback` bars — i.e. recent downside exhaustion, the timing
    window the operator buys into.

    We roll the COMPLETION EVENT, not the raw running count. Using `count >= min_count` would
    stay true for every bar of a long selloff (the count keeps climbing past min_count), which
    would admit entries many bars after the actual TD7/9 print. A completion event is the bar
    where the run first reaches `min_count` (count == min_count, the prior bar was below it).
    """
    c = df["close"].to_numpy()
    n = len(df)
    run = 0
    event = np.zeros(n, dtype=bool)
    for t in range(n):
        prev_run = run
        if t >= 4 and c[t] < c[t - 4]:
            run += 1
        else:
            run = 0
        # completion: the run crosses UP through min_count exactly once per setup
        if run == min_count and prev_run == min_count - 1:
            event[t] = True
    out = np.zeros(n, dtype=bool)
    for t in range(n):
        lo = max(0, t - lookback)
        if event[lo : t + 1].any():
            out[t] = True
    return out


def ribbon_bullish(df: pd.DataFrame) -> np.ndarray:
    """MA-ribbon TREND-CONTEXT filter: price above sma44 AND sma44 rising.

    The operator's ribbon is sma5/ema25/sma22/sma44/sma120/sma200/sma233. The discretionary
    "ribbon is bullish" read is: price riding above the mid ribbon and the ribbon sloping up.
    We encode the robust, simple version: close > sma44 and sma44[t] > sma44[t-1]. (sma44 is
    the operator's "MA60" label — the mid-trend line price pulls back to and reclaims.)
    """
    c = df["close"].to_numpy()
    s44 = sma(df["close"], 44).to_numpy()
    n = len(df)
    out = np.zeros(n, dtype=bool)
    for t in range(1, n):
        if np.isfinite(s44[t]) and np.isfinite(s44[t - 1]):
            out[t] = c[t] > s44[t] and s44[t] > s44[t - 1]
    return out


def ribbon_stacked(df: pd.DataFrame) -> np.ndarray:
    """Stricter ribbon read: stacked bullish sma22 > sma44 > sma120 (short over mid over long)."""
    s22 = sma(df["close"], 22).to_numpy()
    s44 = sma(df["close"], 44).to_numpy()
    s120 = sma(df["close"], 120).to_numpy()
    n = len(df)
    out = np.zeros(n, dtype=bool)
    for t in range(n):
        if np.isfinite(s22[t]) and np.isfinite(s44[t]) and np.isfinite(s120[t]):
            out[t] = s22[t] > s44[t] > s120[t]
    return out


def above_vwma(df: pd.DataFrame, n: int = 55) -> np.ndarray:
    """True where close >= the rolling VWMA55 (price reclaiming / holding the dynamic S/R)."""
    v = vwma(df, n).to_numpy()
    c = df["close"].to_numpy()
    return np.where(np.isfinite(v), c >= v, False)


def hvn_proximity(df: pd.DataFrame, window: int = 240, bins: int = 24,
                  top_frac: float = 0.30) -> np.ndarray:
    """VRVP approximation: True where price sits near a high-volume node (HVN).

    Rolling volume-by-price histogram over the last `window` bars (≈ a session-ish visible
    range). Bins the close range, sums volume per bin, flags the bins in the top
    `top_frac` of volume as HVNs; True at t if close[t] falls in an HVN bin. LOWER priority
    (VRVP is discretionary) — wired as an OPTIONAL add-on on the anchor configs via
    `EntryConfig.require_hvn`, not crossed with every combo.
    """
    c = df["close"].to_numpy()
    vol = df["volume"].to_numpy()
    n = len(df)
    out = np.zeros(n, dtype=bool)
    for t in range(window, n):
        seg_c = c[t - window : t]
        seg_v = vol[t - window : t]
        lo, hi = seg_c.min(), seg_c.max()
        if hi <= lo:
            continue
        edges = np.linspace(lo, hi, bins + 1)
        idx = np.clip(np.digitize(seg_c, edges) - 1, 0, bins - 1)
        vol_by_bin = np.zeros(bins)
        np.add.at(vol_by_bin, idx, seg_v)
        # If the current close is OUTSIDE the measured price range, it is a breakout beyond
        # the visible volume profile — there is no measured node there, so it is NOT an HVN.
        # (Clipping it into the nearest edge bin would falsely tag breakouts as support.)
        if c[t] < lo or c[t] > hi:
            continue
        thresh = np.quantile(vol_by_bin[vol_by_bin > 0], 1.0 - top_frac) if (vol_by_bin > 0).any() else np.inf
        price_bin = int(np.clip(np.digitize([c[t]], edges)[0] - 1, 0, bins - 1))
        out[t] = vol_by_bin[price_bin] >= thresh
    return out


def in_rth(df: pd.DataFrame) -> np.ndarray:
    """RTH (US cash session) filter — TIMEZONE-AWARE so it tracks the EST↔EDT change.

    Converts UTC timestamps to US/Eastern and gates on 09:30 ≤ ET < 16:00 (regular S&P cash
    hours). This is correct for BOTH the EST (Jan–early Mar: cash = 14:30–21:00 UTC) and the
    EDT (Mar–Jun: cash = 13:30–20:00 UTC) portions of the sample — a fixed UTC window would be
    wrong by an hour for ~40% of the data.
    """
    ts_utc = df["timestamp"]
    if ts_utc.dt.tz is None:
        ts_utc = ts_utc.dt.tz_localize("UTC")
    et = ts_utc.dt.tz_convert("America/New_York")
    minute_of_day = et.dt.hour.to_numpy() * 60 + et.dt.minute.to_numpy()
    open_min = 9 * 60 + 30   # 09:30 ET
    close_min = 16 * 60      # 16:00 ET
    return (minute_of_day >= open_min) & (minute_of_day < close_min)


def session_last_bar(df: pd.DataFrame) -> np.ndarray:
    """True at the last bar of each CME equity-futures trading day (for end-of-session flatten).

    The CME equity-index session runs 18:00 ET → 17:00 ET next day (with a 17:00–18:00 ET
    break). The trading "day" therefore rolls at 17:00 ET, NOT at 00:00 UTC. Bucketing by UTC
    calendar date would force flats around 00:00 UTC mid-session and hold across the real
    break; we bucket by ET trading-day (a bar at/after 17:00 ET belongs to the next day) so the
    flatten lands at the actual session close.
    """
    ts_utc = df["timestamp"]
    if ts_utc.dt.tz is None:
        ts_utc = ts_utc.dt.tz_localize("UTC")
    et = ts_utc.dt.tz_convert("America/New_York")
    # shift forward 7h so 17:00 ET maps to 00:00 of the next trading day → date() buckets it
    trading_day = (et + pd.Timedelta(hours=7)).dt.date
    td = pd.Series(trading_day, index=df.index)
    return (td != td.shift(-1)).to_numpy()


def rth_last_bar(df: pd.DataFrame) -> np.ndarray:
    """True at the last RTH bar of each day — the bar after which `in_rth` turns False.

    Used as an additional flatten boundary for RTH-gated configs so an RTH entry is closed
    at the cash-session close (~20:00 UTC), NOT held for hours into the overnight book.
    """
    rth = in_rth(df)
    nxt = np.roll(rth, -1)
    nxt[-1] = False
    return rth & ~nxt


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntryConfig:
    """Confluence filters layered on top of the base fractalUpTrend entry."""
    require_vwma: bool
    require_ribbon: bool      # close>sma44 & sma44 rising
    require_stacked: bool     # sma22>sma44>sma120 (stricter; implies a trend)
    require_td: bool          # recent TD buy-setup (>=7) within lookback
    require_rth: bool
    require_hvn: bool = False  # VRVP approximation: price near a high-volume node (optional)

    def label(self) -> str:
        bits = ["fractal"]
        if self.require_vwma:
            bits.append(">vwma55")
        if self.require_ribbon:
            bits.append("ribbon↑")
        if self.require_stacked:
            bits.append("stacked")
        if self.require_td:
            bits.append("TD7")
        if self.require_rth:
            bits.append("RTH")
        if self.require_hvn:
            bits.append("HVN")
        return "+".join(bits)


@dataclass(frozen=True)
class ExitConfig:
    kind: str            # 'atr' | 'time' | 'vwma'
    atr_k: float = 0.0   # stop = entry - k*ATR
    rr: float = 0.0      # target = entry + rr*(entry-stop)
    bars: int = 0        # time-stop hold length (5m bars)

    def label(self) -> str:
        if self.kind == "atr":
            return f"ATR{self.atr_k:g}×RR{self.rr:g}"
        if self.kind == "time":
            return f"time{self.bars}"
        return "vwmaCross"


# ---------------------------------------------------------------------------
# Entry assembly
# ---------------------------------------------------------------------------

def compute_entries(df: pd.DataFrame, ec: EntryConfig,
                    cache: dict | None = None) -> np.ndarray:
    """Base fractalUpTrend AND all requested confluence filters → entry-trigger bars."""
    cache = cache if cache is not None else {}

    def get(key, fn):
        if key not in cache:
            cache[key] = fn()
        return cache[key]

    sig = get("fractal", lambda: fractal_up_trend(df)).copy()
    if ec.require_vwma:
        sig &= get("vwma", lambda: above_vwma(df, 55))
    if ec.require_ribbon:
        sig &= get("ribbon", lambda: ribbon_bullish(df))
    if ec.require_stacked:
        sig &= get("stacked", lambda: ribbon_stacked(df))
    if ec.require_td:
        sig &= get("td", lambda: td_buy_context(df, lookback=6, min_count=7))
    if ec.require_rth:
        rth = get("rth", lambda: in_rth(df))
        # The entry fills NEXT bar (t+1), so the entry bar must also be in RTH — otherwise an
        # entry on the last RTH bar (e.g. 19:55) fills at 20:00 (outside RTH) and is never
        # flattened at the cash close. Require both the signal bar AND its successor in RTH.
        next_rth = np.roll(rth, -1)
        next_rth[-1] = False
        sig &= rth & next_rth
    if ec.require_hvn:
        sig &= get("hvn", lambda: hvn_proximity(df))
    return sig


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    ret: float       # net of cost, fractional
    reason: str


@dataclass
class Result:
    trades: list[Trade]
    equity: np.ndarray
    n: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    mar: float = 0.0       # window total-return / max-DD (NOT annualized — see note)
    max_dd: float = 0.0
    avg_hold: float = 0.0
    exposure: float = 0.0
    total_pts: float = 0.0


def simulate(df: pd.DataFrame, entries: np.ndarray, xc: ExitConfig,
             atr_series: np.ndarray, vwma_series: np.ndarray,
             flatten: np.ndarray, bars_per_yr: float, n_years: float,
             score_start: int = 0) -> Result:
    """Event-driven long-only single-position sim.

    Timing (no lookahead): signal confirms at close[t]; ENTER at open[t+1]. Exits:
      - atr  : intrabar stop/target fill at the level on a later bar.
      - time : exit at close after `bars` 5m bars.
      - vwma : exit at close when close < VWMA55.
    Any open position is force-closed at the bar's close on a `flatten[j]` bar (no overnight
    hold) and at end-of-data. `flatten` = session-end bars, plus the RTH-close bars for
    RTH-gated configs (so an RTH entry is closed at the cash-session close, not held into the
    overnight book). Cost: ROUND_TRIP_PTS index points per round-trip (fractional on entry).
    """
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    n = len(df)

    equity = np.ones(n)
    eq = 1.0
    trades: list[Trade] = []
    held_bars = 0

    t = 0
    while t < n - 1:
        if not entries[t]:
            equity[t] = eq
            t += 1
            continue

        entry_bar = t + 1            # enter NEXT bar's open (no same-close fill)
        entry_price = o[entry_bar]
        stop = tp = None
        if xc.kind == "atr":
            a = atr_series[t]
            if not np.isfinite(a) or a <= 0:
                equity[t] = eq
                t += 1
                continue
            stop = entry_price - xc.atr_k * a
            tp = entry_price + xc.rr * (entry_price - stop)

        exit_bar = None
        exit_price = None
        reason = ""
        j = entry_bar
        while j < n:
            # Intrabar risk (ATR stop/target) takes precedence over the close-based
            # exits, INCLUDING the session-end flatten: a stop touched on the session's
            # last bar must fill at the stop, not at that bar's close. Within a bar a
            # stop is assumed hit before a target (conservative).
            if xc.kind == "atr":
                if low[j] <= stop:
                    exit_bar, exit_price, reason = j, min(o[j], stop), "stop"
                    break
                if h[j] >= tp:
                    exit_bar, exit_price, reason = j, max(o[j], tp), "target"
                    break
            if flatten[j]:  # force-close at session end / RTH close (covers gap/overnight)
                exit_bar, exit_price, reason = j, c[j], "session_end"
                break
            if xc.kind == "time":
                # Hold exactly `bars` 5-minute bars: the entry bar (filled at its open) is
                # held bar #1, so exit at the close of bar entry_bar + bars - 1. `time6` thus
                # holds 6 bars / 30 minutes, matching its label.
                if j - entry_bar >= xc.bars - 1:
                    exit_bar, exit_price, reason = j, c[j], "time"
                    break
            elif xc.kind == "vwma":
                if np.isfinite(vwma_series[j]) and c[j] < vwma_series[j]:
                    exit_bar, exit_price, reason = j, c[j], "vwmaCross"
                    break
            j += 1

        if exit_bar is None:
            exit_bar, exit_price, reason = n - 1, c[-1], "end_of_data"

        gross = exit_price / entry_price - 1.0
        cost = ROUND_TRIP_PTS / entry_price
        net = gross - cost
        eq_before = eq
        eq = eq_before * (1.0 + net)
        trades.append(Trade(entry_bar, exit_bar, entry_price, exit_price, net, reason))
        held_bars += (exit_bar - entry_bar + 1)  # entry bar is held bar #1

        # MARK-TO-MARKET the equity curve through the hold so in-trade drawdown is captured
        # (max-DD / Sharpe must see adverse excursion, not just the final outcome). Pre-entry
        # flat bars [t, entry_bar) hold eq_before; held bars [entry_bar, exit_bar) mark at the
        # bar close vs entry; the exit bar takes the final cost-adjusted equity.
        for k in range(t, entry_bar):
            equity[k] = eq_before
        for k in range(entry_bar, exit_bar):
            equity[k] = eq_before * (c[k] / entry_price)
        equity[exit_bar] = eq
        t = exit_bar + 1

    # forward-fill any remaining flat bars to the last equity level
    prev = 1.0
    for k in range(n):
        if equity[k] == 1.0 and prev != 1.0:
            equity[k] = prev
        prev = equity[k]

    res = Result(trades=trades, equity=equity)
    _finalize(res, df, bars_per_yr, n_years, n, held_bars, score_start)
    return res


def _finalize(res: Result, df: pd.DataFrame, bars_per_yr: float, n_years: float,
              n_bars: int, held_bars: int, score_start: int = 0) -> None:
    """Compute aggregate stats. `score_start` excludes leading warm-up bars (which carry no
    trades and flat equity) from the equity-based risk metrics + exposure denominator, so a
    warm-up buffer prepended for indicator state does not dilute Sharpe / max-DD / exposure."""
    tr = res.trades
    res.n = len(tr)
    if res.n == 0:
        res.equity = np.ones(n_bars)
        return
    rets = np.array([x.ret for x in tr])
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    res.win_rate = len(wins) / res.n
    gw = wins.sum()
    gl = abs(losses.sum())
    res.profit_factor = (gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0)
    res.expectancy = float(rets.mean())
    res.total_return = float(np.prod(1.0 + rets) - 1.0)
    res.cagr = (1.0 + res.total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    res.avg_hold = held_bars / res.n
    res.exposure = held_bars / max(1, n_bars - score_start)
    res.total_pts = float(sum((x.exit_price - x.entry_price) - ROUND_TRIP_PTS for x in tr))

    # equity-based risk metrics over the SCORED region only (drop warm-up bars), rebased to 1.0
    eq_full = res.equity[score_start:]
    eq = eq_full / eq_full[0] if len(eq_full) and eq_full[0] != 0 else eq_full
    dr = np.diff(eq) / eq[:-1]
    dr = dr[np.isfinite(dr)]
    if len(dr) > 1 and dr.std() > 0:
        res.sharpe = float(dr.mean() / dr.std() * math.sqrt(bars_per_yr))
    peak = -np.inf
    mdd = 0.0
    for v in eq:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    res.max_dd = mdd
    res.mar = (res.total_return / mdd) if mdd > 0 else (float("inf") if res.total_return > 0 else 0.0)


# ---------------------------------------------------------------------------
# Buy & hold baseline
# ---------------------------------------------------------------------------

def buy_and_hold(df: pd.DataFrame, bars_per_yr: float, n_years: float) -> Result:
    c = df["close"].to_numpy()
    eq = c / c[0]
    res = Result(trades=[], equity=eq)
    res.n = 1
    res.total_return = float(c[-1] / c[0] - 1.0)
    res.cagr = (1.0 + res.total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    dr = np.diff(eq) / eq[:-1]
    if dr.std() > 0:
        res.sharpe = float(dr.mean() / dr.std() * math.sqrt(bars_per_yr))
    peak = -np.inf
    mdd = 0.0
    for v in eq:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)
    res.max_dd = mdd
    res.mar = (res.total_return / mdd) if mdd > 0 else (float("inf") if res.total_return > 0 else 0.0)
    res.exposure = 1.0
    res.total_pts = float(c[-1] - c[0])
    return res


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def entry_grid() -> list[EntryConfig]:
    """Baseline (bare fractal) + the confluence layers the screenshots suggested.

    require_stacked implies a trend, so we don't combine it with require_ribbon (redundant);
    require_rth is an orthogonal time-of-day filter toggled across the meaningful combos.
    The VRVP/HVN approximation is the lowest-priority signal, so it is NOT crossed with every
    combo (that would double the grid); it is added as an optional add-on on the bare fractal
    and on the full-confluence config so its marginal effect is genuinely measured.
    """
    out: list[EntryConfig] = []
    base_combos = [
        # (vwma, ribbon, stacked, td)
        (False, False, False, False),  # bare fractal (baseline)
        (True, False, False, False),   # + price>vwma55
        (False, True, False, False),   # + ribbon rising
        (True, True, False, False),    # + vwma + ribbon
        (False, False, True, False),   # + stacked ribbon
        (True, False, True, False),    # + vwma + stacked
        (False, False, False, True),   # + TD buy context
        (True, True, False, True),     # + vwma + ribbon + TD (full confluence)
        (True, False, True, True),     # + vwma + stacked + TD
    ]
    for (v, r, s, td), rth in itertools.product(base_combos, [False, True]):
        out.append(EntryConfig(v, r, s, td, rth, require_hvn=False))
    # Optional VRVP/HVN add-on on the two anchor configs (bare + full confluence), RTH off/on.
    for (v, r, s, td) in [(False, False, False, False), (True, True, False, True)]:
        for rth in (False, True):
            out.append(EntryConfig(v, r, s, td, rth, require_hvn=True))
    return out


def exit_grid() -> list[ExitConfig]:
    out: list[ExitConfig] = []
    for k in (1.0, 1.5, 2.0):
        for r in (1.0, 1.5, 2.0, 3.0):
            out.append(ExitConfig("atr", atr_k=k, rr=r))
    for nb in (6, 12, 24, 48):  # ~30min, 1h, 2h, 4h on 5m bars
        out.append(ExitConfig("time", bars=nb))
    out.append(ExitConfig("vwma"))
    return out


@dataclass
class Row:
    entry: EntryConfig
    exit: ExitConfig
    res: Result


def run_grid(df: pd.DataFrame, bars_per_yr: float, n_years: float,
             score_start: int = 0) -> list[Row]:
    """Sweep every entry×exit config. `score_start` (>0 when `df` carries a leading warm-up
    buffer) masks entries before it and excludes warm-up bars from the risk metrics, so a
    warmed grid is comparable to a warmed single run."""
    atr_s = atr(df, 14).to_numpy()
    vwma_s = vwma(df, 55).to_numpy()
    sess_last = session_last_bar(df)
    rth_last = rth_last_bar(df)
    cache: dict = {}
    rows: list[Row] = []
    exits = exit_grid()
    for ec in entry_grid():
        entries = compute_entries(df, ec, cache)
        if score_start > 0:
            entries = entries.copy()
            entries[:score_start] = False
        if entries.sum() == 0:
            continue
        # RTH-gated configs also flatten at the cash-session close, not just session end.
        flatten = (sess_last | rth_last) if ec.require_rth else sess_last
        for xc in exits:
            res = simulate(df, entries, xc, atr_s, vwma_s, flatten,
                           bars_per_yr, n_years, score_start=score_start)
            if res.n > 0:
                rows.append(Row(ec, xc, res))
    return rows


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def _mar_key(r: Row) -> float:
    """Rank key: finite Ret/DD ratio; treat inf (zero-DD) as large-but-finite so a real
    drawdown-bearing winner isn't beaten by a tiny no-DD fluke."""
    cal = r.res.mar
    if not np.isfinite(cal):
        return 1e6 if cal > 0 else -1e6
    return cal


# Longest composed lookback across all signals (sma233, hvn window 240) — the OOS frozen-config
# run prepends this many pre-split bars so rolling indicators are warmed, matching a real
# forward test (where history before the split IS available). Warm-up bars never trade.
WARMUP_BARS = 240


def walk_forward(df: pd.DataFrame, bars_per_yr: float):
    n = len(df)
    split = int(n * 0.6)
    is_df = df.iloc[:split].reset_index(drop=True)
    oos_df = df.iloc[split:].reset_index(drop=True)
    is_years = len(is_df) / bars_per_yr
    oos_years = len(oos_df) / bars_per_yr

    is_rows = run_grid(is_df, bars_per_yr, is_years)
    if not is_rows:
        return None
    eligible = [r for r in is_rows if r.res.n >= MIN_TRADES] or \
               [r for r in is_rows if r.res.n >= 10] or is_rows
    is_best = max(eligible, key=_mar_key)

    # Run that exact config untouched on OOS, but warm the rolling indicators from pre-split
    # history (a real forward test would have it). Build a frame = [warm-up | OOS], compute on
    # it, then zero out any entry inside the warm-up region so warm-up bars never trade and
    # only post-split trades are scored.
    warm_start = max(0, split - WARMUP_BARS)
    warm_len = split - warm_start
    oos_warm = df.iloc[warm_start:].reset_index(drop=True)
    atr_o = atr(oos_warm, 14).to_numpy()
    vwma_o = vwma(oos_warm, 55).to_numpy()
    sess_o = session_last_bar(oos_warm)
    flatten_o = (sess_o | rth_last_bar(oos_warm)) if is_best.entry.require_rth else sess_o
    entries = compute_entries(oos_warm, is_best.entry)
    entries[:warm_len] = False  # no entries before the true OOS start
    # score_start=warm_len so the warm-up bars are excluded from the OOS risk metrics.
    oos_res = simulate(oos_warm, entries, is_best.exit, atr_o, vwma_o, flatten_o,
                       bars_per_yr, oos_years, score_start=warm_len)

    # best-with-hindsight on OOS (the honesty benchmark) — warmed the SAME way as the frozen
    # run (oos_warm + score_start) so the two rows compare like with like, not warmed-vs-cold.
    oos_rows = run_grid(oos_warm, bars_per_yr, oos_years, score_start=warm_len)
    oos_eligible = [r for r in oos_rows if r.res.n >= MIN_TRADES] or \
                   [r for r in oos_rows if r.res.n >= 10] or oos_rows
    oos_best = max(oos_eligible, key=_mar_key)

    label = f"{is_best.entry.label()} · {is_best.exit.label()}"
    return ((label, is_best.res), oos_res,
            (f"{oos_best.entry.label()} · {oos_best.exit.label()}", oos_best.res))


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _row_cells(r: Row) -> str:
    x = r.res
    return (
        f"<td>{r.entry.label()}</td><td>{r.exit.label()}</td>"
        f"<td class='r'>{x.n}</td><td class='r'>{x.win_rate*100:.0f}%</td>"
        f"<td class='r'>{num(x.profit_factor)}</td>"
        f"<td class='r {'pos' if x.total_return>0 else 'neg'}'>{pct(x.total_return)}</td>"
        f"<td class='r'>{num(x.mar)}</td><td class='r'>{num(x.sharpe)}</td>"
        f"<td class='r'>{pct(x.max_dd)}</td><td class='r'>{x.exposure*100:.0f}%</td>"
    )


def _spark(series, color, scale=None):
    """Polyline for `series`. `scale=(mn,mx)` forces a shared y-range so two overlaid
    series are visually comparable (default: self-scale)."""
    if len(series) < 2:
        return ""
    mn, mx = scale if scale is not None else (min(series), max(series))
    rng = (mx - mn) or 1
    w, h = 760, 200
    pts = " ".join(
        f"{i/(len(series)-1)*w:.1f},{h-(v-mn)/rng*h:.1f}"
        for i, v in enumerate(series))
    return f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{pts}'/>"


def pick_sweet_spot(rows: list[Row]) -> Row:
    """The HEADLINE sweet spot: best Ret/DD among configs that are NOT a trade-count outlier.

    The single highest-Ret/DD config can be a thin 30-ish-trade fluke; per the brief, the
    sweet spot must be robust AND risk-adjusted-good AND not a count outlier. So we pick the
    best Ret/DD among configs with n >= ROBUST_TRADES (= 2× the floor); if none clear that
    bar, fall back to the floor, then to all rows.
    """
    pool = ([r for r in rows if r.res.n >= ROBUST_TRADES]
            or [r for r in rows if r.res.n >= MIN_TRADES]
            or rows)
    return max(pool, key=_mar_key)


def build_report(df, rows, bh, wf, window, n_years, bars_per_yr,
                 baseline_row, verdict_html, screenshot_html) -> str:
    robust = [r for r in rows if r.res.n >= MIN_TRADES]
    enough = bool(robust)
    ranked_pool = robust if robust else rows
    ranked = sorted(ranked_pool, key=_mar_key, reverse=True)
    best = pick_sweet_spot(rows)  # headline = robust (not the thin top-of-leaderboard fluke)
    top = ranked[:25]
    n_thin = len(rows) - len(robust)

    hl = " style='background:#0e1714'"  # highlight the featured sweet-spot row
    top_rows = "\n".join(
        f"<tr{hl if r is best else ''}>{_row_cells(r)}</tr>" for r in top)

    bh_row = (
        f"<tr><td>Buy &amp; hold MES</td><td>—</td><td class='r'>—</td><td class='r'>—</td>"
        f"<td class='r'>—</td><td class='r {'pos' if bh.total_return>0 else 'neg'}'>{pct(bh.total_return)}</td>"
        f"<td class='r'>{num(bh.mar)}</td><td class='r'>{num(bh.sharpe)}</td>"
        f"<td class='r'>{pct(bh.max_dd)}</td><td class='r'>100%</td></tr>"
    )
    base = baseline_row.res
    base_row = (
        f"<tr><td>{baseline_row.entry.label()} (bare fractal, best exit)</td>"
        f"<td>{baseline_row.exit.label()}</td><td class='r'>{base.n}</td>"
        f"<td class='r'>{base.win_rate*100:.0f}%</td><td class='r'>{num(base.profit_factor)}</td>"
        f"<td class='r {'pos' if base.total_return>0 else 'neg'}'>{pct(base.total_return)}</td>"
        f"<td class='r'>{num(base.mar)}</td><td class='r'>{num(base.sharpe)}</td>"
        f"<td class='r'>{pct(base.max_dd)}</td><td class='r'>{base.exposure*100:.0f}%</td></tr>"
    )

    # equity spark: best confluence vs buy & hold (downsampled)
    we = best.res.equity
    bhe = bh.equity
    step = max(1, len(we) // 240)
    we_pts = [we[i] for i in range(0, len(we), step)]
    bhe_pts = [bhe[i] for i in range(0, len(bhe), step)]
    # shared y-scale so the two equity curves are visually comparable (same axis)
    spark_scale = (min(min(we_pts), min(bhe_pts)), max(max(we_pts), max(bhe_pts)))

    wf_html = ""
    if wf:
        is_r, oos_r, oos_best = wf
        wf_html = f"""
      <h2 class="sec" id="wf">Walk-forward — is the edge real, or in-sample fit?</h2>
      <p>Pick the risk-adjusted-best config (by Ret/DD, with a {MIN_TRADES}-trade floor)
      on the first 60% of bars (<b>in-sample</b>). Then run that exact config, untouched,
      on the last 40% (<b>out-of-sample</b>). If OOS collapses, the leaderboard was fit.</p>
      <table>
        <tr><th>Window</th><th>Config</th><th class="r">Trades</th><th class="r">Win</th>
        <th class="r">PF</th><th class="r">Return</th><th class="r">Ret/DD</th>
        <th class="r">Sharpe</th><th class="r">MaxDD</th></tr>
        <tr><td>In-sample (60%)</td><td>{is_r[0]}</td><td class="r">{is_r[1].n}</td>
        <td class="r">{is_r[1].win_rate*100:.0f}%</td><td class="r">{num(is_r[1].profit_factor)}</td>
        <td class="r {'pos' if is_r[1].total_return>0 else 'neg'}">{pct(is_r[1].total_return)}</td>
        <td class="r">{num(is_r[1].mar)}</td><td class="r">{num(is_r[1].sharpe)}</td>
        <td class="r">{pct(is_r[1].max_dd)}</td></tr>
        <tr><td>→ same config, OOS (40%)</td><td>{is_r[0]}</td><td class="r">{oos_r.n}</td>
        <td class="r">{oos_r.win_rate*100:.0f}%</td><td class="r">{num(oos_r.profit_factor)}</td>
        <td class="r {'pos' if oos_r.total_return>0 else 'neg'}">{pct(oos_r.total_return)}</td>
        <td class="r">{num(oos_r.mar)}</td><td class="r">{num(oos_r.sharpe)}</td>
        <td class="r">{pct(oos_r.max_dd)}</td></tr>
        <tr><td><i>best config fit ON the OOS window</i></td><td><i>{oos_best[0]}</i></td>
        <td class="r"><i>{oos_best[1].n}</i></td><td class="r"><i>{oos_best[1].win_rate*100:.0f}%</i></td>
        <td class="r"><i>{num(oos_best[1].profit_factor)}</i></td>
        <td class="r"><i>{pct(oos_best[1].total_return)}</i></td>
        <td class="r"><i>{num(oos_best[1].mar)}</i></td>
        <td class="r"><i>{num(oos_best[1].sharpe)}</i></td><td class="r"><i>{pct(oos_best[1].max_dd)}</i></td></tr>
      </table>
      <p class="muted">If the OOS "same config" row is far worse than in-sample — and worse
      than the config you'd have picked WITH hindsight on the OOS window — the edge is mostly fit.</p>
    """

    b = best.res
    sample_note = "" if enough else (
        f"<div class='box warn'><h4>⚠ Sample too small to trust</h4>"
        f"<p style='margin:0'>NO config reached {MIN_TRADES} trades on this 5-month window. "
        f"Every number below is a handful-of-trades sample; 100% win rates and ∞ profit "
        f"factors are luck. Directional only.</p></div>")

    return f"""{PAGE_HEAD}
<h1>MES 5-minute — Sweet-Spot Study</h1>
<p class="sub">Operator's discretionary stack (fractal + MA ribbon + VWMA55 + 九轉序列 TD + VRVP),
decoded &amp; backtested · MES · round-trip cost {ROUND_TRIP_PTS:g} pt · {len(df)} bars ·
{window} (~{n_years:.2f}y) · generated {datetime.now(timezone.utc):%Y-%m-%d}</p>

{verdict_html}

{screenshot_html}

<h2 class="sec" id="results">Backtest results — ranked by Ret/DD (risk-adjusted)</h2>
<p class="muted">{len(df)} bars · {len(rows)} config combos · {n_thin} had &lt;{MIN_TRADES} trades (dropped).
Buy &amp; hold here = <b class="{'pos' if bh.total_return>0 else 'neg'}">{pct(bh.total_return)}</b>
(Ret/DD {num(bh.mar)}, Sharpe {num(bh.sharpe)}, MaxDD {pct(bh.max_dd)}).</p>
{sample_note}
<div class="grid">
  <div class="stat"><div class="v {'pos' if b.total_return>0 else 'neg'}">{pct(b.total_return)}</div><div class="k">Best-config return</div></div>
  <div class="stat"><div class="v">{num(b.mar)}</div><div class="k">Ret / MaxDD</div></div>
  <div class="stat"><div class="v">{num(b.sharpe)}</div><div class="k">Sharpe</div></div>
  <div class="stat"><div class="v">{pct(b.max_dd)}</div><div class="k">Max drawdown</div></div>
  <div class="stat"><div class="v">{b.n}</div><div class="k">Trades</div></div>
  <div class="stat"><div class="v">{b.win_rate*100:.0f}%</div><div class="k">Win rate</div></div>
  <div class="stat"><div class="v">{num(b.profit_factor)}</div><div class="k">Profit factor</div></div>
  <div class="stat"><div class="v">{b.exposure*100:.0f}%</div><div class="k">Time in market</div></div>
</div>
<p>Risk-adjusted best: <code>{best.entry.label()}</code> entry · <code>{best.exit.label()}</code> exit
({b.n} trades, {b.total_pts:+.0f} pts ≈ ${b.total_pts*POINT_VALUE:+,.0f}/contract).</p>

<svg viewBox="0 0 760 200" width="100%" height="200">{_spark(we_pts,'#3ddc97',spark_scale)}{_spark(bhe_pts,'#5aa9ff',spark_scale)}</svg>
<div class="legend"><b style="color:#3ddc97">▬</b> best confluence config &nbsp;&nbsp;
<b style="color:#5aa9ff">▬</b> buy &amp; hold MES &nbsp;·&nbsp; {window}</div>

<h3>Top configs ({'n≥'+str(MIN_TRADES) if enough else 'all'}, ranked by Ret/DD)</h3>
<table>
<tr><th>Entry</th><th>Exit</th><th class="r">Trades</th><th class="r">Win</th><th class="r">PF</th>
<th class="r">Return</th><th class="r">Ret/DD</th><th class="r">Sharpe</th><th class="r">MaxDD</th><th class="r">Expo</th></tr>
{top_rows}
{base_row}
{bh_row}
</table>
<p class="muted"><b>Highlighted = the featured sweet spot</b>, chosen as the best Ret/DD among
configs with ≥{ROBUST_TRADES} trades — NOT necessarily the single top row. The literal #1 by
Ret/DD can be a thin ~{MIN_TRADES}-trade config whose ratio is a small-sample fluke; we
deliberately do not crown a trade-count outlier. The "bare fractal" row is the baseline:
confluence filters must beat IT (and buy &amp; hold) to be worth the added complexity.</p>

{wf_html}

<h2 class="sec" id="knobs">What the signals + knobs do</h2>
<table>
<tr><th>Signal / knob</th><th>What it encodes</th></tr>
<tr><td>fractal (base)</td><td>Pine v4 <code>fractalUpTrend</code>: bullish bottom-reversal triangle — down-fractal shape + 3-bar bullish thrust + body expansion + new high + prior EMA(5)&lt;EMA(11). Long-only.</td></tr>
<tr><td>&gt;vwma55</td><td>close ≥ rolling 55-bar VWMA (the operator's "VWAP 55"). Price reclaiming dynamic S/R.</td></tr>
<tr><td>ribbon↑</td><td>close &gt; SMA44 AND SMA44 rising — pulled back to the mid ribbon and the ribbon slopes up.</td></tr>
<tr><td>stacked</td><td>SMA22 &gt; SMA44 &gt; SMA120 — ribbon stacked bullish (a genuine uptrend).</td></tr>
<tr><td>TD7</td><td>a DeMark buy-setup count ≥7 completed within the last 6 bars — recent downside exhaustion.</td></tr>
<tr><td>RTH</td><td>time-of-day filter: only the US cash session (≈13:30-20:00 UTC). Approximate (DST).</td></tr>
<tr><td>exit ATR k×RR</td><td>stop = entry − k·ATR(14); target = entry + RR·(entry−stop). k∈{{1,1.5,2}}, RR∈{{1,1.5,2,3}}.</td></tr>
<tr><td>exit time</td><td>flat after N 5m bars (6/12/24/48 ≈ 30min/1h/2h/4h).</td></tr>
<tr><td>exit vwmaCross</td><td>flat when close falls back below VWMA55.</td></tr>
</table>

<div class="foot">Source: scripts/mes_5m_sweetspot_study.py · signals decoded from the operator's Pine
source + chart screenshots · exploratory study, NOT a productionized emitter. Returns =
single-instrument long-only timing on MES, one position at a time, full equity per trade, net of
{ROUND_TRIP_PTS:g}-pt round-trip cost, positions flattened at session end — NOT leveraged-futures
P&amp;L. 5-month sample = one regime; treat as directional. Past performance is not predictive.</div>
</main></body></html>"""


# ---------------------------------------------------------------------------
# Verdict + screenshot narrative (filled from the numbers + the chart reads)
# ---------------------------------------------------------------------------

def build_verdict(best: Row, baseline: Row, bh: Result, wf) -> str:
    b = best.res
    base = baseline.res
    beats_bh = b.total_return > bh.total_return
    beats_base = _mar_key(best) > _mar_key(baseline)
    oos_note = ""
    if wf:
        _, oos_r, _ = wf
        oos_note = (f"Walk-forward OOS on the held-out 40%: the in-sample winner returned "
                    f"<b class='{'pos' if oos_r.total_return>0 else 'neg'}'>{pct(oos_r.total_return)}</b> "
                    f"(Ret/DD {num(oos_r.mar)}, {oos_r.n} trades) — "
                    + ("the edge survived." if oos_r.total_return > 0 and oos_r.n >= 10
                       else "the edge did NOT hold up; this is in-sample fit, not a tradeable system."))
    return f"""
<div class="box warn"><h4>Read this first</h4>
<p style="margin:0">The fractal Pine script is <b>entry-only</b> — every return here depends on exits
we added. "Return" = a single-instrument long-only timing strategy on MES, one position at a time,
full equity per trade, net of {ROUND_TRIP_PTS:g}-pt round-trip cost, flattened at session end —
<b>not</b> leveraged-futures P&amp;L. The window is <b>~5 months (one regime)</b>: high overfitting
risk. The walk-forward row, not the leaderboard top, is the honest read.</p></div>
<div class="box key"><h4>Bottom line</h4>
<p style="margin:0 0 8px"><b>Sweet spot: <code>{best.entry.label()}</code> entry · <code>{best.exit.label()}</code> exit</b>
— {b.n} trades, {pct(b.total_return)} return, Ret/DD {num(b.mar)}, {b.win_rate*100:.0f}% win,
{b.total_pts:+.0f} pts ≈ ${b.total_pts*POINT_VALUE:+,.0f}/contract.</p>
<ul style="margin:0">
<li>vs <b>buy &amp; hold MES</b> ({pct(bh.total_return)}, Ret/DD {num(bh.mar)}): the sweet spot
{'beats' if beats_bh else 'does NOT beat'} buy-and-hold on raw return, with
{'lower' if b.max_dd < bh.max_dd else 'higher'} drawdown ({pct(b.max_dd)} vs {pct(bh.max_dd)}).</li>
<li>vs <b>bare fractal</b> ({pct(base.total_return)}, Ret/DD {num(base.mar)}): confluence
{'improves' if beats_base else 'does NOT improve'} the risk-adjusted result.
{('This matches the screenshots — a green triangle is worth more WITH trend + VWMA confluence.' if beats_base else 'This CONTRADICTS the screenshots intuition (which said confluence should help): on this single up-trending window the trend/VWMA/TD filters discarded winning dip-buys without removing proportionally more losers. Likely because the filters are a downtrend discriminator and there were few sustained downtrends here to filter — in a genuine bear regime they would probably matter, but this 5-month sample cannot show it.')}</li>
<li>{oos_note}</li>
</ul>
<p style="margin:8px 0 0">Honest caveat: 5 months is one broad regime. Even a config that survives this
walk-forward is one bear market away from a different verdict. Treat the sweet spot as a
<i>hypothesis to forward-test</i>, not a deployable edge.</p></div>
"""


def build_screenshot_section() -> str:
    """Concrete observations from the operator's chart screenshots (read as analysis)."""
    return """
<h2 class="sec" id="screenshots">What the chart screenshots show</h2>
<p>Reading ~7 of the operator's 26 MES 5-minute screenshots, the indicator stack is consistent:
green/red <b>fractal triangles</b> at swing lows/highs, the multi-line <b>MA ribbon</b> (white SMA5
hugging price, then the slower lines fanning out), a heavier white <b>VWMA55</b> line, the
<b>九轉序列 TD count</b> numbers (small 7s and 9s printed at exhaustion points), and the
<b>VRVP volume profile</b> histogram down the left edge. The recurring lesson:</p>
<ul>
<li><b>Winning fractal long (e.g. the steady up-leg screenshots):</b> a green triangle prints at a
swing low <i>while price is at or reclaiming the VWMA55</i> and the ribbon is rising / stacked bullish
(short MAs above long MAs). Price then rides up the ribbon. The TD count often shows a recent 7/9
buy-setup just before — downside exhaustion into the entry.</li>
<li><b>Failed fractal long (the choppy-top &amp; down-leg screenshots):</b> a green triangle prints
but price is <i>below a falling ribbon</i> and below the VWMA55 — a counter-trend "catch the falling
knife" that the ribbon context vetoes. Several reds and greens cluster with no follow-through.</li>
<li><b>VRVP / high-volume nodes:</b> price stalls and reverses at the fat part of the left-edge
histogram (high-volume nodes act as S/R); thin nodes get traversed fast. This is discretionary —
we approximate it with a rolling volume-by-price node-proximity flag and wire it as an optional
add-on on the anchor configs (its marginal effect is measured, but it is the lowest-priority signal).</li>
</ul>
<p class="muted">These observations drove the confluence rules tested below: fractal + price&gt;VWMA55
+ ribbon-rising/stacked + recent-TD-buy is the "good entry" the screenshots depict; bare fractal is
the "every triangle" baseline it must beat.</p>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_clean_5m(CSV_PATH)
    span_days = (df["timestamp"].iat[-1] - df["timestamp"].iat[0]).days
    n_years = span_days / 365.25
    bars_per_yr = len(df) / n_years if n_years > 0 else 0.0
    window = f"{str(df['timestamp'].iat[0])[:16]} → {str(df['timestamp'].iat[-1])[:16]}"
    print(f"=== MES 5-min: {len(df)} bars, {window}, {span_days}d (~{n_years:.2f}y, "
          f"{bars_per_yr:.0f} bars/yr) ===")

    bh = buy_and_hold(df, bars_per_yr, n_years)
    print(f"  buy & hold: {pct(bh.total_return)}  Ret/DD {num(bh.mar)}  "
          f"Sharpe {num(bh.sharpe)}  MaxDD {pct(bh.max_dd)}")

    n_fractal = int(fractal_up_trend(df).sum())
    print(f"  bare fractalUpTrend signals: {n_fractal} ({n_fractal/n_years:.0f}/yr)")

    rows = run_grid(df, bars_per_yr, n_years)
    print(f"  {len(rows)} config combos traded.")

    robust = [r for r in rows if r.res.n >= MIN_TRADES]
    ranked = sorted(robust if robust else rows, key=_mar_key, reverse=True)
    best = pick_sweet_spot(rows)  # headline = robust pick (not a thin top-of-leaderboard fluke)
    print(f"  Top 5 by Ret/DD (n≥{MIN_TRADES}):")
    for r in ranked[:5]:
        x = r.res
        flag = "  <- SWEET SPOT" if r is best else ""
        print(f"    {r.entry.label():28} {r.exit.label():12} n={x.n:>3} win={x.win_rate*100:>3.0f}% "
              f"ret={pct(x.total_return):>8} mar={num(x.mar):>6} sharpe={num(x.sharpe):>5}{flag}")
    if best not in ranked[:5]:
        x = best.res
        print(f"    [sweet spot] {best.entry.label():16} {best.exit.label():12} n={x.n:>3} "
              f"win={x.win_rate*100:>3.0f}% ret={pct(x.total_return):>8} mar={num(x.mar):>6}")

    # baseline = best exit family on the BARE fractal entry
    bare = [r for r in rows if r.entry == EntryConfig(False, False, False, False, False)]
    baseline = max(bare, key=_mar_key) if bare else best

    wf = walk_forward(df, bars_per_yr)
    if wf:
        is_r, oos_r, oos_best = wf
        print(f"  WF in-sample: {is_r[0]:44} ret {pct(is_r[1].total_return)} mar {num(is_r[1].mar)}")
        print(f"  WF OOS (same config): ret {pct(oos_r.total_return)} mar {num(oos_r.mar)} n={oos_r.n}")
        print(f"  WF OOS best-with-hindsight: {oos_best[0]:30} ret {pct(oos_best[1].total_return)}")

    verdict = build_verdict(best, baseline, bh, wf)
    screenshots = build_screenshot_section()
    html = build_report(df, rows, bh, wf, window, n_years, bars_per_yr,
                        baseline, verdict, screenshots)
    OUT_HTML.write_text(html)
    print(f"\nWrote {OUT_HTML}")


if __name__ == "__main__":
    main()
