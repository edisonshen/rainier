"""SignalPanel — the frozen ≤66-lookback, trend-only signal list for v1.

Each member is a daily *risk-on* indicator returning a ``{0.0, 1.0}`` series
(1 = uptrend-confirming). The panel feeds ``ResonanceScorer`` (power-weighted
score) which feeds ``ResonanceGate`` (dual-threshold in/out of TQQQ).

Source-of-truth: the enumerated ``_MEMBERS`` list below (design
``docs/DESIGN-multi-signal-resonance.md`` §5.2). It was seeded from the ≤66
subset of the exploratory ``scripts/regime_signal_search.build_signals``
catalog, then **deduplicated** (one threshold per indicator — RSI>50 not
RSI>55, VIX<25 not VIX<20/<30, one of ADX/+DI) so no indicator is
triple-counted, with two changes required by the lookback invariant:

  * the two ``expanding(60).median()`` volatility members are **rewritten** to
    bounded rolling-median form (realizedVol(20) vs its rolling-40 median;
    ATR%(14) vs its rolling-46 median) — expanding windows are unbounded and
    fail the finite-window truncation test;
  * ATR% uses a **simple rolling mean** of true range, NOT Wilder/EWMA.

Added vs the prototype: ``price>SMA66``, SPY trend signals, and an optional
breadth member (``% of a frozen universe above SMA50``).

Lookback invariant (§5.5): every member's *composed* lookback ≤ 66. Nested
indicators compose: realizedVol(20) into a rolling-40 median spans 20+40 = 60.
Finite-window members (SMA, rolling-median, Donchian, ROC) have exact composed
support ≤66 and are bit-identical when history before ``t-66`` is dropped
(test b). EMA-family members (EMA, RSI, MACD, ADX) have *bounded effective*
memory — they never widen with more history — and are covered by the frozen
``WARMUP_BARS`` pre-registered constant rather than exact truncation.

The panel computes **decision** series at ``close[t]`` — it does NOT shift. The
sim (``backtest/daily_mtm.py``) owns the +1 shift.

```
   df (QQQ OHLC + VIX + SPY [+ breadth])
            │
            ▼
   SignalPanel.compute()  ──►  {member_name: {0,1} series}   (no shift)
            │
            ▼
   ResonanceScorer / ResonanceGate
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Lookback constants (pre-registered; do NOT make these data-dependent)
# ---------------------------------------------------------------------------

MAX_LOOKBACK = 66
"""Operator constraint: no moving-average / composed lookback may exceed this."""

WARMUP_BARS = 150
"""Frozen warmup for EMA-family members (ADX double-smoothed, MACD hist =
EMA-of-EMA). Empirically the slowest member (ADX(14)) converges well within
this; pinned as an integer constant so ``t0`` is data-independent (the gate
determinism test asserts this). It must NOT shift with the data."""


# ---------------------------------------------------------------------------
# Indicator primitives (finite-window unless noted)
# ---------------------------------------------------------------------------

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    """EMA — EMA-family (bounded effective memory, not finite-window)."""
    return s.ewm(span=n, adjust=False).mean()


def roc(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change(n)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI — EMA-family (bounded effective memory)."""
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd_hist(s: pd.Series, f: int = 12, sl: int = 26, sig: int = 9) -> pd.Series:
    """MACD histogram — EMA-family (EMA-of-EMA, bounded effective memory)."""
    line = ema(s, f) - ema(s, sl)
    return line - ema(line, sig)


def realized_vol(s: pd.Series, n: int = 20) -> pd.Series:
    """Annualized rolling-std of daily returns (finite window n+1 via pct_change)."""
    return s.pct_change().rolling(n).std() * np.sqrt(252)


def atr_pct_simple(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """ATR%/price using a **simple rolling mean** of true range (finite window).

    Deliberately NOT Wilder/EWMA — the design pins ATR% to a bounded
    finite-window form so it passes the truncation test.
    """
    h, lo, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean() / c


def adx(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX / +DI / -DI — EMA-family (double-smoothed, bounded effective memory)."""
    h, lo, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -lo.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0), pdi, mdi


def donchian_mid(df: pd.DataFrame, n: int) -> pd.Series:
    return (df["high"].rolling(n).max() + df["low"].rolling(n).min()) / 2


def _b(x) -> np.ndarray:
    """Coerce a bool/float series to a clean {0,1} float ndarray (NaN→0)."""
    return np.nan_to_num(np.asarray(x, dtype=float))


# ---------------------------------------------------------------------------
# Member registry — THE source of truth (§5.2)
# ---------------------------------------------------------------------------

# Six categories, each weighted equally by the category-balanced prior (§5.3).
CATEGORIES = ("trend", "momentum", "volatility", "structure", "cross_asset", "breadth")


@dataclass(frozen=True)
class PanelMember:
    name: str
    category: str
    composed_lookback: int  # exact finite support, or bounded effective memory
    finite_window: bool     # True → exact-support member (truncation-testable)
    fn: Callable[["PanelInputs"], np.ndarray]


@dataclass
class PanelInputs:
    """Aligned inputs for one ``compute`` call. ``spy`` and ``breadth`` optional."""
    qqq: pd.DataFrame          # columns: open, high, low, close
    vix: pd.Series             # VIX close, aligned to qqq index
    spy: pd.Series | None = None      # SPY close, aligned
    breadth: pd.Series | None = None  # %-above-SMA50 already computed, aligned ∈ [0,1]


def _trend_members() -> list[PanelMember]:
    def p_sma(n):
        return lambda i: _b(i.qqq["close"] > sma(i.qqq["close"], n))

    def p_ema(n):
        return lambda i: _b(i.qqq["close"] > ema(i.qqq["close"], n))

    out = [
        PanelMember(f"price>SMA{n}", "trend", n, True, p_sma(n)) for n in (20, 50, 66)
    ]
    out += [
        PanelMember(f"price>EMA{n}", "trend", n, False, p_ema(n)) for n in (20, 50)
    ]
    out.append(PanelMember(
        "SMA22>SMA44", "trend", 44, True,
        lambda i: _b(sma(i.qqq["close"], 22) > sma(i.qqq["close"], 44)),
    ))
    # SMA50 rising: 50-day MA vs 10 bars ago → composed 50+10 = 60.
    out.append(PanelMember(
        "SMA50 rising", "trend", 60, True,
        lambda i: _b(sma(i.qqq["close"], 50) > sma(i.qqq["close"], 50).shift(10)),
    ))
    return out


def _momentum_members() -> list[PanelMember]:
    return [
        PanelMember("RSI14>50", "momentum", 14, False,
                    lambda i: _b(rsi(i.qqq["close"], 14) > 50)),
        PanelMember("MACD hist>0", "momentum", 26, False,
                    lambda i: _b(macd_hist(i.qqq["close"]) > 0)),
        PanelMember("ROC20>0", "momentum", 20, True,
                    lambda i: _b(roc(i.qqq["close"], 20) > 0)),
        PanelMember("ROC60>0", "momentum", 60, True,
                    lambda i: _b(roc(i.qqq["close"], 60) > 0)),
        PanelMember("price>Donchian50mid", "momentum", 50, True,
                    lambda i: _b(i.qqq["close"] > donchian_mid(i.qqq, 50))),
    ]


def _vol_realized(i: PanelInputs) -> np.ndarray:
    # realizedVol(20) below its rolling-40 median → composed 20+40 = 60 (finite).
    rv = realized_vol(i.qqq["close"], 20)
    return _b(rv < rv.rolling(40).median())


def _vol_atr(i: PanelInputs) -> np.ndarray:
    # ATR%(14, simple mean) below its rolling-46 median → composed 14+46 = 60 (finite).
    ap = atr_pct_simple(i.qqq, 14)
    return _b(ap < ap.rolling(46).median())


def _volatility_members() -> list[PanelMember]:
    return [
        PanelMember("realizedVol<rollMedian", "volatility", 60, True, _vol_realized),
        PanelMember("ATR%<rollMedian", "volatility", 60, True, _vol_atr),
        PanelMember("VIX<25", "volatility", 1, True,
                    lambda i: _b(i.vix < 25)),
        PanelMember("VIX<SMA20", "volatility", 20, True,
                    lambda i: _b(i.vix < sma(i.vix, 20))),
        PanelMember("VIX falling", "volatility", 5, True,
                    lambda i: _b(i.vix < i.vix.shift(5))),
    ]


def _hh_hl(i: PanelInputs) -> np.ndarray:
    # 20-day higher-high AND higher-low vs 20 bars ago → composed 20+20 = 40 (finite).
    hh = i.qqq["high"].rolling(20).max() > i.qqq["high"].rolling(20).max().shift(20)
    hl = i.qqq["low"].rolling(20).min() > i.qqq["low"].rolling(20).min().shift(20)
    return _b(hh & hl)


def _adx_up(i: PanelInputs) -> np.ndarray:
    adxv, pdi, mdi = adx(i.qqq, 14)
    return _b((adxv > 20) & (pdi > mdi))


def _structure_members() -> list[PanelMember]:
    return [
        PanelMember("within5%of60d-high", "structure", 60, True,
                    lambda i: _b(i.qqq["close"] >= 0.95 * i.qqq["close"].rolling(60).max())),
        PanelMember("ADX>20 & +DI>-DI", "structure", 14, False, _adx_up),
        PanelMember("HH+HL structure", "structure", 40, True, _hh_hl),
    ]


def _cross_asset_members() -> list[PanelMember]:
    # Cross-asset members require SPY; skipped at compute time when spy is None.
    return [
        PanelMember("SPY>SMA22", "cross_asset", 22, True,
                    lambda i: _b(i.spy > sma(i.spy, 22))),
        PanelMember("SPY>SMA44", "cross_asset", 44, True,
                    lambda i: _b(i.spy > sma(i.spy, 44))),
    ]


def _breadth_members() -> list[PanelMember]:
    # Breadth requires a precomputed %-above-SMA50 series ∈ [0,1]; skipped when None.
    # Risk-on when > half the (frozen) universe is above its SMA50.
    return [
        PanelMember("breadth>50%above-SMA50", "breadth", 50, True,
                    lambda i: _b(i.breadth > 0.5)),
    ]


def _all_members() -> list[PanelMember]:
    return (
        _trend_members()
        + _momentum_members()
        + _volatility_members()
        + _structure_members()
        + _cross_asset_members()
        + _breadth_members()
    )


# Frozen, ordered member list. Length ≤ 66 by construction (currently 21).
_MEMBERS: tuple[PanelMember, ...] = tuple(_all_members())


# ---------------------------------------------------------------------------
# SignalPanel
# ---------------------------------------------------------------------------

class SignalPanel:
    """Computes the frozen ≤66-lookback trend-only signal members.

    ``compute`` returns an ordered ``dict[name -> {0,1} ndarray]`` for the
    members whose inputs are present. Cross-asset members need ``spy``; the
    breadth member needs a precomputed ``breadth`` series — absent inputs drop
    those members (and the design requires breadth be excluded from the OOS
    claims when only a frozen ex-ante universe is available; see §6.2).
    """

    def __init__(self, members: tuple[PanelMember, ...] = _MEMBERS) -> None:
        self.members = members
        self._validate_lookback_invariant()

    def _validate_lookback_invariant(self) -> None:
        bad = [m.name for m in self.members if m.composed_lookback > MAX_LOOKBACK]
        if bad:
            raise ValueError(
                f"panel members exceed MAX_LOOKBACK={MAX_LOOKBACK}: {bad}"
            )

    def available_members(self, inputs: PanelInputs) -> list[PanelMember]:
        out = []
        for m in self.members:
            if m.category == "cross_asset" and inputs.spy is None:
                continue
            if m.category == "breadth" and inputs.breadth is None:
                continue
            out.append(m)
        return out

    def compute(self, inputs: PanelInputs) -> dict[str, np.ndarray]:
        n = len(inputs.qqq)
        out: dict[str, np.ndarray] = {}
        for m in self.available_members(inputs):
            series = m.fn(inputs)
            if len(series) != n:
                raise ValueError(f"member {m.name} produced length {len(series)} != {n}")
            out[m.name] = series
        return out

    def category_of(self, name: str) -> str:
        for m in self.members:
            if m.name == name:
                return m.category
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Breadth helper (frozen ex-ante universe path)
# ---------------------------------------------------------------------------

def breadth_pct_above_sma(prices: pd.DataFrame, n: int = 50) -> pd.Series:
    """% of a frozen universe above its own SMA(n), per row, ∈ [0,1].

    ``prices`` is a wide frame: one column of close prices per symbol, indexed
    by date. The universe is whatever columns are present (frozen ex-ante — the
    caller is responsible for picking a survivorship-aware set). Returns the
    daily fraction of symbols trading above their own ``n``-day SMA.

    This is the v1 breadth feed when point-in-time QQQ constituents are
    unavailable; per design §6.2, breadth computed from a frozen universe is
    **excluded from the OOS claims**.

    Dashboard integration point (design §7 / Q5): the QQQ market-breadth
    dashboard already computes the sibling metric via
    ``rainier.market_breadth.indicators.pct_above_ma`` (long-format, returns
    [0,100]). To surface this panel member on that dashboard, feed a top-N-QQQ
    holdings frame through ``pct_above_ma(df, window=50)`` and divide by 100 to
    get the [0,1] series this panel consumes. Building the point-in-time
    QQQ-holdings OHLCV substrate + the dashboard panel is tracked as a [P2]
    follow-up (out of scope for v1 per the task brief).
    """
    if n > MAX_LOOKBACK:
        raise ValueError(f"breadth SMA period {n} exceeds MAX_LOOKBACK={MAX_LOOKBACK}")
    above = prices > prices.rolling(n).mean()
    # Only count symbols that have a defined SMA on a given row (avoid early-NaN bias).
    valid = prices.rolling(n).mean().notna()
    num_above = (above & valid).sum(axis=1)
    denom = valid.sum(axis=1).replace(0, np.nan)
    return (num_above / denom).clip(0.0, 1.0)
