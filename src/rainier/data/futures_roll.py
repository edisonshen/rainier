"""Front-month futures series: stitch per-contract bars, and back-adjust them at rolls."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

# CME Globex sessions open 17:00 Chicago time; shifting by 7h maps each bar to its trade date.
_SESSION_TZ = "America/Chicago"
_SESSION_SHIFT = pd.Timedelta(hours=7)


def session_date(timestamps: pd.Series) -> pd.Series:
    """CME trade date of each UTC timestamp (a Sunday 17:00 CT open belongs to Monday)."""
    return (timestamps.dt.tz_convert(_SESSION_TZ) + _SESSION_SHIFT).dt.date


def stitch_front_month(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Keep, for each CME session, the bars of the contract that traded the most volume.

    ``frames`` maps contract expiry (``YYYYMMDD``) to its OHLCV bars. The front month only
    rolls forward, so a thin late session of an expiring contract can't switch back to it.
    Prices are not back-adjusted (same convention as yfinance ``MES=F``).
    """
    parts = [df.assign(expiry=int(exp)) for exp, df in frames.items() if not df.empty]
    if not parts:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    bars = pd.concat(parts, ignore_index=True)
    bars["session"] = session_date(bars["timestamp"])

    volume = bars.groupby(["session", "expiry"])["volume"].sum().unstack(fill_value=0)
    front = volume.idxmax(axis=1).cummax()

    stitched = bars[bars["expiry"] == bars["session"].map(front)]
    return (
        stitched.drop(columns=["expiry", "session"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# --- Back-adjustment ------------------------------------------------------------------------
# A continuous front-month series jumps at every quarterly roll by the new contract's premium
# (its carry), so older bars sit below where the same price action trades in today's contract.
# Difference back-adjustment adds each later roll's gap to every bar traded before that roll,
# so the whole series is priced in the newest contract (TradingView MES1! "back-adjust").

# Bars from this long before to this long after an expiry may be in either contract: IBKR's
# volume stitch rolls the Monday before expiry, yfinance's MES=F only after it.
ROLL_WINDOW = (pd.Timedelta(days=12), pd.Timedelta(days=4))
ROLL_GAP_COLUMNS = ["expiry", "gap", "source", "price", "rate", "asof"]
CARRY = "carry"  # source of estimated (not observed) gaps


def quarterly_expiries(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """CME equity-index expiries (third Friday of Mar/Jun/Sep/Dec) in ``[start, end]``, UTC."""
    months = pd.date_range(start.tz_localize(None).normalize().replace(day=1),
                           end.tz_localize(None) + pd.Timedelta(days=31), freq="MS")
    out = []
    for m in months[months.month % 3 == 0]:
        first_friday = m + pd.Timedelta(days=(4 - m.weekday()) % 7)
        exp = (first_friday + pd.Timedelta(weeks=2)).tz_localize("UTC")
        if start <= exp <= end:
            out.append(exp)
    return out


def contract_gaps(frames: dict[str, pd.DataFrame], sessions: int = 5) -> dict[str, float]:
    """Next-minus-expiring contract close spread at each roll, keyed by the expiring contract.

    Median over the ``sessions`` CME sessions up to the volume roll, on bars both traded.
    """
    keys = sorted(k for k, df in frames.items() if not df.empty)
    gaps: dict[str, float] = {}
    for old, new in zip(keys, keys[1:], strict=False):
        both = frames[old].merge(frames[new], on="timestamp", suffixes=("_old", "_new"))
        if both.empty:
            continue
        both["session"] = session_date(both["timestamp"])
        lead = both.groupby("session")[["volume_old", "volume_new"]].sum()
        rolled = lead.index[lead.volume_new > lead.volume_old]
        last = rolled[0] if len(rolled) else lead.index[-1]
        recent = sorted(s for s in lead.index if s <= last)[-sessions:]
        spread = both[both.session.isin(recent)]
        gaps[old] = float((spread.close_new - spread.close_old).median())
    return gaps


def carry_gap(price: float, rate: float, div_yield: float, days: float) -> float:
    """Fair next-minus-expiring spread: price x (financing rate - dividend yield) x term."""
    return price * (rate - div_yield) * days / 365


def implied_div_yield(gap: float, price: float, rate: float, days: float) -> float:
    """Dividend yield that makes :func:`carry_gap` equal an observed ``gap``."""
    return rate - gap * 365 / (price * days)


def fill_roll_gaps(
    gaps: pd.DataFrame,
    expiries: list[pd.Timestamp],
    price_at: Callable[[pd.Timestamp], float],
    rate_at: Callable[[pd.Timestamp], float],
) -> pd.DataFrame:
    """Observed gaps, plus a carry estimate for every expiry in ``expiries`` none covers.

    The estimate's dividend yield is the median implied by the observed rolls of the same
    calendar month (dividends cluster by quarter), else of all observed rolls. Estimates are
    redone on every call, so they sharpen as observations accumulate.
    """
    obs = gaps[gaps["source"] != CARRY].copy()
    obs["expiry"] = obs["expiry"].astype(str)
    exp_ts = pd.to_datetime(obs["expiry"], format="%Y%m%d").dt.tz_localize("UTC")
    for i, ts in zip(obs.index, exp_ts, strict=True):
        if pd.isna(obs.at[i, "price"]):
            obs.at[i, "price"] = price_at(ts - pd.Timedelta(days=7))
        if pd.isna(obs.at[i, "rate"]):
            obs.at[i, "rate"] = rate_at(ts - pd.Timedelta(days=7))
    term = 91.0
    obs["q"] = [implied_div_yield(g, p, r, term)
                for g, p, r in zip(obs["gap"], obs["price"], obs["rate"], strict=True)]
    obs["month"] = exp_ts.dt.month.to_numpy()
    rows = []
    for exp in expiries:
        if (abs(exp_ts - exp) <= pd.Timedelta(days=3)).any():
            continue
        same = obs.loc[obs["month"] == exp.month, "q"]
        q = same.median() if len(same) else (obs["q"].median() if len(obs) else 0.0)
        when = exp - pd.Timedelta(days=7)
        price, rate = price_at(when), rate_at(when)
        rows.append({"expiry": exp.strftime("%Y%m%d"),
                     "gap": round(carry_gap(price, rate, q, term) * 4) / 4,
                     "source": CARRY, "price": price, "rate": rate, "asof": None})
    out = pd.concat([obs.drop(columns=["q", "month"]), pd.DataFrame(rows)], ignore_index=True)
    return out[ROLL_GAP_COLUMNS].sort_values("expiry").reset_index(drop=True)


def load_roll_gaps(path) -> pd.DataFrame:
    """The ``{symbol}_rolls.csv`` table (one row per roll, keyed by the expiring contract)."""
    from pathlib import Path

    if not Path(path).exists():
        return pd.DataFrame(columns=ROLL_GAP_COLUMNS)
    return pd.read_csv(path, dtype={"expiry": str, "source": str})


def save_roll_gaps(path, gaps: pd.DataFrame) -> None:
    gaps.sort_values("expiry")[ROLL_GAP_COLUMNS].to_csv(path, index=False)


def _old_contract_mask(close: np.ndarray, gap: float, prev: float | None,
                       nxt: float | None) -> np.ndarray:
    """Which bars of a roll window still trade the expiring contract.

    Two-state Viterbi: the labelling whose back-adjusted closes move least bar to bar, joined
    to the already-adjusted bars just before (``prev``) and after (``nxt``) the window.
    Handles sources that flip contracts more than once around a roll.
    """
    adj = np.stack([close, close + gap])  # state 0 = new contract, 1 = old (raised by gap)
    cost = (adj[:, 0] - prev) ** 2 if prev is not None else np.zeros(2)
    back = np.zeros((len(close), 2), dtype=int)
    for k in range(1, len(close)):
        step = cost[:, None] + (adj[None, :, k] - adj[:, k - 1, None]) ** 2  # [from, to]
        back[k] = step.argmin(axis=0)
        cost = step.min(axis=0)
    if nxt is not None:
        cost = cost + (nxt - adj[:, -1]) ** 2
    s = int(cost.argmin())
    old = np.zeros(len(close), dtype=bool)
    for k in range(len(close) - 1, -1, -1):
        old[k] = s == 1
        s = back[k, s]
    return old


def back_adjust(df: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    """``df`` priced in its newest contract: each roll's gap added to the bars before it.

    Only rolls whose whole window is in ``df`` apply, so a roll in progress never moves the
    latest price; its gap lands once the window has passed.
    """
    if df.empty or gaps.empty:
        return df
    ts = pd.to_datetime(df["timestamp"], utc=True)
    close = df["close"].to_numpy(float)
    offset = np.zeros(len(df))
    lo_win, hi_win = ROLL_WINDOW
    for expiry, gap in zip(gaps["expiry"], gaps["gap"].astype(float), strict=True):
        exp = pd.Timestamp(str(expiry), tz="UTC")
        start, end = exp - lo_win, exp + hi_win
        if ts.iloc[-1] <= end or ts.iloc[0] >= end:
            continue
        before = (ts < start).to_numpy()
        inside = ((ts >= start) & (ts <= end)).to_numpy()
        offset[before] += gap
        if inside.any():
            idx = np.flatnonzero(inside)
            adj = close + offset
            prev = adj[idx[0] - 1] if idx[0] > 0 else None
            nxt = adj[idx[-1] + 1] if idx[-1] + 1 < len(df) else None
            offset[idx[_old_contract_mask(adj[idx], gap, prev, nxt)]] += gap
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype(float) + offset
    return out
