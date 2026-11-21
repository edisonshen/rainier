"""Front-month futures series: stitch per-contract bars, and back-adjust them at rolls."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from rainier.core.types import CSV_TZ

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
        if (pd.Timestamp(new) - pd.Timestamp(old)).days > 100:
            continue  # a missing quarter between them: their spread spans two rolls
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


def contract_closes(raw_dir, symbol: str) -> dict[str, pd.Series]:
    """Per-contract 5m closes (UTC index) from the IBKR cache, keyed by contract expiry."""
    from pathlib import Path

    out = {}
    for f in sorted(Path(raw_dir).glob(f"{symbol}_*_5m.csv")):
        x = pd.read_csv(f, usecols=["timestamp", "close"])
        idx = pd.DatetimeIndex(pd.to_datetime(x["timestamp"], utc=True))
        out[f.stem.split("_")[1]] = pd.Series(x["close"].to_numpy(float), index=idx).sort_index()
    return out


def _roll_contracts(
    contracts: dict[str, pd.Series], exp: pd.Timestamp
) -> tuple[pd.Series, pd.Series] | None:
    """(expiring, next) contract closes for the roll at ``exp``, if both are cached."""
    keys = sorted(contracts)
    for old, new in zip(keys, keys[1:], strict=False):
        if abs(pd.Timestamp(old, tz="UTC") - exp) <= pd.Timedelta(days=3):
            if (pd.Timestamp(new) - pd.Timestamp(old)).days > 100:
                return None  # the next quarter isn't cached
            return contracts[old], contracts[new]
    return None


def _close_in_bar(c: pd.Series, start: pd.Series, span: pd.Timedelta) -> np.ndarray:
    """``c``'s last close inside each bar ``[start, start + span)``; NaN where it has none.

    Daily bars take the close of their CME session: a midnight-to-midnight span would run
    into the next session's evening trade.
    """
    if span >= pd.Timedelta(days=1):
        last = c.groupby(session_date(c.index.to_series())).last()
        return session_date(start).map(last).to_numpy(float)
    pos = c.index.searchsorted(start + span, side="left") - 1
    ok = pos >= 0
    ok[ok] = c.index[pos[ok]] >= start.to_numpy()[ok]
    out = np.full(len(start), np.nan)
    out[ok] = c.to_numpy()[pos[ok]]
    return out


def _matched_old(close: np.ndarray, start: pd.Series, span: pd.Timedelta, gap: float,
                 old: pd.Series, new: pd.Series) -> np.ndarray:
    """1 where a bar's close is the expiring contract's, 0 the next one's, NaN if neither."""
    d_old = np.abs(close - _close_in_bar(old, start, span))
    d_new = np.abs(close - _close_in_bar(new, start, span))
    tol = abs(gap) / 4
    out = np.full(len(close), np.nan)
    out[(d_old <= tol) & ~(d_new < d_old)] = 1.0
    out[(d_new <= tol) & ~(d_old <= d_new)] = 0.0
    return out


def _single_switch(close: np.ndarray, gap: float, prev: float | None,
                   nxt: float | None) -> np.ndarray:
    """Old-contract mask with one switch, placed where the adjusted closes move least.

    A price-only guess: a real move as large as the gap can fool it, so it is the fallback
    when no per-contract closes cover the window.
    """
    n = len(close)
    head = [prev] if prev is not None else []
    tail = [nxt] if nxt is not None else []
    costs = [(np.diff(np.concatenate([head, close + gap * (np.arange(n) < k), tail])) ** 2).sum()
             for k in range(n + 1)]
    return np.arange(n) < int(np.argmin(costs))


def _nearest_fill(labels: np.ndarray) -> np.ndarray:
    """NaNs take the label of the nearest non-NaN position (ties: the earlier one)."""
    known = np.flatnonzero(~np.isnan(labels))
    pos = np.clip(np.searchsorted(known, np.arange(len(labels))), 1, len(known) - 1) \
        if len(known) > 1 else np.zeros(len(labels), dtype=int)
    if len(known) > 1:
        left, right = known[pos - 1], known[pos]
        here = np.arange(len(labels))
        pick = np.where(here - left <= right - here, left, right)
    else:
        pick = np.full(len(labels), known[0])
    return labels[pick]


def back_adjust(df: pd.DataFrame, gaps: pd.DataFrame,
                contracts: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """``df`` priced in its newest contract: each roll's gap added to the bars before it.

    Inside a roll window each bar is matched to the contract whose close it carries
    (``contracts``, per-contract 5m closes), which handles sources that flip back and forth;
    bars no contract covers take the nearest matched bar's label, and windows with no
    contract data fall back to :func:`_single_switch`. Only rolls whose whole window is in
    ``df`` apply, so a roll in progress never moves the latest price.
    """
    if df.empty or gaps.empty:
        return df
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:  # CSVProvider bars are naive CSV_TZ
        ts = ts.dt.tz_localize(CSV_TZ, ambiguous=np.ones(len(ts), dtype=bool),
                               nonexistent="shift_forward")
    ts = ts.dt.tz_convert("UTC").reset_index(drop=True)
    span = ts.diff().median() if len(ts) > 1 else pd.Timedelta(days=1)
    close = df["close"].to_numpy(float)
    offset = np.zeros(len(df))
    lo_win, hi_win = ROLL_WINDOW
    for expiry, gap in zip(gaps["expiry"], gaps["gap"].astype(float), strict=True):
        exp = pd.Timestamp(str(expiry), tz="UTC")
        start, end = exp - lo_win, exp + hi_win
        if ts.iloc[-1] <= end or ts.iloc[0] >= end:
            continue
        offset[(ts < start).to_numpy()] += gap
        idx = np.flatnonzero(((ts >= start) & (ts <= end)).to_numpy())
        if not len(idx):
            continue
        pair = _roll_contracts(contracts or {}, exp)
        old = np.full(len(idx), np.nan)
        if pair:
            old = _matched_old(close[idx], ts.iloc[idx].reset_index(drop=True), span, gap, *pair)
        if np.isnan(old).all():
            adj = close + offset
            prev = adj[idx[0] - 1] if idx[0] > 0 else None
            nxt = adj[idx[-1] + 1] if idx[-1] + 1 < len(df) else None
            mask = _single_switch(close[idx], gap, prev, nxt)
        else:
            mask = _nearest_fill(old) > 0.5
        offset[idx[mask]] += gap
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].astype(float) + offset
    return out
