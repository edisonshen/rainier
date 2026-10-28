"""Stitch per-contract futures bars into one unadjusted front-month series."""

from __future__ import annotations

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
