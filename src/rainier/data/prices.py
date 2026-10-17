"""Adjusted daily close loader for strategy inputs."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from math import ceil
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


def warmup_start(start: date, sessions: int) -> date:
    """Fetch enough calendar days to warm up the longest rolling window."""
    return start - timedelta(days=ceil(sessions * 1.5) + 30)


def load_daily_close(symbol: str, start: date) -> pd.Series:
    """Load auto-adjusted daily closes, indexed by timezone-naive session date."""
    history = yf.Ticker(symbol).history(start=start.isoformat(), auto_adjust=True)
    if history.empty:
        return pd.Series(dtype=float, name=symbol)
    index = pd.DatetimeIndex(pd.to_datetime(history.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    close = history["Close"].astype(float).copy()
    close.index = index.normalize()
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="last")]
    close.name = symbol
    return close


def load_last_price(symbol: str) -> tuple[float, datetime]:
    """Load the most recent one-minute close with a timezone-aware timestamp."""
    history = yf.Ticker(symbol).history(period="1d", interval="1m")
    if history.empty:
        raise ValueError(f"No intraday price available for {symbol}")

    timestamp = pd.Timestamp(history.index[-1])
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(ZoneInfo("America/New_York"))
    return float(history["Close"].iloc[-1]), timestamp.to_pydatetime()
