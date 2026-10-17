"""Adjusted daily close loader for strategy inputs."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf


def warmup_start(start: date) -> date:
    """Fetch well over 120 trading sessions before the strategy's replay start."""
    return start - timedelta(days=260)


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
