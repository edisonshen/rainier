"""YFinance data provider — fetches OHLCV from Yahoo Finance."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog
import yfinance as yf

from rainier.core.types import Timeframe

log = structlog.get_logger()

# Map our symbols to yfinance tickers
SYMBOL_TO_TICKER: dict[str, str] = {
    "NQ": "NQ=F",
    "MNQ": "MNQ=F",
    "ES": "ES=F",
    "MES": "MES=F",
    "GC": "GC=F",
}

# Map our timeframes to yfinance intervals + sensible periods
TF_TO_YF: dict[Timeframe, tuple[str, str]] = {
    Timeframe.M5: ("5m", "60d"),
    Timeframe.H1: ("1h", "730d"),
    Timeframe.H4: ("1h", "730d"),  # yfinance has no 4h; we resample from 1h
    Timeframe.D1: ("1d", "max"),
}


class YFinanceProvider:
    """DataProvider implementation backed by yfinance."""

    def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch candles from yfinance, return normalized DataFrame."""
        ticker = SYMBOL_TO_TICKER.get(symbol, symbol)
        if timeframe not in TF_TO_YF:
            raise ValueError(f"Unsupported timeframe {timeframe} for yfinance")

        yf_interval, yf_period = TF_TO_YF[timeframe]
        raw = yf.Ticker(ticker).history(period=yf_period, interval=yf_interval)

        if raw.empty:
            log.warning("yfinance_no_data", symbol=symbol, timeframe=timeframe.value)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = _normalize(raw)

        # Resample 1h → 4h if needed
        if timeframe == Timeframe.H4:
            df = _resample_4h(df)

        # Filter date range
        if start:
            df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]

        return df.reset_index(drop=True)


def fetch_symbol(
    symbol: str,
    timeframes: list[Timeframe],
    data_dir: Path,
) -> dict[Timeframe, int]:
    """Fetch data for a symbol across timeframes, merge with existing CSVs.

    Returns dict of {timeframe: total_rows} after merge.
    Backward-compatible wrapper around YFinanceProvider.
    """
    from rainier.data.persistence import save_candles

    provider = YFinanceProvider()
    results: dict[Timeframe, int] = {}

    for tf in timeframes:
        df = provider.get_candles(symbol, tf)
        if df.empty:
            continue
        results[tf] = save_candles(df, symbol, tf, data_dir)

    return results


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance DataFrame to our standard format."""
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "timestamp"
    df = df.reset_index()
    # yfinance index name is 'Datetime' for intraday, 'Date' for daily
    df = df.rename(columns={"Datetime": "timestamp", "Date": "timestamp"})
    # Keep timezone info in the CSV for proper merging
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h data to 4h bars."""
    df = df.set_index("timestamp")
    resampled = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return resampled.reset_index()


# Equity-index futures roll quarterly; their CSVs are back-adjusted with {symbol}_rolls.csv.
QUARTERLY_ROLL_SYMBOLS = ("ES", "MES", "NQ", "MNQ")
_MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}


def contract_ticker(symbol: str, expiry: pd.Timestamp) -> str:
    """Yahoo ticker of one quarterly CME contract, e.g. MES Dec 2026 -> ``MESZ26.CME``."""
    return f"{symbol}{_MONTH_CODES[expiry.month]}{expiry.year % 100:02d}.CME"


def front_roll_spread(symbol: str, now: pd.Timestamp) -> dict | None:
    """Next-minus-front contract close spread (median of the last 3 shared sessions).

    Recorded daily until the front contract expires, so each roll's gap is observed from
    the contracts themselves rather than estimated. None if Yahoo lacks either contract.
    """
    from rainier.data.futures_roll import quarterly_expiries

    front, nxt = quarterly_expiries(now - pd.Timedelta(days=3), now + pd.Timedelta(days=200))[:2]
    closes = []
    for exp in (front, nxt):
        raw = yf.Ticker(contract_ticker(symbol, exp)).history(period="10d", interval="1d")
        if raw.empty:
            return None
        closes.append(raw["Close"].set_axis(raw.index.date).rename(len(closes)))
    both = pd.concat(closes, axis=1, join="inner").dropna().tail(3)
    if both.empty:
        return None
    return {"expiry": front.strftime("%Y%m%d"), "gap": float((both[1] - both[0]).median()),
            "source": "yfinance", "price": float(both[0].iloc[-1]), "rate": None,
            "asof": now.strftime("%Y-%m-%d")}


def rate_history(start: pd.Timestamp) -> pd.Series:
    """13-week T-bill yield (^IRX) as a fraction, indexed by UTC date: the futures carry rate."""
    raw = yf.Ticker("^IRX").history(start=start.strftime("%Y-%m-%d"), interval="1d")
    rates = raw["Close"] / 100
    rates.index = pd.to_datetime(rates.index.date).tz_localize("UTC")
    return rates
