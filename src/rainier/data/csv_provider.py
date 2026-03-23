"""Read OHLCV data from CSV files (yfinance exports or other sources)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from rainier.core.types import Timeframe


class CSVProvider:
    """Reads OHLCV CSV files from yfinance or other sources.

    Expected CSV format:
        timestamp, open, high, low, close, volume
    Also accepts TradingView format (time, Volume with capital V).
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        csv_path = self._find_csv(symbol, timeframe)
        if csv_path is None:
            raise FileNotFoundError(
                f"No CSV found for {symbol} {timeframe.value} in {self.data_dir}"
            )
        return self._read_csv(csv_path, start, end)

    def _find_csv(self, symbol: str, timeframe: Timeframe) -> Path | None:
        """Find CSV file matching symbol and timeframe.

        Tries patterns:
          - {symbol}_{timeframe}.csv  (e.g., NQ_1H.csv)
          - {symbol}.csv              (fallback)
        """
        patterns = [
            f"{symbol}_{timeframe.value}.csv",
            f"{symbol.upper()}_{timeframe.value}.csv",
            f"{symbol}.csv",
        ]
        for pattern in patterns:
            path = self.data_dir / pattern
            if path.exists():
                return path
        # Try glob for partial matches
        for path in sorted(self.data_dir.glob(f"*{symbol}*{timeframe.value}*")):
            if path.suffix == ".csv":
                return path
        return None

    def _read_csv(
        self,
        path: Path,
        start: datetime | None,
        end: datetime | None,
    ) -> pd.DataFrame:
        df = pd.read_csv(path)

        # Normalize column names (TradingView uses 'time', 'Volume')
        col_map = {}
        for col in df.columns:
            lower = col.strip().lower()
            if lower in ("time", "date", "datetime", "timestamp"):
                col_map[col] = "timestamp"
            elif lower == "volume":
                col_map[col] = "volume"
            elif lower in ("open", "high", "low", "close"):
                col_map[col] = lower
        df = df.rename(columns=col_map)

        # Ensure required columns
        required = {"timestamp", "open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing} in {path}")

        if "volume" not in df.columns:
            df["volume"] = 0.0

        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], utc=True)
            .dt.tz_convert("US/Pacific")
            .dt.tz_localize(None)
        )
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Filter date range
        if start:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end:
            df = df[df["timestamp"] <= pd.Timestamp(end)]

        return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
