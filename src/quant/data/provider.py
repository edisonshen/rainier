"""Data provider protocols — swappable sources for OHLCV and supplementary data."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd

from quant.core.types import Timeframe


@runtime_checkable
class DataProvider(Protocol):
    """Provides OHLCV candle data."""

    def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Return DataFrame with columns: timestamp, open, high, low, close, volume."""
        ...


@runtime_checkable
class SupplementaryProvider(Protocol):
    """Future: provides supplementary data (options flow, money flow, dark pool).
    Each implementation returns its own schema. Not used in v1."""

    def get_data(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame: ...
