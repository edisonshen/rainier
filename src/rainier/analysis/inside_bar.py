"""Inside bar / range compression detection."""

from __future__ import annotations

import pandas as pd

from rainier.core.config import InsideBarConfig
from rainier.core.types import Candle, InsideBar, Timeframe


def detect_inside_bars(
    df: pd.DataFrame,
    config: InsideBarConfig | None = None,
    symbol: str = "",
    timeframe: Timeframe = Timeframe.H1,
) -> list[InsideBar]:
    """Detect inside bars (current bar's range is entirely within previous bar's range).

    An inside bar: high <= mother_high AND low >= mother_low.
    """
    if config is None:
        config = InsideBarConfig()

    inside_bars: list[InsideBar] = []

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]

        mother_high = float(prev["high"])
        mother_low = float(prev["low"])

        if float(curr["high"]) <= mother_high and float(curr["low"]) >= mother_low:
            mother_range = mother_high - mother_low
            if mother_range == 0:
                continue

            curr_range = float(curr["high"]) - float(curr["low"])
            compression = curr_range / mother_range

            candle = Candle(
                timestamp=pd.Timestamp(curr["timestamp"]).to_pydatetime(),
                open=float(curr["open"]),
                high=float(curr["high"]),
                low=float(curr["low"]),
                close=float(curr["close"]),
                volume=float(curr.get("volume", 0)),
                symbol=symbol,
                timeframe=timeframe,
            )

            mother = Candle(
                timestamp=pd.Timestamp(prev["timestamp"]).to_pydatetime(),
                open=float(prev["open"]),
                high=float(prev["high"]),
                low=float(prev["low"]),
                close=float(prev["close"]),
                volume=float(prev.get("volume", 0)),
                symbol=symbol,
                timeframe=timeframe,
            )

            inside_bars.append(
                InsideBar(
                    candle=candle,
                    index=i,
                    mother_candle=mother,
                    mother_index=i - 1,
                    compression_ratio=compression,
                )
            )

    return inside_bars
