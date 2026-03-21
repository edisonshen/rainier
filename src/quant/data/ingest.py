"""Ingest CSV data into PostgreSQL."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from quant.core.models import CandleRecord
from quant.core.types import Timeframe


def ingest_csv(
    session: Session,
    csv_path: Path,
    symbol: str,
    timeframe: Timeframe,
) -> int:
    """Load a CSV file into the candles table. Returns number of rows upserted."""
    df = pd.read_csv(csv_path)

    # Normalize columns
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

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe.value,
                "timestamp": row["timestamp"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )

    if not rows:
        return 0

    # Upsert: on conflict update OHLCV
    stmt = pg_insert(CandleRecord).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "timeframe", "timestamp"],
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
        },
    )
    session.execute(stmt)
    session.commit()

    return len(rows)
