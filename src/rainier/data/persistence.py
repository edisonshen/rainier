"""Shared persistence utilities — CSV merge + database upsert for candle data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

from rainier.core.types import Timeframe

log = structlog.get_logger()


def save_candles(
    df: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    data_dir: Path,
    *,
    prefer_existing: bool = False,
) -> int:
    """Save candles to CSV (merging with existing) and DB. Returns total row count.

    With ``prefer_existing`` the new bars only fill timestamps the CSV doesn't have yet.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{symbol}_{timeframe.value}.csv"
    df = merge_with_existing(df, csv_path, keep="first" if prefer_existing else "last")
    df.to_csv(csv_path, index=False)
    persist_to_db(df, symbol, timeframe)
    return len(df)


def merge_with_existing(
    new_df: pd.DataFrame, csv_path: Path, keep: str = "last",
) -> pd.DataFrame:
    """Merge new data with existing CSV, dedup on timestamp.

    ``keep="last"`` lets new bars overwrite existing ones; ``"first"`` keeps existing ones.
    """
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="timestamp", keep=keep)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        return combined
    return new_df


_UPSERT_CHUNK = 2000  # rows per INSERT; keeps bind params under SQLite's 32k limit
_OHLCV = ("open", "high", "low", "close", "volume")


def persist_to_db(df: pd.DataFrame, symbol: str, tf: Timeframe) -> None:
    """Upsert candle data into the CandleRecord table (bulk ON CONFLICT, naive-UTC timestamps)."""
    try:
        from rainier.core.database import get_session
        from rainier.core.models import CandleRecord

        ts = pd.to_datetime(df["timestamp"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        rows = [
            {"symbol": symbol, "timeframe": tf.value, "timestamp": t.to_pydatetime(),
             **{c: float(v) for c, v in zip(_OHLCV, vals, strict=True)}}
            for t, vals in zip(ts, df[list(_OHLCV)].itertuples(index=False), strict=True)
        ]

        with get_session() as db:
            if db.get_bind().dialect.name == "postgresql":
                from sqlalchemy.dialects.postgresql import insert
            else:
                from sqlalchemy.dialects.sqlite import insert
            for i in range(0, len(rows), _UPSERT_CHUNK):
                stmt = insert(CandleRecord).values(rows[i:i + _UPSERT_CHUNK])
                db.execute(stmt.on_conflict_do_update(
                    index_elements=["symbol", "timeframe", "timestamp"],
                    set_={c: stmt.excluded[c] for c in _OHLCV},
                ))

        log.info("db_persisted", symbol=symbol, timeframe=tf.value, rows=len(df))
    except Exception as exc:
        log.warning("db_persist_skipped", symbol=symbol, reason=str(exc))
