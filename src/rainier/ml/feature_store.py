"""Feature store — exports ML-ready features + labels to Parquet files.

Pipeline: DB (StockPrice) → OHLCV DataFrame → analyze() → FeatureExtractor → labels → Parquet

Each Parquet file contains one row per bar per symbol with ~50 features,
forward-return labels, and metadata (symbol, date).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from rainier.analysis.analyzer import analyze
from rainier.core.config import AnalysisConfig
from rainier.core.database import get_session
from rainier.core.models import StockPrice
from rainier.core.types import Timeframe
from rainier.features.extractor import FeatureExtractor

logger = logging.getLogger(__name__)

# Minimum bars needed for meaningful feature extraction
MIN_BARS = 100

# Forward return horizons for labels
LABEL_HORIZONS = [5, 10, 20]

# Threshold for binary label (1% move)
LABEL_THRESHOLD = 0.01


def load_ohlcv_from_db(
    symbol: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Load daily OHLCV for a single symbol from the StockPrice table."""
    with get_session() as db:
        rows = db.execute(
            select(
                StockPrice.date,
                StockPrice.open,
                StockPrice.high,
                StockPrice.low,
                StockPrice.close,
                StockPrice.volume,
            ).where(
                StockPrice.symbol == symbol,
                StockPrice.date >= start.isoformat(),
                StockPrice.date <= end.isoformat(),
            ).order_by(StockPrice.date)
        ).all()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Fill missing volume with 0
    df["volume"] = df["volume"].fillna(0).astype(float)
    df = df.dropna(subset=["close"])
    return df.reset_index(drop=True)


def compute_forward_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward-return labels for multiple horizons.

    For each horizon N, creates:
    - fwd_return_{N}d: raw forward return (float)
    - label_{N}d: binary label (1 if return > threshold, 0 if < -threshold, NaN if neutral)
    """
    close = df["close"].values
    labels = pd.DataFrame(index=df.index)

    for horizon in LABEL_HORIZONS:
        fwd = np.full(len(close), np.nan)
        for i in range(len(close) - horizon):
            fwd[i] = (close[i + horizon] - close[i]) / close[i]

        labels[f"fwd_return_{horizon}d"] = fwd
        binary = np.full(len(close), np.nan)
        binary[fwd > LABEL_THRESHOLD] = 1.0
        binary[fwd < -LABEL_THRESHOLD] = 0.0
        labels[f"label_{horizon}d"] = binary

    return labels


def export_symbol_features(
    symbol: str,
    start: date,
    end: date,
    config: AnalysisConfig | None = None,
) -> pd.DataFrame | None:
    """Extract features + labels for a single symbol.

    Returns DataFrame with features, labels, and metadata columns,
    or None if insufficient data.
    """
    df = load_ohlcv_from_db(symbol, start, end)
    if len(df) < MIN_BARS:
        logger.debug("%s: only %d bars (need %d), skipping", symbol, len(df), MIN_BARS)
        return None

    # Run analysis pipeline
    try:
        result = analyze(df, symbol, Timeframe.D1, config=config)
    except Exception:
        logger.exception("%s: analysis failed", symbol)
        return None

    # Extract features
    extractor = FeatureExtractor()
    try:
        features = extractor.extract(result, df)
    except Exception:
        logger.exception("%s: feature extraction failed", symbol)
        return None

    # Compute forward-return labels
    labels = compute_forward_labels(df)

    # Combine features + labels + metadata
    out = features.copy()
    for col in labels.columns:
        out[col] = labels[col].values

    out["symbol"] = symbol
    out["date"] = df["timestamp"].values
    out["close"] = df["close"].values
    out["volume"] = df["volume"].values

    return out


def export_training_data(
    symbols: list[str],
    start: date,
    end: date,
    output_dir: Path,
    config: AnalysisConfig | None = None,
) -> Path:
    """Export features + labels for multiple symbols to a single Parquet file.

    Args:
        symbols: Stock tickers to process.
        start: Start date for historical data.
        end: End date for historical data.
        output_dir: Directory to write the Parquet file.
        config: Analysis config (uses defaults if None).

    Returns:
        Path to the output Parquet file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    processed = 0
    skipped = 0

    for i, symbol in enumerate(symbols, 1):
        if i % 50 == 0 or i == len(symbols):
            logger.info("Progress: %d/%d symbols (exported %d)", i, len(symbols), processed)

        result = export_symbol_features(symbol, start, end, config)
        if result is not None:
            frames.append(result)
            processed += 1
        else:
            skipped += 1

    if not frames:
        raise ValueError(f"No features exported — all {len(symbols)} symbols skipped")

    combined = pd.concat(frames, ignore_index=True)

    # Write Parquet
    filename = f"features_{start.isoformat()}_{end.isoformat()}_{len(symbols)}sym.parquet"
    output_path = output_dir / filename
    combined.to_parquet(output_path, index=False, engine="pyarrow")

    logger.info(
        "Feature store exported: %d rows, %d symbols, %d features → %s",
        len(combined), processed, len(combined.columns), output_path,
    )
    return output_path


def get_qu100_symbols() -> list[str]:
    """Get all distinct symbols from QU100 money flow rankings."""
    from sqlalchemy import func

    from rainier.core.models import MoneyFlowSnapshot

    with get_session() as db:
        symbols = db.execute(
            select(func.distinct(MoneyFlowSnapshot.symbol)).order_by(
                MoneyFlowSnapshot.symbol
            )
        ).scalars().all()
    return list(symbols)


def get_symbols_with_prices(start: date, end: date, min_bars: int = MIN_BARS) -> list[str]:
    """Get symbols that have sufficient price data in the given date range."""
    from sqlalchemy import func

    with get_session() as db:
        rows = db.execute(
            select(
                StockPrice.symbol,
                func.count().label("bar_count"),
            ).where(
                StockPrice.date >= start.isoformat(),
                StockPrice.date <= end.isoformat(),
            ).group_by(StockPrice.symbol).having(
                func.count() >= min_bars
            ).order_by(StockPrice.symbol)
        ).all()
    return [row[0] for row in rows]


def validate_parquet(path: Path) -> dict:
    """Validate an exported Parquet file and return summary stats."""
    df = pd.read_parquet(path)

    nan_counts = df.isna().sum()
    feature_cols = [c for c in df.columns if c not in (
        "symbol", "date", "close", "volume",
    ) and not c.startswith("fwd_return_") and not c.startswith("label_")]

    feature_nans = nan_counts[feature_cols]
    label_cols = [c for c in df.columns if c.startswith("label_")]

    return {
        "rows": len(df),
        "symbols": df["symbol"].nunique(),
        "features": len(feature_cols),
        "date_range": f"{df['date'].min()} to {df['date'].max()}",
        "feature_nan_total": int(feature_nans.sum()),
        "label_coverage": {
            col: f"{df[col].notna().mean():.1%}" for col in label_cols
        },
        "label_positive_rate": {
            col: f"{df[col].mean():.1%}" for col in label_cols if df[col].notna().any()
        },
    }
