"""Trade export — BacktestMetrics → CSV / Parquet for external analysis."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from rainier.core.protocols import BacktestMetrics


def trades_to_dataframe(metrics: BacktestMetrics) -> pd.DataFrame:
    """Convert trade records to a DataFrame for analysis."""
    if not metrics.trades:
        return pd.DataFrame()
    return pd.DataFrame([asdict(t) for t in metrics.trades])


def export_trades_csv(metrics: BacktestMetrics, path: Path) -> Path:
    """Export trade log to CSV."""
    df = trades_to_dataframe(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def export_trades_parquet(metrics: BacktestMetrics, path: Path) -> Path:
    """Export trade log to Parquet (columnar, compressed)."""
    df = trades_to_dataframe(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    return path
