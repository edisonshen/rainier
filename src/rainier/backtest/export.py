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


def export_equity_curve(metrics: BacktestMetrics, path: Path) -> Path:
    """Export equity curve to CSV."""
    df = pd.DataFrame({
        "bar": range(len(metrics.equity_curve)),
        "equity": metrics.equity_curve,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def export_summary(metrics: BacktestMetrics, path: Path) -> Path:
    """Export aggregate metrics summary to CSV (single row, useful for sweep comparisons)."""
    row = {
        "total_trades": metrics.total_trades,
        "winners": metrics.winners,
        "losers": metrics.losers,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "total_net_pnl": metrics.total_net_pnl,
        "total_commission": metrics.total_commission,
        "total_slippage": metrics.total_slippage,
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "sharpe_ratio": metrics.sharpe_ratio,
        "avg_win": metrics.avg_win,
        "avg_loss": metrics.avg_loss,
        "avg_hold_bars": metrics.avg_hold_bars,
        "avg_mae": metrics.avg_mae,
        "avg_mfe": metrics.avg_mfe,
        "largest_win": metrics.largest_win,
        "largest_loss": metrics.largest_loss,
        "initial_capital": metrics.initial_capital,
        "final_equity": metrics.final_equity,
        "slippage_pct": metrics.slippage_pct,
        "commission_per_trade": metrics.commission_per_trade,
        "min_confidence": metrics.min_confidence,
        "min_rr_ratio": metrics.min_rr_ratio,
    }
    df = pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
