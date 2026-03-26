"""Parameter sweep runner — test backtest across config combinations.

Depends only on core/ and backtest/engine. The SignalEmitter factory
is injected so sweep doesn't know which signal strategy is being tested.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from rainier.core.config import BacktestConfig
from rainier.core.protocols import SignalEmitter
from rainier.core.types import Timeframe

from .engine import run_backtest


@dataclass(frozen=True)
class SweepParams:
    """A single parameter combination to test."""
    min_confidence: float
    min_rr_ratio: float
    slippage_pct: float = 0.0005
    commission_per_trade: float = 2.50


@dataclass
class SweepResult:
    """Results from a full parameter sweep."""
    rows: list[dict] = field(default_factory=list)  # one dict per param combo
    best_by_pnl: SweepParams | None = None
    best_by_sharpe: SweepParams | None = None
    best_by_profit_factor: SweepParams | None = None

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# Type alias: factory that creates a SignalEmitter given confidence + rr params
EmitterFactory = Callable[[float, float], SignalEmitter]


def run_sweep(
    df: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    emitter_factory: EmitterFactory,
    config: BacktestConfig | None = None,
    confidence_values: list[float] | None = None,
    rr_values: list[float] | None = None,
) -> SweepResult:
    """Run backtest across all combinations of confidence × rr_ratio.

    Args:
        df: OHLCV data
        symbol: Instrument symbol
        timeframe: Bar timeframe
        emitter_factory: Creates a SignalEmitter for given (min_confidence, min_rr_ratio).
                        This keeps sweep decoupled from any specific signal strategy.
        config: Base backtest config (sweep overrides confidence/rr)
        confidence_values: Confidence thresholds to test
        rr_values: Min R:R ratios to test

    Returns:
        SweepResult with comparison table and best params
    """
    if config is None:
        config = BacktestConfig()
    if confidence_values is None:
        confidence_values = config.sweep_min_confidence
    if rr_values is None:
        rr_values = config.sweep_min_rr_ratio

    result = SweepResult()
    best_pnl = float("-inf")
    best_sharpe = float("-inf")
    best_pf = float("-inf")

    combos = list(itertools.product(confidence_values, rr_values))

    for idx, (conf, rr) in enumerate(combos, 1):
        emitter = emitter_factory(conf, rr)
        metrics = run_backtest(df, symbol, timeframe, emitter, config)

        # Tag metrics with the params used
        metrics.min_confidence = conf
        metrics.min_rr_ratio = rr

        params = SweepParams(
            min_confidence=conf,
            min_rr_ratio=rr,
            slippage_pct=config.slippage_pct,
            commission_per_trade=config.commission_per_trade,
        )

        row = {
            "min_confidence": conf,
            "min_rr_ratio": rr,
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "total_net_pnl": metrics.total_net_pnl,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "avg_win": metrics.avg_win,
            "avg_loss": metrics.avg_loss,
            "avg_hold_bars": metrics.avg_hold_bars,
            "final_equity": metrics.final_equity,
        }
        result.rows.append(row)

        if metrics.total_net_pnl > best_pnl:
            best_pnl = metrics.total_net_pnl
            result.best_by_pnl = params
        if metrics.sharpe_ratio > best_sharpe:
            best_sharpe = metrics.sharpe_ratio
            result.best_by_sharpe = params
        if metrics.profit_factor > best_pf and metrics.total_trades >= 5:
            best_pf = metrics.profit_factor
            result.best_by_profit_factor = params

    return result


def format_sweep_table(sweep: SweepResult) -> str:
    """Format sweep results as a readable text table."""
    if not sweep.rows:
        return "No sweep results."

    df = sweep.to_dataframe()
    df = df.sort_values("total_net_pnl", ascending=False)

    lines = [
        "=" * 100,
        "PARAMETER SWEEP RESULTS",
        "=" * 100,
        f"{'Conf':>6} {'R:R':>5} {'Trades':>7} {'WinRate':>8} {'PF':>7} "
        f"{'NetPnL':>12} {'MaxDD%':>8} {'Sharpe':>8} {'AvgWin':>10} {'AvgLoss':>10}",
        "-" * 100,
    ]

    for _, row in df.iterrows():
        lines.append(
            f"{row['min_confidence']:>6.2f} {row['min_rr_ratio']:>5.1f} "
            f"{row['total_trades']:>7.0f} {row['win_rate']:>7.1%} "
            f"{row['profit_factor']:>7.2f} {row['total_net_pnl']:>+12,.2f} "
            f"{row['max_drawdown_pct']:>7.2%} {row['sharpe_ratio']:>8.2f} "
            f"{row['avg_win']:>+10,.2f} {row['avg_loss']:>+10,.2f}"
        )

    lines.append("=" * 100)

    if sweep.best_by_pnl:
        lines.append(
            f"Best by P&L:    conf={sweep.best_by_pnl.min_confidence:.2f}, "
            f"rr={sweep.best_by_pnl.min_rr_ratio:.1f}"
        )
    if sweep.best_by_sharpe:
        lines.append(
            f"Best by Sharpe: conf={sweep.best_by_sharpe.min_confidence:.2f}, "
            f"rr={sweep.best_by_sharpe.min_rr_ratio:.1f}"
        )
    if sweep.best_by_profit_factor:
        lines.append(
            f"Best by PF:     conf={sweep.best_by_profit_factor.min_confidence:.2f}, "
            f"rr={sweep.best_by_profit_factor.min_rr_ratio:.1f}"
        )

    return "\n".join(lines)
