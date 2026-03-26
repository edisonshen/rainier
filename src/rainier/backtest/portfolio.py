"""Multi-symbol portfolio backtest — runs backtest per symbol and aggregates.

Dependency rule: imports only from core/ and backtest/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rainier.core.config import BacktestConfig
from rainier.core.protocols import BacktestMetrics, SignalEmitter
from rainier.core.types import Timeframe

from .engine import run_backtest

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SymbolResult:
    """Per-symbol backtest result with allocation weight."""
    symbol: str
    metrics: BacktestMetrics
    weight: float  # allocation weight (e.g., 0.33 for 3 symbols)


@dataclass
class PortfolioResult:
    """Aggregate portfolio backtest results."""
    symbol_results: list[SymbolResult] = field(default_factory=list)
    combined_equity_curve: list[float] = field(default_factory=list)
    total_net_pnl: float = 0.0
    total_trades: int = 0
    per_symbol_pnl: dict[str, float] = field(default_factory=dict)
    portfolio_sharpe: float = 0.0
    portfolio_max_drawdown_pct: float = 0.0
    initial_capital: float = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_portfolio_backtest(
    data: dict[str, pd.DataFrame],
    timeframes: dict[str, Timeframe],
    signal_emitter: SignalEmitter,
    config: BacktestConfig | None = None,
) -> PortfolioResult:
    """Run backtest per symbol and aggregate portfolio metrics.

    Args:
        data: symbol -> OHLCV DataFrame
        timeframes: symbol -> Timeframe
        signal_emitter: Shared emitter (or per-symbol if wrapping)
        config: Base backtest config (capital split equally)

    Returns:
        PortfolioResult with per-symbol and combined metrics
    """
    if config is None:
        config = BacktestConfig()

    symbols = list(data.keys())
    n_symbols = len(symbols)
    if n_symbols == 0:
        return PortfolioResult(initial_capital=config.initial_capital)

    # Equal capital allocation
    per_symbol_capital = config.initial_capital / n_symbols
    weight = 1.0 / n_symbols

    result = PortfolioResult(initial_capital=config.initial_capital)
    all_equity_deltas: list[list[float]] = []

    for sym in symbols:
        sym_config = config.model_copy()
        sym_config.initial_capital = per_symbol_capital

        tf = timeframes.get(sym, Timeframe.H1)
        metrics = run_backtest(data[sym], sym, tf, signal_emitter, sym_config)

        sym_result = SymbolResult(
            symbol=sym, metrics=metrics, weight=weight,
        )
        result.symbol_results.append(sym_result)
        result.per_symbol_pnl[sym] = metrics.total_net_pnl
        result.total_trades += metrics.total_trades

        # Collect equity deltas
        if len(metrics.equity_curve) > 1:
            deltas = [
                metrics.equity_curve[i] - metrics.equity_curve[i - 1]
                for i in range(1, len(metrics.equity_curve))
            ]
        else:
            deltas = []
        all_equity_deltas.append(deltas)

    # Build combined equity curve
    max_len = max(len(d) for d in all_equity_deltas) if all_equity_deltas else 0
    combined = [config.initial_capital]
    for i in range(max_len):
        bar_delta = 0.0
        for deltas in all_equity_deltas:
            if i < len(deltas):
                bar_delta += deltas[i]
        combined.append(combined[-1] + bar_delta)

    result.combined_equity_curve = combined
    result.total_net_pnl = sum(result.per_symbol_pnl.values())

    # Portfolio Sharpe
    if len(combined) > 1:
        returns = np.diff(combined) / np.array(combined[:-1])
        if len(returns) > 0 and np.std(returns) > 0:
            result.portfolio_sharpe = float(
                np.mean(returns) / np.std(returns) * math.sqrt(252)
            )

    # Portfolio max drawdown
    peak = combined[0]
    max_dd_pct = 0.0
    for eq in combined:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd_pct = (peak - eq) / peak
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
    result.portfolio_max_drawdown_pct = max_dd_pct

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_portfolio_report(result: PortfolioResult) -> str:
    """Format portfolio results as a readable text report."""
    lines = [
        "=" * 80,
        "PORTFOLIO BACKTEST RESULTS",
        "=" * 80,
        "",
        f"  Initial capital: ${result.initial_capital:,.2f}",
        f"  Symbols: {len(result.symbol_results)}",
        f"  Total trades: {result.total_trades}",
        f"  Total net P&L: ${result.total_net_pnl:+,.2f}",
        f"  Portfolio Sharpe: {result.portfolio_sharpe:.2f}",
        f"  Portfolio max DD: {result.portfolio_max_drawdown_pct:.2%}",
        "",
        "PER-SYMBOL BREAKDOWN",
        "-" * 80,
        f"  {'Symbol':<10} {'Weight':>7} {'Trades':>7} {'WinRate':>8} "
        f"{'PF':>7} {'NetPnL':>12} {'Sharpe':>8} {'MaxDD':>8}",
        "-" * 80,
    ]

    for sr in result.symbol_results:
        m = sr.metrics
        lines.append(
            f"  {sr.symbol:<10} {sr.weight:>6.1%} {m.total_trades:>7} "
            f"{m.win_rate:>7.1%} {m.profit_factor:>7.2f} "
            f"{m.total_net_pnl:>+12,.2f} {m.sharpe_ratio:>8.2f} "
            f"{m.max_drawdown_pct:>7.2%}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)
