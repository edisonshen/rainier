"""Compare signal emitters by running backtests side-by-side.

Answers the question: "Does the ML model produce better trading results
than the rule-based BookScorer on the same historical data?"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from rainier.backtest.engine import run_backtest
from rainier.core.config import BacktestConfig
from rainier.core.protocols import BacktestMetrics, SignalEmitter
from rainier.core.types import Timeframe


@dataclass
class ComparisonResult:
    """Side-by-side backtest results for multiple emitters."""

    rows: list[tuple[str, BacktestMetrics]] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        return [label for label, _ in self.rows]

    def get_metrics(self, label: str) -> BacktestMetrics | None:
        for lbl, metrics in self.rows:
            if lbl == label:
                return metrics
        return None


def compare_emitters(
    df: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    emitters: dict[str, SignalEmitter],
    config: BacktestConfig | None = None,
) -> ComparisonResult:
    """Run backtest with each emitter and collect results.

    Args:
        df: OHLCV DataFrame
        symbol: Instrument symbol
        timeframe: Bar timeframe
        emitters: Mapping of label → SignalEmitter (e.g. "BookScorer" → emitter)
        config: Shared backtest configuration

    Returns:
        ComparisonResult with one (label, BacktestMetrics) per emitter
    """
    if config is None:
        config = BacktestConfig()

    result = ComparisonResult()

    for label, emitter in emitters.items():
        metrics = run_backtest(df, symbol, timeframe, emitter, config)
        result.rows.append((label, metrics))

    return result


def format_comparison_table(result: ComparisonResult) -> str:
    """Format comparison results as a side-by-side ASCII table."""
    if not result.rows:
        return "No results to compare."

    # Metrics to display
    metric_defs = [
        ("Total trades", lambda m: f"{m.total_trades}"),
        ("Winners", lambda m: f"{m.winners}"),
        ("Losers", lambda m: f"{m.losers}"),
        ("Win rate", lambda m: f"{m.win_rate:.1%}"),
        ("Profit factor", lambda m: f"{m.profit_factor:.2f}"),
        ("Net P&L", lambda m: f"{m.total_net_pnl:+,.2f}"),
        ("Gross P&L", lambda m: f"{m.total_gross_pnl:+,.2f}"),
        ("Commission", lambda m: f"{m.total_commission:,.2f}"),
        ("Slippage", lambda m: f"{m.total_slippage:,.2f}"),
        ("Max drawdown", lambda m: f"{m.max_drawdown_pct:.1%}"),
        ("Sharpe ratio", lambda m: f"{m.sharpe_ratio:.2f}"),
        ("Avg win", lambda m: f"{m.avg_win:+,.2f}"),
        ("Avg loss", lambda m: f"{m.avg_loss:+,.2f}"),
        ("Avg hold bars", lambda m: f"{m.avg_hold_bars:.1f}"),
        ("Largest win", lambda m: f"{m.largest_win:+,.2f}"),
        ("Largest loss", lambda m: f"{m.largest_loss:+,.2f}"),
        ("Final equity", lambda m: f"{m.final_equity:,.2f}"),
    ]

    labels = result.labels
    col_width = max(14, *(len(label) + 2 for label in labels))
    label_col_width = max(len(name) for name, _ in metric_defs) + 2

    # Header
    lines: list[str] = []
    sep = "=" * (label_col_width + col_width * len(labels) + 4)
    lines.append(sep)
    lines.append("SCORER COMPARISON")
    lines.append(sep)

    # Column headers
    header = f"{'':>{label_col_width}}"
    for label in labels:
        header += f"  {label:>{col_width}}"
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for metric_name, fmt_fn in metric_defs:
        row = f"{metric_name:>{label_col_width}}"
        for _, metrics in result.rows:
            row += f"  {fmt_fn(metrics):>{col_width}}"
        lines.append(row)

    lines.append(sep)

    # Delta summary (if exactly 2 emitters)
    if len(result.rows) == 2:
        _, m1 = result.rows[0]
        _, m2 = result.rows[1]
        lines.append("")
        lines.append(f"Delta ({labels[1]} vs {labels[0]}):")

        pf_delta = m2.profit_factor - m1.profit_factor
        wr_delta = m2.win_rate - m1.win_rate
        pnl_delta = m2.total_net_pnl - m1.total_net_pnl
        dd_delta = m2.max_drawdown_pct - m1.max_drawdown_pct
        sharpe_delta = m2.sharpe_ratio - m1.sharpe_ratio

        lines.append(f"  Profit factor: {pf_delta:+.2f}")
        lines.append(f"  Win rate:      {wr_delta:+.1%}")
        lines.append(f"  Net P&L:       {pnl_delta:+,.2f}")
        lines.append(f"  Max drawdown:  {dd_delta:+.1%}")
        lines.append(f"  Sharpe ratio:  {sharpe_delta:+.2f}")

    return "\n".join(lines)
