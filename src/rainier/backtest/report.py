"""Backtest report: equity curve, stats summary."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from .engine import BacktestResult


def format_report(result: BacktestResult) -> str:
    """Format backtest results as a text report."""
    lines = [
        "=" * 50,
        "BACKTEST REPORT",
        "=" * 50,
        f"Total trades:    {result.total_trades}",
        f"Winners:         {len(result.winners)}",
        f"Losers:          {len(result.losers)}",
        f"Win rate:        {result.win_rate:.1%}",
        f"Profit factor:   {result.profit_factor:.2f}",
        f"Total P&L:       {result.total_pnl:+,.2f}",
        f"Max drawdown:    {result.max_drawdown:.2%}",
        f"Final equity:    {result.equity_curve[-1]:,.2f}" if result.equity_curve else "",
        "=" * 50,
    ]

    if result.trades:
        pnls = [t.pnl for t in result.trades]
        lines.extend([
            f"Avg win:         {sum(t.pnl for t in result.winners) / len(result.winners):+,.2f}" if result.winners else "",
            f"Avg loss:        {sum(t.pnl for t in result.losers) / len(result.losers):+,.2f}" if result.losers else "",
            f"Largest win:     {max(pnls):+,.2f}",
            f"Largest loss:    {min(pnls):+,.2f}",
        ])

    return "\n".join(line for line in lines if line)


def plot_equity_curve(result: BacktestResult, output_path: Path | None = None) -> go.Figure:
    """Plot the equity curve."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=result.equity_curve,
            mode="lines",
            name="Equity",
            line=dict(color="#26a69a", width=2),
        )
    )

    fig.update_layout(
        title="Equity Curve",
        yaxis_title="Equity ($)",
        xaxis_title="Bar",
        template="plotly_dark",
        height=400,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

    return fig
