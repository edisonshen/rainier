"""Backtest report: text summary, equity curve chart, per-trade detail."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from rainier.core.protocols import BacktestMetrics


def format_report(metrics: BacktestMetrics) -> str:
    """Format backtest results as a text report."""
    lines = [
        "=" * 60,
        "BACKTEST REPORT",
        "=" * 60,
        f"Total trades:       {metrics.total_trades}",
        f"Winners:            {metrics.winners}",
        f"Losers:             {metrics.losers}",
        f"Win rate:           {metrics.win_rate:.1%}",
        f"Profit factor:      {metrics.profit_factor:.2f}",
        "-" * 60,
        f"Gross P&L:          {metrics.total_gross_pnl:+,.2f}",
        f"Commission:         {metrics.total_commission:,.2f}",
        f"Slippage:           {metrics.total_slippage:,.2f}",
        f"Net P&L:            {metrics.total_net_pnl:+,.2f}",
        "-" * 60,
        f"Max drawdown:       {metrics.max_drawdown_pct:.2%} ({metrics.max_drawdown:+,.2f})",
        f"Sharpe ratio:       {metrics.sharpe_ratio:.2f}",
        "-" * 60,
        f"Avg win:            {metrics.avg_win:+,.2f}",
        f"Avg loss:           {metrics.avg_loss:+,.2f}",
        f"Largest win:        {metrics.largest_win:+,.2f}",
        f"Largest loss:       {metrics.largest_loss:+,.2f}",
        f"Avg hold (bars):    {metrics.avg_hold_bars:.1f}",
        f"Avg MAE:            {metrics.avg_mae:,.2f}",
        f"Avg MFE:            {metrics.avg_mfe:,.2f}",
        "-" * 60,
        f"Initial capital:    {metrics.initial_capital:,.2f}",
        f"Final equity:       {metrics.final_equity:,.2f}",
        f"Return:             {(metrics.final_equity / metrics.initial_capital - 1):.2%}"
        if metrics.initial_capital > 0 else "",
        "=" * 60,
        f"Config: slippage={metrics.slippage_pct:.4f}, "
        f"commission={metrics.commission_per_trade:.2f}/side",
    ]

    return "\n".join(line for line in lines if line)


def format_trade_log(metrics: BacktestMetrics, max_trades: int = 50) -> str:
    """Format individual trade details as a text table."""
    if not metrics.trades:
        return "No trades."

    lines = [
        "=" * 120,
        "TRADE LOG",
        "=" * 120,
        f"{'#':>4} {'Dir':>5} {'Entry':>10} {'Exit':>10} {'NetPnL':>10} "
        f"{'Conf':>5} {'R:R':>5} {'MAE':>8} {'MFE':>8} {'Bars':>5} {'Reason':>12} {'Why':>25}",
        "-" * 120,
    ]

    for t in metrics.trades[:max_trades]:
        lines.append(
            f"{t.trade_id:>4} {t.direction:>5} {t.entry_price:>10.2f} "
            f"{t.exit_price:>10.2f} {t.net_pnl:>+10.2f} "
            f"{t.confidence:>5.2f} {t.rr_ratio:>5.1f} "
            f"{t.mae:>8.2f} {t.mfe:>8.2f} {t.hold_bars:>5} "
            f"{t.exit_reason:>12} {t.entry_reason:>25}"
        )

    if len(metrics.trades) > max_trades:
        lines.append(f"  ... and {len(metrics.trades) - max_trades} more trades")

    lines.append("=" * 120)
    return "\n".join(lines)


def plot_equity_curve(
    metrics: BacktestMetrics, output_path: Path | None = None,
) -> go.Figure:
    """Plot the equity curve."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=metrics.equity_curve,
            mode="lines",
            name="Equity",
            line=dict(color="#26a69a", width=2),
        )
    )

    # Add drawdown shading
    peak = metrics.equity_curve[0]
    peaks = []
    for eq in metrics.equity_curve:
        if eq > peak:
            peak = eq
        peaks.append(peak)

    fig.add_trace(
        go.Scatter(
            y=peaks,
            mode="lines",
            name="Peak",
            line=dict(color="#ef5350", width=1, dash="dot"),
            opacity=0.5,
        )
    )

    fig.update_layout(
        title=f"Equity Curve — Net P&L: {metrics.total_net_pnl:+,.2f} | "
              f"Sharpe: {metrics.sharpe_ratio:.2f} | "
              f"MaxDD: {metrics.max_drawdown_pct:.1%}",
        yaxis_title="Equity ($)",
        xaxis_title="Bar",
        template="plotly_dark",
        height=400,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

    return fig
