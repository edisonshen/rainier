"""Event-driven backtest engine with pre-computed S/R for performance."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from rainier.analysis.analyzer import analyze
from rainier.core.config import AnalysisConfig, SignalConfig
from rainier.core.types import Direction, Signal, Timeframe
from rainier.signals.generator import generate_signals


@dataclass
class BacktestTrade:
    signal: Signal
    entry_bar: int
    exit_bar: int | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    initial_capital: float = 100_000.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winners(self) -> list[BacktestTrade]:
        return [t for t in self.trades if t.pnl > 0]

    @property
    def losers(self) -> list[BacktestTrade]:
        return [t for t in self.trades if t.pnl <= 0]

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return len(self.winners) / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.winners)
        gross_loss = abs(sum(t.pnl for t in self.losers))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    analysis_config: AnalysisConfig | None = None,
    signal_config: SignalConfig | None = None,
    sr_recompute_interval: int = 50,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Run a backtest over historical data.

    Optimization: S/R levels are pre-computed every `sr_recompute_interval` bars,
    not on every bar. Pin bar detection runs per-bar against cached levels.

    Handles gap-through-stop: fills at actual price, not stop price.
    """
    if analysis_config is None:
        analysis_config = AnalysisConfig()
    if signal_config is None:
        signal_config = SignalConfig()

    result = BacktestResult(initial_capital=initial_capital)
    capital = initial_capital
    result.equity_curve.append(capital)

    open_trades: list[BacktestTrade] = []
    min_bars = max(analysis_config.pivot.lookback * 2 + 1, 30)

    for i in range(min_bars, len(df)):
        bar = df.iloc[i]

        # Check open trades for exit
        closed = []
        for trade in open_trades:
            exit_price, reason = _check_exit(trade.signal, bar)
            if exit_price is not None:
                trade.exit_bar = i
                trade.exit_price = exit_price
                trade.exit_reason = reason

                if trade.signal.direction == Direction.LONG:
                    trade.pnl = exit_price - trade.signal.entry_price
                else:
                    trade.pnl = trade.signal.entry_price - exit_price

                capital += trade.pnl
                closed.append(trade)
                result.trades.append(trade)

        for t in closed:
            open_trades.remove(t)

        # Re-compute S/R at intervals
        if i % sr_recompute_interval == 0:
            lookback_df = df.iloc[max(0, i - 500) : i + 1].reset_index(drop=True)
            analysis = analyze(lookback_df, symbol, timeframe, analysis_config)
            signals = generate_signals(analysis, lookback_df, signal_config)

            # Check for new entries
            for signal in signals:
                # Only take the most recent signal (last bar)
                if signal.timestamp != pd.Timestamp(bar["timestamp"]).to_pydatetime():
                    continue

                # Check if limit order would fill on this bar
                if _would_fill(signal, bar):
                    bt_trade = BacktestTrade(signal=signal, entry_bar=i)
                    open_trades.append(bt_trade)

        result.equity_curve.append(capital)

    # Close any remaining open trades at last bar's close
    last_bar = df.iloc[-1]
    for trade in open_trades:
        trade.exit_bar = len(df) - 1
        trade.exit_price = float(last_bar["close"])
        trade.exit_reason = "end_of_data"
        if trade.signal.direction == Direction.LONG:
            trade.pnl = trade.exit_price - trade.signal.entry_price
        else:
            trade.pnl = trade.signal.entry_price - trade.exit_price
        capital += trade.pnl
        result.trades.append(trade)
        result.equity_curve.append(capital)

    return result


def _check_exit(signal: Signal, bar: pd.Series) -> tuple[float | None, str]:
    """Check if a bar triggers SL or TP.

    Gap-through-stop: if bar opens beyond SL, fill at open (gap price), not stop price.
    """
    bar_open = float(bar["open"])
    bar_high = float(bar["high"])
    bar_low = float(bar["low"])

    if signal.direction == Direction.LONG:
        # Check SL (gap through = fill at open, not SL price)
        if bar_low <= signal.stop_loss:
            fill = min(bar_open, signal.stop_loss)  # gap through → worse fill
            return fill, "stop_loss"
        # Check TP
        if bar_high >= signal.take_profit:
            fill = max(bar_open, signal.take_profit)
            return fill, "take_profit"
    else:
        # Short: SL is above entry
        if bar_high >= signal.stop_loss:
            fill = max(bar_open, signal.stop_loss)  # gap through → worse fill
            return fill, "stop_loss"
        # Check TP
        if bar_low <= signal.take_profit:
            fill = min(bar_open, signal.take_profit)
            return fill, "take_profit"

    return None, ""


def _would_fill(signal: Signal, bar: pd.Series) -> bool:
    """Check if a limit order at entry_price would fill on this bar."""
    bar_high = float(bar["high"])
    bar_low = float(bar["low"])

    if signal.direction == Direction.LONG:
        return bar_low <= signal.entry_price
    else:
        return bar_high >= signal.entry_price
