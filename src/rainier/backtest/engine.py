"""Event-driven backtest engine with protocol-based signal injection.

Dependency rule: this module imports ONLY from core/.
Signal generation is injected via the SignalEmitter protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rainier.core.config import BacktestConfig
from rainier.core.protocols import BacktestMetrics, SignalEmitter, TradeRecord
from rainier.core.types import Direction, Signal, Timeframe

# ---------------------------------------------------------------------------
# Internal trade tracking (not exported)
# ---------------------------------------------------------------------------


@dataclass
class _OpenTrade:
    """Mutable state for a trade that hasn't exited yet."""
    signal: Signal
    trade_id: int
    entry_bar: int
    entry_price: float  # actual fill after slippage
    slippage_cost: float
    mae: float = 0.0  # tracks worst adverse move
    mfe: float = 0.0  # tracks best favorable move


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    signal_emitter: SignalEmitter,
    config: BacktestConfig | None = None,
) -> BacktestMetrics:
    """Run a backtest over historical data.

    Args:
        df: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
        symbol: Instrument symbol
        timeframe: Bar timeframe
        signal_emitter: Protocol implementation that produces signals from data
        config: Backtest configuration (slippage, commission, etc.)

    Returns:
        BacktestMetrics with full trade log and aggregate statistics
    """
    if config is None:
        config = BacktestConfig()

    capital = config.initial_capital
    equity_curve: list[float] = [capital]
    open_trades: list[_OpenTrade] = []
    closed_trades: list[TradeRecord] = []
    trade_counter = 0

    min_bars = 30  # need enough history for analysis

    for i in range(min_bars, len(df)):
        bar = df.iloc[i]

        # --- Phase 1: Check exits on open trades ---
        still_open: list[_OpenTrade] = []
        for ot in open_trades:
            _update_mae_mfe(ot, bar)

            exit_price, reason = _check_exit(ot.signal, bar)
            if exit_price is not None:
                exit_price = _apply_exit_slippage(
                    exit_price, ot.signal.direction, config.slippage_pct,
                )
                record = _close_trade(ot, exit_price, reason, i, bar, config)
                capital += record.net_pnl
                closed_trades.append(record)
            else:
                still_open.append(ot)

        open_trades = still_open

        # --- Phase 2: Emit new signals at recompute intervals ---
        if i % config.sr_recompute_interval == 0:
            lookback_df = df.iloc[max(0, i - 500) : i + 1].reset_index(drop=True)
            signals = signal_emitter.emit(lookback_df, symbol, timeframe)

            for signal in signals:
                # Only take signals from the current bar
                if signal.timestamp != pd.Timestamp(bar["timestamp"]).to_pydatetime():
                    continue

                # Position limit
                if len(open_trades) >= config.max_open_positions:
                    break

                # Check if limit order fills on this bar
                if _would_fill(signal, bar):
                    trade_counter += 1
                    entry_price, slippage_cost = _apply_entry_slippage(
                        signal.entry_price, signal.direction, config.slippage_pct,
                    )
                    ot = _OpenTrade(
                        signal=signal,
                        trade_id=trade_counter,
                        entry_bar=i,
                        entry_price=entry_price,
                        slippage_cost=slippage_cost,
                    )
                    open_trades.append(ot)

        equity_curve.append(capital)

    # --- Close remaining trades at last bar's close ---
    if len(df) > 0:
        last_bar = df.iloc[-1]
        for ot in open_trades:
            _update_mae_mfe(ot, last_bar)
            exit_price = float(last_bar["close"])
            record = _close_trade(ot, exit_price, "end_of_data", len(df) - 1, last_bar, config)
            capital += record.net_pnl
            closed_trades.append(record)
            equity_curve.append(capital)

    return compute_metrics(closed_trades, equity_curve, config)


# ---------------------------------------------------------------------------
# Slippage modeling
# ---------------------------------------------------------------------------


def _apply_entry_slippage(
    price: float, direction: Direction, slippage_pct: float,
) -> tuple[float, float]:
    """Apply slippage to entry. Returns (fill_price, slippage_cost)."""
    slippage = price * slippage_pct
    if direction == Direction.LONG:
        fill = price + slippage  # buy higher
    else:
        fill = price - slippage  # sell lower
    return fill, slippage


def _apply_exit_slippage(
    price: float, direction: Direction, slippage_pct: float,
) -> float:
    """Apply slippage to exit. Exits are always adverse."""
    slippage = price * slippage_pct
    if direction == Direction.LONG:
        return price - slippage  # sell lower
    else:
        return price + slippage  # cover higher


# ---------------------------------------------------------------------------
# Exit logic
# ---------------------------------------------------------------------------


def _check_exit(signal: Signal, bar: pd.Series) -> tuple[float | None, str]:
    """Check if a bar triggers SL or TP.

    Gap-through-stop: if bar opens beyond SL, fill at open (gap price).
    """
    bar_open = float(bar["open"])
    bar_high = float(bar["high"])
    bar_low = float(bar["low"])

    if signal.direction == Direction.LONG:
        if bar_low <= signal.stop_loss:
            fill = min(bar_open, signal.stop_loss)
            return fill, "stop_loss"
        if bar_high >= signal.take_profit:
            fill = max(bar_open, signal.take_profit)
            return fill, "take_profit"
    else:
        if bar_high >= signal.stop_loss:
            fill = max(bar_open, signal.stop_loss)
            return fill, "stop_loss"
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


# ---------------------------------------------------------------------------
# MAE/MFE tracking
# ---------------------------------------------------------------------------


def _update_mae_mfe(ot: _OpenTrade, bar: pd.Series) -> None:
    """Update max adverse/favorable excursion for an open trade."""
    bar_high = float(bar["high"])
    bar_low = float(bar["low"])

    if ot.signal.direction == Direction.LONG:
        # Favorable = price goes up, Adverse = price goes down
        favorable = bar_high - ot.entry_price
        adverse = ot.entry_price - bar_low
    else:
        # Short: Favorable = price goes down, Adverse = price goes up
        favorable = ot.entry_price - bar_low
        adverse = bar_high - ot.entry_price

    ot.mfe = max(ot.mfe, favorable)
    ot.mae = max(ot.mae, adverse)


# ---------------------------------------------------------------------------
# Trade closing
# ---------------------------------------------------------------------------


def _close_trade(
    ot: _OpenTrade,
    exit_price: float,
    reason: str,
    exit_bar: int,
    exit_bar_data: pd.Series,
    config: BacktestConfig,
) -> TradeRecord:
    """Create a TradeRecord from a closed _OpenTrade."""
    sig = ot.signal
    commission = config.commission_per_trade * 2  # entry + exit

    if sig.direction == Direction.LONG:
        gross_pnl = exit_price - ot.entry_price
    else:
        gross_pnl = ot.entry_price - exit_price

    net_pnl = gross_pnl - commission - ot.slippage_cost

    # Build entry reason from signal context
    entry_reason = ""
    sr_price = None
    sr_type = None
    pattern_type = None

    if sig.pin_bar is not None:
        entry_reason = f"pin_bar_{sig.direction.value.lower()}"
        if sig.pin_bar.nearest_sr is not None:
            sr_price = sig.pin_bar.nearest_sr.price
            sr_type = sig.pin_bar.nearest_sr.sr_type.value
            entry_reason += f"_at_{sig.pin_bar.nearest_sr.role.value}"

    if sig.sr_level is not None and sr_price is None:
        sr_price = sig.sr_level.price
        sr_type = sig.sr_level.sr_type.value

    # Parse pattern_type from notes (e.g. "pattern:w_bottom")
    if sig.notes and sig.notes.startswith("pattern:"):
        pattern_type = sig.notes.split("pattern:", 1)[1]
        if not entry_reason:
            entry_reason = f"pattern_breakout_{pattern_type}"

    return TradeRecord(
        trade_id=ot.trade_id,
        symbol=sig.symbol,
        timeframe=sig.timeframe.value,
        direction=sig.direction.value,
        entry_price=ot.entry_price,
        exit_price=exit_price,
        stop_loss=sig.stop_loss,
        take_profit=sig.take_profit,
        entry_bar=ot.entry_bar,
        exit_bar=exit_bar,
        entry_timestamp=str(sig.timestamp),
        exit_timestamp=str(exit_bar_data["timestamp"]),
        hold_bars=exit_bar - ot.entry_bar,
        gross_pnl=gross_pnl,
        commission=commission,
        slippage_cost=ot.slippage_cost,
        net_pnl=net_pnl,
        confidence=sig.confidence,
        rr_ratio=sig.rr_ratio,
        risk=abs(sig.entry_price - sig.stop_loss),
        mae=ot.mae,
        mfe=ot.mfe,
        exit_reason=reason,
        entry_reason=entry_reason,
        sr_level_price=sr_price,
        sr_level_type=sr_type,
        pattern_type=pattern_type,
    )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(
    trades: list[TradeRecord],
    equity_curve: list[float],
    config: BacktestConfig,
) -> BacktestMetrics:
    """Compute aggregate metrics from completed trades."""
    winners = [t for t in trades if t.net_pnl > 0]
    losers = [t for t in trades if t.net_pnl <= 0]

    total_gross = sum(t.gross_pnl for t in trades)
    total_comm = sum(t.commission for t in trades)
    total_slip = sum(t.slippage_cost for t in trades)

    gross_win = sum(t.net_pnl for t in winners)
    gross_loss = abs(sum(t.net_pnl for t in losers))

    if gross_loss > 0:
        pf = gross_win / gross_loss
    elif gross_win > 0:
        pf = float("inf")
    else:
        pf = 0.0

    # Max drawdown
    max_dd = 0.0
    max_dd_pct = 0.0
    peak = equity_curve[0] if equity_curve else 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd_pct = (peak - eq) / peak
            dd_abs = peak - eq
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd = dd_abs

    # Sharpe ratio (annualized, using per-bar returns)
    sharpe = 0.0
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * math.sqrt(252))

    n = len(trades)
    return BacktestMetrics(
        total_trades=n,
        winners=len(winners),
        losers=len(losers),
        win_rate=len(winners) / n if n > 0 else 0.0,
        profit_factor=pf,
        total_gross_pnl=total_gross,
        total_commission=total_comm,
        total_slippage=total_slip,
        total_net_pnl=sum(t.net_pnl for t in trades),
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        sharpe_ratio=sharpe,
        avg_win=sum(t.net_pnl for t in winners) / len(winners) if winners else 0.0,
        avg_loss=sum(t.net_pnl for t in losers) / len(losers) if losers else 0.0,
        avg_hold_bars=sum(t.hold_bars for t in trades) / n if n > 0 else 0.0,
        avg_mae=sum(t.mae for t in trades) / n if n > 0 else 0.0,
        avg_mfe=sum(t.mfe for t in trades) / n if n > 0 else 0.0,
        largest_win=max((t.net_pnl for t in trades), default=0.0),
        largest_loss=min((t.net_pnl for t in trades), default=0.0),
        initial_capital=config.initial_capital,
        final_equity=equity_curve[-1] if equity_curve else config.initial_capital,
        equity_curve=equity_curve,
        slippage_pct=config.slippage_pct,
        commission_per_trade=config.commission_per_trade,
        min_confidence=0.0,  # set by caller from signal config
        min_rr_ratio=0.0,
        trades=trades,
    )
