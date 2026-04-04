"""Protocol contracts — boundary interfaces between modules.

Modules depend on these protocols, not on each other's concrete implementations.
This enables:
  - backtest/ to run any signal strategy without importing signals/
  - trader/ to execute any signal source without knowing how signals are generated
  - features/ to extract from any analysis result without importing analysis/

Dependency rule:
  core/ ← analysis/ ← (nothing above)
  core/ ← signals/  ← (nothing above)
  core/ ← backtest/ ← (nothing above, receives SignalEmitter via DI)
  core/ ← trader/   ← (nothing above, receives signals via protocol)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from rainier.core.types import AnalysisResult, PatternSignal, Signal, Timeframe

# ---------------------------------------------------------------------------
# Analysis boundary: DataFrame → AnalysisResult
# ---------------------------------------------------------------------------


@runtime_checkable
class Analyzer(Protocol):
    """Produces technical analysis from OHLCV data."""

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
    ) -> AnalysisResult: ...


# ---------------------------------------------------------------------------
# Signal boundary: AnalysisResult + DataFrame → Signal[]
# ---------------------------------------------------------------------------


@runtime_checkable
class SignalEmitter(Protocol):
    """Generates trade signals from OHLCV data.

    This is the main contract between signal generation and consumers
    (backtest engine, live trader, research tools). Implementations decide
    internally how to analyze and score — the consumer only sees signals.
    """

    def emit(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
    ) -> list[Signal]: ...


# ---------------------------------------------------------------------------
# Scoring boundary: PatternSignal + features → confidence score
# ---------------------------------------------------------------------------


@runtime_checkable
class ScoringStrategy(Protocol):
    """Scores a pattern setup, returning a confidence value in [0, 1].

    Two implementations:
    - BookScorer: rule-based weighted sum (production-ready now)
    - MLScorer: XGBoost model (requires trained model + feature store)
    """

    def score(self, pattern: PatternSignal, features: pd.DataFrame) -> float: ...


# ---------------------------------------------------------------------------
# Backtest result contract (shared output type)
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """A completed trade with full analytics — the universal trade output format.

    Used by backtest engine, trade journal, and research tools.
    Intentionally flat (no nested objects) for easy export to CSV/Parquet.
    """
    # Identity
    trade_id: int = 0
    symbol: str = ""
    timeframe: str = ""
    direction: str = ""  # "LONG" or "SHORT"

    # Prices
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    # Timing
    entry_bar: int = 0
    exit_bar: int = 0
    entry_timestamp: str = ""
    exit_timestamp: str = ""
    hold_bars: int = 0

    # P&L
    gross_pnl: float = 0.0
    commission: float = 0.0
    slippage_cost: float = 0.0
    net_pnl: float = 0.0

    # Analytics
    confidence: float = 0.0
    rr_ratio: float = 0.0
    risk: float = 0.0  # |entry - stop_loss|
    mae: float = 0.0  # max adverse excursion (worst unrealized loss)
    mfe: float = 0.0  # max favorable excursion (best unrealized profit)
    exit_reason: str = ""  # "stop_loss", "take_profit", "end_of_data"

    # Context (what generated this signal)
    entry_reason: str = ""  # "pin_bar_at_support", "pattern_breakout", etc.
    sr_level_price: float | None = None
    sr_level_type: str | None = None  # "horizontal", "diagonal"
    pattern_type: str | None = None  # "w_bottom", "bull_flag", etc.


@dataclass
class BacktestMetrics:
    """Aggregate metrics from a backtest run — the standard output contract."""
    # Core stats
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # P&L
    total_gross_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_net_pnl: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0

    # Per-trade averages
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_hold_bars: float = 0.0
    avg_mae: float = 0.0
    avg_mfe: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Equity
    initial_capital: float = 0.0
    final_equity: float = 0.0
    equity_curve: list[float] = field(default_factory=list)

    # Config used
    slippage_pct: float = 0.0
    commission_per_trade: float = 0.0
    min_confidence: float = 0.0
    min_rr_ratio: float = 0.0

    # Full trade log
    trades: list[TradeRecord] = field(default_factory=list)
