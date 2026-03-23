"""Tests for backtest engine."""

from datetime import datetime, timedelta

import pandas as pd

from rainier.backtest.engine import BacktestResult, BacktestTrade, run_backtest
from rainier.core.config import AnalysisConfig, SignalConfig
from rainier.core.types import Direction, Signal, SignalStatus, Timeframe


def _make_long_dataset(n_bars: int = 200) -> pd.DataFrame:
    """Create a dataset with trending and reversal patterns."""
    rows = []
    price = 100.0
    base = datetime(2025, 1, 1)

    for i in range(n_bars):
        # Zigzag: up 20 bars, down 20 bars
        cycle = i % 40
        if cycle < 20:
            move = 1.0
        else:
            move = -1.0

        o = price
        h = price + abs(move) + 1.0
        l = price - abs(move) - 0.5
        c = price + move

        rows.append({
            "timestamp": base + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": 1000.0 + (i % 10) * 100,
        })
        price = c

    return pd.DataFrame(rows)


class TestBacktestResult:
    def test_empty_result(self):
        r = BacktestResult()
        assert r.total_trades == 0
        assert r.win_rate == 0.0
        assert r.total_pnl == 0.0

    def test_win_rate_calculation(self):
        sig = Signal(
            symbol="NQ", timeframe=Timeframe.H1, direction=Direction.LONG,
            entry_price=100, stop_loss=95, take_profit=110, confidence=0.8,
            timestamp=datetime(2025, 1, 1),
        )
        r = BacktestResult(trades=[
            BacktestTrade(signal=sig, entry_bar=0, exit_bar=10, pnl=5.0),
            BacktestTrade(signal=sig, entry_bar=0, exit_bar=10, pnl=-3.0),
            BacktestTrade(signal=sig, entry_bar=0, exit_bar=10, pnl=2.0),
        ])
        assert r.win_rate == pytest.approx(2 / 3)
        assert r.profit_factor == pytest.approx(7.0 / 3.0)

    def test_max_drawdown(self):
        r = BacktestResult(equity_curve=[100, 110, 90, 95, 80, 120])
        # Peak 110, trough 80 → dd = 30/110 ≈ 0.2727
        assert r.max_drawdown == pytest.approx(30 / 110, rel=1e-3)


class TestRunBacktest:
    def test_runs_without_error(self):
        df = _make_long_dataset()
        result = run_backtest(df, "NQ", Timeframe.H1)
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_no_signals_on_short_data(self):
        """Very short data → probably no signals."""
        df = _make_long_dataset(n_bars=15)
        result = run_backtest(df, "NQ", Timeframe.H1)
        assert isinstance(result, BacktestResult)

    def test_equity_curve_starts_at_initial_capital(self):
        df = _make_long_dataset()
        result = run_backtest(df, "NQ", Timeframe.H1, initial_capital=50_000)
        assert result.equity_curve[0] == 50_000


import pytest
