"""Tests for ml/compare.py — side-by-side backtest comparison."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from rainier.core.config import BacktestConfig
from rainier.core.protocols import BacktestMetrics
from rainier.core.types import Signal, Timeframe
from rainier.ml.compare import ComparisonResult, compare_emitters, format_comparison_table

# ---------------------------------------------------------------------------
# Fake emitters for deterministic testing
# ---------------------------------------------------------------------------


class _FixedEmitter:
    """Emits a fixed set of signals regardless of input."""

    def __init__(self, signals: list[Signal] | None = None):
        self._signals = signals or []

    def emit(self, df, symbol, timeframe) -> list[Signal]:
        # Return signals whose timestamp falls within the df range
        if self._signals:
            ts_set = set(df["timestamp"].tolist())
            return [s for s in self._signals if s.timestamp in ts_set]
        return []


class _NeverEmitter:
    """Emits zero signals — baseline for comparison."""

    def emit(self, df, symbol, timeframe) -> list[Signal]:
        return []


def _make_df(n: int = 200) -> pd.DataFrame:
    """Simple trending dataset for backtest."""
    rows = []
    price = 100.0
    base = datetime(2025, 1, 1)
    for i in range(n):
        cycle = i % 40
        move = 1.0 if cycle < 20 else -1.0
        o = price
        h = price + abs(move) + 1.0
        low = price - abs(move) - 0.5
        c = price + move
        rows.append({
            "timestamp": base + timedelta(hours=i),
            "open": o, "high": h, "low": low, "close": c,
            "volume": 1000.0 + (i % 10) * 100,
        })
        price = c
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompareEmitters:
    """compare_emitters runs backtest with each emitter."""

    def test_returns_result_for_each_emitter(self):
        df = _make_df()
        result = compare_emitters(
            df=df,
            symbol="TEST",
            timeframe=Timeframe.H1,
            emitters={"A": _NeverEmitter(), "B": _NeverEmitter()},
            config=BacktestConfig(),
        )
        assert len(result.rows) == 2
        assert result.labels == ["A", "B"]

    def test_metrics_have_correct_type(self):
        df = _make_df()
        result = compare_emitters(
            df=df,
            symbol="TEST",
            timeframe=Timeframe.H1,
            emitters={"Base": _NeverEmitter()},
            config=BacktestConfig(),
        )
        _, metrics = result.rows[0]
        assert isinstance(metrics, BacktestMetrics)

    def test_no_signals_yields_zero_trades(self):
        df = _make_df()
        result = compare_emitters(
            df=df,
            symbol="TEST",
            timeframe=Timeframe.H1,
            emitters={"Empty": _NeverEmitter()},
            config=BacktestConfig(),
        )
        _, metrics = result.rows[0]
        assert metrics.total_trades == 0

    def test_get_metrics_by_label(self):
        df = _make_df()
        result = compare_emitters(
            df=df,
            symbol="TEST",
            timeframe=Timeframe.H1,
            emitters={"A": _NeverEmitter(), "B": _NeverEmitter()},
            config=BacktestConfig(),
        )
        assert result.get_metrics("A") is not None
        assert result.get_metrics("B") is not None
        assert result.get_metrics("C") is None


class TestComparisonResult:
    """ComparisonResult dataclass methods."""

    def test_labels_property(self):
        m = BacktestMetrics()
        result = ComparisonResult(rows=[("X", m), ("Y", m)])
        assert result.labels == ["X", "Y"]

    def test_empty_result(self):
        result = ComparisonResult()
        assert result.labels == []
        assert result.get_metrics("anything") is None


class TestFormatComparisonTable:
    """format_comparison_table produces readable output."""

    def test_empty_result(self):
        result = ComparisonResult()
        output = format_comparison_table(result)
        assert "No results" in output

    def test_single_emitter(self):
        metrics = BacktestMetrics(
            total_trades=10, winners=6, losers=4,
            win_rate=0.6, profit_factor=1.5,
            total_net_pnl=1500.0, total_gross_pnl=2000.0,
            total_commission=300.0, total_slippage=200.0,
            max_drawdown_pct=0.05, sharpe_ratio=1.2,
            avg_win=500.0, avg_loss=-250.0, avg_hold_bars=8.0,
            largest_win=1000.0, largest_loss=-500.0,
            final_equity=101500.0,
        )
        result = ComparisonResult(rows=[("BookScorer", metrics)])
        output = format_comparison_table(result)

        assert "SCORER COMPARISON" in output
        assert "BookScorer" in output
        assert "10" in output  # total_trades
        assert "60.0%" in output  # win_rate
        assert "1.50" in output  # profit_factor

    def test_two_emitters_shows_delta(self):
        m1 = BacktestMetrics(
            total_trades=10, win_rate=0.5, profit_factor=1.2,
            total_net_pnl=1000.0, max_drawdown_pct=0.08, sharpe_ratio=0.8,
        )
        m2 = BacktestMetrics(
            total_trades=8, win_rate=0.625, profit_factor=1.6,
            total_net_pnl=2000.0, max_drawdown_pct=0.05, sharpe_ratio=1.1,
        )
        result = ComparisonResult(rows=[("Book", m1), ("ML", m2)])
        output = format_comparison_table(result)

        assert "Delta" in output
        assert "Profit factor" in output
        assert "Win rate" in output

    def test_three_emitters_no_delta(self):
        m = BacktestMetrics()
        result = ComparisonResult(rows=[("A", m), ("B", m), ("C", m)])
        output = format_comparison_table(result)
        assert "Delta" not in output
