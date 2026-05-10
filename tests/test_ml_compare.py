"""Tests for ml/compare.py — side-by-side backtest comparison."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from rainier.core.config import BacktestConfig
from rainier.core.protocols import BacktestMetrics
from rainier.core.types import Signal, Timeframe
from rainier.ml.compare import (
    ComparisonResult,
    WalkForwardResult,
    WalkForwardWindow,
    compare_emitters,
    format_comparison_table,
    format_walkforward_table,
    run_walkforward_compare,
)

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


# ---------------------------------------------------------------------------
# Walk-forward tests
# ---------------------------------------------------------------------------


class _ConstantEmitter:
    """Emits zero signals — gives deterministic backtest with no trades."""

    def emit(self, df, symbol, timeframe) -> list[Signal]:
        return []


class TestRunWalkforwardCompare:
    """run_walkforward_compare slides train/test windows and aggregates."""

    def test_yields_expected_window_count(self):
        df = _make_df(500)
        # train=200, test=100, step=100 → folds at train_starts 0,100,200
        # need train_start + 300 <= 500 → 0,100,200 valid
        result = run_walkforward_compare(
            df=df,
            symbol="TEST",
            timeframe=Timeframe.H1,
            emitter_factories={"empty": lambda _train: _ConstantEmitter()},
            train_bars=200,
            test_bars=100,
            step_bars=100,
            config=BacktestConfig(),
        )
        assert len(result.windows) == 3

    def test_default_step_equals_test_bars(self):
        df = _make_df(400)
        # train=200, test=100, default step=100 → folds at 0, 100
        result = run_walkforward_compare(
            df=df,
            symbol="TEST",
            timeframe=Timeframe.H1,
            emitter_factories={"empty": lambda _train: _ConstantEmitter()},
            train_bars=200,
            test_bars=100,
            config=BacktestConfig(),
        )
        assert len(result.windows) == 2

    def test_window_indices_are_correct(self):
        df = _make_df(400)
        result = run_walkforward_compare(
            df=df,
            symbol="TEST",
            timeframe=Timeframe.H1,
            emitter_factories={"empty": lambda _train: _ConstantEmitter()},
            train_bars=200,
            test_bars=100,
            step_bars=100,
            config=BacktestConfig(),
        )
        w0 = result.windows[0]
        assert (w0.train_start, w0.train_end) == (0, 200)
        assert (w0.test_start, w0.test_end) == (200, 300)
        w1 = result.windows[1]
        assert (w1.train_start, w1.train_end) == (100, 300)
        assert (w1.test_start, w1.test_end) == (300, 400)

    def test_factory_receives_train_slice_only(self):
        """Factory must be called with the training slice, not the full df."""
        df = _make_df(400)
        captured: list[int] = []

        def factory(train_df):
            captured.append(len(train_df))
            return _ConstantEmitter()

        run_walkforward_compare(
            df=df,
            symbol="TEST",
            timeframe=Timeframe.H1,
            emitter_factories={"e": factory},
            train_bars=200,
            test_bars=100,
            step_bars=100,
            config=BacktestConfig(),
        )
        assert captured == [200, 200]  # called once per fold with train_bars

    def test_aggregates_per_label(self):
        df = _make_df(400)
        result = run_walkforward_compare(
            df=df,
            symbol="TEST",
            timeframe=Timeframe.H1,
            emitter_factories={
                "A": lambda _train: _ConstantEmitter(),
                "B": lambda _train: _ConstantEmitter(),
            },
            train_bars=200,
            test_bars=100,
            step_bars=100,
            config=BacktestConfig(),
        )
        assert result.labels == ["A", "B"]
        assert result.get_aggregate("A") is not None
        assert result.get_aggregate("B") is not None
        # No signals → 0 trades aggregated
        assert result.get_aggregate("A").total_trades == 0

    def test_raises_on_empty_factories(self):
        df = _make_df(400)
        with pytest.raises(ValueError, match="non-empty"):
            run_walkforward_compare(
                df=df,
                symbol="TEST",
                timeframe=Timeframe.H1,
                emitter_factories={},
                train_bars=200,
                test_bars=100,
            )

    def test_raises_on_too_short_data(self):
        df = _make_df(100)
        with pytest.raises(ValueError, match="need at least"):
            run_walkforward_compare(
                df=df,
                symbol="TEST",
                timeframe=Timeframe.H1,
                emitter_factories={"e": lambda _t: _ConstantEmitter()},
                train_bars=200,
                test_bars=100,
            )

    def test_raises_on_invalid_bar_counts(self):
        df = _make_df(400)
        with pytest.raises(ValueError, match="must be positive"):
            run_walkforward_compare(
                df=df,
                symbol="TEST",
                timeframe=Timeframe.H1,
                emitter_factories={"e": lambda _t: _ConstantEmitter()},
                train_bars=0,
                test_bars=100,
            )


class TestWalkForwardResult:
    def test_empty_result(self):
        result = WalkForwardResult()
        assert result.labels == []
        assert result.get_aggregate("anything") is None

    def test_window_dataclass(self):
        w = WalkForwardWindow(
            fold=0, train_start=0, train_end=200,
            test_start=200, test_end=300,
        )
        assert w.fold == 0
        assert w.metrics_by_label == {}


class TestFormatWalkforwardTable:
    def test_empty_returns_message(self):
        out = format_walkforward_table(WalkForwardResult())
        assert "No walk-forward" in out

    def test_renders_per_window_and_aggregate(self):
        df = _make_df(400)
        result = run_walkforward_compare(
            df=df,
            symbol="TEST",
            timeframe=Timeframe.H1,
            emitter_factories={"Book": lambda _t: _ConstantEmitter()},
            train_bars=200,
            test_bars=100,
            step_bars=100,
            config=BacktestConfig(),
        )
        out = format_walkforward_table(result)
        assert "WALK-FORWARD COMPARISON" in out
        assert "Book" in out
        assert "Aggregate" in out
        assert "Profit factor" in out
