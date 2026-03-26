"""Tests for multi-symbol portfolio backtest."""

from datetime import datetime, timedelta

import pandas as pd

from rainier.backtest.portfolio import (
    PortfolioResult,
    format_portfolio_report,
    run_portfolio_backtest,
)
from rainier.core.config import BacktestConfig
from rainier.core.types import Direction, Signal, Timeframe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dataset(n_bars: int = 200, seed_price: float = 100.0) -> pd.DataFrame:
    rows = []
    price = seed_price
    base = datetime(2025, 1, 1)
    for i in range(n_bars):
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


class FakeEmitter:
    def emit(self, df: pd.DataFrame, symbol: str, timeframe: Timeframe) -> list[Signal]:
        last_bar = df.iloc[-1]
        return [
            Signal(
                symbol=symbol,
                timeframe=timeframe,
                direction=Direction.LONG,
                entry_price=float(last_bar["close"]) - 0.5,
                stop_loss=float(last_bar["close"]) - 5.0,
                take_profit=float(last_bar["close"]) + 5.0,
                confidence=0.75,
                timestamp=pd.Timestamp(last_bar["timestamp"]).to_pydatetime(),
            ),
        ]


class EmptyEmitter:
    def emit(self, df: pd.DataFrame, symbol: str, timeframe: Timeframe) -> list[Signal]:
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPortfolioBacktest:
    def test_single_symbol_runs(self):
        df = _make_dataset(200)
        data = {"MES": df}
        tfs = {"MES": Timeframe.H1}
        config = BacktestConfig(sr_recompute_interval=50)

        result = run_portfolio_backtest(data, tfs, FakeEmitter(), config)

        assert isinstance(result, PortfolioResult)
        assert len(result.symbol_results) == 1
        assert result.symbol_results[0].symbol == "MES"
        assert result.initial_capital == config.initial_capital

    def test_single_symbol_matches_standalone(self):
        """Portfolio with 1 symbol should produce same metrics."""
        from rainier.backtest.engine import run_backtest

        df = _make_dataset(200)
        config = BacktestConfig(sr_recompute_interval=50)
        emitter = FakeEmitter()

        # Standalone
        standalone = run_backtest(df, "MES", Timeframe.H1, emitter, config)

        # Portfolio (single symbol gets full capital)
        emitter2 = FakeEmitter()
        port = run_portfolio_backtest(
            {"MES": df}, {"MES": Timeframe.H1}, emitter2, config,
        )

        assert port.symbol_results[0].metrics.total_trades == standalone.total_trades
        assert abs(port.total_net_pnl - standalone.total_net_pnl) < 0.01

    def test_multi_symbol_capital_split(self):
        data = {
            "MES": _make_dataset(200, 100.0),
            "NQ": _make_dataset(200, 200.0),
            "ES": _make_dataset(200, 150.0),
        }
        tfs = {s: Timeframe.H1 for s in data}
        config = BacktestConfig(
            initial_capital=300_000.0,
            sr_recompute_interval=50,
        )

        result = run_portfolio_backtest(data, tfs, FakeEmitter(), config)

        assert len(result.symbol_results) == 3
        for sr in result.symbol_results:
            assert sr.weight == 1.0 / 3
            assert sr.metrics.initial_capital == 100_000.0

    def test_pnl_aggregation(self):
        data = {
            "MES": _make_dataset(200, 100.0),
            "NQ": _make_dataset(200, 200.0),
        }
        tfs = {s: Timeframe.H1 for s in data}
        config = BacktestConfig(sr_recompute_interval=50)

        result = run_portfolio_backtest(data, tfs, FakeEmitter(), config)

        per_sym_sum = sum(result.per_symbol_pnl.values())
        assert abs(result.total_net_pnl - per_sym_sum) < 0.01

    def test_total_trades_aggregation(self):
        data = {
            "MES": _make_dataset(200, 100.0),
            "NQ": _make_dataset(200, 200.0),
        }
        tfs = {s: Timeframe.H1 for s in data}
        config = BacktestConfig(sr_recompute_interval=50)

        result = run_portfolio_backtest(data, tfs, FakeEmitter(), config)

        trade_sum = sum(sr.metrics.total_trades for sr in result.symbol_results)
        assert result.total_trades == trade_sum

    def test_combined_equity_curve_starts_at_capital(self):
        data = {"MES": _make_dataset(200)}
        tfs = {"MES": Timeframe.H1}
        config = BacktestConfig(sr_recompute_interval=50)

        result = run_portfolio_backtest(data, tfs, FakeEmitter(), config)

        assert result.combined_equity_curve[0] == config.initial_capital

    def test_empty_data_returns_empty_result(self):
        config = BacktestConfig()
        result = run_portfolio_backtest({}, {}, FakeEmitter(), config)

        assert result.total_trades == 0
        assert result.total_net_pnl == 0.0
        assert len(result.symbol_results) == 0

    def test_empty_emitter_zero_trades(self):
        data = {"MES": _make_dataset(200)}
        tfs = {"MES": Timeframe.H1}
        config = BacktestConfig(sr_recompute_interval=50)

        result = run_portfolio_backtest(data, tfs, EmptyEmitter(), config)

        assert result.total_trades == 0

    def test_different_data_lengths(self):
        data = {
            "MES": _make_dataset(200, 100.0),
            "NQ": _make_dataset(300, 200.0),
        }
        tfs = {s: Timeframe.H1 for s in data}
        config = BacktestConfig(sr_recompute_interval=50)

        # Should not crash
        result = run_portfolio_backtest(data, tfs, FakeEmitter(), config)
        assert len(result.symbol_results) == 2


class TestPortfolioReport:
    def test_report_contains_key_sections(self):
        data = {
            "MES": _make_dataset(200, 100.0),
            "NQ": _make_dataset(200, 200.0),
        }
        tfs = {s: Timeframe.H1 for s in data}
        config = BacktestConfig(sr_recompute_interval=50)

        result = run_portfolio_backtest(data, tfs, FakeEmitter(), config)
        report = format_portfolio_report(result)

        assert "PORTFOLIO" in report
        assert "MES" in report
        assert "NQ" in report
        assert "PER-SYMBOL" in report
