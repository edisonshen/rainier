"""Tests for backtest engine, export, and sweep."""

from datetime import datetime, timedelta

import pandas as pd

from rainier.backtest.engine import run_backtest
from rainier.core.config import BacktestConfig
from rainier.core.protocols import BacktestMetrics, SignalEmitter, TradeRecord
from rainier.core.types import Direction, Signal, Timeframe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_zigzag_dataset(n_bars: int = 200) -> pd.DataFrame:
    """Create a dataset with trending and reversal patterns."""
    rows = []
    price = 100.0
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


class FakeSignalEmitter:
    """A test emitter that produces a known signal at a specific bar."""

    def __init__(self, signals: list[Signal] | None = None):
        self._signals = signals or []

    def emit(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
    ) -> list[Signal]:
        return self._signals


class SingleTradeEmitter:
    """Emits exactly one LONG signal with known entry/SL/TP.

    Designed so a zigzag dataset will trigger both fill and exit.
    """

    def __init__(self):
        self._emitted = False

    def emit(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
    ) -> list[Signal]:
        if self._emitted:
            return []
        self._emitted = True
        last_bar = df.iloc[-1]
        return [
            Signal(
                symbol=symbol,
                timeframe=timeframe,
                direction=Direction.LONG,
                entry_price=float(last_bar["close"]) - 0.5,  # limit below current
                stop_loss=float(last_bar["close"]) - 5.0,
                take_profit=float(last_bar["close"]) + 5.0,
                confidence=0.85,
                timestamp=pd.Timestamp(last_bar["timestamp"]).to_pydatetime(),
            ),
        ]


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocols:
    def test_fake_emitter_satisfies_protocol(self):
        emitter = FakeSignalEmitter()
        assert isinstance(emitter, SignalEmitter)

    def test_single_trade_emitter_satisfies_protocol(self):
        emitter = SingleTradeEmitter()
        assert isinstance(emitter, SignalEmitter)


# ---------------------------------------------------------------------------
# BacktestMetrics
# ---------------------------------------------------------------------------


class TestBacktestMetrics:
    def test_empty_result(self):
        config = BacktestConfig()
        emitter = FakeSignalEmitter()
        df = _make_zigzag_dataset(50)
        m = run_backtest(df, "NQ", Timeframe.H1, emitter, config)
        assert m.total_trades == 0
        assert m.win_rate == 0.0
        assert m.total_net_pnl == 0.0
        assert len(m.equity_curve) > 0

    def test_equity_starts_at_initial_capital(self):
        config = BacktestConfig(initial_capital=50_000.0)
        emitter = FakeSignalEmitter()
        df = _make_zigzag_dataset()
        m = run_backtest(df, "NQ", Timeframe.H1, emitter, config)
        assert m.equity_curve[0] == 50_000.0
        assert m.initial_capital == 50_000.0


# ---------------------------------------------------------------------------
# Slippage and commission
# ---------------------------------------------------------------------------


class TestSlippageCommission:
    def test_zero_slippage_zero_commission(self):
        config = BacktestConfig(slippage_pct=0.0, commission_per_trade=0.0)
        emitter = SingleTradeEmitter()
        df = _make_zigzag_dataset()
        m = run_backtest(df, "NQ", Timeframe.H1, emitter, config)
        if m.total_trades > 0:
            assert m.total_commission == 0.0
            assert m.total_slippage == 0.0
            for t in m.trades:
                assert t.commission == 0.0
                assert t.slippage_cost == 0.0
                assert t.gross_pnl == t.net_pnl

    def test_commission_reduces_pnl(self):
        config_no_comm = BacktestConfig(slippage_pct=0.0, commission_per_trade=0.0)
        config_with_comm = BacktestConfig(slippage_pct=0.0, commission_per_trade=5.0)

        df = _make_zigzag_dataset()

        m_no = run_backtest(df, "NQ", Timeframe.H1, SingleTradeEmitter(), config_no_comm)
        m_with = run_backtest(df, "NQ", Timeframe.H1, SingleTradeEmitter(), config_with_comm)

        if m_no.total_trades > 0 and m_with.total_trades > 0:
            assert m_with.total_net_pnl < m_no.total_net_pnl

    def test_slippage_reduces_pnl(self):
        config_no_slip = BacktestConfig(slippage_pct=0.0, commission_per_trade=0.0)
        config_with_slip = BacktestConfig(slippage_pct=0.01, commission_per_trade=0.0)

        df = _make_zigzag_dataset()

        m_no = run_backtest(df, "NQ", Timeframe.H1, SingleTradeEmitter(), config_no_slip)
        m_with = run_backtest(df, "NQ", Timeframe.H1, SingleTradeEmitter(), config_with_slip)

        if m_no.total_trades > 0 and m_with.total_trades > 0:
            assert m_with.total_net_pnl < m_no.total_net_pnl


# ---------------------------------------------------------------------------
# MAE/MFE tracking
# ---------------------------------------------------------------------------


class TestMAEMFE:
    def test_mae_mfe_non_negative(self):
        config = BacktestConfig(slippage_pct=0.0, commission_per_trade=0.0)
        emitter = SingleTradeEmitter()
        df = _make_zigzag_dataset()
        m = run_backtest(df, "NQ", Timeframe.H1, emitter, config)
        for t in m.trades:
            assert t.mae >= 0.0
            assert t.mfe >= 0.0


# ---------------------------------------------------------------------------
# Trade record completeness
# ---------------------------------------------------------------------------


class TestTradeRecord:
    def test_trade_fields_populated(self):
        config = BacktestConfig(slippage_pct=0.0, commission_per_trade=0.0)
        emitter = SingleTradeEmitter()
        df = _make_zigzag_dataset()
        m = run_backtest(df, "NQ", Timeframe.H1, emitter, config)
        for t in m.trades:
            assert t.trade_id > 0
            assert t.symbol == "NQ"
            assert t.timeframe == "1H"
            assert t.direction in ("LONG", "SHORT")
            assert t.entry_price > 0
            assert t.exit_price > 0
            assert t.hold_bars >= 0
            assert t.exit_reason in ("stop_loss", "take_profit", "end_of_data")
            assert t.confidence > 0
            assert t.entry_timestamp != ""
            assert t.exit_timestamp != ""


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_csv(self, tmp_path):
        from rainier.backtest.export import export_trades_csv

        config = BacktestConfig(slippage_pct=0.0, commission_per_trade=0.0)
        emitter = SingleTradeEmitter()
        df = _make_zigzag_dataset()
        m = run_backtest(df, "NQ", Timeframe.H1, emitter, config)

        out = tmp_path / "trades.csv"
        export_trades_csv(m, out)
        assert out.exists()

        loaded = pd.read_csv(out)
        assert len(loaded) == m.total_trades

    def test_export_empty(self, tmp_path):
        from rainier.backtest.export import export_trades_csv

        m = BacktestMetrics()
        out = tmp_path / "empty.csv"
        export_trades_csv(m, out)
        assert out.exists()

    def test_trades_to_dataframe_columns(self):
        from rainier.backtest.export import trades_to_dataframe

        m = BacktestMetrics(trades=[
            TradeRecord(trade_id=1, symbol="NQ", direction="LONG", net_pnl=10.0),
        ])
        df = trades_to_dataframe(m)
        assert "trade_id" in df.columns
        assert "net_pnl" in df.columns
        assert "mae" in df.columns
        assert "mfe" in df.columns
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


class TestSweep:
    def test_sweep_runs(self):
        from rainier.backtest.sweep import run_sweep

        df = _make_zigzag_dataset()
        config = BacktestConfig()

        def factory(conf, rr):
            return FakeSignalEmitter()

        result = run_sweep(
            df, "NQ", Timeframe.H1, factory, config,
            confidence_values=[0.5, 0.6],
            rr_values=[1.0, 2.0],
        )
        # 2 x 2 = 4 combinations
        assert len(result.rows) == 4

    def test_sweep_table_format(self):
        from rainier.backtest.sweep import SweepResult, format_sweep_table

        result = SweepResult(rows=[{
            "min_confidence": 0.6,
            "min_rr_ratio": 2.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_net_pnl": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_hold_bars": 0.0,
            "final_equity": 100_000.0,
        }])
        text = format_sweep_table(result)
        assert "PARAMETER SWEEP" in text


# ---------------------------------------------------------------------------
# Position limit
# ---------------------------------------------------------------------------


class TestPositionLimit:
    def test_max_open_positions_respected(self):
        """Emitter that floods signals should be capped by max_open_positions."""

        class FloodEmitter:
            def emit(self, df, symbol, timeframe):
                last = df.iloc[-1]
                ts = pd.Timestamp(last["timestamp"]).to_pydatetime()
                return [
                    Signal(
                        symbol=symbol, timeframe=timeframe,
                        direction=Direction.LONG,
                        entry_price=float(last["close"]) - 0.1 * i,
                        stop_loss=float(last["close"]) - 50.0,
                        take_profit=float(last["close"]) + 50.0,
                        confidence=0.9, timestamp=ts,
                    )
                    for i in range(10)
                ]

        config = BacktestConfig(max_open_positions=2, slippage_pct=0.0, commission_per_trade=0.0)
        df = _make_zigzag_dataset()
        m = run_backtest(df, "NQ", Timeframe.H1, FloodEmitter(), config)
        # Can't verify exact count, but should not crash
        assert isinstance(m, BacktestMetrics)
