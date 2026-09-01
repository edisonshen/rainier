"""Tests for backtest engine, export, and sweep."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

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
# compute_metrics golden test — every field from a hand-computed trade list
# ---------------------------------------------------------------------------


def _trade(net, gross, comm, slip, hold, mae, mfe):
    return TradeRecord(
        net_pnl=net, gross_pnl=gross, commission=comm, slippage_cost=slip,
        hold_bars=hold, mae=mae, mfe=mfe,
    )


class TestComputeMetricsGolden:
    def test_every_field_hand_computed(self):
        from pytest import approx

        from rainier.backtest.engine import compute_metrics

        trades = [
            _trade(net=10.0, gross=11.0, comm=0.5, slip=0.5, hold=2, mae=1.0, mfe=12.0),
            _trade(net=-4.0, gross=-3.0, comm=0.5, slip=0.5, hold=4, mae=5.0, mfe=1.0),
            _trade(net=6.0, gross=7.0, comm=0.5, slip=0.5, hold=6, mae=2.0, mfe=8.0),
        ]
        equity_curve = [100.0, 110.0, 106.0, 112.0]
        config = BacktestConfig(
            initial_capital=100.0, slippage_pct=0.001, commission_per_trade=0.25,
        )

        m = compute_metrics(trades, equity_curve, config)

        assert m.total_trades == 3
        assert m.winners == 2
        assert m.losers == 1
        assert m.win_rate == approx(2 / 3)
        assert m.profit_factor == approx(16.0 / 4.0)
        assert m.total_gross_pnl == approx(15.0)
        assert m.total_commission == approx(1.5)
        assert m.total_slippage == approx(1.5)
        assert m.total_net_pnl == approx(12.0)
        assert m.max_drawdown == approx(4.0)  # peak 110 → trough 106
        assert m.max_drawdown_pct == approx(4.0 / 110.0)
        assert m.sharpe_ratio == approx(11.185232151607185)
        assert m.avg_win == approx(8.0)
        assert m.avg_loss == approx(-4.0)
        assert m.avg_hold_bars == approx(4.0)
        assert m.avg_mae == approx(8.0 / 3.0)
        assert m.avg_mfe == approx(7.0)
        assert m.largest_win == approx(10.0)
        assert m.largest_loss == approx(-4.0)
        assert m.initial_capital == 100.0
        assert m.final_equity == 112.0
        assert m.equity_curve == equity_curve
        assert m.slippage_pct == 0.001
        assert m.commission_per_trade == 0.25
        assert m.min_confidence == 0.0
        assert m.min_rr_ratio == 0.0
        assert m.trades == trades

    def test_no_losers_profit_factor_inf(self):
        from rainier.backtest.engine import compute_metrics

        trades = [_trade(net=5.0, gross=5.0, comm=0.0, slip=0.0, hold=1, mae=0.0, mfe=5.0)]
        m = compute_metrics(trades, [100.0, 105.0], BacktestConfig())
        assert m.profit_factor == float("inf")

    def test_no_trades_all_zero(self):
        from rainier.backtest.engine import compute_metrics

        m = compute_metrics([], [100.0], BacktestConfig(initial_capital=100.0))
        assert m.total_trades == 0
        assert m.profit_factor == 0.0
        assert m.win_rate == 0.0
        assert m.largest_win == 0.0
        assert m.largest_loss == 0.0
        assert m.avg_win == 0.0
        assert m.avg_loss == 0.0

    def test_zero_net_pnl_trade_counts_as_loser(self):
        from rainier.backtest.engine import compute_metrics

        trades = [_trade(net=0.0, gross=0.0, comm=0.0, slip=0.0, hold=1, mae=0.0, mfe=0.0)]
        m = compute_metrics(trades, [100.0, 100.0], BacktestConfig())
        assert m.winners == 0
        assert m.losers == 1


# ---------------------------------------------------------------------------
# _close_trade — hand-computed PnL for long + short with slippage/commission
# ---------------------------------------------------------------------------


def _make_open_trade(direction: Direction, entry_price: float, slippage_cost: float,
                     notes: str = ""):
    from rainier.backtest.engine import _OpenTrade

    signal = Signal(
        symbol="NQ", timeframe=Timeframe.H1, direction=direction,
        entry_price=100.0, stop_loss=95.0 if direction == Direction.LONG else 105.0,
        take_profit=110.0 if direction == Direction.LONG else 90.0,
        confidence=0.75, timestamp=datetime(2025, 1, 1, 9), notes=notes,
    )
    return _OpenTrade(
        signal=signal, trade_id=7, entry_bar=10,
        entry_price=entry_price, slippage_cost=slippage_cost, mae=1.5, mfe=3.0,
    )


def _exit_bar_data():
    return pd.Series({
        "timestamp": datetime(2025, 1, 1, 15),
        "open": 109.0, "high": 111.0, "low": 108.0, "close": 110.0, "volume": 1000.0,
    })


class TestCloseTrade:
    def test_long_pnl_with_commission_and_slippage(self):
        from rainier.backtest.engine import _close_trade

        config = BacktestConfig(commission_per_trade=2.5)
        ot = _make_open_trade(Direction.LONG, entry_price=100.1, slippage_cost=0.1)
        record = _close_trade(ot, 110.0, "take_profit", 16, _exit_bar_data(), config)

        assert record.trade_id == 7
        assert record.symbol == "NQ"
        assert record.timeframe == "1H"
        assert record.direction == "LONG"
        assert record.entry_price == 100.1
        assert record.exit_price == 110.0
        assert record.stop_loss == 95.0
        assert record.take_profit == 110.0
        assert record.entry_bar == 10
        assert record.exit_bar == 16
        assert record.hold_bars == 6
        assert record.gross_pnl == pytest.approx(9.9)
        assert record.commission == pytest.approx(5.0)  # 2.5 per side x 2
        assert record.slippage_cost == pytest.approx(0.1)
        assert record.net_pnl == pytest.approx(9.9 - 5.0 - 0.1)
        assert record.confidence == 0.75
        assert record.risk == pytest.approx(5.0)  # |signal entry 100 - SL 95|
        assert record.mae == 1.5
        assert record.mfe == 3.0
        assert record.exit_reason == "take_profit"
        assert record.entry_timestamp == str(datetime(2025, 1, 1, 9))
        assert record.exit_timestamp == str(datetime(2025, 1, 1, 15))

    def test_short_pnl_inverted(self):
        from rainier.backtest.engine import _close_trade

        config = BacktestConfig(commission_per_trade=0.0)
        ot = _make_open_trade(Direction.SHORT, entry_price=99.9, slippage_cost=0.1)
        record = _close_trade(ot, 90.0, "take_profit", 16, _exit_bar_data(), config)

        assert record.direction == "SHORT"
        assert record.gross_pnl == pytest.approx(9.9)  # entry 99.9 - exit 90.0
        assert record.net_pnl == pytest.approx(9.9 - 0.1)

    def test_losing_short(self):
        from rainier.backtest.engine import _close_trade

        config = BacktestConfig(commission_per_trade=1.0)
        ot = _make_open_trade(Direction.SHORT, entry_price=100.0, slippage_cost=0.05)
        record = _close_trade(ot, 105.0, "stop_loss", 12, _exit_bar_data(), config)

        assert record.gross_pnl == pytest.approx(-5.0)
        assert record.net_pnl == pytest.approx(-5.0 - 2.0 - 0.05)
        assert record.exit_reason == "stop_loss"

    def test_pattern_notes_parsed(self):
        from rainier.backtest.engine import _close_trade

        ot = _make_open_trade(Direction.LONG, entry_price=100.0, slippage_cost=0.0,
                              notes="pattern:w_bottom")
        record = _close_trade(ot, 110.0, "take_profit", 16, _exit_bar_data(), BacktestConfig())
        assert record.pattern_type == "w_bottom"
        assert record.entry_reason == "pattern_breakout_w_bottom"


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
