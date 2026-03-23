"""Tests for label generation from backtest results."""

from datetime import datetime

from rainier.backtest.engine import BacktestTrade
from rainier.core.types import Direction, Signal, SignalStatus, Timeframe
from rainier.features.labels import LabelGenerator, LabelPolicy


def _make_signal(direction: Direction = Direction.LONG) -> Signal:
    return Signal(
        symbol="NQ", timeframe=Timeframe.H1, direction=direction,
        entry_price=100.0, stop_loss=98.0, take_profit=104.0,
        confidence=0.9, timestamp=datetime(2025, 1, 1),
        status=SignalStatus.PENDING,
    )


def _make_trade(
    exit_reason: str, pnl: float, direction: Direction = Direction.LONG,
) -> BacktestTrade:
    return BacktestTrade(
        signal=_make_signal(direction),
        entry_bar=10, exit_bar=20,
        exit_price=100.0 + pnl, pnl=pnl,
        exit_reason=exit_reason,
    )


class TestLabelGenerator:
    def test_take_profit_labeled_1(self):
        trades = [_make_trade("take_profit", pnl=4.0)]
        labels = LabelGenerator().generate(trades)
        assert len(labels) == 1
        assert labels.iloc[0]["label"] == 1
        assert not labels.iloc[0]["is_soft_label"]

    def test_stop_loss_labeled_0(self):
        trades = [_make_trade("stop_loss", pnl=-2.0)]
        labels = LabelGenerator().generate(trades)
        assert len(labels) == 1
        assert labels.iloc[0]["label"] == 0
        assert not labels.iloc[0]["is_soft_label"]

    def test_end_of_data_excluded_by_default(self):
        trades = [
            _make_trade("take_profit", pnl=4.0),
            _make_trade("end_of_data", pnl=1.0),
        ]
        labels = LabelGenerator().generate(trades)
        assert len(labels) == 1  # end_of_data excluded

    def test_end_of_data_included_when_policy_allows(self):
        trades = [
            _make_trade("take_profit", pnl=4.0),
            _make_trade("end_of_data", pnl=1.5),
        ]
        policy = LabelPolicy(exclude_end_of_data=False)
        labels = LabelGenerator(policy).generate(trades)
        assert len(labels) == 2
        eod_row = labels[labels["exit_reason"] == "end_of_data"].iloc[0]
        assert eod_row["label"] == 1  # positive PnL → 1
        assert eod_row["is_soft_label"]

    def test_end_of_data_negative_pnl(self):
        trades = [_make_trade("end_of_data", pnl=-0.5)]
        policy = LabelPolicy(exclude_end_of_data=False)
        labels = LabelGenerator(policy).generate(trades)
        assert labels.iloc[0]["label"] == 0

    def test_empty_trades(self):
        labels = LabelGenerator().generate([])
        assert len(labels) == 0

    def test_summary(self):
        trades = [
            _make_trade("take_profit", pnl=4.0),
            _make_trade("take_profit", pnl=3.0),
            _make_trade("stop_loss", pnl=-2.0),
        ]
        gen = LabelGenerator()
        labels = gen.generate(trades)
        summary = gen.summary(labels)
        assert summary["total"] == 3
        assert summary["positive"] == 2
        assert summary["negative"] == 1
        assert summary["soft"] == 0
        assert summary["positive_rate"] == 2 / 3

    def test_mixed_directions(self):
        trades = [
            _make_trade("take_profit", pnl=4.0, direction=Direction.LONG),
            _make_trade("stop_loss", pnl=-2.0, direction=Direction.SHORT),
        ]
        labels = LabelGenerator().generate(trades)
        assert len(labels) == 2
        assert labels.iloc[0]["direction"] == "LONG"
        assert labels.iloc[1]["direction"] == "SHORT"
