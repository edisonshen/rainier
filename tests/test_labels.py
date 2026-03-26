"""Tests for label generation from backtest results."""

from rainier.core.protocols import TradeRecord
from rainier.features.labels import LabelGenerator, LabelPolicy


def _make_trade(exit_reason: str, net_pnl: float, direction: str = "LONG") -> TradeRecord:
    return TradeRecord(
        trade_id=1,
        symbol="NQ",
        timeframe="1H",
        direction=direction,
        entry_price=100.0,
        exit_price=100.0 + net_pnl,
        stop_loss=98.0,
        take_profit=104.0,
        entry_bar=10,
        exit_bar=20,
        hold_bars=10,
        gross_pnl=net_pnl,
        net_pnl=net_pnl,
        confidence=0.9,
        rr_ratio=2.0,
        risk=2.0,
        exit_reason=exit_reason,
    )


class TestLabelGenerator:
    def test_take_profit_labeled_1(self):
        trades = [_make_trade("take_profit", net_pnl=4.0)]
        labels = LabelGenerator().generate(trades)
        assert len(labels) == 1
        assert labels.iloc[0]["label"] == 1
        assert not labels.iloc[0]["is_soft_label"]

    def test_stop_loss_labeled_0(self):
        trades = [_make_trade("stop_loss", net_pnl=-2.0)]
        labels = LabelGenerator().generate(trades)
        assert len(labels) == 1
        assert labels.iloc[0]["label"] == 0
        assert not labels.iloc[0]["is_soft_label"]

    def test_end_of_data_excluded_by_default(self):
        trades = [
            _make_trade("take_profit", net_pnl=4.0),
            _make_trade("end_of_data", net_pnl=1.0),
        ]
        labels = LabelGenerator().generate(trades)
        assert len(labels) == 1

    def test_end_of_data_included_when_policy_allows(self):
        trades = [
            _make_trade("take_profit", net_pnl=4.0),
            _make_trade("end_of_data", net_pnl=1.5),
        ]
        policy = LabelPolicy(exclude_end_of_data=False)
        labels = LabelGenerator(policy).generate(trades)
        assert len(labels) == 2
        eod_row = labels[labels["exit_reason"] == "end_of_data"].iloc[0]
        assert eod_row["label"] == 1
        assert eod_row["is_soft_label"]

    def test_end_of_data_negative_pnl(self):
        trades = [_make_trade("end_of_data", net_pnl=-0.5)]
        policy = LabelPolicy(exclude_end_of_data=False)
        labels = LabelGenerator(policy).generate(trades)
        assert labels.iloc[0]["label"] == 0

    def test_empty_trades(self):
        labels = LabelGenerator().generate([])
        assert len(labels) == 0

    def test_summary(self):
        trades = [
            _make_trade("take_profit", net_pnl=4.0),
            _make_trade("take_profit", net_pnl=3.0),
            _make_trade("stop_loss", net_pnl=-2.0),
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
            _make_trade("take_profit", net_pnl=4.0, direction="LONG"),
            _make_trade("stop_loss", net_pnl=-2.0, direction="SHORT"),
        ]
        labels = LabelGenerator().generate(trades)
        assert len(labels) == 2
        assert labels.iloc[0]["direction"] == "LONG"
        assert labels.iloc[1]["direction"] == "SHORT"
