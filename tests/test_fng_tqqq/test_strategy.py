from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from rainier.core.config import FngTqqqConfig
from rainier.signals.fng_tqqq import evaluate, replay

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "fng_tqqq"
    / "history.csv"
)


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, parse_dates=["date"]).set_index("date")


@pytest.mark.parametrize(
    (
        "buy_below",
        "regime_sma_window",
        "stop_loss_pct",
        "start",
        "final",
        "trade_count",
        "stop_count",
        "first_two",
        "last",
    ),
    [
        (
            30.0,
            180,
            0.10,
            "2021-01-22",
            107890.99,
            25,
            2,
            None,
            None,
        ),
        (
            30.0,
            180,
            0.10,
            "2023-10-16",
            46573.52,
            15,
            2,
            None,
            None,
        ),
        (
            30.0,
            200,
            0.10,
            "2021-01-22",
            95719.81,
            26,
            3,
            None,
            None,
        ),
        (
            30.0,
            None,
            None,
            "2021-01-22",
            93282.91,
            28,
            0,
            [("2021-01-28", "2021-02-23"), ("2021-05-13", "2021-05-28")],
            ("2026-09-16", "2026-10-16"),
        ),
        (
            30.0,
            None,
            None,
            "2023-10-16",
            69672.44,
            14,
            0,
            [("2023-10-20", "2024-04-05"), ("2024-04-22", "2024-07-22")],
            ("2026-09-16", "2026-10-16"),
        ),
    ],
)
def test_fixture_replay_matches_original_backtest(
    history,
    buy_below,
    regime_sma_window,
    stop_loss_pct,
    start,
    final,
    trade_count,
    stop_count,
    first_two,
    last,
):
    """The full-history regime/SMA, fills, stops, and costs match the golden replay."""
    cfg = FngTqqqConfig(
        buy_below=buy_below,
        sma_window=38,
        regime_sma_window=regime_sma_window,
        stop_loss_pct=stop_loss_pct,
        start=date.fromisoformat(start),
    )
    scores = history["fg"].dropna()
    days = evaluate(
        scores,
        history["qqq_close"],
        history["tqqq_close"],
        cfg,
    )
    trades, final_equity = replay(days, history["tqqq_close"])

    assert all(not day.armed for day in days if not day.holding)
    assert abs(final_equity - final) <= 1
    assert len(trades) == trade_count
    assert sum(day.action == "STOP" for day in days) == stop_count
    if first_two is not None:
        assert [
            (trade.entry_date.isoformat(), trade.exit_date.isoformat())
            for trade in trades[:2]
        ] == first_two
        assert (
            trades[-1].entry_date.isoformat(),
            trades[-1].exit_date.isoformat(),
        ) == last
        assert all(trade.exit_date is not None for trade in trades)


def test_stop_locks_reentry_and_bear_days_block_buys(history):
    cfg = FngTqqqConfig()
    days = evaluate(
        history["fg"].dropna(),
        history["qqq_close"],
        history["tqqq_close"],
        cfg,
    )

    assert any(day.action == "WAIT" and day.bear for day in days)
    assert any(
        previous.action == "STOP"
        and day.action == "WAIT"
        and day.fg < cfg.buy_below
        and day.locked
        for previous, day in zip(days, days[1:])
    )
