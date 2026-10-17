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
    ("buy_below", "start", "final", "trade_count", "first_two", "last"),
    [
        (
            30.0,
            "2021-01-22",
            93283,
            28,
            [("2021-01-28", "2021-02-23"), ("2021-05-13", "2021-05-28")],
            ("2026-09-16", "2026-10-16"),
        ),
        (
            30.0,
            "2023-10-16",
            69672,
            14,
            [("2023-10-20", "2024-04-05"), ("2024-04-22", "2024-07-22")],
            ("2026-09-16", "2026-10-16"),
        ),
        (
            25.0,
            "2021-01-22",
            44368,
            19,
            [("2021-07-20", "2021-08-19"), ("2021-08-20", "2021-09-21")],
            ("2026-06-29", "2026-07-06"),
        ),
        (
            25.0,
            "2023-10-16",
            36460,
            9,
            [("2023-10-23", "2024-04-05"), ("2024-08-07", "2024-08-29")],
            ("2026-06-29", "2026-07-06"),
        ),
    ],
)
def test_fixture_replay_matches_original_backtest(
    history,
    buy_below,
    start,
    final,
    trade_count,
    first_two,
    last,
):
    """The full-history SMA, action timing, fills, and costs match the golden replay."""
    cfg = FngTqqqConfig(
        buy_below=buy_below,
        sma_window=38,
        start=date.fromisoformat(start),
    )
    scores = history["fg"].dropna()
    days = evaluate(scores, history["qqq_close"], cfg)
    trades, final_equity = replay(days, history["tqqq_close"])

    assert round(final_equity) == final
    assert len(trades) == trade_count
    assert [
        (trade.entry_date.isoformat(), trade.exit_date.isoformat())
        for trade in trades[:2]
    ] == first_two
    assert (trades[-1].entry_date.isoformat(), trades[-1].exit_date.isoformat()) == last
    assert all(trade.exit_date is not None for trade in trades)
