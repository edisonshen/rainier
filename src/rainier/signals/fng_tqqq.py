"""Pure F&G/QQQ signal evaluation and TQQQ replay accounting."""

from __future__ import annotations

from datetime import date

import pandas as pd

from rainier.core.config import FngTqqqConfig
from rainier.core.types import StrategyDay, StrategyTrade


def _normalize_series(series: pd.Series) -> pd.Series:
    result = series.copy()
    index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    result = result.sort_index()
    return result[~result.index.duplicated(keep="last")]


def evaluate(
    fg: pd.Series,
    qqq_close: pd.Series,
    cfg: FngTqqqConfig,
    start: date | str | None = None,
) -> list[StrategyDay]:
    """Evaluate close decisions over the F&G/QQQ date intersection."""
    qqq = _normalize_series(qqq_close).astype(float)
    scores = _normalize_series(fg).astype(float)
    qqq_sma = qqq.rolling(cfg.sma_window).mean()
    joined = pd.concat(
        [
            scores.rename("fg"),
            qqq.rename("qqq_close"),
            qqq_sma.rename("qqq_sma"),
        ],
        axis=1,
        join="inner",
    ).dropna(subset=["fg", "qqq_close"])
    replay_start = pd.Timestamp(start or cfg.start)
    joined = joined.loc[joined.index >= replay_start]

    holding = False
    armed = False
    pending: str | None = None
    days: list[StrategyDay] = []
    for timestamp, row in joined.iterrows():
        if pending == "BUY":
            holding = True
            armed = False
        elif pending == "SELL":
            holding = False

        if not holding:
            action = "BUY" if row.fg < cfg.buy_below else "WAIT"
        else:
            if row.qqq_close > row.qqq_sma:
                armed = True
            action = "SELL" if armed and row.qqq_close < row.qqq_sma else "HOLD"

        days.append(
            StrategyDay(
                date=timestamp.date(),
                fg=float(row.fg),
                qqq_close=float(row.qqq_close),
                qqq_sma=float(row.qqq_sma),
                holding=holding,
                armed=armed,
                action=action,
            )
        )
        pending = action
    return days


def replay(
    days: list[StrategyDay],
    tqqq_close: pd.Series,
    cost: float = 0.001,
    initial: float = 10_000.0,
) -> tuple[list[StrategyTrade], float]:
    """Replay close decisions with next-session-close fills and all-in sizing."""
    prices = _normalize_series(tqqq_close).astype(float)
    trades: list[StrategyTrade] = []
    equity = initial
    holding = False
    entry_date: date | None = None
    entry_px: float | None = None
    pending: str | None = None
    last_px: float | None = None

    for day in days:
        timestamp = pd.Timestamp(day.date)
        if timestamp not in prices.index:
            raise ValueError(f"Missing TQQQ close for strategy date {day.date}")
        px = float(prices.loc[timestamp])
        last_px = px

        if pending == "BUY" and not holding:
            equity *= 1 - cost
            holding = True
            entry_date = day.date
            entry_px = px
        elif pending == "SELL" and holding:
            assert entry_date is not None and entry_px is not None
            equity *= px / entry_px * (1 - cost)
            trades.append(
                StrategyTrade(
                    entry_date=entry_date,
                    exit_date=day.date,
                    entry_px=entry_px,
                    exit_px=px,
                    ret=px / entry_px * (1 - cost) ** 2 - 1,
                )
            )
            holding = False
            entry_date = None
            entry_px = None

        pending = day.action

    if holding:
        assert entry_date is not None and entry_px is not None and last_px is not None
        equity *= last_px / entry_px
        trades.append(
            StrategyTrade(
                entry_date=entry_date,
                exit_date=None,
                entry_px=entry_px,
                exit_px=None,
                ret=last_px / entry_px * (1 - cost) - 1,
            )
        )
    return trades, equity
