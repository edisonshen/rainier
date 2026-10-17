from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from rainier.data.prices import load_last_price, warmup_start


def test_warmup_start_covers_session_window():
    assert warmup_start(date(2021, 1, 22), 200) == date(2020, 2, 27)


def test_load_last_price_returns_latest_close_and_aware_timestamp(monkeypatch):
    from rainier.data import prices

    timestamp = pd.Timestamp("2026-10-17 12:58:00", tz=ZoneInfo("America/New_York"))

    class FakeTicker:
        def __init__(self, symbol):
            assert symbol == "TQQQ"

        def history(self, *, period, interval):
            assert period == "1d"
            assert interval == "1m"
            return pd.DataFrame({"Close": [61.2]}, index=[timestamp])

    monkeypatch.setattr(prices.yf, "Ticker", FakeTicker)
    price, price_time = load_last_price("TQQQ")

    assert price == 61.2
    assert isinstance(price_time, datetime)
    assert price_time.tzinfo is not None
