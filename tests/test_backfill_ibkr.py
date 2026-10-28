"""Spec: `rainier backfill-ibkr` fills MES 5m history from per-contract IBKR bars.

A fake TWS serves two quarterly contracts (MESZ5, MESH6) whose volume leadership
flips at the Dec 2025 roll. MESH6 trades 50 points above MESZ5, so every stitched
bar tells which contract it came from. The CSV already holds yfinance bars with a gap.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner

MARKET_START = datetime(2025, 12, 1, tzinfo=UTC)
ROLL = datetime(2025, 12, 14, 23, tzinfo=UTC)  # Sunday 17:00 CT open of the Dec 15 session
H6_PREMIUM = 50.0


@pytest.fixture(autouse=True)
def _event_loop():
    asyncio.set_event_loop(asyncio.new_event_loop())
    yield


class FakeTWS:
    """Serves 5m bars per contract; honors endDateTime + durationStr like TWS."""

    contracts = {"20251219": 0.0, "20260320": H6_PREMIUM}

    def __init__(self):
        self.requests: list[tuple[str, datetime]] = []

    def reqContractDetails(self, query):  # noqa: N802 (TWS API name)
        assert query.includeExpired
        return [
            SimpleNamespace(contract=SimpleNamespace(lastTradeDateOrContractMonth=exp))
            for exp in self.contracts
        ]

    def reqHistoricalData(self, contract, endDateTime, durationStr, **_):  # noqa: N802, N803
        from ib_insync import BarData

        expiry = contract.lastTradeDateOrContractMonth
        self.requests.append((expiry, endDateTime))
        last = min(endDateTime, datetime.strptime(expiry, "%Y%m%d").replace(tzinfo=UTC))
        first = max(endDateTime - timedelta(days=int(durationStr.split()[0])), MARKET_START)
        bars, ts = [], first
        while ts < last:
            is_front = (ts < ROLL) == (expiry == "20251219")
            price = 6800.0 + self.contracts[expiry] + (ts - MARKET_START).days
            bars.append(BarData(date=ts, open=price, high=price + 1, low=price - 1,
                                close=price, volume=1000 if is_front else 10))
            ts += timedelta(minutes=5)
        return bars


def _run(tmp_path, tws):
    from rainier.cli import cli
    from rainier.data.ibkr_provider import IBKRProvider

    @contextmanager
    def connection(self):
        yield tws

    with patch.object(IBKRProvider, "_connection", connection):
        return CliRunner().invoke(cli, [
            "backfill-ibkr", "--symbol", "MES", "--port", "7496",
            "--data-dir", str(tmp_path / "csv"), "--raw-dir", str(tmp_path / "raw"),
        ])


@pytest.fixture(autouse=True)
def candles_db(tmp_path):
    """A real `candles` table (SQLite) behind `get_session`, as `persist_to_db` sees it."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from rainier.core.models import CandleRecord

    engine = create_engine(f"sqlite:///{tmp_path / 'candles.db'}")
    CandleRecord.__table__.create(engine)
    factory = sessionmaker(bind=engine)

    @contextmanager
    def get_session():
        with factory() as session, session.begin():
            yield session

    with patch("rainier.core.database.get_session", get_session):
        yield engine


def _db_candles(engine) -> pd.DataFrame:
    df = pd.read_sql("SELECT timestamp, close FROM candles WHERE symbol='MES' AND timeframe='5m'"
                     " ORDER BY timestamp", engine)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
    return df


@pytest.fixture
def existing_csv(tmp_path):
    """yfinance bars for Dec 17 only (a 0.25 off the IBKR price), nothing before."""
    ts = pd.date_range("2025-12-17 00:00", "2025-12-17 23:55", freq="5min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0,
                       "close": 1.25, "volume": 7.0})
    (tmp_path / "csv").mkdir()
    df.to_csv(tmp_path / "csv" / "MES_5m.csv", index=False)
    return df


def test_backfill_stitches_front_month_and_keeps_existing_bars(tmp_path, existing_csv,
                                                               candles_db):
    result = _run(tmp_path, FakeTWS())
    assert result.exit_code == 0, result.output

    out = pd.read_csv(tmp_path / "csv" / "MES_5m.csv", parse_dates=["timestamp"])
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    assert out["timestamp"].is_unique and out["timestamp"].is_monotonic_increasing
    assert out["timestamp"].min() == pd.Timestamp(MARKET_START)

    yf_day = out["timestamp"].dt.date == pd.Timestamp("2025-12-17").date()
    assert (out.loc[yf_day, "close"] == 1.25).all(), "existing yfinance bars must win"

    ib = out[~yf_day]
    from_h6 = (ib["close"] - 6800.0 - (ib["timestamp"] - pd.Timestamp(MARKET_START)).dt.days
               ) == H6_PREMIUM
    assert not from_h6[ib["timestamp"] < pd.Timestamp(ROLL)].any(), "MESZ5 until the roll"
    assert from_h6[ib["timestamp"] >= pd.Timestamp(ROLL)].all(), "MESH6 from the roll on"
    assert (ib["volume"] == 1000).all(), "no back-month bars"
    assert "new" in result.output

    db = _db_candles(candles_db)
    assert db["timestamp"].tolist() == out["timestamp"].tolist(), "candles 5m mirrors the CSV"
    assert db["close"].tolist() == out["close"].tolist()


def test_second_run_reads_expired_contracts_from_cache(tmp_path, existing_csv, candles_db):
    assert _run(tmp_path, FakeTWS()).exit_code == 0
    first = pd.read_csv(tmp_path / "csv" / "MES_5m.csv")

    tws = FakeTWS()
    result = _run(tmp_path, tws)
    assert result.exit_code == 0, result.output
    assert tws.requests == [], "both contracts expired, so no new IBKR requests"
    assert pd.read_csv(tmp_path / "csv" / "MES_5m.csv").equals(first)
    assert len(_db_candles(candles_db)) == len(first), "re-run upserts, no duplicate rows"
