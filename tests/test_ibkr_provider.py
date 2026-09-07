"""Tests for IBKR data provider."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rainier.core.config import IBKRConfig, InstrumentConfig
from rainier.core.types import Timeframe
from rainier.data.ibkr_provider import TF_TO_IB_BAR_SIZE, IBKRProvider


@pytest.fixture(scope="module", autouse=True)
def _ib_insync_event_loop():
    """ib_insync (via eventkit) calls ``asyncio.get_event_loop()`` at import
    time, which raises on 3.12 once an earlier ``asyncio.run()`` in the same
    process has cleared the current loop. Provide one so test order (and
    xdist worker assignment) doesn't matter."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    asyncio.set_event_loop(None)
    loop.close()


@pytest.fixture
def ibkr_config():
    return IBKRConfig(host="127.0.0.1", port=7497, client_id=1, timeout=10, readonly=True)


@pytest.fixture
def sample_bars():
    """Simulate ib_insync bar objects as a DataFrame (what util.df returns)."""
    return pd.DataFrame([
        {"date": "2025-01-02 09:00:00+00:00", "open": 5000.0, "high": 5010.0,
         "low": 4990.0, "close": 5005.0, "volume": 1000, "barCount": 50, "average": 5002.0},
        {"date": "2025-01-02 10:00:00+00:00", "open": 5005.0, "high": 5020.0,
         "low": 4995.0, "close": 5015.0, "volume": 1200, "barCount": 60, "average": 5010.0},
        {"date": "2025-01-02 11:00:00+00:00", "open": 5015.0, "high": 5025.0,
         "low": 5000.0, "close": 5020.0, "volume": 900, "barCount": 45, "average": 5012.0},
    ])


class TestIBKRProviderContract:
    """Test contract building from watchlist config."""

    @patch("rainier.data.ibkr_provider.get_settings")
    @patch("rainier.data.ibkr_provider.load_watchlist")
    def test_make_contract_contfut(self, mock_watchlist, mock_settings, ibkr_config):
        mock_settings.return_value.ibkr = ibkr_config
        mock_watchlist.return_value = {
            "MES": InstrumentConfig(
                symbol="MES", exchange="CME", ib_sec_type="CONTFUT", ib_currency="USD",
            ),
        }
        provider = IBKRProvider(config=ibkr_config)

        with patch("ib_insync.ContFuture") as mock_cf:
            mock_cf.return_value = MagicMock()
            provider._make_contract("MES")
            mock_cf.assert_called_once_with(symbol="MES", exchange="CME", currency="USD")

    @patch("rainier.data.ibkr_provider.get_settings")
    @patch("rainier.data.ibkr_provider.load_watchlist")
    def test_make_contract_stock(self, mock_watchlist, mock_settings, ibkr_config):
        mock_settings.return_value.ibkr = ibkr_config
        mock_watchlist.return_value = {
            "AAPL": InstrumentConfig(
                symbol="AAPL", exchange="SMART", ib_sec_type="STK", ib_currency="USD",
            ),
        }
        provider = IBKRProvider(config=ibkr_config)

        with patch("ib_insync.Stock") as mock_stock:
            mock_stock.return_value = MagicMock()
            provider._make_contract("AAPL")
            mock_stock.assert_called_once_with(symbol="AAPL", exchange="SMART", currency="USD")

    @patch("rainier.data.ibkr_provider.get_settings")
    @patch("rainier.data.ibkr_provider.load_watchlist")
    def test_make_contract_unknown_defaults_contfut(
        self, mock_watchlist, mock_settings, ibkr_config,
    ):
        mock_settings.return_value.ibkr = ibkr_config
        mock_watchlist.return_value = {}
        provider = IBKRProvider(config=ibkr_config)

        with patch("ib_insync.ContFuture") as mock_cf:
            mock_cf.return_value = MagicMock()
            provider._make_contract("UNKNOWN")
            mock_cf.assert_called_once_with(symbol="UNKNOWN", exchange="CME", currency="USD")


class TestIBKRProviderNormalize:
    """Test DataFrame normalization."""

    @patch("rainier.data.ibkr_provider.get_settings")
    @patch("rainier.data.ibkr_provider.load_watchlist")
    def test_normalize_output_format(self, mock_watchlist, mock_settings, ibkr_config, sample_bars):
        mock_settings.return_value.ibkr = ibkr_config
        mock_watchlist.return_value = {}
        provider = IBKRProvider(config=ibkr_config)

        df = provider._normalize(sample_bars)

        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert len(df) == 3
        assert df["timestamp"].dtype == "datetime64[ns, UTC]"
        assert df["open"].iloc[0] == 5000.0
        assert df["close"].iloc[-1] == 5020.0


class TestIBKRProviderGetCandles:
    """Test get_candles with mocked IB connection."""

    @patch("rainier.data.ibkr_provider.get_settings")
    @patch("rainier.data.ibkr_provider.load_watchlist")
    def test_get_candles_success(self, mock_watchlist, mock_settings, ibkr_config, sample_bars):
        mock_settings.return_value.ibkr = ibkr_config
        mock_watchlist.return_value = {
            "MES": InstrumentConfig(symbol="MES", exchange="CME"),
        }

        mock_bar = MagicMock()
        mock_bars = [mock_bar, mock_bar, mock_bar]

        provider = IBKRProvider(config=ibkr_config)

        with patch.object(provider, "_connection") as mock_conn, \
             patch("ib_insync.util") as mock_util:
            mock_ib = MagicMock()
            mock_conn.return_value.__enter__ = MagicMock(return_value=mock_ib)
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_ib.reqHistoricalData.return_value = mock_bars
            mock_util.df.return_value = sample_bars

            df = provider.get_candles("MES", Timeframe.H1)

            assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
            assert len(df) == 3
            mock_ib.qualifyContracts.assert_called_once()
            mock_ib.reqHistoricalData.assert_called_once()

    @patch("rainier.data.ibkr_provider.get_settings")
    @patch("rainier.data.ibkr_provider.load_watchlist")
    def test_get_candles_empty(self, mock_watchlist, mock_settings, ibkr_config):
        mock_settings.return_value.ibkr = ibkr_config
        mock_watchlist.return_value = {}

        provider = IBKRProvider(config=ibkr_config)

        with patch.object(provider, "_connection") as mock_conn:
            mock_ib = MagicMock()
            mock_conn.return_value.__enter__ = MagicMock(return_value=mock_ib)
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_ib.reqHistoricalData.return_value = []

            df = provider.get_candles("MES", Timeframe.H1)

            assert df.empty
            assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


class TestTimeframeMapping:
    """Test that all supported timeframes have IBKR mappings."""

    def test_all_core_timeframes_mapped(self):
        core_tfs = [Timeframe.M5, Timeframe.H1, Timeframe.H4, Timeframe.D1]
        for tf in core_tfs:
            assert tf in TF_TO_IB_BAR_SIZE, f"{tf} not mapped to IBKR bar size"
