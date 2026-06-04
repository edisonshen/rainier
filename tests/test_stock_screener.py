"""Regression: screen_stocks() returns tuple[list[StockCandidate], dict[str, DataFrame]]."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from rainier.analysis.stock_screener import screen_stocks


def _patched_session_with_no_signals():
    fake = MagicMock()
    # Layer 1 _screen_money_flow returns [] when there is no data. It probes
    # the latest top100 trading day via `query(...).filter(...).scalar()` and
    # also via the unfiltered `query(...).scalar()` chain, so BOTH must resolve
    # to None to exercise the no-data early return.
    fake.query.return_value.scalar.return_value = None
    fake.query.return_value.filter.return_value.scalar.return_value = None
    cm = MagicMock()
    cm.__enter__.return_value = fake
    cm.__exit__.return_value = False
    return cm


def test_returns_tuple_of_list_and_dict_when_empty():
    cm = _patched_session_with_no_signals()
    with patch("rainier.analysis.stock_screener.get_session", return_value=cm):
        out = screen_stocks()
    assert isinstance(out, tuple)
    candidates, ohlcv = out
    assert isinstance(candidates, list)
    assert candidates == []
    assert isinstance(ohlcv, dict)
    assert ohlcv == {}


def test_tuple_unpacking_supports_existing_call_pattern():
    cm = _patched_session_with_no_signals()
    with patch("rainier.analysis.stock_screener.get_session", return_value=cm):
        candidates, ohlcv = screen_stocks()
    assert candidates == []
    assert ohlcv == {}


def test_full_pipeline_yields_dict_keyed_by_symbol():
    """Layer 1 signals + mocked yfinance -> dict has the same keys as fetched symbols."""
    from rainier.core.types import MoneyFlowSignal, SectorTrend

    sig = MoneyFlowSignal(
        symbol="NVDA", rank=5, rank_change=0, long_short="Long in",
        capital_flow_direction="+", days_in_top100=3, sector="Technology",
        industry="Semiconductors", signal_strength=0.9,
    )
    sector_trends = [
        SectorTrend(
            sector="Technology", long_in_count=10, short_in_count=2,
            net_sentiment=0.6, top_stocks=["NVDA"], trend_direction="bullish",
            sector_rank=1,
        )
    ]
    df = pd.DataFrame(
        {"open": [1.0] * 80, "high": [1.0] * 80, "low": [1.0] * 80,
         "close": [1.0] * 80, "volume": [1] * 80}
    )

    cm = MagicMock()
    cm.__enter__.return_value = MagicMock()
    cm.__exit__.return_value = False

    with patch(
        "rainier.analysis.stock_screener._screen_money_flow", return_value=[sig]
    ), patch(
        "rainier.analysis.stock_screener.analyze_sectors", return_value=sector_trends
    ), patch(
        "rainier.analysis.stock_screener._fetch_stock_data", return_value={"NVDA": df}
    ), patch(
        "rainier.analysis.stock_screener.detect_patterns", return_value=[]
    ), patch(
        "rainier.analysis.stock_screener.get_session", return_value=cm
    ):
        candidates, ohlcv = screen_stocks()

    assert "NVDA" in ohlcv
    assert isinstance(ohlcv["NVDA"], pd.DataFrame)
    assert any(c.symbol == "NVDA" for c in candidates)
