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


def _daily_frame(n: int, end: str | None = None) -> pd.DataFrame:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1},
        index=idx,
    )


def test_fetch_keeps_a_year_but_patterns_see_only_six_months():
    from rainier.analysis.stock_screener import _fetch_stock_data, pattern_window

    with patch(
        "rainier.analysis.stock_screener.yf.download", return_value=_daily_frame(252)
    ) as dl:
        data = _fetch_stock_data(["NVDA"], min_bars=60)

    assert dl.call_args.kwargs["period"] == "1y"
    history = data["NVDA"]
    assert len(history) == 252  # enough for the 200-bar thesis chart
    window = pattern_window(history)
    cutoff = history.index[-1] - pd.DateOffset(months=6)
    assert window.index.min() > cutoff
    assert len(window) == (history.index > cutoff).sum()
    assert 120 <= len(window) <= 135


def test_min_bars_applies_to_the_pattern_window_not_the_year():
    from rainier.analysis.stock_screener import _fetch_stock_data

    # A recent IPO with 50 bars is still skipped, as with the old 6mo fetch.
    with patch(
        "rainier.analysis.stock_screener.yf.download", return_value=_daily_frame(50)
    ):
        assert _fetch_stock_data(["NEWCO"], min_bars=60) == {}


def test_stale_series_is_not_shifted_back_into_a_full_window():
    """A symbol whose bars stopped 4 months ago keeps only ~2 months in the
    today-anchored 6mo window, so it is skipped exactly as the old fetch did."""
    from rainier.analysis.stock_screener import _fetch_stock_data

    stale_end = (pd.Timestamp.today() - pd.DateOffset(months=4)).strftime("%Y-%m-%d")
    with patch(
        "rainier.analysis.stock_screener.yf.download",
        return_value=_daily_frame(252, end=stale_end),
    ):
        assert _fetch_stock_data(["HALTED"], min_bars=60) == {}
