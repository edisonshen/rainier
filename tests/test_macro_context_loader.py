"""Unit tests for src/rainier/research/macro_context_loader.py."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rainier.research import macro_context_loader as loader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cache(path: Path) -> None:
    """Write a tiny synthetic parquet matching the production schema."""
    rows = []
    for sym in ["VIX", "XLK", "QQQ"]:
        for d in pd.date_range("2024-10-01", "2024-10-15", freq="B").date:
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1_000_000,
                    "adjusted_close": 100.5,
                    "fetched_at": datetime(2026, 5, 22, tzinfo=timezone.utc),
                    "yfinance_version": "1.2.0",
                }
            )
    df = pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_identical_query_identical_bytes(tmp_path):
    cache = tmp_path / "macro.parquet"
    _make_cache(cache)

    df_a = loader.read_range("VIX", "2024-10-01", "2024-10-15", path=cache)
    df_b = loader.read_range("VIX", "2024-10-01", "2024-10-15", path=cache)

    pd.testing.assert_frame_equal(df_a, df_b)


def test_sort_key_symbol_date_ascending(tmp_path):
    cache = tmp_path / "macro.parquet"
    _make_cache(cache)

    df = loader.read_all(path=cache)
    sorted_view = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), sorted_view)


def test_missing_symbol_returns_empty(tmp_path):
    cache = tmp_path / "macro.parquet"
    _make_cache(cache)

    df = loader.read_range("ZZZZ", "2024-10-01", "2024-10-15", path=cache)
    # Empty DataFrame with the ledger-registered columns intact.
    # adjusted_close is INTENTIONALLY absent (look-ahead control; codex iter-7).
    assert df.empty
    for col in ["symbol", "date", "open", "high", "low", "close", "volume"]:
        assert col in df.columns
    assert "adjusted_close" not in df.columns


def test_date_range_filter(tmp_path):
    cache = tmp_path / "macro.parquet"
    _make_cache(cache)

    df = loader.read_range("XLK", "2024-10-07", "2024-10-11", path=cache)
    # Trading-day range 2024-10-07..2024-10-11 = Mon..Fri = 5 days.
    assert len(df) == 5
    assert df["date"].min() == date(2024, 10, 7)
    assert df["date"].max() == date(2024, 10, 11)
    # All rows are the requested symbol.
    assert set(df["symbol"].unique()) == {"XLK"}


def test_inclusive_range_bounds(tmp_path):
    cache = tmp_path / "macro.parquet"
    _make_cache(cache)

    df = loader.read_range("VIX", "2024-10-01", "2024-10-01", path=cache)
    # Inclusive bounds on both sides.
    assert len(df) == 1
    assert df["date"].iloc[0] == date(2024, 10, 1)


def test_missing_cache_file_raises(tmp_path):
    cache = tmp_path / "nope.parquet"
    with pytest.raises(FileNotFoundError):
        loader.read_range("VIX", "2024-10-01", "2024-10-15", path=cache)


def test_read_all_returns_all_symbols(tmp_path):
    cache = tmp_path / "macro.parquet"
    _make_cache(cache)

    df = loader.read_all(path=cache)
    assert set(df["symbol"].unique()) == {"VIX", "XLK", "QQQ"}


def test_loader_excludes_adjusted_close_by_default(tmp_path):
    """Default loader output omits look-ahead-leaky columns. Codex iter-7.

    yfinance's ``Adj Close`` is back-adjusted for future splits/dividends
    and the ledger explicitly does NOT register it as as-of observable. The
    loader must not return it by default — otherwise an L3 evaluator that
    serializes/hashes the returned frame wholesale would silently include
    look-ahead-adjusted values.
    """
    cache = tmp_path / "macro.parquet"
    _make_cache(cache)

    df_range = loader.read_range("VIX", "2024-10-01", "2024-10-15", path=cache)
    df_all = loader.read_all(path=cache)
    for frame in (df_range, df_all):
        assert "adjusted_close" not in frame.columns
        assert "fetched_at" not in frame.columns
        assert "yfinance_version" not in frame.columns
        # As-of fields remain.
        for col in ["symbol", "date", "open", "high", "low", "close", "volume"]:
            assert col in frame.columns


def test_loader_include_unregistered_opt_in_returns_full_schema(tmp_path):
    """Explicit opt-in surfaces the full parquet schema for non-L3 analyses."""
    cache = tmp_path / "macro.parquet"
    _make_cache(cache)

    df_range = loader.read_range(
        "VIX", "2024-10-01", "2024-10-15", path=cache, include_unregistered=True
    )
    df_all = loader.read_all(path=cache, include_unregistered=True)
    for frame in (df_range, df_all):
        assert "adjusted_close" in frame.columns
        assert "fetched_at" in frame.columns
        assert "yfinance_version" in frame.columns


def test_read_range_accepts_datetime_bounds(tmp_path):
    """datetime is a subclass of date; passing one must NOT raise TypeError.

    Per codex iter-5: ``_coerce_date(datetime(...))`` used to return the
    datetime unchanged because ``isinstance(d, date)`` is true. Then the
    parquet ``date`` column (loaded as ``datetime.date`` objects) couldn't be
    compared against a ``datetime``, raising ``TypeError`` mid-query. Now we
    truncate datetime → date defensively.
    """
    cache = tmp_path / "macro.parquet"
    _make_cache(cache)

    df = loader.read_range(
        "VIX",
        datetime(2024, 10, 7, 14, 30, 0),
        datetime(2024, 10, 11, 23, 59, 59),
        path=cache,
    )
    # 5 weekday rows in the inclusive range.
    assert len(df) == 5
    assert df["date"].min() == date(2024, 10, 7)
    assert df["date"].max() == date(2024, 10, 11)
