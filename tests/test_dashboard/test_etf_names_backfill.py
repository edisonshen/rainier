"""Unit tests for the ETF names backfill (yfinance longName cache).

Backfill module:
    src/rainier/dashboard/etf_names_backfill.py

Public surface under test:
    backfill_names(symbols, out_path, *, fetch_fn=..., retries=1,
                   retry_sleep=lambda _: None) -> Path

Tests:
    1. parquet schema matches (symbol, long_name, short_name, fetched_at)
    2. re-running upserts on symbol (no duplicates)
    3. RateLimitError → one retry → succeed
    4. RateLimitError twice → surfaces a clean error (no infinite loop)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _stub_fetch(payload: dict[str, dict[str, str | None]]):
    """Build a stub fetcher that returns ``payload`` for any symbol list.

    Each value is a dict with keys ``long_name`` and ``short_name``. Missing
    symbols default to (None, None) — matching yfinance behaviour for
    delisted / unknown tickers.
    """

    def _fetch(symbols: list[str]) -> dict[str, dict[str, str | None]]:
        out: dict[str, dict[str, str | None]] = {}
        for sym in symbols:
            out[sym] = payload.get(sym, {"long_name": None, "short_name": None})
        return out

    return _fetch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_backfill_names_writes_parquet_with_expected_schema(tmp_path):
    """Parquet has the right columns + dtypes (symbol, long_name, short_name, fetched_at)."""
    from rainier.dashboard.etf_names_backfill import backfill_names

    fetch_fn = _stub_fetch(
        {
            "EFA": {
                "long_name": "iShares MSCI EAFE ETF",
                "short_name": "iShares MSCI EAFE",
            },
            "SPY": {
                "long_name": "SPDR S&P 500 ETF Trust",
                "short_name": "SPDR S&P 500",
            },
        }
    )
    out = tmp_path / "etf_names.parquet"
    written = backfill_names(
        symbols=["EFA", "SPY"],
        out_path=out,
        fetch_fn=fetch_fn,
        retry_sleep=lambda _: None,
    )
    assert written == out
    assert out.exists(), "expected parquet to be written"

    df = pd.read_parquet(out)
    assert set(df.columns) == {"symbol", "long_name", "short_name", "fetched_at"}, (
        f"unexpected columns: {df.columns.tolist()}"
    )
    assert len(df) == 2
    assert set(df["symbol"]) == {"EFA", "SPY"}
    # long_name pulled from the payload.
    efa = df.loc[df["symbol"] == "EFA"].iloc[0]
    assert efa["long_name"] == "iShares MSCI EAFE ETF"
    assert efa["short_name"] == "iShares MSCI EAFE"
    # fetched_at must be a UTC tz-aware timestamp column so re-reads stay
    # comparable across runs.
    assert pd.api.types.is_datetime64_any_dtype(df["fetched_at"]), (
        "fetched_at must be a datetime column"
    )


def test_backfill_names_upserts_on_symbol(tmp_path):
    """Re-running on the same symbol replaces the row (no duplicates)."""
    from rainier.dashboard.etf_names_backfill import backfill_names

    out = tmp_path / "etf_names.parquet"

    # First write — EFA only.
    backfill_names(
        symbols=["EFA"],
        out_path=out,
        fetch_fn=_stub_fetch(
            {"EFA": {"long_name": "iShares MSCI EAFE ETF", "short_name": "EAFE"}}
        ),
        retry_sleep=lambda _: None,
    )
    df1 = pd.read_parquet(out)
    assert len(df1) == 1

    # Second write — EFA again (refreshed long_name) + SPY (new). After the
    # upsert there should be exactly TWO rows: EFA replaced + SPY appended.
    backfill_names(
        symbols=["EFA", "SPY"],
        out_path=out,
        fetch_fn=_stub_fetch(
            {
                "EFA": {
                    "long_name": "iShares MSCI EAFE ETF (renamed)",
                    "short_name": "EAFE",
                },
                "SPY": {
                    "long_name": "SPDR S&P 500 ETF Trust",
                    "short_name": "SPDR",
                },
            }
        ),
        retry_sleep=lambda _: None,
    )
    df2 = pd.read_parquet(out)
    assert len(df2) == 2, f"expected 2 rows after upsert, got {len(df2)}: {df2}"
    assert set(df2["symbol"]) == {"EFA", "SPY"}
    efa = df2.loc[df2["symbol"] == "EFA"].iloc[0]
    assert efa["long_name"] == "iShares MSCI EAFE ETF (renamed)", (
        "EFA row should reflect the latest fetch payload"
    )


def test_backfill_names_handles_yfinance_rate_limit_with_one_retry(tmp_path):
    """One transient RateLimitError → one retry → success."""
    from rainier.dashboard.etf_names_backfill import (
        RateLimitError,
        backfill_names,
    )

    calls = {"n": 0}

    def _fetch_flaky(symbols: list[str]) -> dict[str, dict[str, str | None]]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("synthetic yfinance rate limit")
        return {
            sym: {"long_name": f"{sym} long name", "short_name": sym}
            for sym in symbols
        }

    out = tmp_path / "etf_names.parquet"
    written = backfill_names(
        symbols=["EFA"],
        out_path=out,
        fetch_fn=_fetch_flaky,
        retry_sleep=lambda _: None,
    )
    assert written == out
    assert calls["n"] == 2, (
        f"fetcher should have been called twice (1 fail + 1 retry); got {calls['n']}"
    )
    df = pd.read_parquet(out)
    assert df.loc[df["symbol"] == "EFA"].iloc[0]["long_name"] == "EFA long name"


def test_backfill_names_surfaces_persistent_rate_limit(tmp_path):
    """After ONE retry, a persistent RateLimitError must surface cleanly.

    Acceptance gate from TASK-PLAN: "yfinance rate-limit triggers ONE retry,
    then surfaces a clean error" — no infinite loop, no silent skip.
    """
    from rainier.dashboard.etf_names_backfill import (
        RateLimitError,
        backfill_names,
    )

    calls = {"n": 0}

    def _always_rate_limit(symbols: list[str]) -> dict[str, dict[str, str | None]]:
        calls["n"] += 1
        raise RateLimitError("synthetic persistent yfinance rate limit")

    out = tmp_path / "etf_names.parquet"
    with pytest.raises(RateLimitError):
        backfill_names(
            symbols=["EFA"],
            out_path=out,
            fetch_fn=_always_rate_limit,
            retry_sleep=lambda _: None,
        )
    # Should have called twice (initial + one retry), then surfaced.
    assert calls["n"] == 2, (
        f"expected exactly 2 calls (1 initial + 1 retry) before raising; "
        f"got {calls['n']}"
    )
    # No partial parquet on the failure path — prior file (if any) is preserved.
    assert not out.exists(), "no parquet should be written when backfill fails"
