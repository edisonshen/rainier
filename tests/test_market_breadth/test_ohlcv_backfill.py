"""Unit tests for src/rainier/market_breadth/ohlcv_backfill.py.

Mirrors the shape of `tests/research/test_backfill_thematic_universe.py`:
yfinance is never called in tests — a `fetch_fn` injection point keeps CI
offline. Each test exercises one fail-mode from the task plan §Tests.

The five tests below correspond 1:1 to the plan acceptance:

  * test_backfill_writes_long_format_parquet — schema + row count
  * test_backfill_idempotent                  — re-run does not duplicate rows
  * test_incremental_fetches_last_5_days_only — `incremental=True` window
  * test_yfinance_rate_limit_retries          — single retry on rate-limit
  * test_partial_failure_does_not_corrupt_parquet — atomic-write semantics

Design refs:
    docs/TASK-PLAN-sp500-ohlcv-backfill-47b8.md §Tests
    scripts/backfill_thematic_universe.py (pattern mirror)
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pyarrow.parquet as pq
import pytest

from rainier.market_breadth import ohlcv_backfill as ob

# ---------------------------------------------------------------------------
# Stub fetcher — deterministic per-symbol DataFrames
# ---------------------------------------------------------------------------


def _stub_fetch_factory(days: int = 5, anchor: str = "2024-01-02"):
    """Return a `fetch_fn` callable that records the (symbols, start, end)
    of its most recent invocation on ``.last_args`` for window assertions.

    Yields `days` business days starting at `anchor` for every symbol.
    """
    state: dict[str, object] = {"last_args": None}

    def _fetch(symbols, start, end):
        state["last_args"] = (list(symbols), start, end)
        ds = pd.bdate_range(anchor, periods=days).date.tolist()
        out: dict[str, pd.DataFrame] = {}
        for i, sym in enumerate(symbols):
            base = 100.0 + i * 10.0
            rows = [
                {
                    "date": d,
                    "open": base + j,
                    "high": base + j + 0.5,
                    "low": base + j - 0.5,
                    "close": base + j + 0.25,
                    "volume": 1000 * (j + 1),
                }
                for j, d in enumerate(ds)
            ]
            out[sym] = pd.DataFrame(rows)
        return out

    _fetch.last_args = lambda: state["last_args"]  # type: ignore[attr-defined]
    return _fetch


# ---------------------------------------------------------------------------
# 1. Long-format parquet — schema + row count
# ---------------------------------------------------------------------------


def test_backfill_writes_long_format_parquet(tmp_path):
    """3 symbols × 5 days → 15 rows, schema matches the design contract."""
    out = tmp_path / "sp500.parquet"
    fetch = _stub_fetch_factory(days=5)

    ob.backfill(
        symbols=["AAPL", "MSFT", "NVDA"],
        since="2024-01-01",
        out_path=out,
        fetch_fn=fetch,
    )

    df = pd.read_parquet(out)
    assert len(df) == 15
    assert list(df.columns) == [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "fetched_at",
        "yfinance_version",
    ]

    # pyarrow-level schema check (not just pandas dtypes)
    schema = pq.read_schema(out)
    field_types = {f.name: str(f.type) for f in schema}
    assert field_types["symbol"] == "string"
    assert field_types["date"] == "date32[day]"
    assert field_types["close"] == "double"
    assert field_types["volume"] == "int64"
    assert field_types["fetched_at"].startswith("timestamp[ns")
    assert field_types["yfinance_version"] == "string"

    # Sort key (symbol, date) ascending — same convention as thematic_universe.
    syms = df["symbol"].tolist()
    assert syms == sorted(syms)
    for sym in df["symbol"].unique():
        dates = df.loc[df["symbol"] == sym, "date"].tolist()
        assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# 2. Idempotency — re-run produces same row count, no dups
# ---------------------------------------------------------------------------


def test_backfill_idempotent(tmp_path):
    """Two consecutive one-shot runs over the same window do NOT duplicate.

    The backfill upserts on `(symbol, date)` so re-invoking with identical
    parameters leaves the parquet at the same row count.
    """
    out = tmp_path / "sp500.parquet"
    fetch = _stub_fetch_factory(days=5)

    ob.backfill(
        symbols=["AAPL", "MSFT"],
        since="2024-01-01",
        out_path=out,
        fetch_fn=fetch,
    )
    rows_before = len(pd.read_parquet(out))

    ob.backfill(
        symbols=["AAPL", "MSFT"],
        since="2024-01-01",
        out_path=out,
        fetch_fn=fetch,
    )
    df_after = pd.read_parquet(out)

    assert len(df_after) == rows_before
    # No duplicate (symbol, date) pairs.
    assert not df_after.duplicated(subset=["symbol", "date"]).any()


# ---------------------------------------------------------------------------
# 3. Incremental — last 5 calendar days only
# ---------------------------------------------------------------------------


def test_incremental_fetches_last_5_days_only(tmp_path):
    """`incremental=True` passes a 5-calendar-day window from today to fetch_fn.

    The full-backfill `since` arg is ignored; the windowing decision lives
    inside the backfill function so the cron call site doesn't have to
    re-derive "today minus 5 days".
    """
    out = tmp_path / "sp500.parquet"
    fetch = _stub_fetch_factory(days=2)

    today = date.today()

    ob.backfill(
        symbols=["AAPL"],
        since="2020-01-01",  # would be the full-backfill anchor; ignored
        out_path=out,
        incremental=True,
        fetch_fn=fetch,
        today=today,  # injected for determinism
    )

    syms, start, end = fetch.last_args()
    assert syms == ["AAPL"]
    assert start == (today - timedelta(days=5)).isoformat()
    assert end == today.isoformat()


# ---------------------------------------------------------------------------
# 4. Rate-limit retry — one transient failure, success on retry
# ---------------------------------------------------------------------------


def test_yfinance_rate_limit_retries(tmp_path):
    """Fetch raises RateLimitError on first call, succeeds on the second.

    Backfill completes; the retry counter exposes that exactly two attempts
    happened. Deterministic — no `time.sleep` assertion.
    """
    out = tmp_path / "sp500.parquet"

    stub = _stub_fetch_factory(days=3)
    state = {"calls": 0}

    def _flaky(symbols, start, end):
        state["calls"] += 1
        if state["calls"] == 1:
            raise ob.RateLimitError("yfinance rate limit (synthetic)")
        return stub(symbols, start, end)

    ob.backfill(
        symbols=["AAPL", "MSFT"],
        since="2024-01-01",
        out_path=out,
        fetch_fn=_flaky,
        retry_sleep=lambda _attempt: None,  # disable timing in tests
    )

    assert state["calls"] == 2
    df = pd.read_parquet(out)
    assert len(df) == 6  # 2 symbols x 3 days


# ---------------------------------------------------------------------------
# 5. Partial failure — atomic write, no half-written parquet visible
# ---------------------------------------------------------------------------


def test_partial_failure_does_not_corrupt_parquet(tmp_path):
    """If the fetch raises mid-run, the destination parquet is untouched.

    Atomic write convention: write to `.tmp` first, only rename on success.
    A pre-existing parquet on disk is NOT overwritten by a half-run.
    """
    out = tmp_path / "sp500.parquet"

    # Seed with a "good" prior backfill.
    fetch_good = _stub_fetch_factory(days=2)
    ob.backfill(
        symbols=["AAPL"],
        since="2024-01-01",
        out_path=out,
        fetch_fn=fetch_good,
    )
    pre_bytes = out.read_bytes()

    # Re-run with a fetcher that always blows up.
    def _broken(symbols, start, end):
        raise RuntimeError("synthetic yfinance failure mid-batch")

    with pytest.raises(RuntimeError):
        ob.backfill(
            symbols=["AAPL", "MSFT"],
            since="2024-01-01",
            out_path=out,
            fetch_fn=_broken,
            retry_sleep=lambda _attempt: None,
        )

    # On-disk parquet is byte-identical to the pre-existing one.
    assert out.read_bytes() == pre_bytes

    # And no `.tmp` sibling lingers next to it.
    assert not (tmp_path / "sp500.parquet.tmp").exists()


# ---------------------------------------------------------------------------
# 6. End-bound bump — yfinance.end is EXCLUSIVE, so the prod fetcher must
#    bump end by +1 calendar day before calling yf.download. Without the
#    bump, today's bar is silently dropped on every run (the 12:40 PT
#    cron's whole purpose is capturing today's bar pre-close, so this
#    would be a silent data-loss bug).
#
#    Regression test for /review iter-1 finding.
# ---------------------------------------------------------------------------


def test_yfinance_fetch_bumps_end_by_one_day(monkeypatch):
    """`_yfinance_fetch(end='2024-03-15')` must call `yf.download(end='2024-03-16')`.

    yfinance's `end=` is EXCLUSIVE; the caller contract passes inclusive
    end-date strings (matches scripts/backfill_thematic_universe.py:110).
    Without the +1 day bump the caller's "today" bar is dropped.
    """
    import yfinance as yf

    captured: dict[str, object] = {}

    def _fake_download(symbols, start, end, **kwargs):  # noqa: ARG001
        captured["start"] = start
        captured["end"] = end
        # Return an empty frame; we only care about the end= we received.
        return pd.DataFrame()

    monkeypatch.setattr(yf, "download", _fake_download)

    ob._yfinance_fetch(["AAPL"], start="2024-03-01", end="2024-03-15")

    assert captured["start"] == "2024-03-01"
    assert captured["end"] == "2024-03-16", (
        f"expected end bumped to 2024-03-16 (yfinance end is exclusive); "
        f"got {captured['end']!r}"
    )
