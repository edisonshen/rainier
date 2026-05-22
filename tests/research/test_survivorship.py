"""Survivorship-gate tests — D-016 PASS gate proof of concept.

The check verifies (a) every trading day in [from, to] has > 0 rows in
the QU100 snapshot history, (b) tickers that appear in earlier snapshots
but vanish before `to` are preserved (delisted but row-present), and
(c) the schema is insert-only (no UPDATE/DELETE on historical rows).

We model the table with an in-memory SQLite session here so the test is
hermetic (no Postgres dependency) — the production CLI takes any
SQLAlchemy session, so the same code path runs against either backend.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rainier.research import survivorship


@pytest.fixture
def in_memory_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    survivorship.bootstrap_sqlite(engine)
    with Session(engine, future=True) as s:
        yield s


def _seed_alive(session, symbols, start: date, end: date):
    """Insert one snapshot per (symbol, trading day) — alive across the window."""
    survivorship.seed_snapshots(session, symbols=symbols, start=start, end=end)


def _seed_delisted(session, symbol, from_date: date, until_date: date):
    """Insert snapshots for a symbol that disappears after `until_date`."""
    survivorship.seed_snapshots(session, symbols=[symbol], start=from_date, end=until_date)


def test_pass_when_full_coverage_no_drops(in_memory_session):
    start = date(2025, 1, 1)
    end = date(2025, 2, 1)
    _seed_alive(in_memory_session, ["AAPL", "NVDA", "TSLA"], start, end)
    result = survivorship.check(
        session=in_memory_session, from_date=start, to_date=end,
    )
    assert result.verdict == "PASS"
    assert result.delisted_tickers == []
    assert result.missing_days == []


def test_delisted_ticker_preserved_in_history(in_memory_session):
    # First NYSE session of 2025 is Jan 2 (Jan 1 is NYE holiday).
    start = date(2025, 1, 2)
    end = date(2025, 3, 1)
    drop_date = date(2025, 2, 3)  # first NYSE session in Feb 2025
    # OLDCO present Jan but gone Feb onwards.
    _seed_delisted(in_memory_session, "OLDCO", start, drop_date)
    # Anchors keep every trading day covered.
    _seed_alive(in_memory_session, ["AAPL", "NVDA"], start, end)
    result = survivorship.check(
        session=in_memory_session, from_date=start, to_date=end,
    )
    assert result.verdict == "PASS"
    assert "OLDCO" in result.delisted_tickers
    # OLDCO rows must still be queryable (preserved per D-009 / D-016).
    assert survivorship.ticker_present(in_memory_session, "OLDCO", start)


def test_missing_day_returns_conditional(in_memory_session):
    start = date(2025, 1, 1)
    end = date(2025, 1, 31)
    # Seed Jan 1–Jan 15, then skip a chunk, then Jan 25–Jan 31.
    _seed_alive(in_memory_session, ["AAPL"], start, date(2025, 1, 15))
    _seed_alive(in_memory_session, ["AAPL"], date(2025, 1, 25), end)
    result = survivorship.check(
        session=in_memory_session, from_date=start, to_date=end,
    )
    assert result.verdict == "CONDITIONAL"
    assert result.missing_days  # at least one gap
    assert result.next_step  # concrete remediation surfaced


def test_insert_only_violation_returns_conditional(in_memory_session):
    """Mutating a historical row trips the immutability proof."""
    # Use a known NYSE session as the mutation anchor (Jan 1 is NYE holiday).
    start = date(2025, 1, 2)
    end = date(2025, 1, 31)
    _seed_alive(in_memory_session, ["AAPL"], start, end)
    survivorship.mutate_historical_row_for_test(in_memory_session, "AAPL", start)
    result = survivorship.check(
        session=in_memory_session, from_date=start, to_date=end,
    )
    # Insert-only proof failed — return CONDITIONAL with concrete next-step.
    assert result.verdict == "CONDITIONAL"
    assert "insert-only" in result.next_step.lower() or "immutab" in result.next_step.lower()


def test_to_dict_shape(in_memory_session):
    start = date(2025, 1, 1)
    end = date(2025, 2, 1)
    _seed_alive(in_memory_session, ["AAPL"], start, end)
    result = survivorship.check(
        session=in_memory_session, from_date=start, to_date=end,
    )
    d = result.as_dict()
    assert d["from_date"] == "2025-01-01"
    assert d["to_date"] == "2025-02-01"
    assert d["verdict"] in {"PASS", "CONDITIONAL", "FAIL"}
    assert isinstance(d["delisted_tickers"], list)
    assert isinstance(d["missing_days"], list)


def test_nyse_holidays_not_reported_as_missing(in_memory_session):
    """Regression: codex iter-1 [P1] — NYE / MLK / Presidents Day / Good
    Friday / Memorial Day are NYSE-closed sessions. The gate must not
    flag them as missing on otherwise-healthy data. Pre-fix the check
    used weekday-only `_trading_days` and flipped PASS → CONDITIONAL
    on default-window runs starting at 2025-01-01.
    """
    # Default-ish window that straddles several NYSE holidays:
    # NYE (Jan 1), MLK Day (Jan 20), Presidents Day (Feb 17),
    # Good Friday (Apr 18), Memorial Day (May 26).
    start = date(2025, 1, 1)
    end = date(2025, 6, 1)
    # Seed via the same helper — which itself filters to NYSE sessions —
    # so the data represents an honest insert-only history.
    _seed_alive(in_memory_session, ["AAPL"], start, end)
    result = survivorship.check(
        session=in_memory_session, from_date=start, to_date=end,
    )
    assert result.verdict == "PASS", (
        f"unexpected verdict {result.verdict} with missing={result.missing_days}"
    )
    assert result.missing_days == []
