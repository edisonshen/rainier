"""Fear & Greed ingest — the core user functions and the contracts they protect.

Core user functions: ``rainier fear-greed fetch`` and ``... backfill`` (driven
here as ``fetch()`` / ``backfill()`` with an injected fake CNN client).

One focused test per contract (table-driven where cases share a driver):
  1. fetch writes a correct observation (composite + all 9 components + raw).
  2. append-on-change / no duplicates  (unchanged / changed / dup-current-day).
  3. backfill loads history as ``source_version=backfill``.
  4. fail-loud on a bad CNN response — persist nothing.
  5. point-in-time boundary — daily vs backfill distinguishable.

Deterministic + offline: a recorded JSON fixture, SQLite-backed DB (the ORM
downgrades JSONB→JSON and BigInteger→INTEGER on SQLite), no live network, no
timing sleeps.
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime, timezone

import pytest
from sqlalchemy import func, select

from rainier.core.models import FearGreedIndex
from rainier.data.fear_greed import (
    COMPONENT_KEYS,
    EARLIEST_DATE,
    FearGreedError,
    backfill,
    fetch,
    parse_observations,
    persist_observations,
)

# Two distinct ingest instants — the append-on-change / PIT contracts depend on
# observed_at ordering, so tests pin it rather than lean on wall-clock now().
T1 = datetime(2026, 7, 29, 22, 10, tzinfo=timezone.utc)
T2 = datetime(2026, 7, 30, 22, 10, tzinfo=timezone.utc)
T3 = datetime(2026, 7, 31, 22, 10, tzinfo=timezone.utc)

# Fixture trading days (2020-09-21..23); D_LAST is the "current" day CNN dupes.
D_FIRST = date(2020, 9, 21)
D_LAST = date(2020, 9, 23)


class _FakeResponse:
    def __init__(self, status_code: int, body, bad_json: bool = False):
        self.status_code = status_code
        self._body = body
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:  # a 200 with an HTML interstitial body
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._body


class _FakeClient:
    """httpx.Client stand-in recording the URLs it was asked to GET."""

    def __init__(self, status_code: int = 200, body=None, bad_json: bool = False):
        self._status = status_code
        self._body = body
        self._bad_json = bad_json
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, headers=None, **kwargs):
        self.calls.append((url, headers or {}))
        return _FakeResponse(self._status, self._body, self._bad_json)


def _rows_for_date(session_factory, d: date) -> list[FearGreedIndex]:
    with session_factory() as s:
        return (
            s.execute(
                select(FearGreedIndex)
                .where(FearGreedIndex.date == d)
                .order_by(FearGreedIndex.observed_at, FearGreedIndex.id)
            )
            .scalars()
            .all()
        )


def _row_count(session_factory) -> int:
    with session_factory() as s:
        return s.execute(select(func.count()).select_from(FearGreedIndex)).scalar_one()


# --------------------------------------------------------------------------
# 1. fetch writes a correct observation
# --------------------------------------------------------------------------


def test_fetch_writes_correct_observation(session_factory, payload):
    """CONTRACT: `fear-greed fetch` parses+persists an observation with the
    composite score+rating, all 9 component `*_score` columns populated, and
    every rating label preserved in `raw`."""
    client = _FakeClient(body=payload)
    assert fetch(session_factory=session_factory, client=client) == 3

    row = _rows_for_date(session_factory, D_FIRST)[0]
    assert row.score == 45.0
    assert row.rating == "neutral"
    assert row.source_version == "daily"
    # All 9 components stored — sp125 and vix_50 are the easy-to-drop pair.
    for col in COMPONENT_KEYS.values():
        assert getattr(row, col) is not None
    assert row.momentum_sp125_score is not None
    assert row.volatility_vix_50_score is not None
    # Composite + per-component rating labels recoverable from raw.
    assert row.raw["composite"]["rating"] == "neutral"
    assert set(row.raw["components"]) == set(COMPONENT_KEYS)
    for key in COMPONENT_KEYS:
        assert "rating" in row.raw["components"][key]


# --------------------------------------------------------------------------
# 2. append-on-change / no duplicates  (load-bearing dedup + the live bug)
# --------------------------------------------------------------------------


def _case_unchanged(payload, dup):
    obs = parse_observations(payload)
    return dict(first=obs, second=obs, n1=3, n2=0, target=D_FIRST, scores=[45.0])


def _case_changed(payload, dup):
    obs = parse_observations(payload)
    revised = list(obs)
    revised[0] = dataclasses.replace(revised[0], score=99.0)
    return dict(first=obs, second=revised, n1=3, n2=1, target=D_FIRST, scores=[45.0, 99.0])


def _case_dup_current_day(payload, dup):
    # CNN serves the current (unsettled) day twice with different scores; parse
    # collapses to the latest reading (63.0) → a single persist writes one row.
    obs = parse_observations(dup)
    return dict(first=obs, second=None, n1=3, n2=0, target=D_LAST, scores=[63.0])


@pytest.mark.parametrize(
    "build",
    [_case_unchanged, _case_changed, _case_dup_current_day],
    ids=["unchanged->0-new", "changed->1-new-immutable", "dup-current-day->collapse-1"],
)
def test_append_on_change(session_factory, payload, payload_dup_current_day, build):
    """CONTRACT (append-only-on-change; the load-bearing correctness property and
    the live current-day-twice bug): append a new immutable row ONLY when the
    value changed; an identical re-pull is a no-op; CNN's duplicated current day
    collapses to exactly one row; a prior row is never mutated."""
    c = build(payload, payload_dup_current_day)
    with session_factory() as s:
        assert persist_observations(s, c["first"], source_version="daily", observed_at=T1) == c["n1"]
    if c["second"] is not None:
        with session_factory() as s:
            assert (
                persist_observations(s, c["second"], source_version="daily", observed_at=T2)
                == c["n2"]
            )
    rows = _rows_for_date(session_factory, c["target"])
    assert [r.score for r in rows] == c["scores"]
    assert rows[0].score == c["scores"][0]  # earliest row immutable


# --------------------------------------------------------------------------
# 3. backfill loads history
# --------------------------------------------------------------------------


def test_backfill_loads_history(session_factory, payload):
    """CONTRACT: `fear-greed backfill` loads one row per trading day over the
    range, tagged `source_version=backfill`, starting at the earliest CNN date."""
    client = _FakeClient(body=payload)
    assert backfill(session_factory=session_factory, client=client) == 3
    assert _row_count(session_factory) == 3
    with session_factory() as s:
        versions = set(s.execute(select(FearGreedIndex.source_version)).scalars())
        span = s.execute(
            select(func.min(FearGreedIndex.date), func.max(FearGreedIndex.date))
        ).one()
    assert versions == {"backfill"}
    assert span == (D_FIRST, D_LAST)
    assert client.calls[0][0].endswith(f"/{EARLIEST_DATE.isoformat()}")


# --------------------------------------------------------------------------
# 4. fail-loud — a bad CNN response persists nothing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "client",
    [
        _FakeClient(status_code=418, body=None),
        _FakeClient(status_code=200, body={}),
        _FakeClient(status_code=200, bad_json=True),
    ],
    ids=["non-200", "empty-body", "non-json-body"],
)
def test_fetch_fails_loud_persists_nothing(session_factory, client):
    """CONTRACT: a non-200, empty, or non-JSON CNN response raises FearGreedError
    and writes zero rows — the fail-loud boundary owns every bad response."""
    with pytest.raises(FearGreedError):
        fetch(session_factory=session_factory, client=client)
    assert _row_count(session_factory) == 0


# --------------------------------------------------------------------------
# 5. point-in-time boundary — daily vs backfill
# --------------------------------------------------------------------------


def test_pit_boundary_daily_vs_backfill(session_factory, payload):
    """CONTRACT: backfill (revised) and daily (live capture) rows are
    distinguishable; the first live daily fetch after backfill records a `daily`
    row even when the value is unchanged, so the derived boundary
    `MIN(observed_at) WHERE source_version='daily'` is the live-capture instant;
    a second identical daily re-pull no-ops."""
    obs = parse_observations(payload)
    with session_factory() as s:
        assert persist_observations(s, obs, source_version="backfill", observed_at=T1) == 3
    with session_factory() as s:
        assert persist_observations(s, obs, source_version="daily", observed_at=T2) == 3
    with session_factory() as s:
        assert persist_observations(s, obs, source_version="daily", observed_at=T3) == 0

    rows = _rows_for_date(session_factory, D_FIRST)
    assert [r.source_version for r in rows] == ["backfill", "daily"]
    with session_factory() as s:
        boundary = s.execute(
            select(func.min(FearGreedIndex.observed_at)).where(
                FearGreedIndex.source_version == "daily"
            )
        ).scalar_one()
    # SQLite drops tzinfo on round-trip; compare naive-to-naive.
    assert boundary.replace(tzinfo=None) == T2.replace(tzinfo=None)
    assert _row_count(session_factory) == 6  # 3 backfill + 3 daily
