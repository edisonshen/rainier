"""Ingest behaviour: parse, append-on-change dedup, revision immutability,
fail-loud, and the backfill/fetch orchestrators.

All 9 CNN component series are stored as `*_score` columns; the composite plus
each component's rating label live in `raw`. Point-in-time capture is
append-only-on-change: an unchanged re-pull is a no-op, a changed value appends
a new immutable observation.
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
    fetch_graphdata,
    parse_observations,
    persist_observations,
)

T1 = datetime(2026, 7, 29, 22, 10, tzinfo=timezone.utc)
T2 = datetime(2026, 7, 30, 22, 10, tzinfo=timezone.utc)


class _FakeResponse:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _FakeClient:
    """Minimal httpx.Client stand-in recording the URL + headers it saw."""

    def __init__(self, status_code: int = 200, body=None):
        self._status = status_code
        self._body = body
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, headers=None, **kwargs):
        self.calls.append((url, headers or {}))
        return _FakeResponse(self._status, self._body)


# --------------------------------------------------------------------------
# Parse
# --------------------------------------------------------------------------


def test_parse_yields_one_observation_per_date(payload):
    obs = parse_observations(payload)
    assert [o.date for o in obs] == [
        date(2020, 9, 21),
        date(2020, 9, 22),
        date(2020, 9, 23),
    ]


def test_parse_composite_and_all_nine_components(payload):
    first = parse_observations(payload)[0]
    assert first.score == 45.0
    assert first.rating == "neutral"  # CNN band: 45 ≤ score < 55
    # All 9 component columns present and non-null.
    assert set(first.components) == set(COMPONENT_KEYS.values())
    assert all(v is not None for v in first.components.values())
    # The two easy-to-drop series are explicitly present.
    assert first.components["momentum_sp125_score"] is not None
    assert first.components["volatility_vix_50_score"] is not None


def test_parse_keeps_rating_labels_in_raw(payload):
    first = parse_observations(payload)[0]
    # 9 component rating labels recoverable from raw, plus the composite rating.
    assert first.raw["composite"]["rating"] == "neutral"
    assert set(first.raw["components"]) == set(COMPONENT_KEYS)
    for key in COMPONENT_KEYS:
        assert "rating" in first.raw["components"][key]


# --------------------------------------------------------------------------
# Persist — append-on-change
# --------------------------------------------------------------------------


def _row_count(session_factory) -> int:
    with session_factory() as s:
        return s.execute(select(func.count()).select_from(FearGreedIndex)).scalar_one()


def test_all_nine_score_columns_persisted(session_factory, payload):
    obs = parse_observations(payload)
    with session_factory() as s:
        persist_observations(s, obs, source_version="backfill", observed_at=T1)
    with session_factory() as s:
        row = s.execute(
            select(FearGreedIndex).where(FearGreedIndex.date == date(2020, 9, 21))
        ).scalar_one()
    assert row.score == 45.0
    assert row.rating == "neutral"
    assert row.momentum_sp125_score is not None
    assert row.volatility_vix_50_score is not None
    for col in COMPONENT_KEYS.values():
        assert getattr(row, col) is not None
    # rating labels survive in raw
    assert row.raw["components"]["market_volatility_vix_50"]["rating"] is not None


def test_idempotent_double_fire_identical_value(session_factory, payload):
    obs = parse_observations(payload)
    with session_factory() as s:
        n1 = persist_observations(s, obs, source_version="daily", observed_at=T1)
    with session_factory() as s:
        n2 = persist_observations(s, obs, source_version="daily", observed_at=T2)
    assert n1 == 3
    assert n2 == 0  # identical re-pull → no new rows
    assert _row_count(session_factory) == 3


def test_revision_appends_new_observation_original_unchanged(session_factory, payload):
    obs = parse_observations(payload)
    with session_factory() as s:
        persist_observations(s, obs, source_version="daily", observed_at=T1)

    # Simulate a revised re-pull: the first date's composite score moved.
    revised = list(obs)
    revised[0] = dataclasses.replace(revised[0], score=99.0)
    with session_factory() as s:
        n = persist_observations(s, revised, source_version="daily", observed_at=T2)
    assert n == 1  # only the changed date appends

    d0 = date(2020, 9, 21)
    with session_factory() as s:
        rows = (
            s.execute(
                select(FearGreedIndex)
                .where(FearGreedIndex.date == d0)
                .order_by(FearGreedIndex.observed_at)
            )
            .scalars()
            .all()
        )
    assert len(rows) == 2
    assert rows[0].score == 45.0  # original immutable
    assert rows[1].score == 99.0  # new observation


def test_revision_on_component_score_appends(session_factory, payload):
    """A change in a component score (composite unchanged) still appends."""
    obs = parse_observations(payload)
    with session_factory() as s:
        persist_observations(s, obs, source_version="daily", observed_at=T1)
    revised = list(obs)
    comps = dict(revised[0].components)
    comps["put_call_score"] = comps["put_call_score"] + 1.0
    revised[0] = dataclasses.replace(revised[0], components=comps)
    with session_factory() as s:
        n = persist_observations(s, revised, source_version="daily", observed_at=T2)
    assert n == 1


# --------------------------------------------------------------------------
# Fail-loud
# --------------------------------------------------------------------------


def test_fetch_graphdata_raises_on_non_200():
    client = _FakeClient(status_code=418, body=None)
    with pytest.raises(FearGreedError):
        fetch_graphdata(EARLIEST_DATE, client=client)


def test_fetch_graphdata_raises_on_empty_payload():
    client = _FakeClient(status_code=200, body={})
    with pytest.raises(FearGreedError):
        fetch_graphdata(EARLIEST_DATE, client=client)


def test_fetch_graphdata_sends_cnn_headers_and_dated_url(payload):
    client = _FakeClient(status_code=200, body=payload)
    got = fetch_graphdata(date(2020, 9, 21), client=client)
    assert got is payload
    url, headers = client.calls[0]
    assert url.endswith("/2020-09-21")
    assert headers["Origin"] == "https://edition.cnn.com"
    assert headers["Referer"] == "https://edition.cnn.com/"
    assert "User-Agent" in headers
    assert headers["Accept"] == "application/json"


def test_fetch_failure_persists_nothing(session_factory):
    client = _FakeClient(status_code=418, body=None)
    with pytest.raises(FearGreedError):
        fetch(session_factory=session_factory, client=client)
    assert _row_count(session_factory) == 0


# --------------------------------------------------------------------------
# Orchestrators
# --------------------------------------------------------------------------


def test_backfill_persists_range_as_backfill(session_factory, payload):
    client = _FakeClient(status_code=200, body=payload)
    n = backfill(session_factory=session_factory, client=client)
    assert n == 3
    with session_factory() as s:
        versions = set(
            s.execute(select(FearGreedIndex.source_version)).scalars().all()
        )
        span = s.execute(
            select(func.min(FearGreedIndex.date), func.max(FearGreedIndex.date))
        ).one()
    assert versions == {"backfill"}
    assert span == (date(2020, 9, 21), date(2020, 9, 23))
    # backfill default start is the earliest CNN date.
    assert client.calls[0][0].endswith(f"/{EARLIEST_DATE.isoformat()}")


def test_fetch_tags_rows_daily(session_factory, payload):
    client = _FakeClient(status_code=200, body=payload)
    n = fetch(session_factory=session_factory, client=client)
    assert n == 3
    with session_factory() as s:
        versions = set(
            s.execute(select(FearGreedIndex.source_version)).scalars().all()
        )
    assert versions == {"daily"}
