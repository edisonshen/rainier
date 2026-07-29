"""CNN Fear & Greed Index ingest — point-in-time, append-only-on-change.

Fetches the CNN graphdata endpoint, parses the composite plus 9 component
series, and appends a new ``(date, observed_at)`` observation to
``fear_greed_index`` ONLY when the value changed since the last stored
observation for that date.

    CNN graphdata  ──fetch──▶  parse  ──▶  persist (append-on-change)
      (fail-loud)              (align       │
                               by date)     ├─ latest row for date differs? → INSERT
                                            └─ identical? → no-op

Point-in-time contract (read by Phase 2): ``backfill`` rows carry
``source_version='backfill'`` and ``observed_at = ingest time`` — these are the
CNN endpoint's *revised* values, NOT true point-in-time. ``fetch`` rows carry
``source_version='daily'``. The true-PIT boundary is DERIVED, never stored:
``SELECT MIN(observed_at) FROM fear_greed_index WHERE source_version='daily'``.

The engine is the LEGACY ``core.database`` one (where ``stock_prices`` lives)
so Phase 2 can co-locate the join — NOT the canonical Neon ``db.engine``
(memory ``project_two_database_url_engines``).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from rainier.core.database import get_session
from rainier.core.models import FearGreedIndex

BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

# Earliest date the CNN endpoint serves; earlier dates return HTTP 500.
EARLIEST_DATE = date(2020, 9, 21)

# Trailing window pulled by the daily `fetch` (self-heals a missed cron day).
FETCH_LOOKBACK_DAYS = 30

# Origin/Referer = cnn.com + a browser UA are mandatory; without them the
# endpoint returns HTTP 418 (bot-block). Verified live 2026-07-28.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Origin": "https://edition.cnn.com",
    "Referer": "https://edition.cnn.com/",
}

_HISTORICAL_KEY = "fear_and_greed_historical"

# CNN payload series key → our `*_score` column. All 9 are stored; sp125 and
# vix_50 are easy to drop by accident — they are NOT optional here.
COMPONENT_KEYS: dict[str, str] = {
    "market_momentum_sp500": "momentum_sp500_score",
    "market_momentum_sp125": "momentum_sp125_score",
    "stock_price_strength": "price_strength_score",
    "stock_price_breadth": "price_breadth_score",
    "put_call_options": "put_call_score",
    "market_volatility_vix": "volatility_vix_score",
    "market_volatility_vix_50": "volatility_vix_50_score",
    "junk_bond_demand": "junk_bond_demand_score",
    "safe_haven_demand": "safe_haven_demand_score",
}


class FearGreedError(RuntimeError):
    """Raised on a bad CNN response (non-200 / empty) — persist nothing."""


@dataclass(frozen=True)
class FearGreedObservation:
    """One trading day's composite + 9 component scores, plus the raw slice."""

    date: date
    score: float
    rating: str | None
    components: dict[str, float | None]  # column name → score
    raw: dict


SessionFactory = Callable[[], AbstractContextManager[Session]]


def _epoch_ms_to_date(x) -> date:
    """CNN `x` is epoch milliseconds; map to its UTC calendar date."""
    return datetime.fromtimestamp(int(x) / 1000, tz=timezone.utc).date()


def _point_value(point: dict) -> float | None:
    y = point.get("y")
    return None if y is None else float(y)


# --------------------------------------------------------------------------
# Fetch (fail-loud)
# --------------------------------------------------------------------------


def fetch_graphdata(start: date, *, client: httpx.Client | None = None) -> dict:
    """GET the CNN graphdata payload for ``start``→today. Raise on failure.

    A non-200 status or an empty/malformed payload raises ``FearGreedError``;
    the caller persists nothing.
    """
    url = f"{BASE_URL}/{start.isoformat()}"
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        resp = client.get(url, headers=_HEADERS)
    finally:
        if owns_client:
            client.close()

    if resp.status_code != 200:
        raise FearGreedError(
            f"CNN Fear & Greed fetch failed: HTTP {resp.status_code} for {url}"
        )
    try:
        payload = resp.json()
    except ValueError as exc:  # 200 with a non-JSON body (e.g. an HTML interstitial)
        raise FearGreedError(
            f"CNN Fear & Greed returned a non-JSON body for {url}"
        ) from exc
    if not payload or not payload.get(_HISTORICAL_KEY, {}).get("data"):
        raise FearGreedError(f"CNN Fear & Greed returned an empty payload for {url}")
    return payload


# --------------------------------------------------------------------------
# Parse (align composite + components by date)
# --------------------------------------------------------------------------


def parse_observations(payload: dict) -> list[FearGreedObservation]:
    """Turn a CNN graphdata payload into one observation per composite date."""
    historical = payload.get(_HISTORICAL_KEY, {}).get("data")
    if not historical:
        raise FearGreedError("payload has no fear_and_greed_historical data")

    # date → raw point, per component series. The dict comprehension collapses
    # any same-date duplicates to the LAST point (last-write-wins), matching the
    # composite collapse below.
    comp_by_date: dict[str, dict[date, dict]] = {}
    for key in COMPONENT_KEYS:
        series = payload.get(key, {}).get("data", []) or []
        comp_by_date[key] = {_epoch_ms_to_date(pt["x"]): pt for pt in series}

    # Collapse the composite series to one point per date. CNN emits MULTIPLE
    # points for the current (unsettled) trading day with different scores
    # (verified live 2026-07-29: today present as both 33.4531 and 33.4694).
    # Keep the LAST/most-recent point per date — CNN's latest reading for that
    # day, which matches the `fear_and_greed.score` current composite. Without
    # this collapse, parse emits a row per duplicate, persist stamps them with a
    # single batch observed_at, and the ambiguous "latest by observed_at" read
    # makes every later fetch append again (current day grows unbounded).
    # Dict insertion order preserves chronological date order (a repeated date
    # keeps its first-seen position but takes the last value).
    composite_by_date: dict[date, dict] = {}
    for pt in historical:
        composite_by_date[_epoch_ms_to_date(pt["x"])] = pt

    observations: list[FearGreedObservation] = []
    for d, pt in composite_by_date.items():
        components: dict[str, float | None] = {}
        raw_components: dict[str, dict | None] = {}
        for key, column in COMPONENT_KEYS.items():
            cpt = comp_by_date[key].get(d)
            components[column] = _point_value(cpt) if cpt else None
            raw_components[key] = cpt  # includes the component's rating label
        observations.append(
            FearGreedObservation(
                date=d,
                score=float(pt["y"]),
                rating=pt.get("rating"),
                components=components,
                raw={
                    "date": d.isoformat(),
                    "composite": {"y": pt.get("y"), "rating": pt.get("rating")},
                    "components": raw_components,
                },
            )
        )
    return observations


# --------------------------------------------------------------------------
# Persist (append-only-on-change)
# --------------------------------------------------------------------------


def _differs(latest: FearGreedIndex, obs: FearGreedObservation) -> bool:
    """True if the pulled observation differs from the latest stored row."""
    if latest.score != obs.score or latest.rating != obs.rating:
        return True
    return any(
        getattr(latest, column) != value for column, value in obs.components.items()
    )


def persist_observations(
    session: Session,
    observations: list[FearGreedObservation],
    *,
    source_version: str,
    observed_at: datetime | None = None,
) -> int:
    """Append changed observations; return the number of rows inserted.

    For each date, the latest stored observation (by ``observed_at`` DESC, then
    ``id`` DESC to break same-``observed_at`` ties deterministically) is compared
    against the pulled value: identical → no-op; changed or absent → a new
    immutable ``(date, observed_at)`` row. Never mutates a prior row.
    """
    observed_at = observed_at or datetime.now(tz=timezone.utc)
    inserted = 0
    for obs in observations:
        latest = session.execute(
            select(FearGreedIndex)
            .where(FearGreedIndex.date == obs.date)
            .order_by(FearGreedIndex.observed_at.desc(), FearGreedIndex.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        if latest is not None and not _differs(latest, obs):
            # Value unchanged. Normally a no-op — EXCEPT the first live `daily`
            # fetch that confirms a date previously seen only via `backfill`
            # (revised, research-grade). Record that transition so the derived
            # PIT boundary `MIN(observed_at) WHERE source_version='daily'` is
            # populated when live capture starts, even if CNN hasn't revised the
            # value since the backfill. Without this, an operator who backfills
            # then arms the daily cron never establishes a true point-in-time
            # boundary until CNN happens to change a number. A subsequent
            # daily→daily identical re-pull still no-ops (idempotent).
            provenance_upgrade = (
                source_version == "daily" and latest.source_version != "daily"
            )
            if not provenance_upgrade:
                continue
        session.add(
            FearGreedIndex(
                date=obs.date,
                observed_at=observed_at,
                score=obs.score,
                rating=obs.rating,
                raw=obs.raw,
                source_version=source_version,
                **obs.components,
            )
        )
        inserted += 1
    session.flush()
    return inserted


# --------------------------------------------------------------------------
# Orchestrators
# --------------------------------------------------------------------------


def backfill(
    start: date = EARLIEST_DATE,
    *,
    session_factory: SessionFactory = get_session,
    client: httpx.Client | None = None,
) -> int:
    """Pull ``start``→today in one call and append as ``source_version=backfill``."""
    payload = fetch_graphdata(start, client=client)
    observations = parse_observations(payload)
    with session_factory() as session:
        return persist_observations(session, observations, source_version="backfill")


def fetch(
    *,
    session_factory: SessionFactory = get_session,
    client: httpx.Client | None = None,
    lookback_days: int = FETCH_LOOKBACK_DAYS,
) -> int:
    """Pull a trailing window and append today's observation as ``daily``."""
    start = datetime.now(tz=timezone.utc).date() - timedelta(days=lookback_days)
    if start < EARLIEST_DATE:
        start = EARLIEST_DATE
    payload = fetch_graphdata(start, client=client)
    observations = parse_observations(payload)
    with session_factory() as session:
        return persist_observations(session, observations, source_version="daily")
