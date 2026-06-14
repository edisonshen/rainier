"""WS B — reconsider invalidated bull-traps (`false_breakout` only).

A `false_breakout` is a CORRECT bearish setup: price pokes above a resistance
high (the "trap high" / `false_high`), then rejects and declines. The detector
is correct and untouched. But when a LATER daily close reclaims the trap high,
the bear thesis is dead — the symbol may now be a fresh LONG. The scheduled LLM
run only ever sees today's top-5, so a reclaimed name outside it would be lost.

This module:

  1. detect_and_enqueue_reclaims — daily post-ingest check. For each recently
     declined `false_breakout` thesis with a persisted trap high, if a close in
     the lookback window reclaimed the level, durably enqueue the symbol
     (idempotent + once-per-K-days dedup).
  2. audit_reclaims — historical hit-rate of reclaim events (mean forward
     return over N days). The auto-thesis gate reads this.
  3. process_reclaim_queue — when auto-thesis is enabled AND the audit gate
     passes, inject queued symbols into the thesis path as FRESH long theses;
     otherwise leave them queued for human review (detection-only fallback).

ASCII flow:

    daily ingest done
        │
        ▼
    detect_and_enqueue_reclaims(eval_date)
        │  (declined false_breakout, trap reclaimed, K-day dedup)
        ▼
    paper_reclaim_queue rows  ── status: queued
        │
        ▼
    process_reclaim_queue(eval_date)
        │  gate = reclaim_auto_thesis AND audit passes
        ├─ gate FAILS → leave queued (detection-only)
        └─ gate PASSES → spawn fresh long thesis, status: reenqueued→done
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rainier.core.database import get_session
from rainier.core.models import (
    LLMAnalysisRecord,
    PaperReclaimQueue,
    ScreenedStockRecord,
    StockPrice,
)

log = logging.getLogger(__name__)

# Only the exact bearish trap pattern is in scope. `false_breakout_hs` has no
# `false_high` key-point, so it never persists a `bearish_invalidation_level`
# and never enters this path (scope regression in the tests).
RECLAIM_PATTERN = "false_breakout"


@dataclass(frozen=True)
class ReclaimCandidate:
    """A declined false_breakout whose trap high might be reclaimed."""

    symbol: str
    invalidated_thesis_id: int
    invalidation_level: float
    scan_date: date


def is_reclaimed(closes: list[float], invalidation_level: float) -> bool:
    """Pure: True iff any close strictly exceeds the trap high.

    A close back ABOVE the trap high invalidates the bear setup (the breakout
    is no longer false). Equality does not reclaim — the bear setup needs price
    to clear, not merely touch, the level.
    """
    return any(c > invalidation_level for c in closes)


def _closes_in_window(
    session, symbol: str, start: date, end: date
) -> list[tuple[date, float]]:
    from rainier.paper.ingest import canonical_instant

    rows = session.execute(
        select(StockPrice.date, StockPrice.close)
        .where(
            StockPrice.symbol == symbol,
            StockPrice.date >= canonical_instant(start),
            StockPrice.date <= canonical_instant(end),
            StockPrice.close.is_not(None),
        )
        .order_by(StockPrice.date)
    ).all()
    out: list[tuple[date, float]] = []
    for d, c in rows:
        dd = d.date() if hasattr(d, "date") else d
        out.append((dd, float(c)))
    return out


def _find_reclaim_candidates(
    session, *, eval_date: date, window_days: int
) -> list[ReclaimCandidate]:
    """Declined false_breakout theses in the window with a persisted trap high.

    A row qualifies when its pattern is exactly `false_breakout`, it carries a
    `bearish_invalidation_level`, it has an LLM thesis, and that thesis did NOT
    recommend a long (a `setup_long` was already acted on the bullish side, so
    there is no bear trap to reconsider).
    """
    earliest = eval_date - timedelta(days=window_days)
    rows = session.execute(
        select(
            ScreenedStockRecord.symbol,
            ScreenedStockRecord.thesis_id,
            ScreenedStockRecord.bearish_invalidation_level,
            ScreenedStockRecord.scan_date,
            LLMAnalysisRecord.recommendation,
        )
        .join(
            LLMAnalysisRecord,
            LLMAnalysisRecord.id == ScreenedStockRecord.thesis_id,
        )
        .where(
            ScreenedStockRecord.pattern_type == RECLAIM_PATTERN,
            ScreenedStockRecord.bearish_invalidation_level.is_not(None),
            ScreenedStockRecord.thesis_id.is_not(None),
            ScreenedStockRecord.scan_date >= earliest,
            ScreenedStockRecord.scan_date <= eval_date,
        )
    ).all()

    candidates: list[ReclaimCandidate] = []
    for symbol, thesis_id, level, scan_date, recommendation in rows:
        if recommendation == "setup_long":
            continue
        sd = scan_date.date() if hasattr(scan_date, "date") else scan_date
        candidates.append(
            ReclaimCandidate(
                symbol=symbol,
                invalidated_thesis_id=int(thesis_id),
                invalidation_level=float(level),
                scan_date=sd,
            )
        )
    return candidates


def _recent_queue_row(
    session, symbol: str, *, since: date
) -> PaperReclaimQueue | None:
    """Most-recent queue row for `symbol` with reclaim_date >= `since` (K-day
    dedup). Any such row means we already enqueued this symbol within K days."""
    return session.execute(
        select(PaperReclaimQueue)
        .where(
            PaperReclaimQueue.symbol == symbol,
            PaperReclaimQueue.reclaim_date >= since,
        )
        .order_by(PaperReclaimQueue.reclaim_date.desc())
    ).scalars().first()


def detect_and_enqueue_reclaims(
    *,
    eval_date: date,
    window_days: int,
    dedup_days: int,
) -> dict[str, int]:
    """Detect reclaimed bull-traps and durably enqueue them (idempotent).

    For each declined `false_breakout` in the last `window_days`, look for a
    close in (scan_date .. eval_date] that exceeds the trap high. On a reclaim,
    insert a `paper_reclaim_queue` row UNLESS:
      * the same (symbol, invalidated_thesis_id) is already queued
        (UNIQUE constraint → ON CONFLICT DO NOTHING), or
      * the symbol was already enqueued within the last `dedup_days`
        (whipsaw dedup: below→above→below→above within K days → one row).

    Returns {"detected": n, "enqueued": m}.
    """
    detected = enqueued = 0
    dedup_since = eval_date - timedelta(days=dedup_days)

    with get_session() as session:
        candidates = _find_reclaim_candidates(
            session, eval_date=eval_date, window_days=window_days
        )

    for cand in candidates:
        with get_session() as session:
            # Only consider closes AFTER the bearish scan (the trap formed then).
            closes = _closes_in_window(
                session, cand.symbol, cand.scan_date, eval_date
            )
            reclaim_close = None
            reclaim_date = None
            for d, c in closes:
                if c > cand.invalidation_level:
                    reclaim_close = c
                    reclaim_date = d
                    break
            if reclaim_close is None:
                continue
            detected += 1

            # Whipsaw / K-day dedup: skip if this symbol was enqueued recently.
            if _recent_queue_row(session, cand.symbol, since=dedup_since) is not None:
                continue

            stmt = (
                pg_insert(PaperReclaimQueue)
                .values(
                    symbol=cand.symbol,
                    invalidated_thesis_id=cand.invalidated_thesis_id,
                    invalidation_level=cand.invalidation_level,
                    reclaim_date=reclaim_date,
                    reclaim_close=reclaim_close,
                    status="queued",
                )
                .on_conflict_do_nothing(
                    constraint="uq_paper_reclaim_symbol_thesis"
                )
                .returning(PaperReclaimQueue.id)
            )
            inserted = session.execute(stmt).first() is not None
            if inserted:
                enqueued += 1

    return {"detected": detected, "enqueued": enqueued}


@dataclass(frozen=True)
class AuditResult:
    """Post-reclaim audit summary feeding the auto-thesis gate."""

    events: int
    mean_forward_return: float | None
    passes: bool


def compute_audit_pass(
    forward_returns: list[float], *, min_events: int
) -> AuditResult:
    """Pure: the auto-thesis gate decision.

    Gate passes iff there are at least `min_events` historical reclaim events
    AND their mean forward return is strictly positive. Below the floor or a
    non-positive mean → fail (detection + queue only).
    """
    events = len(forward_returns)
    if events == 0:
        return AuditResult(events=0, mean_forward_return=None, passes=False)
    mean = sum(forward_returns) / events
    passes = events >= min_events and mean > 0.0
    return AuditResult(events=events, mean_forward_return=mean, passes=passes)


def audit_reclaims(*, eval_date: date, horizon_days: int, min_events: int) -> AuditResult:
    """Audit historical reclaim events: forward return `horizon_days` after each.

    For every queue row whose reclaim is old enough to have a full forward
    window, compute the close-to-close return from the reclaim close to the
    close `horizon_days` later. Feeds `compute_audit_pass`.
    """
    cutoff = eval_date - timedelta(days=horizon_days)
    with get_session() as session:
        rows = session.execute(
            select(
                PaperReclaimQueue.symbol,
                PaperReclaimQueue.reclaim_date,
                PaperReclaimQueue.reclaim_close,
            ).where(PaperReclaimQueue.reclaim_date <= cutoff)
        ).all()

        forward_returns: list[float] = []
        for symbol, reclaim_date, reclaim_close in rows:
            rd = reclaim_date.date() if hasattr(reclaim_date, "date") else reclaim_date
            target_day = rd + timedelta(days=horizon_days)
            closes = _closes_in_window(session, symbol, rd, target_day)
            if not closes or reclaim_close in (None, 0):
                continue
            # Last available close at/after the horizon.
            _, fwd_close = closes[-1]
            forward_returns.append((fwd_close - reclaim_close) / reclaim_close)

    return compute_audit_pass(forward_returns, min_events=min_events)


def process_reclaim_queue(
    *,
    eval_date: date,
    auto_thesis: bool,
    audit_horizon_days: int,
    audit_min_events: int,
    spawn_fn=None,
) -> dict[str, int]:
    """Act on queued reclaims, gated on config + the post-reclaim audit.

    Default behavior (auto_thesis False OR audit gate fails): leave every queued
    row as-is (detection + queue only) — the designed fallback. The queue is
    surfaced for review but no fresh thesis is spawned.

    When `auto_thesis` is True AND the audit gate passes, each `queued` row is
    transitioned `queued -> reenqueued` and `spawn_fn(symbol, queue_id)` is
    invoked to create a fresh long thesis via the normal screen→LLM path. On a
    truthy return (the new thesis_id), the row is finalized
    `reenqueued -> done` with `fresh_thesis_id` set. `spawn_fn` is injected so
    the scheduled top-5 run is never disturbed by this single-symbol path and so
    tests can drive it deterministically.

    Returns {"gate_passed": 0|1, "processed": n, "spawned": m}.
    """
    audit = audit_reclaims(
        eval_date=eval_date,
        horizon_days=audit_horizon_days,
        min_events=audit_min_events,
    )
    gate_passed = bool(auto_thesis and audit.passes)
    result = {"gate_passed": int(gate_passed), "processed": 0, "spawned": 0}
    if not gate_passed or spawn_fn is None:
        return result

    with get_session() as session:
        queued = session.execute(
            select(PaperReclaimQueue).where(PaperReclaimQueue.status == "queued")
        ).scalars().all()
        items = [(q.id, q.symbol) for q in queued]

    for queue_id, symbol in items:
        # Claim the row (queued -> reenqueued) race-safely; only proceed if we
        # won the transition.
        with get_session() as session:
            res = session.execute(
                update(PaperReclaimQueue)
                .where(
                    PaperReclaimQueue.id == queue_id,
                    PaperReclaimQueue.status == "queued",
                )
                .values(status="reenqueued", reenqueued_at=func.now())
            )
            if int(res.rowcount or 0) == 0:
                continue
        result["processed"] += 1

        try:
            fresh_thesis_id = spawn_fn(symbol, queue_id)
        except Exception as exc:  # spawn must never poison the rest of the queue
            log.error("reclaim_spawn_failed symbol=%s error=%s", symbol, exc)
            continue
        if fresh_thesis_id:
            with get_session() as session:
                session.execute(
                    update(PaperReclaimQueue)
                    .where(PaperReclaimQueue.id == queue_id)
                    .values(status="done", fresh_thesis_id=int(fresh_thesis_id))
                )
            result["spawned"] += 1

    return result
