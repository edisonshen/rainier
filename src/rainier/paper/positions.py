"""Thesis-time paper-position creation, fill, and exit-application.

Creation (design §5(1), D1/D3/D4/D7/D10):
  * Only `setup_long` + `llm_confidence >= CONF_GATE` + session in
    {afternoon, close} is a *buy signal* eligible to become a position. A
    morning pick or a sub-gate confidence is NOT acted on AND NOT a skip (it's
    excluded from the acted-vs-skipped denominator, D11).
  * Same symbol with multiple eligible theses in one day → open the highest
    `llm_confidence`; tie → earliest session (afternoon before close). The
    losers are recorded `paper_skip(same_symbol_lower_conviction)` (D1).
  * Levels are read from the **persisted `screened_stocks` row** (Tier-1 cache
    replay safety, D4) — never the in-memory candidate. Missing screened row →
    skip `missing_screened_record`; any required level NULL → skip
    `missing_levels` (and zero `paper_trade` rows — the guard runs BEFORE the
    insert so a later C2 backfill + re-run CAN create the position, D8).
  * The insert is `ON CONFLICT(thesis_id) DO NOTHING` (idempotent per thesis)
    AND each insert runs in its **own `get_session()` scope** so a
    partial-unique conflict (symbol already active) rolls back cleanly and the
    rest of the batch still processes (D7c). The active-symbol pre-check is a
    fast path; the DB partial-unique index is the race-safe guard.
  * `time_stop_days` snapshots `learned_time_stop_days` config at fill time
    (NULL until Phase 2 learns one) — set at FILL, not creation (D6).
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from rainier.core.database import get_session
from rainier.core.models import PaperSkip, PaperTrade, ScreenedStockRecord, StockPrice

from .calendar import DEFAULT_CALENDAR, TradingCalendar
from .exit import evaluate_exit

log = logging.getLogger(__name__)

CONF_GATE = 6
ACTIONABLE_SESSIONS = ("afternoon", "close")
# Earlier session wins ties — afternoon before close (design D1).
_SESSION_ORDER = {"afternoon": 0, "close": 1}

_REQUIRED_LEVELS = ("entry_price", "stop_loss", "target_price")

# Fixed $10k notional per position (design D2).
NOTIONAL_USD = 10_000.0
# A pending position with no fill after this many trading sessions → expired (D3).
PENDING_EXPIRY_SESSIONS = 2
# Adjustment basis the fill is booked under (design D5 split-freeze). yfinance
# auto_adjust=True gives a split-adjusted series, so the booked basis is
# "adjusted"; evaluate_exit walks the same-basis series for an open position.
PRICE_BASIS = "adjusted"


def _as_date(d: Any) -> date:
    if isinstance(d, datetime):
        return d.date()
    return d


def _is_buy_signal(verdict: str | None, llm_confidence: int | None, session: str) -> bool:
    """A `setup_long` passing the confidence + session gate. Failing this is
    'not a buy signal' (NOT a skip)."""
    return (
        verdict == "setup_long"
        and session in ACTIONABLE_SESSIONS
        and llm_confidence is not None
        and int(llm_confidence) >= CONF_GATE
    )


def _record_skip(thesis_id: int, symbol: str, scan_date: date, reason: str) -> None:
    """Idempotent skip-ledger write (own txn, ON CONFLICT(thesis_id) DO NOTHING)."""
    with get_session() as session:
        stmt = (
            pg_insert(PaperSkip)
            .values(
                thesis_id=thesis_id, symbol=symbol, scan_date=scan_date, reason=reason
            )
            .on_conflict_do_nothing(constraint="uq_paper_skip_thesis_id")
        )
        session.execute(stmt)


def _persisted_levels(
    session, symbol: str, scan_date: date, session_name: str
) -> ScreenedStockRecord | None:
    return session.execute(
        select(ScreenedStockRecord).where(
            ScreenedStockRecord.symbol == symbol,
            ScreenedStockRecord.scan_date == scan_date,
            ScreenedStockRecord.session_name == session_name,
        )
    ).scalars().first()


def _active_position(session, symbol: str) -> PaperTrade | None:
    """The current pending/open position holding ``symbol``'s active slot, if
    any (the partial-unique index guarantees at most one)."""
    return session.execute(
        select(PaperTrade).where(
            PaperTrade.symbol == symbol,
            PaperTrade.status.in_(("pending", "open")),
        )
    ).scalars().first()


def create_positions_for_theses(
    theses: list[dict[str, Any]],
    *,
    scan_date: date,
    learned_time_stop_days: int | None = None,
) -> dict[str, int]:
    """Create pending paper positions for the eligible theses in one scan day.

    Each ``thesis`` dict carries: ``thesis_id``, ``symbol``, ``verdict``,
    ``llm_confidence``, ``session_name``. ``learned_time_stop_days`` is NOT used
    at creation (fields stay NULL until fill, D6) but is accepted so the caller
    can pass config through uniformly.

    Returns counts ``{"created": n, "skipped": m}``.
    """
    # 1. Filter to buy signals (gate). Non-buy-signals are silently excluded.
    eligible = [
        t
        for t in theses
        if _is_buy_signal(
            t.get("verdict"), t.get("llm_confidence"), t.get("session_name", "")
        )
    ]
    if not eligible:
        return {"created": 0, "skipped": 0}

    # 2. Same-symbol/day dedupe (deterministic regardless of input order):
    #    highest llm_confidence wins; tie → earliest session.
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for t in eligible:
        by_symbol.setdefault(t["symbol"], []).append(t)

    winners: list[dict[str, Any]] = []
    for symbol, group in by_symbol.items():
        group_sorted = sorted(
            group,
            key=lambda t: (
                -int(t["llm_confidence"]),
                _SESSION_ORDER.get(t.get("session_name", ""), 99),
                int(t["thesis_id"]),  # final stable tiebreak
            ),
        )
        winners.append(group_sorted[0])
        for loser in group_sorted[1:]:
            _record_skip(
                int(loser["thesis_id"]), symbol, scan_date,
                "same_symbol_lower_conviction",
            )

    created = 0
    skipped = len(eligible) - len(winners)  # the lower-conviction losers

    # 3. Per-winner: own get_session() scope (clean txn on conflict).
    for t in winners:
        if _create_one(t, scan_date):
            created += 1
        else:
            skipped += 1

    return {"created": created, "skipped": skipped}


def _create_one(thesis: dict[str, Any], scan_date: date) -> bool:
    """Create one position. Returns True iff a row was inserted; otherwise a
    skip-ledger row is written (or the thesis was already opened — idempotent
    no-op). Runs entirely in its own session scope."""
    thesis_id = int(thesis["thesis_id"])
    symbol = thesis["symbol"]
    session_name = thesis["session_name"]
    new_conf = int(thesis["llm_confidence"])

    # Read the durable screened row for levels (D4). missing row → skip.
    with get_session() as session:
        screened = _persisted_levels(session, symbol, scan_date, session_name)
        if screened is None:
            screened_id = None
            levels = None
        else:
            screened_id = screened.id
            levels = {
                "entry_price": screened.entry_price,
                "stop_loss": screened.stop_loss,
                "target_price": screened.target_price,
                "rr_ratio": screened.rr_ratio,
            }

    if levels is None:
        _record_skip(thesis_id, symbol, scan_date, "missing_screened_record")
        return False

    # Any required level NULL → skip missing_levels (guard BEFORE insert so a
    # later C2 backfill + re-run can still create the position, D8).
    if any(levels.get(k) is None for k in _REQUIRED_LEVELS):
        _record_skip(thesis_id, symbol, scan_date, "missing_levels")
        return False

    # Active-symbol resolution (D7 + D10 cross-session). `create_positions_for_
    # theses` runs once PER SESSION in production (afternoon, then close), so the
    # same-day tiebreak must also hold ACROSS sessions, not just within one call:
    #   * existing active row is OPEN, or PENDING from a DIFFERENT scan_date →
    #     never displace; skip the new pick `symbol_already_active` (D7).
    #   * existing active row is PENDING from the SAME scan_date with HIGHER-OR-
    #     EQUAL confidence → the existing pick wins (tie → earlier session it was
    #     created in); skip the new pick `same_symbol_lower_conviction` (D10).
    #   * existing PENDING same-day with LOWER confidence → the new higher-conf
    #     close pick DISPLACES it (expire the old pending, record the old thesis
    #     `same_symbol_lower_conviction`), then fall through to insert the new.
    with get_session() as session:
        active = _active_position(session, symbol)
        if active is not None:
            # Idempotent re-run: the active row IS this thesis's own position
            # (D5). No skip, no displacement — just a no-op.
            if int(active.thesis_id) == thesis_id:
                return False
            # D10 cross-session tiebreak applies ONLY between DIFFERENT sessions
            # of the SAME scan day with a pending (not-yet-filled) incumbent. A
            # same-session re-pick, an open incumbent, or a prior-day pending is
            # a genuine already-active symbol → D7 skip.
            cross_session_pending = (
                active.status == "pending"
                and _as_date(active.scan_date) == scan_date
                and active.session_name != session_name
            )
            existing_conf = active.llm_confidence
            if not cross_session_pending:
                _record_skip(thesis_id, symbol, scan_date, "symbol_already_active")
                return False
            if existing_conf is not None and existing_conf >= new_conf:
                # Incumbent out-convicts (or ties → earliest session, i.e. the
                # already-pending one) → the new pick loses (D10).
                _record_skip(
                    thesis_id, symbol, scan_date, "same_symbol_lower_conviction"
                )
                return False
            # New pick out-convicts the same-day pending → displace it (only if
            # still pending — a concurrent fill may have opened it meanwhile).
            displaced_thesis = int(active.thesis_id)
            session.execute(
                _update_position(
                    active.id, expected_status="pending", status="expired"
                )
            )
        else:
            displaced_thesis = None

    if displaced_thesis is not None:
        _record_skip(
            displaced_thesis, symbol, scan_date, "same_symbol_lower_conviction"
        )

    values = {
        "thesis_id": thesis_id,
        "screened_record_id": screened_id,
        "symbol": symbol,
        "scan_date": scan_date,
        "session_name": session_name,
        "status": "pending",
        "planned_entry_price": float(levels["entry_price"]),
        "stop_loss": float(levels["stop_loss"]),
        "target_price": float(levels["target_price"]),
        "rr_ratio": (
            float(levels["rr_ratio"]) if levels["rr_ratio"] is not None else None
        ),
        "pattern_type": getattr(screened, "pattern_type", None),
        "llm_confidence": int(thesis["llm_confidence"]),
        "verdict": thesis.get("verdict"),
        # fill fields stay NULL while pending (D1).
    }

    try:
        with get_session() as session:
            stmt = (
                pg_insert(PaperTrade)
                .values(**values)
                .on_conflict_do_nothing(constraint="uq_paper_trade_thesis_id")
            )
            result = session.execute(stmt)
        # rowcount 0 → thesis already had a position (idempotent re-run, D5).
        return int(result.rowcount or 0) > 0
    except IntegrityError:
        # Partial-unique conflict (symbol already active) raced past the
        # pre-check. The txn rolled back cleanly (its own scope); record the
        # skip and let the batch continue (D7c).
        _record_skip(thesis_id, symbol, scan_date, "symbol_already_active")
        return False


# ---------------------------------------------------------------------------
# Fill (design §5(1), D3) — pending → open at T+1 trading-session open.
# ---------------------------------------------------------------------------


def _price_on(session, symbol: str, d: date) -> StockPrice | None:
    from rainier.paper.ingest import canonical_instant

    return session.execute(
        select(StockPrice).where(
            StockPrice.symbol == symbol,
            StockPrice.date == canonical_instant(d),
        )
    ).scalars().first()


def fill_pending_positions(
    *,
    as_of: date,
    calendar: TradingCalendar | None = None,
    learned_time_stop_days: int | None = None,
) -> dict[str, int]:
    """Fill each pending position at its T+1 trading-session open.

    For a pending created on `scan_date`, the fill day is the next trading
    session after scan_date (weekend → Monday, E1b). When that session's open
    price exists:
      * **fill-validity guard (E1c, D3):** if the open has gapped past a level
        (`open >= target` or `open <= stop`), the setup is gone → status=expired
        + paper_skip(gap_invalidated). NOT left pending (would hold the slot).
      * otherwise → status=open with sizing `shares = floor(10000/open)` +
        residual cash; `price_basis` recorded; **`time_stop_days` snapshotted
        from `learned_time_stop_days` at FILL time** (NULL until Phase 2 learns
        one — D6/E5). The snapshot is per-fill: an already-open position keeps
        the value it was filled under even if the config later changes (future-
        fills-only invariant, never retroactive).
    A pending with no open after PENDING_EXPIRY_SESSIONS trading sessions →
    status=expired (E3); an expired position never resurrects (E6).
    """
    cal = calendar or DEFAULT_CALENDAR
    filled = expired = 0

    with get_session() as session:
        pending = session.execute(
            select(PaperTrade).where(PaperTrade.status == "pending")
        ).scalars().all()
        # Snapshot the work items (id + immutable plan) so each mutation runs in
        # its own scope below.
        items = [
            {
                "id": p.id, "thesis_id": p.thesis_id, "symbol": p.symbol,
                "scan_date": _as_date(p.scan_date), "stop_loss": p.stop_loss,
                "target_price": p.target_price,
            }
            for p in pending
        ]

    for it in items:
        t1 = cal.next_session(it["scan_date"])
        if t1 > as_of:
            continue  # T+1 not reached yet — stay pending (no look-ahead).

        # Fill at the FIRST priced trading session in the window [T+1 ..
        # expiry-deadline], capped at as_of (no look-ahead). A missing T+1 open
        # must not skip a later priced session before the window closes (E3:
        # "T+2 open appears → fills at T+2"). The window spans
        # PENDING_EXPIRY_SESSIONS sessions starting at T+1; once the last
        # in-window session has passed with no open, the pending expires.
        deadline = cal.add_sessions(t1, PENDING_EXPIRY_SESSIONS - 1)
        candidate_sessions = [
            s for s in cal.sessions_between(t1, deadline) if s <= as_of
        ]

        with get_session() as session:
            fill_day = None
            bar = None
            for cand in candidate_sessions:
                cand_bar = _price_on(session, it["symbol"], cand)
                if cand_bar is not None and cand_bar.open is not None:
                    fill_day, bar = cand, cand_bar
                    break
            open_px = bar.open if bar is not None else None

            if open_px is None:
                # No open in any in-window session reached so far. Expire only
                # once the whole window has elapsed (as_of past the deadline);
                # otherwise stay pending — a later session may still price (E3).
                if as_of >= cal.next_session(deadline):
                    if _expire_locked(session, it["id"]):
                        expired += 1
                continue

            # Fill-validity guard (E1c): open gapped past a level → expired.
            if open_px >= it["target_price"] or open_px <= it["stop_loss"]:
                if _expire_locked(session, it["id"]):
                    _record_skip(
                        it["thesis_id"], it["symbol"], it["scan_date"],
                        "gap_invalidated",
                    )
                    expired += 1
                continue

            shares = int(math.floor(NOTIONAL_USD / open_px))
            allocated = shares * open_px
            residual = NOTIONAL_USD - allocated
            res = session.execute(
                _update_position(
                    it["id"],
                    expected_status="pending",  # concurrency: only fill if still pending
                    status="open",
                    entry_date=fill_day,
                    entry_price=open_px,
                    shares=shares,
                    allocated_amount=allocated,
                    residual_cash=residual,
                    price_basis=PRICE_BASIS,
                    # Snapshot the learned time-stop at FILL time (D6/E5). NULL
                    # in Phase 0+1. Per-fill — never re-read after the position
                    # is open, so a later config change can't retroactively
                    # time-stop an already-open position.
                    time_stop_days=learned_time_stop_days,
                )
            )
            if int(res.rowcount or 0) > 0:
                filled += 1

    return {"filled": filled, "expired": expired}


def _expire_locked(session, position_id: int) -> bool:
    # Only a still-pending row may expire (a concurrent run may have filled it).
    # Returns True iff a row actually transitioned (rowcount), so counters don't
    # over-count a no-op stale expire.
    res = session.execute(
        _update_position(position_id, expected_status="pending", status="expired")
    )
    return int(res.rowcount or 0) > 0


def _update_position(position_id: int, *, expected_status: str | None = None, **values: Any):
    from sqlalchemy import update as sql_update

    # Guard: never mutate a row that is no longer in the expected lifecycle state
    # (immutability of booked fills/exits, E4/E6/F13, + concurrency safety for
    # overlapping daily/manual runs — codex). The caller checks rowcount: 0 means
    # another run already transitioned the row, so the stale write is a no-op.
    stmt = sql_update(PaperTrade).where(PaperTrade.id == position_id)
    if expected_status is not None:
        stmt = stmt.where(PaperTrade.status == expected_status)
    return stmt.values(**values)


# ---------------------------------------------------------------------------
# Exit application (design §5(2)) — open → closed via the pure evaluate_exit.
# ---------------------------------------------------------------------------


def update_open_positions(*, as_of: date) -> dict[str, int]:
    """Close any open position whose path triggered an exit at/through as_of.

    Reads each open position's daily OHLC from entry_date..as_of, runs the pure
    `evaluate_exit`, and on a trigger writes status=closed + exit_* + return_pct
    + pnl. Idempotent: only `status='open'` rows are considered, so a re-run
    never double-closes (F13); booked exit fields are written once and the
    closed row is never revisited.
    """
    from rainier.paper.ingest import canonical_instant

    closed = 0
    same_bar = 0

    with get_session() as session:
        open_positions = session.execute(
            select(PaperTrade).where(PaperTrade.status == "open")
        ).scalars().all()
        items = [
            {
                "id": p.id, "symbol": p.symbol,
                "entry_date": _as_date(p.entry_date),
                "entry_price": p.entry_price, "stop_loss": p.stop_loss,
                "target_price": p.target_price, "shares": p.shares,
                "time_stop_days": p.time_stop_days,
            }
            for p in open_positions
            if p.entry_date is not None and p.entry_price is not None
        ]

    for it in items:
        with get_session() as session:
            rows = session.execute(
                select(StockPrice).where(
                    StockPrice.symbol == it["symbol"],
                    StockPrice.date >= canonical_instant(it["entry_date"]),
                    StockPrice.date <= canonical_instant(as_of),
                )
            ).scalars().all()
            price_rows = [
                {
                    "date": _as_date(r.date), "open": r.open, "high": r.high,
                    "low": r.low, "close": r.close,
                }
                for r in rows
            ]
            result = evaluate_exit(
                entry_date=it["entry_date"],
                entry_price=it["entry_price"],
                stop_loss=it["stop_loss"],
                target_price=it["target_price"],
                shares=it["shares"],
                price_rows=price_rows,
                as_of=as_of,
                time_stop_days=it["time_stop_days"],
            )
            if result is None:
                continue
            # Close only if still open (race-safe + idempotent — F13).
            res = session.execute(
                _update_position(
                    it["id"],
                    expected_status="open",
                    status="closed",
                    exit_date=result.exit_date,
                    exit_price=result.exit_price,
                    exit_reason=result.exit_reason,
                    return_pct=result.return_pct,
                    pnl=result.pnl,
                )
            )
            if int(res.rowcount or 0) > 0:
                closed += 1
                if result.same_bar_ambiguous:
                    same_bar += 1

    return {"closed": closed, "same_bar_ambiguous_exits": same_bar}
