"""Selection reward ledger — daily step (vii) of the QU100-LLM feedback loop (R1).

Design: docs/DESIGN-qu100-selection-reward-loop.md §3.1–3.2.

For every thesis (``analysis_results`` ⋈ ``screened_stocks``) scanned before
``as_of`` this module:

1. **classifies the decision** (the honest denominator)::

       setup_long_filled    live paper_trade open/closed
       setup_long_pending   live paper_trade still pending (not yet scored)
       setup_long_expired   pending never filled (no price data)
       gap_invalidated      fill-day open gapped past a level (paper_skip)
       setup_long_skipped   buy signal, but the position engine skipped it
       setup_long_gated     setup_long verdict below the confidence gate or in
                            a non-actionable session
       watch / no_setup     LLM declined

2. **scores it** with the R-multiple of the committed plan:

   * filled + closed → realized R (``provisional=false``)
   * filled + open   → mark-to-market R off the latest close (``provisional``)
   * everything declined with a valid LONG plan → **counterfactual** R: the
     plan is walked from the next session's open through the same pure
     ``evaluate_exit`` the live book uses, capped at
     ``counterfactual_horizon_days`` sessions. Unresolved → provisional MTM.
   * no / invalid plan, missing prices, basis mismatch → ``value=NULL`` +
     ``reason`` so the decision still counts.

3. **upserts** one ledger row per (thesis, reward_name) for ``r_multiple`` and
   ``pnl_usd``. Provisional rows are rewritten in place; matured rows are
   final unless ``reward_version`` changes.

No look-ahead: only price bars ``<= as_of`` are read, and a decision whose
counterfactual entry session is after ``as_of`` is not scored yet.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rainier.core.database import get_session
from rainier.core.models import (
    LLMAnalysisRecord,
    PaperSkip,
    PaperTrade,
    ScreenedStockRecord,
    SelectionReward,
    StockPrice,
)
from rainier.research.rewards.selection import (
    DEFAULT_COUNTERFACTUAL_HORIZON_DAYS,
    DEFAULT_DRAWDOWN_LAMBDA,
    REWARD_VERSION,
    r_multiple,
    sleeve_pnl,
    summarize,
)

from .calendar import DEFAULT_CALENDAR, TradingCalendar
from .exit import evaluate_exit
from .positions import (
    ACTIONABLE_SESSIONS,
    CONF_GATE,
    NOTIONAL_USD,
    PRICE_BASIS,
    is_long_shape,
)

log = logging.getLogger(__name__)

REWARD_R = "r_multiple"
REWARD_PNL = "pnl_usd"
REWARD_NAMES: tuple[str, ...] = (REWARD_R, REWARD_PNL)

#: Calendar days after the fill session within which a counterfactual entry
#: bar must exist (mirrors the live book's 2-session pending expiry).
_ENTRY_SEARCH_DAYS = 5


@dataclass(frozen=True)
class Decision:
    """One thesis + the state the loop left it in (read side of the ledger)."""

    thesis_id: int
    symbol: str
    scan_date: date
    session_name: str
    verdict: str
    llm_confidence: int | None
    prompt_version: str | None
    model: str | None
    signals_used: tuple[str, ...]
    pattern_type: str | None
    entry_price: float | None
    stop_loss: float | None
    target_price: float | None
    trade: dict[str, Any] | None = None  # live paper_trade (shadow=false) or None
    skip_reason: str | None = None  # latest paper_skip.reason or None


@dataclass(frozen=True)
class RewardRow:
    thesis_id: int
    symbol: str
    scan_date: date
    session_name: str
    decision: str
    lever_context: dict[str, Any]
    reward_name: str
    value: float | None
    reason: str | None
    provisional: bool
    counterfactual: bool
    outcome_date: date | None
    as_of_date: date
    reward_version: int = REWARD_VERSION

    def as_values(self) -> dict[str, Any]:
        return {
            "thesis_id": self.thesis_id,
            "symbol": self.symbol,
            "scan_date": self.scan_date,
            "session_name": self.session_name,
            "decision": self.decision,
            "lever_context": self.lever_context,
            "reward_name": self.reward_name,
            "reward_version": self.reward_version,
            "value": self.value,
            "reason": self.reason,
            "provisional": self.provisional,
            "counterfactual": self.counterfactual,
            "outcome_date": self.outcome_date,
            "as_of_date": self.as_of_date,
        }


@dataclass
class _Scored:
    """Internal: outcome of scoring one decision before fan-out to rows."""

    decision: str
    r: float | None = None
    pnl: float | None = None
    reason: str | None = None
    provisional: bool = False
    counterfactual: bool = False
    outcome_date: date | None = None
    skip: bool = False  # nothing to write yet (pending / not enterable)
    extra_context: dict[str, Any] = field(default_factory=dict)


def _as_date(d: Any) -> date:
    return d.date() if isinstance(d, datetime) else d


# ---------------------------------------------------------------------------
# Classification + lever context
# ---------------------------------------------------------------------------


def is_buy_signal(
    verdict: str, llm_confidence: int | None, session_name: str, *, conf_gate: int = CONF_GATE
) -> bool:
    return (
        verdict == "setup_long"
        and llm_confidence is not None
        and llm_confidence >= conf_gate
        and session_name in ACTIONABLE_SESSIONS
    )


def classify_decision(d: Decision, *, conf_gate: int = CONF_GATE) -> str:
    """Map a thesis + loop state to the ledger's decision class."""
    trade = d.trade
    if trade is not None:
        status = trade["status"]
        if status in ("open", "closed"):
            return "setup_long_filled"
        if status == "pending":
            return "setup_long_pending"
        if d.skip_reason == "gap_invalidated":
            return "gap_invalidated"
        return "setup_long_expired"
    if d.verdict == "setup_long":
        if is_buy_signal(d.verdict, d.llm_confidence, d.session_name, conf_gate=conf_gate):
            return "setup_long_skipped"
        return "setup_long_gated"
    if d.verdict == "watch":
        return "watch"
    return "no_setup"


def signal_set_hash(signals: tuple[str, ...] | list[str]) -> str | None:
    if not signals:
        return None
    joined = ",".join(sorted(signals))
    return hashlib.sha1(joined.encode()).hexdigest()[:12]


def lever_context(d: Decision, *, conf_gate: int = CONF_GATE) -> dict[str, Any]:
    """Lever values live at decision time — snapshotted so later group-bys
    (by prompt_version, confidence bucket, pattern, …) are plain SQL."""
    trade = d.trade or {}
    return {
        "prompt_version": d.prompt_version,
        "model": d.model,
        "llm_confidence": d.llm_confidence,
        "confidence_gate": conf_gate,
        "session": d.session_name,
        "pattern_type": d.pattern_type,
        "signal_set_hash": signal_set_hash(d.signals_used),
        "time_stop_days": trade.get("time_stop_days"),
        "price_basis": trade.get("price_basis"),
    }


# ---------------------------------------------------------------------------
# Scoring (pure over price rows)
# ---------------------------------------------------------------------------


def _bars_sorted(price_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(price_rows, key=lambda r: _as_date(r["date"]))


def _latest_close(price_rows: list[dict[str, Any]], *, start: date, as_of: date):
    """(date, close) of the last bar in [start, as_of] with a close, else None."""
    for r in reversed(_bars_sorted(price_rows)):
        d = _as_date(r["date"])
        if d > as_of:
            continue
        if d < start:
            break
        if r.get("close") is not None:
            return d, float(r["close"])
    return None


def _score_filled(d: Decision, *, as_of: date, price_rows: list[dict[str, Any]]) -> _Scored:
    t = d.trade or {}
    out = _Scored(decision="setup_long_filled")
    entry = t.get("entry_price")
    stop = t.get("stop_loss")
    if entry is None or stop is None or t.get("entry_date") is None:
        out.skip = True
        return out
    if t["status"] == "closed":
        exit_px = t.get("exit_price")
        if exit_px is None:
            out.skip = True
            return out
        out.r = r_multiple(float(entry), float(stop), float(exit_px))
        out.pnl = t.get("pnl")
        if out.pnl is None and t.get("shares"):
            out.pnl = sleeve_pnl(float(entry), float(exit_px), int(t["shares"]))
        out.outcome_date = _as_date(t["exit_date"]) if t.get("exit_date") else None
        if out.r is None:
            out.reason = "invalid_levels"
            out.pnl = None
        return out
    # open → mark-to-market
    out.provisional = True
    if t.get("price_basis") not in (None, PRICE_BASIS):
        out.reason = "basis_mismatch"
        return out
    latest = _latest_close(price_rows, start=_as_date(t["entry_date"]), as_of=as_of)
    if latest is None:
        out.reason = "missing_prices"
        return out
    _, close = latest
    out.r = r_multiple(float(entry), float(stop), close)
    if out.r is None:
        out.reason = "invalid_levels"
        return out
    out.pnl = sleeve_pnl(float(entry), close, int(t.get("shares") or 0))
    return out


def _score_counterfactual(
    d: Decision,
    decision: str,
    *,
    as_of: date,
    price_rows: list[dict[str, Any]],
    calendar: TradingCalendar,
    horizon_days: int | None,
) -> _Scored:
    out = _Scored(decision=decision, counterfactual=True)
    levels = {
        "entry_price": d.entry_price,
        "stop_loss": d.stop_loss,
        "target_price": d.target_price,
    }
    if any(v is None for v in levels.values()):
        out.reason = "no_plan"
        return out
    if not is_long_shape(levels, d.pattern_type):
        out.reason = "invalid_levels"
        return out
    stop = float(d.stop_loss)  # type: ignore[arg-type]
    target = float(d.target_price)  # type: ignore[arg-type]

    fill_day = calendar.next_session(d.scan_date)
    if fill_day > as_of:
        out.skip = True  # not enterable yet — nothing to score
        return out
    entry_bar = None
    for r in _bars_sorted(price_rows):
        bd = _as_date(r["date"])
        if bd < fill_day or bd > as_of:
            continue
        if bd > fill_day + timedelta(days=_ENTRY_SEARCH_DAYS):
            break
        if r.get("open") is not None:
            entry_bar = r
            break
    if entry_bar is None:
        # Bars may still arrive (ingest lag) → provisional so a re-run refreshes.
        out.reason = "missing_prices"
        out.provisional = True
        return out
    entry_date = _as_date(entry_bar["date"])
    entry_px = float(entry_bar["open"])
    if entry_px >= target or entry_px <= stop:
        out.reason = "gap_invalidated"
        return out
    shares = int(math.floor(NOTIONAL_USD / entry_px)) if entry_px > 0 else 0
    if shares <= 0:
        out.reason = "invalid_levels"
        return out
    out.extra_context = {"cf_entry_date": entry_date.isoformat(), "cf_entry_price": entry_px}

    result = evaluate_exit(
        entry_date=entry_date,
        entry_price=entry_px,
        stop_loss=stop,
        target_price=target,
        shares=shares,
        price_rows=price_rows,
        as_of=as_of,
        time_stop_days=horizon_days,
    )
    if result is not None:
        out.r = r_multiple(entry_px, stop, result.exit_price)
        out.pnl = result.pnl
        out.outcome_date = result.exit_date
        out.extra_context["cf_exit_reason"] = result.exit_reason
        return out
    out.provisional = True
    latest = _latest_close(price_rows, start=entry_date, as_of=as_of)
    if latest is None:
        out.reason = "missing_prices"
        return out
    _, close = latest
    out.r = r_multiple(entry_px, stop, close)
    out.pnl = sleeve_pnl(entry_px, close, shares)
    return out


def score_decision(
    d: Decision,
    *,
    as_of: date,
    price_rows: list[dict[str, Any]],
    calendar: TradingCalendar = DEFAULT_CALENDAR,
    counterfactual_horizon_days: int | None = DEFAULT_COUNTERFACTUAL_HORIZON_DAYS,
    conf_gate: int = CONF_GATE,
) -> list[RewardRow]:
    """Pure: classify + score one decision → ledger rows (possibly empty)."""
    decision = classify_decision(d, conf_gate=conf_gate)
    if decision == "setup_long_filled":
        scored = _score_filled(d, as_of=as_of, price_rows=price_rows)
    elif decision == "setup_long_pending":
        scored = _Scored(decision=decision, skip=True)
    elif decision == "gap_invalidated":
        scored = _Scored(decision=decision, reason="gap_invalidated")
    elif decision == "setup_long_expired":
        scored = _Scored(decision=decision, reason="missing_prices")
    else:
        scored = _score_counterfactual(
            d,
            decision,
            as_of=as_of,
            price_rows=price_rows,
            calendar=calendar,
            horizon_days=counterfactual_horizon_days,
        )
    if scored.skip:
        return []

    ctx = lever_context(d, conf_gate=conf_gate)
    ctx.update(scored.extra_context)
    common = {
        "thesis_id": d.thesis_id,
        "symbol": d.symbol,
        "scan_date": d.scan_date,
        "session_name": d.session_name,
        "decision": scored.decision,
        "lever_context": ctx,
        "reason": scored.reason,
        "provisional": scored.provisional,
        "counterfactual": scored.counterfactual,
        "outcome_date": scored.outcome_date,
        "as_of_date": as_of,
    }
    rows = [RewardRow(reward_name=REWARD_R, value=scored.r, **common)]
    if scored.r is not None:
        pnl = float(scored.pnl) if scored.pnl is not None else None
        rows.append(
            RewardRow(
                reward_name=REWARD_PNL,
                value=pnl,
                **{**common, "reason": None if pnl is not None else "missing_prices"},
            )
        )
    return rows


# ---------------------------------------------------------------------------
# DB read side
# ---------------------------------------------------------------------------


def load_decisions(*, as_of: date) -> list[Decision]:
    """Every thesis scanned strictly before ``as_of`` with its loop state."""
    with get_session() as session:
        rows = session.execute(
            select(
                LLMAnalysisRecord.id,
                LLMAnalysisRecord.recommendation,
                LLMAnalysisRecord.confidence,
                LLMAnalysisRecord.prompt_template,
                LLMAnalysisRecord.llm_model,
                LLMAnalysisRecord.signals_used,
                ScreenedStockRecord.symbol,
                ScreenedStockRecord.scan_date,
                ScreenedStockRecord.session_name,
                ScreenedStockRecord.pattern_type,
                ScreenedStockRecord.entry_price,
                ScreenedStockRecord.stop_loss,
                ScreenedStockRecord.target_price,
            )
            .join(ScreenedStockRecord, ScreenedStockRecord.thesis_id == LLMAnalysisRecord.id)
            .where(ScreenedStockRecord.scan_date < as_of)
            .order_by(ScreenedStockRecord.scan_date, LLMAnalysisRecord.id)
        ).all()
        thesis_ids = [int(r.id) for r in rows]
        trades: dict[int, dict[str, Any]] = {}
        skips: dict[int, str] = {}
        if thesis_ids:
            for t in session.execute(
                select(PaperTrade).where(
                    PaperTrade.thesis_id.in_(thesis_ids), PaperTrade.shadow.is_(False)
                )
            ).scalars():
                trades[int(t.thesis_id)] = {
                    "status": t.status,
                    "entry_date": t.entry_date,
                    "entry_price": t.entry_price,
                    "stop_loss": t.stop_loss,
                    "target_price": t.target_price,
                    "shares": t.shares,
                    "time_stop_days": t.time_stop_days,
                    "price_basis": t.price_basis,
                    "exit_date": t.exit_date,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                }
            for s in session.execute(
                select(PaperSkip)
                .where(PaperSkip.thesis_id.in_(thesis_ids))
                .order_by(PaperSkip.created_at)
            ).scalars():
                skips[int(s.thesis_id)] = s.reason

    out: list[Decision] = []
    for r in rows:
        tid = int(r.id)
        out.append(
            Decision(
                thesis_id=tid,
                symbol=r.symbol,
                scan_date=_as_date(r.scan_date),
                session_name=r.session_name,
                verdict=r.recommendation or "no_setup",
                llm_confidence=(
                    int(round(r.confidence)) if r.confidence is not None else None
                ),
                prompt_version=r.prompt_template,
                model=r.llm_model,
                signals_used=tuple(r.signals_used or ()),
                pattern_type=r.pattern_type,
                entry_price=r.entry_price,
                stop_loss=r.stop_loss,
                target_price=r.target_price,
                trade=trades.get(tid),
                skip_reason=skips.get(tid),
            )
        )
    return out


def load_price_rows(symbol: str, *, start: date, as_of: date) -> list[dict[str, Any]]:
    from rainier.paper.ingest import canonical_instant

    with get_session() as session:
        rows = session.execute(
            select(StockPrice).where(
                StockPrice.symbol == symbol,
                StockPrice.date >= canonical_instant(start),
                StockPrice.date <= canonical_instant(as_of),
            )
        ).scalars().all()
    return [
        {
            "date": _as_date(r.date),
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# DB write side
# ---------------------------------------------------------------------------


def upsert_rewards(rows: list[RewardRow]) -> int:
    """Idempotent upsert on (thesis_id, reward_name).

    A conflicting row is rewritten only while it is provisional or was computed
    under a different ``reward_version`` — matured rows are immutable.
    Returns the number of rows inserted or updated.
    """
    if not rows:
        return 0
    written = 0
    with get_session() as session:
        for row in rows:
            stmt = pg_insert(SelectionReward).values(**row.as_values())
            excluded = stmt.excluded
            stmt = stmt.on_conflict_do_update(
                constraint="uq_selection_reward_thesis_reward",
                set_={
                    "decision": excluded.decision,
                    "lever_context": excluded.lever_context,
                    "reward_version": excluded.reward_version,
                    "value": excluded.value,
                    "reason": excluded.reason,
                    "provisional": excluded.provisional,
                    "counterfactual": excluded.counterfactual,
                    "outcome_date": excluded.outcome_date,
                    "as_of_date": excluded.as_of_date,
                    "updated_at": datetime.now().astimezone(),
                },
                where=(
                    SelectionReward.provisional.is_(True)
                    | (SelectionReward.reward_version != excluded.reward_version)
                ),
            ).returning(SelectionReward.id)
            if session.execute(stmt).first() is not None:
                written += 1
    return written


def compute_rewards(
    *,
    as_of: date,
    calendar: TradingCalendar | None = None,
    counterfactual_horizon_days: int | None = DEFAULT_COUNTERFACTUAL_HORIZON_DAYS,
    conf_gate: int = CONF_GATE,
) -> dict[str, int]:
    """Daily step (vii): score every decision as of ``as_of`` and upsert the ledger."""
    cal = calendar or DEFAULT_CALENDAR
    decisions = load_decisions(as_of=as_of)
    counts: dict[str, int] = {
        "decisions": len(decisions),
        "rows_written": 0,
        "scored": 0,
        "unscored": 0,
        "skipped": 0,
        "provisional": 0,
        "counterfactual": 0,
    }
    by_decision: dict[str, int] = {}
    for d in decisions:
        price_rows = load_price_rows(d.symbol, start=d.scan_date, as_of=as_of)
        rows = score_decision(
            d,
            as_of=as_of,
            price_rows=price_rows,
            calendar=cal,
            counterfactual_horizon_days=counterfactual_horizon_days,
            conf_gate=conf_gate,
        )
        if not rows:
            counts["skipped"] += 1
            continue
        head = rows[0]
        by_decision[head.decision] = by_decision.get(head.decision, 0) + 1
        if head.value is None:
            counts["unscored"] += 1
        else:
            counts["scored"] += 1
        if head.provisional:
            counts["provisional"] += 1
        if head.counterfactual:
            counts["counterfactual"] += 1
        counts["rows_written"] += upsert_rewards(rows)
    for k, v in by_decision.items():
        counts[f"decision_{k}"] = v
    log.info("selection_rewards_computed as_of=%s %s", as_of, counts)
    return counts


# ---------------------------------------------------------------------------
# Read-back: matured cohort summary (the three axes)
# ---------------------------------------------------------------------------


def load_matured_rewards(*, as_of: date | None = None) -> list[dict[str, Any]]:
    """Matured (non-provisional, scored) R rows + their $P&L, in outcome order."""
    with get_session() as session:
        q = (
            select(SelectionReward)
            .where(
                SelectionReward.reward_name == REWARD_R,
                SelectionReward.provisional.is_(False),
                SelectionReward.value.is_not(None),
            )
            .order_by(
                SelectionReward.outcome_date, SelectionReward.scan_date, SelectionReward.id
            )
        )
        if as_of is not None:
            q = q.where(SelectionReward.outcome_date <= as_of)
        r_rows = session.execute(q).scalars().all()
        ids = [r.thesis_id for r in r_rows]
        pnl_by_thesis: dict[int, float] = {}
        if ids:
            for p in session.execute(
                select(SelectionReward).where(
                    SelectionReward.reward_name == REWARD_PNL,
                    SelectionReward.thesis_id.in_(ids),
                    SelectionReward.value.is_not(None),
                )
            ).scalars():
                pnl_by_thesis[int(p.thesis_id)] = float(p.value)
    return [
        {
            "thesis_id": int(r.thesis_id),
            "symbol": r.symbol,
            "decision": r.decision,
            "counterfactual": bool(r.counterfactual),
            "outcome_date": r.outcome_date,
            "r": float(r.value),
            "pnl": pnl_by_thesis.get(int(r.thesis_id)),
            "lever_context": dict(r.lever_context or {}),
        }
        for r in r_rows
    ]


def summarize_rewards(
    *, as_of: date | None = None, lam: float = DEFAULT_DRAWDOWN_LAMBDA
) -> dict[str, Any]:
    """Three-axis summary of the matured ledger: all decisions, live-only
    (what the book actually took) and counterfactual-only (what it declined)."""
    rows = load_matured_rewards(as_of=as_of)

    def _agg(subset: list[dict[str, Any]]) -> dict[str, Any]:
        rs = [x["r"] for x in subset]
        pnls = [x["pnl"] for x in subset if x["pnl"] is not None]
        return summarize(rs, pnls if len(pnls) == len(rs) else None, lam=lam)

    return {
        "as_of": as_of.isoformat() if as_of else None,
        "lambda": lam,
        "all": _agg(rows),
        "live": _agg([x for x in rows if not x["counterfactual"]]),
        "counterfactual": _agg([x for x in rows if x["counterfactual"]]),
    }
