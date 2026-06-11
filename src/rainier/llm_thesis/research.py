"""Auto-research loop — weekly job (Friday 09:00 PT) that converts
ThesisEvaluation data into ResearchInsight rows the operator can
accept or reject.

  ┌──────────────────────────┐
  │      run_research        │   (called by scheduler / CLI)
  └──────────┬───────────────┘
             │
             ▼
   ┌─────────────────────────┐    runs all 6 check classes:
   │  check_signal_under-    │      signal_underperform
   │  perform / overperform  │      signal_overperform
   │  verdict_drift          │      verdict_drift
   │  calibration_off        │      calibration_off
   │  new_pattern_discovered │      new_pattern_discovered
   │  prompt_regression      │      prompt_regression
   └──────────┬──────────────┘
              │ each emits via:
              ▼
   ┌─────────────────────────┐    UPSERT semantics (eng review D6):
   │      emit_insight       │      pending row with same (kind, subject)
   └──────────┬──────────────┘      → UPDATE evidence/rationale/recurrence
              │                     else → INSERT new pending row
              ▼
       ResearchInsight row

Accept dispatches via ACTION_EXECUTORS (eng review D3): `insight.action.kind`
maps to a YAML mutator; the executor mutates `config/settings.yaml` via
ruamel.yaml (preserves comments + key order) with atomic temp-file rename.

v1 is recommend-only — accept/reject is a human-in-the-loop CLI step. v2 may
promote certain (kind, severity) tuples to auto-apply on threshold breach.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import and_, select, update

from rainier.core.database import get_session
from rainier.core.models import (
    LLMAnalysisRecord,
    ResearchInsight,
    ThesisEvaluation,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Insight kind / severity / status enums (string literals — no DB enum type
# so a fresh kind can land without a migration)
# ---------------------------------------------------------------------------


INSIGHT_KINDS: tuple[str, ...] = (
    "signal_underperform",
    "signal_overperform",
    "verdict_drift",
    "calibration_off",
    "prompt_regression",
    "new_pattern_discovered",
    # Phase 2 (D6): the learned force-exit horizon discovered by
    # discover_time_stop. Recommend-only — action set_learned_time_stop_days.
    "time_stop_discovered",
    # Phase 2 (D7c): weekly human-readable lessons from the paper-trade record.
    # info/noop — feedable into the prompt later.
    "paper_lessons",
    # Phase 3 (D8 / §5(4)): weekly missed-winner sweep. One insight per as-of
    # ISO week (info/noop); the durable record is the weekly
    # paper_report_snapshot — this row is only the action-queue echo.
    "missed_winner",
)

SEVERITIES: tuple[str, ...] = ("info", "warn", "critical")

STATUSES: tuple[str, ...] = (
    "pending",
    "accepted",
    "rejected",
    "auto_applied",
    "stale",
)


# ---------------------------------------------------------------------------
# UPSERT helper — emit_insight
# ---------------------------------------------------------------------------


def emit_insight(
    *,
    kind: str,
    subject: str,
    severity: str,
    evidence: dict[str, Any],
    action: dict[str, Any],
    rationale: str,
    db_session=None,
) -> ResearchInsight:
    """Insert or refresh a ResearchInsight row.

    Semantics (eng review D6):
      * If a `pending` row already exists with the same (kind, subject):
        UPDATE its evidence + rationale + action + severity + updated_at
        and increment recurrence_count by 1. Status stays `pending` so the
        operator's accept/reject queue doesn't grow unbounded.
      * Otherwise INSERT a fresh `pending` row with recurrence_count=1.

    The Postgres partial unique index `idx_research_insight_pending_kind_subject`
    enforces "at most one pending row per (kind, subject)"; this helper
    implements the application-side UPSERT so SQLite-backed unit tests pass
    too.

    Returns the inserted/updated row (refreshed from DB).
    """
    if kind not in INSIGHT_KINDS:
        raise ValueError(f"Unknown insight kind: {kind!r}; valid={INSIGHT_KINDS}")
    if severity not in SEVERITIES:
        raise ValueError(f"Unknown severity: {severity!r}; valid={SEVERITIES}")
    if not isinstance(action, dict) or "kind" not in action:
        raise ValueError(
            "`action` must be a dict with at least a 'kind' field; "
            f"got {action!r}"
        )

    def _do(session) -> ResearchInsight:
        existing = (
            session.query(ResearchInsight)
            .filter(
                ResearchInsight.kind == kind,
                ResearchInsight.subject == subject,
                ResearchInsight.status == "pending",
            )
            .first()
        )
        if existing is not None:
            existing.evidence = evidence
            existing.action = action
            existing.severity = severity
            existing.rationale = rationale
            existing.recurrence_count = (existing.recurrence_count or 1) + 1
            existing.updated_at = datetime.now(timezone.utc)
            session.flush()
            return existing
        row = ResearchInsight(
            kind=kind,
            subject=subject,
            severity=severity,
            evidence=evidence,
            action=action,
            rationale=rationale,
            recurrence_count=1,
            status="pending",
        )
        session.add(row)
        session.flush()
        return row

    if db_session is not None:
        # Caller owns the session — they're responsible for commit/rollback.
        return _do(db_session)
    # We own the session — go through the contextmanager so a downstream
    # exception triggers rollback() + close() instead of leaking an open
    # transaction. Review iter-1 [P0]: the previous implementation called
    # `get_session().__enter__()` directly, which bypasses the @contextmanager
    # exception handling — a SQL error mid-flush would leave the connection
    # dirty.
    with get_session() as session:
        return _do(session)


# ---------------------------------------------------------------------------
# Stale sweeper
# ---------------------------------------------------------------------------


def mark_stale(days: int = 30, *, eval_date: date | None = None) -> int:
    """Move pending insights older than `days` to `status='stale'`.

    Prevents the recommend-only queue from growing unbounded when the
    operator hasn't touched it in weeks. Returns the count of rows updated.
    """
    anchor = eval_date if eval_date is not None else date.today()
    cutoff = datetime.combine(
        anchor - timedelta(days=days),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )

    with get_session() as session:
        result = session.execute(
            update(ResearchInsight)
            .where(
                ResearchInsight.status == "pending",
                ResearchInsight.created_at < cutoff,
            )
            .values(status="stale")
        )
        return int(result.rowcount or 0)


# ---------------------------------------------------------------------------
# Check classes — the falsifiable per-signal / per-verdict / per-prompt tests
# ---------------------------------------------------------------------------


def _fetch_eval_window(
    *, days: int, horizon: str, eval_date: date | None = None
) -> list[tuple[list[str], float, str, int | None, str]]:
    """Pull (signals_used, return_pct, verdict, llm_confidence, scan_date) tuples
    from ThesisEvaluation for the rolling window.

    Returns the raw rows — callers re-aggregate per check.
    """
    anchor = eval_date if eval_date is not None else date.today()
    cutoff = anchor - timedelta(days=days)
    with get_session() as session:
        rows = session.execute(
            select(
                ThesisEvaluation.signals_used,
                ThesisEvaluation.return_pct,
                ThesisEvaluation.verdict,
                ThesisEvaluation.llm_confidence,
                ThesisEvaluation.scan_date,
            ).where(
                ThesisEvaluation.horizon == horizon,
                ThesisEvaluation.scan_date >= cutoff,
            )
        ).all()
    return [
        (
            list(r[0] or []),
            float(r[1]),
            str(r[2] or ""),
            int(r[3]) if r[3] is not None else None,
            r[4].isoformat() if hasattr(r[4], "isoformat") else str(r[4]),
        )
        for r in rows
    ]


def _mannwhitney_p(used: list[float], absent: list[float]) -> float | None:
    if not used or not absent:
        return None
    try:
        from scipy.stats import mannwhitneyu

        return float(mannwhitneyu(used, absent, alternative="two-sided").pvalue)
    except Exception:
        log.warning("mannwhitneyu_failed", exc_info=True)
        return None


def check_signal_underperform(
    *,
    eval_date: date | None = None,
    days: int = 30,
    horizon: str = "5d",
    p_threshold: float = 0.05,
    lift_threshold: float = 0.001,
    db_session=None,
) -> list[ResearchInsight]:
    """Mann-Whitney U on signal-used vs absent forward returns.

    Emits `signal_underperform` (severity=warn, action=disable_signal) when
    a signal's p<p_threshold AND its lift is < lift_threshold (effectively
    flat or negative).
    """
    rows = _fetch_eval_window(days=days, horizon=horizon, eval_date=eval_date)
    if not rows:
        return []

    all_signals: set[str] = set()
    for sigs, *_ in rows:
        all_signals.update(sigs)

    out: list[ResearchInsight] = []
    for name in sorted(all_signals):
        used = [r[1] for r in rows if name in r[0]]
        absent = [r[1] for r in rows if name not in r[0]]
        if not used or not absent:
            continue
        mean_used = sum(used) / len(used)
        mean_absent = sum(absent) / len(absent)
        lift = mean_used - mean_absent
        p = _mannwhitney_p(used, absent)
        if p is None or p >= p_threshold:
            continue
        if lift >= lift_threshold:
            continue
        evidence = {
            "n_used": len(used),
            "n_absent": len(absent),
            "mean_used": mean_used,
            "mean_absent": mean_absent,
            "lift": lift,
            "p_value": p,
            "horizon": horizon,
            "days": days,
        }
        action = {"kind": "disable_signal", "target": name, "params": {}}
        rationale = (
            f"Signal {name!r} shows lift {lift:+.2%} (p={p:.3f}, n={len(used)} "
            f"used vs {len(absent)} absent) over rolling {days}d at {horizon}. "
            "Recommend disabling — no statistically significant value-add."
        )
        out.append(
            emit_insight(
                kind="signal_underperform",
                subject=name,
                severity="warn",
                evidence=evidence,
                action=action,
                rationale=rationale,
                db_session=db_session,
            )
        )
    return out


def check_signal_overperform(
    *,
    eval_date: date | None = None,
    days: int = 30,
    horizon: str = "5d",
    p_threshold: float = 0.05,
    lift_threshold: float = 0.005,
    db_session=None,
) -> list[ResearchInsight]:
    """Symmetric to underperform — emit `signal_overperform` (info) when a
    signal's p<p_threshold AND lift > lift_threshold (a clear positive
    contribution).
    """
    rows = _fetch_eval_window(days=days, horizon=horizon, eval_date=eval_date)
    if not rows:
        return []

    all_signals: set[str] = set()
    for sigs, *_ in rows:
        all_signals.update(sigs)

    out: list[ResearchInsight] = []
    for name in sorted(all_signals):
        used = [r[1] for r in rows if name in r[0]]
        absent = [r[1] for r in rows if name not in r[0]]
        if not used or not absent:
            continue
        mean_used = sum(used) / len(used)
        mean_absent = sum(absent) / len(absent)
        lift = mean_used - mean_absent
        p = _mannwhitney_p(used, absent)
        if p is None or p >= p_threshold:
            continue
        if lift <= lift_threshold:
            continue
        evidence = {
            "n_used": len(used),
            "n_absent": len(absent),
            "mean_used": mean_used,
            "mean_absent": mean_absent,
            "lift": lift,
            "p_value": p,
            "horizon": horizon,
            "days": days,
        }
        action = {
            "kind": "raise_signal_weight",
            "target": name,
            "params": {"factor": 1.2},
        }
        rationale = (
            f"Signal {name!r} shows lift {lift:+.2%} (p={p:.3f}, n={len(used)} "
            f"used vs {len(absent)} absent) over rolling {days}d at {horizon}. "
            "Recommend raising weight — clear positive contribution."
        )
        out.append(
            emit_insight(
                kind="signal_overperform",
                subject=name,
                severity="info",
                evidence=evidence,
                action=action,
                rationale=rationale,
                db_session=db_session,
            )
        )
    return out


def check_verdict_drift(
    *,
    eval_date: date | None = None,
    horizon: str = "5d",
    short_window: int = 14,
    long_window: int = 30,
    spread_drop_threshold: float = 0.10,
    db_session=None,
) -> list[ResearchInsight]:
    """Detect whether verdict discrimination has collapsed in the last
    short_window vs long_window.

    "Discrimination" = max(hit_rate) - min(hit_rate) across verdicts. If
    the rolling 14d spread is `spread_drop_threshold` BELOW the rolling
    30d spread (e.g. 30d spread 0.30 -> 14d spread 0.05), emit
    `verdict_drift` (critical) suggesting prompt revision.
    """
    rows_long = _fetch_eval_window(
        days=long_window, horizon=horizon, eval_date=eval_date
    )
    rows_short = _fetch_eval_window(
        days=short_window, horizon=horizon, eval_date=eval_date
    )
    if not rows_long or not rows_short:
        return []

    def _by_verdict(rs):
        out: dict[str, list[float]] = {}
        for sigs, ret, verdict, *_ in rs:
            out.setdefault(verdict, []).append(ret)
        return out

    def _hit(verdict: str, return_pct: float) -> bool:
        if verdict == "setup_long":
            return return_pct > 0
        if verdict in ("watch", "no_setup"):
            return return_pct <= 0
        return False

    def _spread(rs):
        buckets = _by_verdict(rs)
        rates = []
        for verdict, returns in buckets.items():
            if not returns:
                continue
            hits = sum(1 for r in returns if _hit(verdict, r))
            rates.append(hits / len(returns))
        if not rates:
            return None
        return max(rates) - min(rates)

    long_spread = _spread(rows_long)
    short_spread = _spread(rows_short)
    if long_spread is None or short_spread is None:
        return []
    drop = long_spread - short_spread
    if drop < spread_drop_threshold:
        return []

    # Look up the current prompt_version so the action targets the right
    # version. Default to "current" if unknown.
    prompt_version = "current"
    try:
        with get_session() as session:
            row = session.execute(
                select(LLMAnalysisRecord.prompt_template)
                .where(LLMAnalysisRecord.created_at.isnot(None))
                .order_by(LLMAnalysisRecord.id.desc())
                .limit(1)
            ).first()
        if row is not None and row[0]:
            prompt_version = str(row[0])
    except Exception:
        log.warning("verdict_drift_prompt_lookup_failed", exc_info=True)

    evidence = {
        "long_window_days": long_window,
        "short_window_days": short_window,
        "horizon": horizon,
        "long_spread": long_spread,
        "short_spread": short_spread,
        "drop": drop,
    }
    action = {
        "kind": "bump_prompt_version",
        "target": prompt_version,
        "params": {},
    }
    rationale = (
        f"Verdict discrimination collapsed: {long_window}d spread "
        f"{long_spread:.2f} -> {short_window}d spread {short_spread:.2f} "
        f"(drop {drop:.2f}). Prompt revision may be needed; recommend "
        "bumping prompt_version."
    )
    return [
        emit_insight(
            kind="verdict_drift",
            subject=prompt_version,
            severity="critical",
            evidence=evidence,
            action=action,
            rationale=rationale,
            db_session=db_session,
        )
    ]


def check_calibration_off(
    *,
    eval_date: date | None = None,
    horizon: str = "5d",
    days: int = 30,
    slope_deviation_threshold: float = 0.3,
    db_session=None,
) -> list[ResearchInsight]:
    """Bin theses by llm_confidence (1-10), compute realized hit-rate per
    bin, regress against expected (where confidence/10 is the predicted
    hit-rate). If the regression slope deviates from 1.0 by more than
    `slope_deviation_threshold`, emit `calibration_off` (warn).

    Action is `noop` — calibration fixes need a prompt edit, not a
    structured config change. The rationale tells the operator what to
    tweak.
    """
    rows = _fetch_eval_window(days=days, horizon=horizon, eval_date=eval_date)
    if not rows:
        return []

    def _hit(verdict: str, return_pct: float) -> bool:
        if verdict == "setup_long":
            return return_pct > 0
        if verdict in ("watch", "no_setup"):
            return return_pct <= 0
        return False

    bins: dict[int, list[float]] = {}
    for sigs, ret, verdict, conf, _scan in rows:
        if conf is None:
            continue
        bins.setdefault(int(conf), []).append(1.0 if _hit(verdict, ret) else 0.0)

    if len(bins) < 3:
        # Need at least three confidence levels to fit a slope.
        return []

    xs = []
    ys = []
    for conf, hits in bins.items():
        if not hits:
            continue
        predicted = conf / 10.0
        observed = sum(hits) / len(hits)
        xs.append(predicted)
        ys.append(observed)

    if len(xs) < 3:
        return []

    # Simple linear regression slope (least-squares). Avoid scipy/numpy to
    # keep this check dependency-light.
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return []
    slope = num / den
    deviation = abs(slope - 1.0)
    if deviation < slope_deviation_threshold:
        return []

    evidence = {
        "horizon": horizon,
        "days": days,
        "n_bins": len(xs),
        "slope": slope,
        "deviation_from_unity": deviation,
        "bins": [
            {"predicted": p, "observed": o} for p, o in zip(xs, ys)
        ],
    }
    action = {
        "kind": "noop",
        "target": "llm_confidence_calibration",
        "params": {},
    }
    direction = "over-confident" if slope < 1.0 else "under-confident"
    rationale = (
        f"LLM confidence is {direction}: regressed slope of observed-vs-predicted "
        f"hit-rate is {slope:.2f} (deviation {deviation:.2f} from 1.0). "
        "Recommend recalibrating the prompt's confidence-scoring instruction."
    )
    return [
        emit_insight(
            kind="calibration_off",
            subject="llm_confidence_calibration",
            severity="warn",
            evidence=evidence,
            action=action,
            rationale=rationale,
            db_session=db_session,
        )
    ]


def check_new_pattern_discovered(
    *,
    eval_date: date | None = None,
    horizon: str = "5d",
    days: int = 30,
    min_occurrences: int = 3,
    db_session=None,
) -> list[ResearchInsight]:
    """Mine LLMAnalysisRecord.structured_output['patterns_in_chart_not_in_indicators']
    for patterns mentioned >= min_occurrences times that correlate with
    positive forward returns. Emit `new_pattern_discovered` (info, action=noop)
    suggesting the operator add a deterministic detector.

    Joins to ThesisEvaluation to get the forward return per thesis.
    """
    anchor = eval_date if eval_date is not None else date.today()
    cutoff = anchor - timedelta(days=days)

    with get_session() as session:
        rows = session.execute(
            select(
                LLMAnalysisRecord.id,
                LLMAnalysisRecord.structured_output,
                ThesisEvaluation.return_pct,
            )
            .join(
                ThesisEvaluation,
                and_(
                    ThesisEvaluation.thesis_id == LLMAnalysisRecord.id,
                    ThesisEvaluation.horizon == horizon,
                ),
            )
            .where(ThesisEvaluation.scan_date >= cutoff)
        ).all()

    if not rows:
        return []

    pattern_returns: dict[str, list[float]] = {}
    for _rec_id, output, ret in rows:
        if not isinstance(output, dict):
            continue
        patterns = output.get("patterns_in_chart_not_in_indicators")
        if not isinstance(patterns, list):
            continue
        for p in patterns:
            if not isinstance(p, str) or not p.strip():
                continue
            pattern_returns.setdefault(p.strip(), []).append(float(ret))

    out: list[ResearchInsight] = []
    for pattern, returns in pattern_returns.items():
        if len(returns) < min_occurrences:
            continue
        mean_ret = sum(returns) / len(returns)
        if mean_ret <= 0:
            continue
        evidence = {
            "occurrences": len(returns),
            "mean_return": mean_ret,
            "horizon": horizon,
            "days": days,
        }
        action = {"kind": "noop", "target": pattern, "params": {}}
        rationale = (
            f"Discovered pattern {pattern!r} appeared in {len(returns)} theses "
            f"with mean {horizon} return {mean_ret:+.2%}. Consider adding a "
            "deterministic detector to the screener."
        )
        out.append(
            emit_insight(
                kind="new_pattern_discovered",
                subject=pattern[:100],  # column is VARCHAR(100)
                severity="info",
                evidence=evidence,
                action=action,
                rationale=rationale,
                db_session=db_session,
            )
        )
    return out


def check_prompt_regression(
    *,
    eval_date: date | None = None,
    horizon: str = "5d",
    days: int = 30,
    p_threshold: float = 0.05,
    db_session=None,
) -> list[ResearchInsight]:
    """A/B compare hit-rates across prompt_template values seen in the
    rolling window. If a NEWER prompt's hit-rate is statistically WORSE
    (chi-square / Mann-Whitney p<p_threshold AND mean is lower) than an
    older prompt's, emit `prompt_regression` (critical) suggesting
    rollback to the older version.

    Skips when fewer than 2 distinct prompt_template values are present.
    """
    anchor = eval_date if eval_date is not None else date.today()
    cutoff = anchor - timedelta(days=days)

    with get_session() as session:
        rows = session.execute(
            select(
                LLMAnalysisRecord.prompt_template,
                LLMAnalysisRecord.created_at,
                ThesisEvaluation.return_pct,
            )
            .join(
                ThesisEvaluation,
                and_(
                    ThesisEvaluation.thesis_id == LLMAnalysisRecord.id,
                    ThesisEvaluation.horizon == horizon,
                ),
            )
            .where(ThesisEvaluation.scan_date >= cutoff)
        ).all()

    if not rows:
        return []

    by_prompt: dict[str, list[float]] = {}
    earliest: dict[str, datetime] = {}
    for prompt, created_at, ret in rows:
        key = str(prompt or "")
        by_prompt.setdefault(key, []).append(float(ret))
        if key not in earliest or created_at < earliest[key]:
            earliest[key] = created_at

    if len(by_prompt) < 2:
        return []

    # Sort prompt templates by first-seen timestamp; older first.
    # Review iter-1 [P2]: emit at most ONE insight per `newer` prompt — pick
    # the strongest evidence (lowest p-value) among all (older, newer) pairs.
    # The prior implementation looped (older, newer) and called emit_insight
    # for each pair; because `subject=newer` is the same key, every later
    # call UPSERT-ed onto the same row, silently dropping earlier evidence
    # AND artificially inflating recurrence_count to N*(N-1)/2 in one job.
    ordered = sorted(by_prompt.keys(), key=lambda k: earliest[k])
    out: list[ResearchInsight] = []
    for newer in ordered[1:]:
        new_returns = by_prompt[newer]
        if not new_returns:
            continue
        mean_new = sum(new_returns) / len(new_returns)
        # Among all older prompts, pick the strongest evidence pair (lowest
        # p) where `newer` is statistically worse.
        # Tuple shape: (p, older, old_returns, mean_old).
        best: tuple[float, str, list[float], float] | None = None
        for older in ordered[: ordered.index(newer)]:
            old_returns = by_prompt[older]
            if not old_returns:
                continue
            mean_old = sum(old_returns) / len(old_returns)
            if mean_new >= mean_old:
                continue
            p = _mannwhitney_p(new_returns, old_returns)
            if p is None or p >= p_threshold:
                continue
            if best is None or p < best[0]:
                best = (p, older, old_returns, mean_old)
        if best is None:
            continue
        p, older, old_returns, mean_old = best
        evidence = {
            "older_prompt": older,
            "newer_prompt": newer,
            "n_old": len(old_returns),
            "n_new": len(new_returns),
            "mean_old": mean_old,
            "mean_new": mean_new,
            "p_value": p,
            "horizon": horizon,
            "days": days,
        }
        action = {
            "kind": "bump_prompt_version",
            "target": older,
            "params": {},
        }
        rationale = (
            f"Prompt {newer!r} shows mean {horizon} return {mean_new:+.2%} "
            f"vs older {older!r} {mean_old:+.2%} (p={p:.3f}). "
            f"Statistically worse — recommend rollback to {older!r}."
        )
        out.append(
            emit_insight(
                kind="prompt_regression",
                subject=newer,
                severity="critical",
                evidence=evidence,
                action=action,
                rationale=rationale,
                db_session=db_session,
            )
        )
    return out


# ---------------------------------------------------------------------------
# D6 — discover_time_stop (learn the force-exit horizon; recommend-only)
# ---------------------------------------------------------------------------

# Subject is fixed so re-emission UPSERTs onto one pending row (stability is
# tracked in evidence across weekly runs, not via duplicate rows).
TIME_STOP_SUBJECT = "paper_time_stop"
# A chosen `k` must repeat across this many consecutive weekly runs before the
# action flips from recommend-only (noop) to executable (set_learned_time_stop_days).
TIME_STOP_STABLE_RUNS = 2


def _is_priced_session(high: Any, low: Any) -> bool:
    """A bar counts as ONE holding session iff its high AND low are present —
    EXACTLY evaluate_exit's rule (exit.py: it skips only when high/low is None
    and advances the session counter on everything else, regardless of close).
    Unifying on this single predicate across _return_by_holding_day,
    _sl_tp_exit_day and _day_k_exit_return keeps session numbering identical to
    the production exit engine even on partial-OHLC bars (codex iter-6/iter-7
    [P2]). close is NOT required to count — a NULL-close session still counts;
    the return curve falls back to entry_price (0%) for it, matching the
    evaluate_exit time-stop close fallback (exit.py:129)."""
    return high is not None and low is not None


def _return_by_holding_day(
    price_rows: list[tuple[Any, float | None, float | None, float | None, float | None]],
    entry_date: date,
    entry_price: float,
    *,
    as_of: date,
) -> list[float]:
    """Reconstruct the close-to-entry return at each holding day (entry day =
    day 1), from entry_date forward, using only bars in [entry_date, as_of]
    (as-of guard — no hindsight). Sessions are counted on high/low presence
    (same _is_priced_session predicate as _sl_tp_exit_day, mirroring
    evaluate_exit). A counted session whose close is NULL contributes a 0%
    (entry_price fallback) return — exactly evaluate_exit's missing-close
    time-stop behaviour (exit.py:129) — so session numbers stay aligned.

    Returns a list where index i = return_pct after holding through the (i+1)th
    priced session.
    """
    bars = sorted(
        (
            (_as_date_local(d), o, h, lo, c)
            for (d, o, h, lo, c) in price_rows
            if entry_date <= _as_date_local(d) <= as_of
        ),
        key=lambda r: r[0],
    )
    out: list[float] = []
    for _d, _o, high, low, close in bars:
        if not _is_priced_session(high, low):
            continue
        # Missing close on a counted session → entry_price fallback (0% return),
        # matching evaluate_exit's time-stop close handling.
        px = float(close) if close is not None else entry_price
        out.append((px - entry_price) / entry_price)
    return out


def _as_date_local(d: Any) -> date:
    if isinstance(d, datetime):
        return d.date()
    return d


def _sl_tp_exit_day(
    price_rows: list[tuple[Any, float | None, float | None, float | None, float | None]],
    entry_date: date,
    stop_loss: float,
    target_price: float,
    *,
    as_of: date,
) -> int | None:
    """The 1-based holding day a trade SL/TP-exits, or None if it never does
    within [entry_date, as_of]. Used to build the SURVIVOR cohort for each
    candidate `k`: a trade exiting before day `k` is excluded from candidate `k`.

    Mirrors evaluate_exit's level rules (low<=stop, high>=target) walking
    priced sessions; partial-OHLC days are skipped (same _is_priced_session
    predicate as the return curve) and do not advance the counter.
    """
    bars = sorted(
        (
            (_as_date_local(d), o, h, lo, c)
            for (d, o, h, lo, c) in price_rows
            if entry_date <= _as_date_local(d) <= as_of
        ),
        key=lambda r: r[0],
    )
    session_n = 0
    for _d, _o, high, low, _close in bars:
        if not _is_priced_session(high, low):
            continue
        session_n += 1
        if low <= stop_loss or high >= target_price:
            return session_n
    return None


def _day_k_exit_return(
    price_rows: list[tuple[Any, float | None, float | None, float | None, float | None]],
    entry_date: date,
    entry_price: float,
    stop_loss: float,
    target_price: float,
    k: int,
    *,
    as_of: date,
) -> float | None:
    """Return realized under a ``time_stop_days=k`` policy, MATCHING evaluate_exit's
    same-session priority (codex iter-4 [P2]).

    evaluate_exit checks SL/TP BEFORE the time-stop on the same session, so a
    trade whose day-k bar touches stop/target exits at that LEVEL, not the day-k
    close. We therefore:
      * if SL/TP fires on a session < k → return None (caller already excludes
        these from candidate k's survivor cohort);
      * if SL/TP fires exactly on session k → score the LEVEL return (stop_loss
        or target_price), conservatively resolving a same-bar SL+TP hit to the
        stop (the F3 downward-bias convention evaluate_exit uses);
      * otherwise (survives past k) → score the day-k close return.

    None when the curve hasn't matured to day k (as-of guard).
    """
    exit_day = _sl_tp_exit_day(
        price_rows, entry_date, stop_loss, target_price, as_of=as_of
    )
    if exit_day is not None and exit_day < k:
        return None  # exited before k — not in candidate k's cohort
    bars = sorted(
        (
            (_as_date_local(d), o, h, lo, c)
            for (d, o, h, lo, c) in price_rows
            if entry_date <= _as_date_local(d) <= as_of
        ),
        key=lambda r: r[0],
    )
    if exit_day == k:
        # Locate the k-th priced (non-NULL-OHLC) session and resolve the level
        # with evaluate_exit's exact precedence (exit.py): an OPEN that already
        # gapped through a level fills AT THE OPEN before any intraday high/low
        # is consulted; only then do intraday touches apply (same-bar SL+TP ->
        # stop, F3 downward bias).
        session_n = 0
        for _d, open_px, high, low, _close in bars:
            if not _is_priced_session(high, low):
                continue
            session_n += 1
            if session_n == k:
                # Open-gap fills first (at the open). stop_loss < target_price,
                # so at most one applies.
                if open_px is not None and open_px <= stop_loss:
                    return (open_px - entry_price) / entry_price
                if open_px is not None and open_px >= target_price:
                    return (open_px - entry_price) / entry_price
                # No open-gap: intraday touches. Stop takes priority on a
                # same-bar SL+TP hit.
                if low <= stop_loss:
                    return (stop_loss - entry_price) / entry_price
                if high >= target_price:
                    return (target_price - entry_price) / entry_price
                break
    curve = _return_by_holding_day(price_rows, entry_date, entry_price, as_of=as_of)
    if len(curve) < k:
        return None
    return curve[k - 1]


def _sharpe_like(returns: list[float]) -> float | None:
    """Risk-adjusted objective (design ★): mean / population-std of the by-day
    returns at candidate `k`. NOT mean return — the gate is return PER UNIT of
    dispersion, so a high-mean/high-variance `k` can lose to a lower-mean/tight
    one. None when undefined (n<2 or zero dispersion)."""
    n = len(returns)
    if n < 2:
        return None
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / n
    std = var ** 0.5
    if std == 0:
        return None
    return mean / std


def _load_time_stop_trades(
    as_of: date,
) -> list[dict[str, Any]]:
    """Pull each paper position's (entry, levels, entry-forward price rows) for
    the curve reconstruction. Only filled positions (entry_date set) qualify."""
    from rainier.core.models import PaperTrade, StockPrice
    from rainier.paper.ingest import canonical_instant

    trades: list[dict[str, Any]] = []
    with get_session() as session:
        positions = session.execute(
            select(PaperTrade).where(PaperTrade.entry_date.isnot(None))
        ).scalars().all()
        meta = [
            {
                "id": p.id,
                "symbol": p.symbol,
                "entry_date": _as_date_local(p.entry_date),
                "entry_price": p.entry_price,
                "stop_loss": p.stop_loss,
                "target_price": p.target_price,
            }
            for p in positions
            if p.entry_price is not None
        ]
        for m in meta:
            rows = session.execute(
                select(
                    StockPrice.date,
                    StockPrice.open,
                    StockPrice.high,
                    StockPrice.low,
                    StockPrice.close,
                ).where(
                    StockPrice.symbol == m["symbol"],
                    StockPrice.date >= canonical_instant(m["entry_date"]),
                    StockPrice.date <= canonical_instant(as_of),
                )
            ).all()
            m["price_rows"] = [tuple(r) for r in rows]
            trades.append(m)
    return trades


def _prior_time_stop_evidence(db_session=None) -> dict[str, Any] | None:
    """The evidence dict of the current pending time_stop_discovered insight (if
    any), so we can read the prior chosen_k + stable_run_count to enforce the
    ≥2-run stability gate."""

    def _do(session):
        row = (
            session.query(ResearchInsight)
            .filter(
                ResearchInsight.kind == "time_stop_discovered",
                ResearchInsight.subject == TIME_STOP_SUBJECT,
                ResearchInsight.status == "pending",
            )
            .first()
        )
        return dict(row.evidence) if row is not None and row.evidence else None

    if db_session is not None:
        return _do(db_session)
    with get_session() as session:
        return _do(session)


def discover_time_stop(
    *,
    eval_date: date | None = None,
    candidate_ks: tuple[int, ...] = (3, 5, 8, 10, 15),
    min_survivors: int = 5,
    db_session=None,
) -> list[ResearchInsight]:
    """Learn the force-exit horizon `k` from realized paper positions (D6).

    For each candidate `k`:
      * reconstruct each trade's return-by-holding-day curve from stock_prices
        (entry-day forward, as-of capped — no hindsight);
      * include a trade in candidate `k`'s cohort ONLY if it SURVIVED to day `k`
        (did not SL/TP-exit before day `k` — survivor cohort);
      * require >= `min_survivors` matured survivors at day `k` (min-sample gate
        is PER candidate `k`, not total trades);
      * score the cohort's day-`k` returns by the RISK-ADJUSTED objective
        (Sharpe-like mean/std — NOT mean return).
    Pick the `k` maximizing the risk-adjusted objective.

    RECOMMEND-ONLY until the chosen `k` is stable across >= TIME_STOP_STABLE_RUNS
    consecutive weekly runs: while unstable the emitted action is `noop`; once
    stable it flips to `set_learned_time_stop_days` (still operator-approvable —
    never auto-applied here). Emits one `time_stop_discovered` insight or [].
    """
    anchor = eval_date if eval_date is not None else date.today()
    trades = _load_time_stop_trades(anchor)
    if not trades:
        return []

    # Score each candidate k over its survivor cohort.
    scored: dict[int, dict[str, Any]] = {}
    for k in candidate_ks:
        cohort_returns: list[float] = []
        survivors = 0
        for t in trades:
            # Score with evaluate_exit's same-session SL/TP-before-time-stop
            # priority (codex iter-4 [P2]): a day-k bar that touches stop/target
            # exits at that LEVEL, not the day-k close. Returns None when the
            # trade exited strictly before k (excluded from candidate k's
            # survivor cohort) or hasn't matured to day k (as-of guard).
            ret_k = _day_k_exit_return(
                t["price_rows"], t["entry_date"], t["entry_price"],
                t["stop_loss"], t["target_price"], k, as_of=anchor,
            )
            if ret_k is None:
                continue
            survivors += 1
            cohort_returns.append(ret_k)
        if survivors < min_survivors:
            continue
        obj = _sharpe_like(cohort_returns)
        if obj is None:
            continue
        mean_ret = sum(cohort_returns) / len(cohort_returns)
        scored[k] = {
            "survivors": survivors,
            "objective": obj,
            "mean_return": mean_ret,
        }

    if not scored:
        return []

    chosen_k = max(scored, key=lambda k: scored[k]["objective"])

    # Stability gate: compare to the prior pending insight's chosen_k. Count
    # only DISTINCT weekly observations — a retry/replay on the same eval_date
    # must not advance the counter (codex iter-1 [P2]): otherwise re-running the
    # job twice for one date would flip noop -> set_learned_time_stop_days
    # without a genuinely new week of data. We key idempotency on the prior
    # run's recorded eval_date.
    prior = _prior_time_stop_evidence(db_session=db_session)
    prior_k = prior.get("chosen_k") if prior else None
    prior_runs = int(prior.get("stable_run_count", 0)) if prior else 0
    prior_run_date = prior.get("last_run_date") if prior else None
    anchor_iso = anchor.isoformat()
    if prior_k != chosen_k:
        # Different pick — stability resets.
        stable_run_count = 1
    elif prior_run_date is not None and anchor_iso <= prior_run_date:
        # Same pick, but this run is NOT strictly forward of the prior recorded
        # observation — a same-date retry OR an out-of-order older replay (codex
        # iter-2 [P2]). Hold the count steady so neither can flip the action to
        # executable without two genuinely forward consecutive runs. Preserve the
        # prior recorded date so a later legitimate run still advances from it.
        stable_run_count = max(prior_runs, 1)
        anchor_iso = str(prior_run_date)
    else:
        # Same pick, a new distinct observation date strictly after the prior —
        # advance.
        stable_run_count = prior_runs + 1
    is_stable = stable_run_count >= TIME_STOP_STABLE_RUNS

    if is_stable:
        action = {
            "kind": "set_learned_time_stop_days",
            "target": str(chosen_k),
            "params": {"k": chosen_k},
        }
        severity = "warn"
    else:
        # Recommend-only — observed but not yet stable enough to apply.
        action = {"kind": "noop", "target": str(chosen_k), "params": {}}
        severity = "info"

    evidence = {
        "chosen_k": chosen_k,
        "stable_run_count": stable_run_count,
        "stable": is_stable,
        # Record the run's logical date so a same-date retry is idempotent
        # against the stability counter (codex iter-1 [P2]).
        "last_run_date": anchor_iso,
        "min_survivors": min_survivors,
        "candidate_ks": list(candidate_ks),
        "per_k": {str(k): v for k, v in sorted(scored.items())},
    }
    rationale = (
        f"Risk-adjusted force-exit horizon k={chosen_k} "
        f"(objective {scored[chosen_k]['objective']:.3f}, "
        f"{scored[chosen_k]['survivors']} survivors, mean "
        f"{scored[chosen_k]['mean_return']:+.2%}). "
        + (
            "Stable across "
            f"{stable_run_count} consecutive runs — recommend adopting "
            "learned_time_stop_days (future fills only)."
            if is_stable
            else f"Observed run {stable_run_count}/{TIME_STOP_STABLE_RUNS} — "
            "recommend-only until stable across consecutive runs."
        )
    )
    return [
        emit_insight(
            kind="time_stop_discovered",
            subject=TIME_STOP_SUBJECT,
            severity=severity,
            evidence=evidence,
            action=action,
            rationale=rationale,
            db_session=db_session,
        )
    ]


# ---------------------------------------------------------------------------
# D7c — weekly paper-trade lessons (human-readable; info/noop)
# ---------------------------------------------------------------------------

PAPER_LESSONS_SUBJECT = "paper_pnl"

# R-C — coarse market-regime tag stamped on each weekly lesson, so a lesson
# learned in one regime isn't blindly applied in another. SPY close vs its
# 200-day SMA on the lesson's as-of date: strictly above → bull, else bear;
# fewer than 200 usable bars → unknown (NEVER a partial-window SMA).
REGIME_SYMBOL = "SPY"
REGIME_SMA_WINDOW = 200
REGIMES: tuple[str, ...] = ("bull", "bear", "unknown")


def compute_market_regime(*, as_of: date) -> str:
    """Classify the market regime on ``as_of``: SPY close vs its 200-day SMA.

    Reads the last ``REGIME_SMA_WINDOW`` usable SPY closes at-or-before
    ``as_of`` from `stock_prices` (legacy engine — same store the paper
    ingest writes). Returns:

    * ``"bull"``    — latest close strictly above the 200-bar SMA
    * ``"bear"``    — latest close at-or-below the 200-bar SMA
    * ``"unknown"`` — fewer than 200 usable bars (no partial-window SMA;
      `ensure_spy_history` backfills coverage so this is transient)
    """
    from rainier.core.models import StockPrice

    # Bars are stored at the canonical 00:00 UTC instant of their trading
    # date (paper.ingest.canonical_instant), so <= this instant includes the
    # as-of bar and excludes anything after it.
    instant = datetime(as_of.year, as_of.month, as_of.day, tzinfo=timezone.utc)
    with get_session() as session:
        closes = session.execute(
            select(StockPrice.close)
            .where(
                StockPrice.symbol == REGIME_SYMBOL,
                StockPrice.date <= instant,
                StockPrice.close.isnot(None),
            )
            .order_by(StockPrice.date.desc())
            .limit(REGIME_SMA_WINDOW)
        ).scalars().all()

    if len(closes) < REGIME_SMA_WINDOW:
        return "unknown"
    latest = float(closes[0])  # newest first (date desc)
    sma = sum(float(c) for c in closes) / len(closes)
    return "bull" if latest > sma else "bear"


def check_paper_lessons(
    *,
    eval_date: date | None = None,
    days: int = 30,
    db_session=None,
) -> list[ResearchInsight]:
    """Read the closed paper-trade record over the rolling window and emit a
    human-readable `paper_lessons` insight (info, action=noop) — D7c.

    Summarizes realized win-rate, exit-reason mix, and the best/worst closed
    trade so the operator (and later the prompt) can see what the paper book
    actually did. Stamped with the coarse market-regime tag (R-C: SPY vs
    200-day SMA → bull/bear/unknown) in both the evidence payload and the
    rendered rationale. No weight-tuning here (that is D7b, deferred)."""
    from rainier.core.models import PaperTrade

    anchor = eval_date if eval_date is not None else date.today()
    cutoff = anchor - timedelta(days=days)

    with get_session() as session:
        closed = session.execute(
            select(PaperTrade).where(
                PaperTrade.status == "closed",
                PaperTrade.exit_date.isnot(None),
                PaperTrade.exit_date >= cutoff,
                PaperTrade.exit_date <= anchor,
            )
        ).scalars().all()
        rows = [
            {
                "symbol": p.symbol,
                "exit_reason": p.exit_reason,
                "return_pct": float(p.return_pct) if p.return_pct is not None else 0.0,
                "pnl": float(p.pnl) if p.pnl is not None else 0.0,
            }
            for p in closed
        ]

    if not rows:
        return []

    n = len(rows)
    wins = sum(1 for r in rows if r["return_pct"] > 0)
    reason_mix: dict[str, int] = {}
    for r in rows:
        reason_mix[r["exit_reason"] or "unknown"] = (
            reason_mix.get(r["exit_reason"] or "unknown", 0) + 1
        )
    total_pnl = sum(r["pnl"] for r in rows)
    best = max(rows, key=lambda r: r["return_pct"])
    worst = min(rows, key=lambda r: r["return_pct"])

    # R-C: stamp the lesson with the market regime on its as-of date so the
    # operator can weigh cross-regime advice. Degrades to "unknown" when SPY
    # coverage is short — never blocks the lesson itself.
    regime = compute_market_regime(as_of=anchor)

    evidence = {
        "regime": regime,
        "days": days,
        "n_closed": n,
        "win_rate": wins / n,
        "exit_reason_mix": reason_mix,
        "total_realized_pnl": round(total_pnl, 2),
        "best": {"symbol": best["symbol"], "return_pct": round(best["return_pct"], 4)},
        "worst": {"symbol": worst["symbol"], "return_pct": round(worst["return_pct"], 4)},
    }
    mix_str = ", ".join(f"{kk}:{vv}" for kk, vv in sorted(reason_mix.items()))
    rationale = (
        f"[regime: {regime}] "
        f"Paper book closed {n} trades over {days}d: win-rate {wins / n:.0%}, "
        f"realized ${total_pnl:,.2f}. Exit mix [{mix_str}]. "
        f"Best {best['symbol']} {best['return_pct']:+.2%}; "
        f"worst {worst['symbol']} {worst['return_pct']:+.2%}."
    )
    return [
        emit_insight(
            kind="paper_lessons",
            subject=PAPER_LESSONS_SUBJECT,
            severity="info",
            evidence=evidence,
            action={"kind": "noop", "target": PAPER_LESSONS_SUBJECT, "params": {}},
            rationale=rationale,
            db_session=db_session,
        )
    ]


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def run_research(
    *,
    eval_date: date | None = None,
    days: int = 30,
    horizon: str = "5d",
) -> list[ResearchInsight]:
    """Run all 6 check classes and return their emitted ResearchInsight rows.

    Each check is wrapped in try/except so one failing check doesn't kill
    the whole job — log and continue.
    """
    out: list[ResearchInsight] = []
    checks: list[tuple[str, Callable]] = [
        ("signal_underperform", check_signal_underperform),
        ("signal_overperform", check_signal_overperform),
        ("verdict_drift", check_verdict_drift),
        ("calibration_off", check_calibration_off),
        ("new_pattern_discovered", check_new_pattern_discovered),
        ("prompt_regression", check_prompt_regression),
        # Phase 2 (D6 / D7c): learn the force-exit horizon + weekly paper lessons.
        ("time_stop_discovered", discover_time_stop),
        ("paper_lessons", check_paper_lessons),
    ]
    for name, fn in checks:
        try:
            kwargs: dict[str, Any] = {"eval_date": eval_date}
            if name in ("signal_underperform", "signal_overperform"):
                kwargs["days"] = days
                kwargs["horizon"] = horizon
            elif name in ("calibration_off", "new_pattern_discovered"):
                kwargs["days"] = days
                kwargs["horizon"] = horizon
            elif name == "prompt_regression":
                kwargs["days"] = days
                kwargs["horizon"] = horizon
            elif name == "verdict_drift":
                kwargs["horizon"] = horizon
            elif name == "paper_lessons":
                kwargs["days"] = days
            # discover_time_stop takes only eval_date (+ its own defaults).
            insights = fn(**kwargs)
            out.extend(insights)
            log.info(
                "research_check_done check=%s emitted=%s", name, len(insights)
            )
        except Exception as exc:
            log.exception("research_check_failed check=%s error=%s", name, exc)
    return out


# ---------------------------------------------------------------------------
# ACTION_EXECUTORS — accept-handler dispatch (eng review D3)
# ---------------------------------------------------------------------------


def _load_yaml_round_trip(path: Path):
    """Return (yaml_loader, parsed_doc). Uses ruamel.yaml so round-trip
    preserves comments + key order.
    """
    from ruamel.yaml import YAML

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    with path.open("r") as f:
        data = yaml.load(f) or {}
    return yaml, data


def _atomic_write_yaml(yaml, data, path: Path) -> None:
    """Dump `data` via `yaml` to a temp file in the target directory, then
    atomic-rename. Caller is responsible for ensuring the parent directory
    is writable.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(parent))
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        # Cleanup on failure.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _disable_signal(target: str, params: dict, settings_path: Path) -> dict[str, Any]:
    """Flip `llm_thesis.signals.<target>.enabled` to false in the YAML.

    If the signal section is missing, create it with sensible defaults
    rather than failing — keeps the executor idempotent.
    """
    yaml, data = _load_yaml_round_trip(settings_path)
    section = data.setdefault("llm_thesis", {}).setdefault("signals", {})
    entry = section.get(target)
    if entry is None:
        entry = {"enabled": False, "params": {}, "weight": 1.0}
        section[target] = entry
    else:
        entry["enabled"] = False
    _atomic_write_yaml(yaml, data, settings_path)
    return {"signal": target, "field": "enabled", "new_value": False}


def _bump_prompt_version(
    target: str, params: dict, settings_path: Path
) -> dict[str, Any]:
    """Increment llm_thesis.prompt_version from 'vN' to 'v(N+1)'.

    `target` is the OLD version we expect to see; if it doesn't match the
    file's current value, we still bump from the file's value (defensive)
    and record the actual transition.
    """
    yaml, data = _load_yaml_round_trip(settings_path)
    section = data.setdefault("llm_thesis", {})
    current = section.get("prompt_version", target or "v1")
    new_version = _increment_prompt_version(str(current))
    section["prompt_version"] = new_version
    _atomic_write_yaml(yaml, data, settings_path)
    return {
        "field": "prompt_version",
        "old_value": current,
        "new_value": new_version,
    }


def _increment_prompt_version(version: str) -> str:
    """Bump 'vN' -> 'v(N+1)'. Falls back to 'v2' for unparseable inputs."""
    if version.startswith("v") and version[1:].isdigit():
        return f"v{int(version[1:]) + 1}"
    return "v2"


def _raise_signal_weight(
    target: str, params: dict, settings_path: Path
) -> dict[str, Any]:
    """Multiply llm_thesis.signals.<target>.weight by params.factor (default 1.2)."""
    factor = float((params or {}).get("factor", 1.2))
    return _scale_signal_weight(target, factor, settings_path)


def _lower_signal_weight(
    target: str, params: dict, settings_path: Path
) -> dict[str, Any]:
    """Multiply llm_thesis.signals.<target>.weight by params.factor (default 0.8)."""
    factor = float((params or {}).get("factor", 0.8))
    return _scale_signal_weight(target, factor, settings_path)


def _scale_signal_weight(
    target: str, factor: float, settings_path: Path
) -> dict[str, Any]:
    """Shared backbone for raise/lower. Clamps to [0.0, 5.0] so noisy
    insights can't push a signal's weight unbounded in either direction.
    """
    yaml, data = _load_yaml_round_trip(settings_path)
    section = data.setdefault("llm_thesis", {}).setdefault("signals", {})
    entry = section.get(target)
    if entry is None:
        entry = {"enabled": True, "params": {}, "weight": 1.0}
        section[target] = entry
    old = float(entry.get("weight", 1.0) or 1.0)
    new = max(0.0, min(5.0, old * factor))
    entry["weight"] = new
    _atomic_write_yaml(yaml, data, settings_path)
    return {
        "signal": target,
        "field": "weight",
        "old_value": old,
        "new_value": new,
        "factor": factor,
    }


def _set_learned_time_stop_days(
    target: str, params: dict, settings_path: Path
) -> dict[str, Any]:
    """Set llm_thesis.learned_time_stop_days to the discovered horizon `k` (D6).

    The chosen `k` comes from `params['k']` (preferred) or, failing that, the
    `target` string. Mutates the EXISTING `LLMThesisConfig.learned_time_stop_days`
    field (core/config.py) — never re-creates it. Only FUTURE fills snapshot the
    new value at fill time (positions.py); already-open positions keep their
    prior NULL/value (future-fills-only invariant). `None`/unparseable clears it
    (back to no time-exit).
    """
    raw = (params or {}).get("k", target)
    new_value: int | None
    try:
        new_value = int(raw) if raw is not None and str(raw) != "" else None
        if new_value is not None and new_value < 1:
            new_value = None
    except (TypeError, ValueError):
        new_value = None

    yaml, data = _load_yaml_round_trip(settings_path)
    section = data.setdefault("llm_thesis", {})
    old = section.get("learned_time_stop_days")
    section["learned_time_stop_days"] = new_value
    _atomic_write_yaml(yaml, data, settings_path)
    return {
        "field": "learned_time_stop_days",
        "old_value": old,
        "new_value": new_value,
    }


def _noop(target: str, params: dict, settings_path: Path) -> dict[str, Any]:
    """Info-only insight — no config change."""
    return {"noop": True, "target": target}


ACTION_EXECUTORS: dict[str, Callable[[str, dict, Path], dict[str, Any]]] = {
    "disable_signal": _disable_signal,
    "bump_prompt_version": _bump_prompt_version,
    "raise_signal_weight": _raise_signal_weight,
    "lower_signal_weight": _lower_signal_weight,
    "set_learned_time_stop_days": _set_learned_time_stop_days,
    "noop": _noop,
}


def apply_action(action: dict[str, Any], settings_path: Path) -> dict[str, Any]:
    """Dispatch one action through ACTION_EXECUTORS.

    Raises ValueError on unknown action.kind.
    """
    if not isinstance(action, dict):
        raise ValueError(f"action must be a dict, got {type(action).__name__}")
    kind = action.get("kind")
    target = action.get("target", "")
    params = action.get("params") or {}
    if kind not in ACTION_EXECUTORS:
        raise ValueError(
            f"Unknown action kind: {kind!r}; valid={sorted(ACTION_EXECUTORS)}"
        )
    return ACTION_EXECUTORS[kind](str(target), dict(params), settings_path)
