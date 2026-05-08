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


def _noop(target: str, params: dict, settings_path: Path) -> dict[str, Any]:
    """Info-only insight — no config change."""
    return {"noop": True, "target": target}


ACTION_EXECUTORS: dict[str, Callable[[str, dict, Path], dict[str, Any]]] = {
    "disable_signal": _disable_signal,
    "bump_prompt_version": _bump_prompt_version,
    "raise_signal_weight": _raise_signal_weight,
    "lower_signal_weight": _lower_signal_weight,
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
