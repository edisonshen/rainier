"""Read-only data loaders for the Streamlit dashboard.

NO Streamlit imports. Every function is a plain DB query that returns
typed dataclasses or pandas DataFrames so the loaders can be exercised
in unit tests without spinning up a Streamlit runtime.

Page layouts in `app.py` consume these helpers; the dashboard never
talks to the database directly. Keeping the data layer pure means a
loader bug fails a deterministic unit test instead of crashing the
Streamlit page at runtime.

  ┌──────────────┐         ┌──────────────┐
  │   app.py     │ ──────▶ │   data.py    │ ──▶ ORM (read-only)
  │ (Streamlit)  │         │ (pure funcs) │
  └──────────────┘         └──────────────┘
        │
        └──▶ actions.py  (write helpers shared with the CLI)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select

from rainier.core.config import load_settings_fresh
from rainier.core.database import get_session
from rainier.core.models import (
    ChartImage,
    LLMAnalysisRecord,
    ResearchInsight,
    ScreenedStockRecord,
    ThesisEvaluation,
)
from rainier.llm_thesis.eval import (
    HitRate,
    SignalContribution,
    compute_signal_contribution,
    compute_verdict_hit_rate,
)
from rainier.llm_thesis.signals import REGISTRY

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignalRow:
    """Tab 1 row — one per signal in the registry."""

    name: str
    enabled: bool
    version: str
    weight: float
    cost_estimate_ms: int
    last_7d_hit_count: int


@dataclass(frozen=True, slots=True)
class ThesisRow:
    """Tab 3 row — one per recent thesis."""

    thesis_id: int
    scan_date: date
    session_name: str
    symbol: str
    verdict: str
    llm_confidence: int | None
    signals_used: list[str]
    return_1d: float | None
    return_5d: float | None
    return_10d: float | None
    chart_image_id: int | None
    structured_output: dict[str, Any] | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class ResearchInsightRow:
    """Tab 4 row — one per ResearchInsight (pending or historical)."""

    insight_id: int
    created_at: datetime
    updated_at: datetime
    kind: str
    severity: str
    subject: str
    rationale: str | None
    evidence: dict[str, Any] | None
    action: dict[str, Any] | None
    recurrence_count: int
    status: str
    decided_at: datetime | None
    decided_by: str | None
    applied_change: dict[str, Any] | None


# ---------------------------------------------------------------------------
# Tab 1 — Signals
# ---------------------------------------------------------------------------


def _hit_count_window_days() -> int:
    """How many days back to count `signals_used` hits for the Tab 1 status.

    Centralized here so the test can override via monkeypatch and so any
    future "last 14d" toggle has one place to change.
    """
    return 7


def load_signal_status(*, settings_path: str | Path | None = None) -> list[SignalRow]:
    """Tab 1: registry x settings.yaml x last-7d hit count.

    For each signal in `REGISTRY`, look up the YAML toggle from a fresh
    settings load, then query `LLMAnalysisRecord.signals_used` over the
    rolling 7d window to count how often the signal contributed to a
    thesis. Counts theses (not symbols) — one thesis per scan/symbol.
    """
    cfg_path = str(settings_path) if settings_path else "config/settings.yaml"
    settings = load_settings_fresh(cfg_path)
    cfg_map = settings.llm_thesis.signals

    days = _hit_count_window_days()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Pull the entire signals_used array from records in the window once,
    # then bucket in Python. Avoids one query per signal.
    counts: dict[str, int] = {name: 0 for name in REGISTRY}
    with get_session() as session:
        rows = session.execute(
            select(LLMAnalysisRecord.signals_used).where(
                LLMAnalysisRecord.created_at >= cutoff,
            )
        ).all()
    for (signals_used,) in rows:
        if not signals_used:
            continue
        for name in signals_used:
            if name in counts:
                counts[name] += 1

    out: list[SignalRow] = []
    for name, sig_cls in REGISTRY.items():
        instance = sig_cls()
        cfg = cfg_map.get(name)
        out.append(
            SignalRow(
                name=name,
                enabled=bool(cfg.enabled) if cfg else True,
                version=str(getattr(instance, "version", "?")),
                weight=float(cfg.weight) if cfg else 1.0,
                cost_estimate_ms=int(getattr(instance, "cost_estimate_ms", 0)),
                last_7d_hit_count=counts.get(name, 0),
            )
        )
    out.sort(key=lambda r: r.name)
    return out


# ---------------------------------------------------------------------------
# Tab 2 — Performance
# ---------------------------------------------------------------------------


def load_signal_contribution(
    days: int = 30, horizon: str = "5d"
) -> list[SignalContribution]:
    """Wrapper around `eval.compute_signal_contribution` for Tab 2.

    Kept as a thin wrapper so the dashboard page imports a single
    `dashboard.data` namespace and can be mocked from tests without
    reaching into `llm_thesis.eval`.
    """
    return compute_signal_contribution(days=days, horizon=horizon)


def load_verdict_hit_rate(days: int = 30) -> dict[str, list[HitRate]]:
    """Wrapper around `eval.compute_verdict_hit_rate` for Tab 2."""
    return compute_verdict_hit_rate(days=days)


# Per-signal value extractor map: signal_name -> (signals_used JSON dict) -> float | None.
# When a new signal lands, add a row here so Tab 2's scatterplot can plot it.
# Returning None drops the row from the scatter (the signal returned no
# numeric quantity for that thesis).
def _extract_rank_trajectory(value: dict[str, Any]) -> float | None:
    delta = value.get("delta_10d")
    return float(delta) if isinstance(delta, (int, float)) else None


def _extract_capital_flow_streak(value: dict[str, Any]) -> float | None:
    streak = value.get("streak_days")
    direction = value.get("direction", "")
    if not isinstance(streak, (int, float)):
        return None
    sign = -1 if direction == "-" else 1
    return float(streak) * sign


def _extract_sector_momentum(value: dict[str, Any]) -> float | None:
    delta = value.get("delta")
    return float(delta) if isinstance(delta, (int, float)) else None


def _extract_fundamentals(value: dict[str, Any]) -> float | None:
    # Forward P/E is the single most-actionable scalar; mirrors how the
    # signal renders for the prompt (eg "Forward P/E: 23.4").
    pe = value.get("pe_forward")
    if not isinstance(pe, (int, float)):
        return None
    return float(pe)


SIGNAL_VALUE_EXTRACTORS: dict[str, Any] = {
    "rank_trajectory": _extract_rank_trajectory,
    "capital_flow_streak": _extract_capital_flow_streak,
    "sector_momentum": _extract_sector_momentum,
    "fundamentals": _extract_fundamentals,
}


def load_signal_value_series(
    signal_name: str, days: int = 30, horizon: str = "5d"
) -> pd.DataFrame:
    """Tab 2 scatter: signal numeric value vs forward return.

    JOINs `LLMAnalysisRecord.structured_output` (where the per-signal
    raw values live alongside the LLM output) with `ThesisEvaluation`
    so we get one row per (thesis, horizon) with a paired x/y.

    Returns an empty DataFrame with the expected columns when no rows
    exist or the signal is unknown — keeps the Streamlit chart code
    branchless.
    """
    cols = ["scan_date", "symbol", "signal_value", "forward_return_5d"]
    extractor = SIGNAL_VALUE_EXTRACTORS.get(signal_name)
    if extractor is None:
        log.info("load_signal_value_series unknown_signal=%s", signal_name)
        return pd.DataFrame(columns=cols)

    cutoff = date.today() - timedelta(days=days)

    with get_session() as session:
        rows = session.execute(
            select(
                ThesisEvaluation.scan_date,
                ThesisEvaluation.symbol,
                ThesisEvaluation.return_pct,
                LLMAnalysisRecord.structured_output,
            )
            .join(
                LLMAnalysisRecord,
                LLMAnalysisRecord.id == ThesisEvaluation.thesis_id,
            )
            .where(
                ThesisEvaluation.horizon == horizon,
                ThesisEvaluation.scan_date >= cutoff,
            )
            .order_by(ThesisEvaluation.scan_date.asc())
        ).all()

    records: list[dict[str, Any]] = []
    for scan_dt, symbol, return_pct, structured in rows:
        if not isinstance(structured, dict):
            continue
        # The signal raw values are nested under structured_output["signals"]
        # when service.py persists them; degrade gracefully if absent.
        signals_blob = structured.get("signals") or {}
        if not isinstance(signals_blob, dict):
            continue
        raw = signals_blob.get(signal_name)
        if not isinstance(raw, dict):
            continue
        try:
            value = extractor(raw)
        except Exception:
            log.warning(
                "signal_value_extract_failed signal=%s symbol=%s",
                signal_name,
                symbol,
                exc_info=True,
            )
            continue
        if value is None:
            continue
        records.append(
            {
                "scan_date": scan_dt,
                "symbol": symbol,
                "signal_value": value,
                "forward_return_5d": float(return_pct),
            }
        )
    return pd.DataFrame(records, columns=cols)


# ---------------------------------------------------------------------------
# Tab 3 — Recent theses
# ---------------------------------------------------------------------------


def _coerce_chart_image_id(value: Any) -> int | None:
    """`LLMAnalysisRecord.chart_image_ids` is ARRAY(Integer); the dashboard
    only renders the first chart. Return the first id or None.
    """
    if not value:
        return None
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, int):
            return first
        try:
            return int(first)
        except (TypeError, ValueError):
            return None
    return None


def load_recent_theses(
    limit: int = 50,
    *,
    verdicts: list[str] | None = None,
    symbol: str | None = None,
    since: date | None = None,
    until: date | None = None,
) -> list[ThesisRow]:
    """Tab 3 list — recent theses with their forward returns.

    JOINs `ScreenedStockRecord` -> `LLMAnalysisRecord` (LEFT) ->
    `ThesisEvaluation` (LEFT, one row per horizon). Returns one
    `ThesisRow` per screened+thesis pair with all three horizons
    flattened into return_1d / return_5d / return_10d (None when the
    eval row hasn't landed yet).

    Filters are applied at the SQL level so the LIMIT is meaningful.
    """
    stmt = (
        select(
            ScreenedStockRecord.id.label("screened_id"),
            ScreenedStockRecord.scan_date,
            ScreenedStockRecord.session_name,
            ScreenedStockRecord.symbol,
            ScreenedStockRecord.thesis_id,
            ScreenedStockRecord.llm_confidence,
            LLMAnalysisRecord.recommendation,
            LLMAnalysisRecord.signals_used,
            LLMAnalysisRecord.chart_image_ids,
            LLMAnalysisRecord.structured_output,
        )
        .join(
            LLMAnalysisRecord,
            LLMAnalysisRecord.id == ScreenedStockRecord.thesis_id,
        )
        .where(ScreenedStockRecord.thesis_id.isnot(None))
    )
    if verdicts:
        stmt = stmt.where(LLMAnalysisRecord.recommendation.in_(verdicts))
    if symbol:
        stmt = stmt.where(ScreenedStockRecord.symbol == symbol.upper())
    if since:
        stmt = stmt.where(ScreenedStockRecord.scan_date >= since)
    if until:
        stmt = stmt.where(ScreenedStockRecord.scan_date <= until)

    stmt = stmt.order_by(
        ScreenedStockRecord.scan_date.desc(),
        ScreenedStockRecord.id.desc(),
    ).limit(int(limit))

    with get_session() as session:
        base_rows = session.execute(stmt).all()
        thesis_ids = [r.thesis_id for r in base_rows if r.thesis_id is not None]

        eval_map: dict[tuple[int, str], float] = {}
        if thesis_ids:
            eval_rows = session.execute(
                select(
                    ThesisEvaluation.thesis_id,
                    ThesisEvaluation.horizon,
                    ThesisEvaluation.return_pct,
                ).where(ThesisEvaluation.thesis_id.in_(thesis_ids))
            ).all()
            for tid, horizon, ret in eval_rows:
                eval_map[(int(tid), str(horizon))] = float(ret)

    out: list[ThesisRow] = []
    for r in base_rows:
        tid = int(r.thesis_id)
        out.append(
            ThesisRow(
                thesis_id=tid,
                scan_date=r.scan_date,
                session_name=r.session_name,
                symbol=r.symbol,
                verdict=str(r.recommendation or ""),
                llm_confidence=(
                    int(r.llm_confidence) if r.llm_confidence is not None else None
                ),
                signals_used=list(r.signals_used or []),
                return_1d=eval_map.get((tid, "1d")),
                return_5d=eval_map.get((tid, "5d")),
                return_10d=eval_map.get((tid, "10d")),
                chart_image_id=_coerce_chart_image_id(r.chart_image_ids),
                structured_output=(
                    dict(r.structured_output)
                    if isinstance(r.structured_output, dict)
                    else None
                ),
            )
        )
    return out


def load_thesis_chart(thesis_id: int) -> bytes | None:
    """Read the first chart PNG associated with a thesis.

    PR5: prefer the inline ``image_bytes`` column (filled by
    ``llm_thesis.chart_persistence.persist_chart_image``) so Discord +
    dashboard never depend on the filesystem. Legacy rows that only have
    ``file_path`` set still resolve via the on-disk read path.

    Returns the file bytes, or None when the thesis has no chart, the
    file is missing on disk, or anything errors. Never raises — Tab 3
    just renders "no chart available" instead.
    """
    with get_session() as session:
        analysis = session.get(LLMAnalysisRecord, int(thesis_id))
        if analysis is None or not analysis.chart_image_ids:
            return None
        first_id = _coerce_chart_image_id(analysis.chart_image_ids)
        if first_id is None:
            return None
        chart = session.get(ChartImage, int(first_id))
        if chart is None:
            return None
        # PR5 path: bytes inline.
        if chart.image_bytes:
            return bytes(chart.image_bytes)
        if not chart.file_path:
            return None
        path = Path(chart.file_path)

    try:
        if not path.exists() or not path.is_file():
            log.info("thesis_chart_missing path=%s", path)
            return None
        return path.read_bytes()
    except OSError:
        log.warning("thesis_chart_read_failed path=%s", path, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Tab 4 — Research insights
# ---------------------------------------------------------------------------


_SEVERITY_ORDER: dict[str, int] = {"critical": 0, "warn": 1, "info": 2}


def _to_jsonable(value: Any) -> Any:
    """Coerce a JSONB column value into a plain dict/list for the UI.

    SQLAlchemy returns either dict/list (Postgres JSONB) or a JSON string
    (SQLite via the test fakes); normalise so the dashboard always sees
    Python primitives.
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return json.loads(bytes(value).decode("utf-8"))
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _row_to_insight(row: ResearchInsight) -> ResearchInsightRow:
    return ResearchInsightRow(
        insight_id=int(row.id),
        created_at=row.created_at,
        updated_at=row.updated_at,
        kind=row.kind,
        severity=row.severity,
        subject=row.subject,
        rationale=row.rationale,
        evidence=_to_jsonable(row.evidence),
        action=_to_jsonable(row.action),
        recurrence_count=int(row.recurrence_count or 0),
        status=row.status,
        decided_at=row.decided_at,
        decided_by=row.decided_by,
        applied_change=_to_jsonable(row.applied_change),
    )


def load_pending_insights() -> list[ResearchInsightRow]:
    """Tab 4 'pending' panel.

    Sorted by severity (critical -> warn -> info), then most-recently
    updated first so a re-fired insight bubbles to the top of the queue.
    """
    with get_session() as session:
        stmt = (
            select(ResearchInsight)
            .where(ResearchInsight.status == "pending")
            .order_by(
                ResearchInsight.updated_at.desc(),
                ResearchInsight.id.desc(),
            )
        )
        rows = session.execute(stmt).scalars().all()

    out = [_row_to_insight(r) for r in rows]
    out.sort(
        key=lambda r: (
            _SEVERITY_ORDER.get(r.severity, 99),
            -int(r.updated_at.timestamp()) if r.updated_at else 0,
            -r.insight_id,
        )
    )
    return out


def load_insight_history(
    status: list[str] | None = None,
) -> list[ResearchInsightRow]:
    """Tab 4 'history' panel — accepted/rejected/auto_applied/stale.

    Default filter excludes `pending` (those go in `load_pending_insights`)
    and `stale` (ignored by the operator). Ordered by `decided_at`
    descending so the most recent decision is on top.
    """
    if status is None:
        status = ["accepted", "rejected", "auto_applied"]
    if not status:
        return []

    with get_session() as session:
        stmt = (
            select(ResearchInsight)
            .where(ResearchInsight.status.in_(list(status)))
            .order_by(
                ResearchInsight.decided_at.desc().nullslast(),
                ResearchInsight.id.desc(),
            )
        )
        rows = session.execute(stmt).scalars().all()
    return [_row_to_insight(r) for r in rows]


# Re-export for callers / tests
__all__ = [
    "HitRate",
    "ResearchInsightRow",
    "SignalContribution",
    "SignalRow",
    "ThesisRow",
    "load_insight_history",
    "load_pending_insights",
    "load_recent_theses",
    "load_signal_contribution",
    "load_signal_status",
    "load_signal_value_series",
    "load_thesis_chart",
    "load_verdict_hit_rate",
    "SIGNAL_VALUE_EXTRACTORS",
]
