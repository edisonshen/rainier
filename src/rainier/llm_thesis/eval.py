"""Daily evaluation — backfill ThesisEvaluation rows + compute rolling stats.

Runs nightly at 17:00 PT. For every thesis from N days ago lacking a
ThesisEvaluation row at the requested horizon, look up entry/exit closes
from `StockPrice`, compute return_pct, and INSERT. Idempotent.

Two analysis helpers consume the resulting ThesisEvaluation rows:

  * `compute_signal_contribution(days)` — per-signal Mann-Whitney U on
    used-vs-absent forward returns at the 5d horizon. Falsifiable: a
    signal with p<0.05 and a non-trivial lift is informative; everything
    else is candidate-disable material.
  * `compute_verdict_hit_rate(days)` — per-verdict win-rate + average
    return at 1d/5d/10d horizons. Drives the rolling base-rate block in
    the Discord eval report.

Both helpers return frozen dataclasses (defined here so the
`alerts/discord.py` renderer doesn't need to import scipy).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal

from sqlalchemy import and_, case, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rainier.core.database import get_session
from rainier.core.models import (
    LLMAnalysisRecord,
    ScreenedStockRecord,
    StockPrice,
    ThesisEvaluation,
)

log = logging.getLogger(__name__)


# Horizons we evaluate. Centralized here so callers don't typo strings.
HORIZONS: tuple[Literal["1d", "5d", "10d"], ...] = ("1d", "5d", "10d")
_HORIZON_DAYS: dict[str, int] = {"1d": 1, "5d": 5, "10d": 10}

# Verdicts the screener / LLM emit. Long-only — `setup_short` is impossible
# under the screener's "Long in" filter.
VERDICTS: tuple[str, ...] = ("setup_long", "watch", "no_setup")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignalContribution:
    """Result of `compute_signal_contribution` for a single signal."""

    name: str
    n_used: int
    n_absent: int
    mean_used: float
    mean_absent: float
    lift: float           # mean_used - mean_absent
    p_value: float | None  # None when sample size is too small for the test


@dataclass(frozen=True, slots=True)
class HitRate:
    """Per-verdict win rate + average return at one horizon."""

    verdict: str
    horizon: str
    n: int
    win_rate: float        # fraction in [0, 1]
    avg_return_pct: float  # mean of return_pct (decimal fraction)


# ---------------------------------------------------------------------------
# evaluate_horizon — backfill ThesisEvaluation rows
# ---------------------------------------------------------------------------


def _close_on_or_before(symbol: str, target: date) -> tuple[date, float] | None:
    """Most recent StockPrice row at or before `target` for `symbol`.

    Falls back to the prior trading day when the exact date is missing
    (weekend / holiday). Returns None when nothing is found within a
    reasonable window (10 calendar days back).
    """
    cutoff = datetime.combine(target, time.max, tzinfo=timezone.utc)
    floor = datetime.combine(
        target - timedelta(days=10), time.min, tzinfo=timezone.utc
    )
    with get_session() as session:
        row = (
            session.query(StockPrice.date, StockPrice.close)
            .filter(
                StockPrice.symbol == symbol,
                StockPrice.date <= cutoff,
                StockPrice.date >= floor,
                StockPrice.close.isnot(None),
            )
            .order_by(StockPrice.date.desc())
            .first()
        )
    if row is None:
        return None
    row_date, row_close = row
    return (row_date.date() if hasattr(row_date, "date") else row_date), float(row_close)


def _hit(verdict: str, return_pct: float) -> bool:
    """Did the thesis direction match the realized return sign?

    setup_long: hit iff return_pct > 0
    watch / no_setup: hit iff return_pct <= 0 (LLM was right NOT to buy)
    Unknown verdicts default to False so we never silently mark them hits.
    """
    if verdict == "setup_long":
        return return_pct > 0
    if verdict in ("watch", "no_setup"):
        return return_pct <= 0
    return False


def evaluate_horizon(
    scan_date: date,
    horizon: Literal["1d", "5d", "10d"],
) -> int:
    """Backfill ThesisEvaluation rows for theses produced ``horizon`` days
    before ``scan_date`` that don't already have a row at this horizon.

    Returns the number of rows INSERTed. Skipped (missing-price, missing-
    screened-row) theses are logged at WARNING and counted out — call
    again later once the data lands and the missing rows fill in.

    Idempotent: re-running fills only the gaps thanks to the unique
    constraint on (thesis_id, horizon).
    """
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon!r}; choose from {HORIZONS}")

    days = _HORIZON_DAYS[horizon]
    target_scan_date = scan_date - timedelta(days=days)
    horizon_date = scan_date  # exit price at the eval date itself

    log.info(
        "evaluate_horizon_starting scan_date=%s horizon=%s target_scan_date=%s",
        scan_date,
        horizon,
        target_scan_date,
    )

    inserted = 0
    skipped_no_screened = 0
    skipped_no_prices = 0

    with get_session() as session:
        # Find every thesis from `target_scan_date` with no eval row at this
        # horizon. We join to ScreenedStockRecord so we can read entry-close
        # day metadata (verdict, llm_confidence) without re-parsing JSON.
        stmt = (
            select(
                LLMAnalysisRecord.id,
                LLMAnalysisRecord.target_symbols,
                LLMAnalysisRecord.recommendation,
                LLMAnalysisRecord.confidence,
                LLMAnalysisRecord.signals_used,
                ScreenedStockRecord.id.label("screened_id"),
                ScreenedStockRecord.scan_date,
                ScreenedStockRecord.symbol,
            )
            .join(
                ScreenedStockRecord,
                ScreenedStockRecord.thesis_id == LLMAnalysisRecord.id,
            )
            .outerjoin(
                ThesisEvaluation,
                and_(
                    ThesisEvaluation.thesis_id == LLMAnalysisRecord.id,
                    ThesisEvaluation.horizon == horizon,
                ),
            )
            .where(
                ScreenedStockRecord.scan_date == target_scan_date,
                ThesisEvaluation.id.is_(None),
            )
        )
        rows = session.execute(stmt).all()

    log.info("evaluate_horizon_candidates count=%s", len(rows))

    for row in rows:
        symbol = row.symbol
        verdict = row.recommendation or "no_setup"
        llm_confidence = (
            int(round(row.confidence)) if row.confidence is not None else None
        )

        entry = _close_on_or_before(symbol, target_scan_date)
        exit_ = _close_on_or_before(symbol, horizon_date)
        if entry is None or exit_ is None:
            log.warning(
                "evaluate_horizon_skip_missing_prices symbol=%s scan_date=%s "
                "horizon=%s entry=%s exit=%s",
                symbol,
                target_scan_date,
                horizon,
                entry,
                exit_,
            )
            skipped_no_prices += 1
            continue

        entry_close = entry[1]
        exit_close = exit_[1]
        if entry_close <= 0:
            log.warning(
                "evaluate_horizon_skip_bad_entry_price symbol=%s entry=%s",
                symbol,
                entry_close,
            )
            skipped_no_prices += 1
            continue

        return_pct = (exit_close - entry_close) / entry_close
        hit = _hit(verdict, return_pct)

        with get_session() as session:
            stmt = (
                pg_insert(ThesisEvaluation)
                .values(
                    thesis_id=int(row.id),
                    screened_record_id=int(row.screened_id),
                    horizon=horizon,
                    scan_date=row.scan_date,
                    symbol=symbol,
                    verdict=verdict,
                    llm_confidence=llm_confidence,
                    entry_price=entry_close,
                    exit_price=exit_close,
                    return_pct=return_pct,
                    hit=hit,
                    signals_used=list(row.signals_used or []),
                )
                .on_conflict_do_nothing(
                    index_elements=["thesis_id", "horizon"]
                )
            )
            result = session.execute(stmt)
            if (result.rowcount or 0) > 0:
                inserted += 1

    if skipped_no_screened or skipped_no_prices:
        log.info(
            "evaluate_horizon_skipped no_screened=%s no_prices=%s",
            skipped_no_screened,
            skipped_no_prices,
        )

    log.info(
        "evaluate_horizon_done horizon=%s inserted=%s",
        horizon,
        inserted,
    )
    return inserted


# ---------------------------------------------------------------------------
# Per-signal contribution — Mann-Whitney U on used-vs-absent forward returns
# ---------------------------------------------------------------------------


def _mannwhitneyu_p(
    used: list[float], absent: list[float]
) -> float | None:
    """Two-sided Mann-Whitney U p-value, or None when either bucket is empty.

    scipy.stats.mannwhitneyu raises on empty input — we want to surface that
    as "not enough data" instead.
    """
    if not used or not absent:
        return None
    try:
        from scipy.stats import mannwhitneyu  # local import to keep top fast

        result = mannwhitneyu(used, absent, alternative="two-sided")
        return float(result.pvalue)
    except Exception:
        log.warning("mannwhitneyu_failed", exc_info=True)
        return None


def compute_signal_contribution(
    days: int = 30, horizon: str = "5d"
) -> list[SignalContribution]:
    """For each signal seen in ThesisEvaluation rows over the last ``days``
    at ``horizon``, partition rows into used-vs-absent buckets, compute mean
    return in each, and run Mann-Whitney U for a p-value.

    Returns one SignalContribution per signal name, sorted by descending lift.
    Signals that never appear in the window are not returned.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon!r}; choose from {HORIZONS}")

    cutoff = date.today() - timedelta(days=days)
    with get_session() as session:
        rows = session.execute(
            select(
                ThesisEvaluation.signals_used,
                ThesisEvaluation.return_pct,
            ).where(
                ThesisEvaluation.horizon == horizon,
                ThesisEvaluation.scan_date >= cutoff,
            )
        ).all()

    if not rows:
        return []

    # Aggregate the union of signal names so we know what to score.
    all_names: set[str] = set()
    for sigs, _ret in rows:
        if sigs:
            all_names.update(sigs)

    out: list[SignalContribution] = []
    for name in sorted(all_names):
        used: list[float] = []
        absent: list[float] = []
        for sigs, ret in rows:
            ret_f = float(ret)
            if sigs and name in sigs:
                used.append(ret_f)
            else:
                absent.append(ret_f)
        if not used and not absent:
            continue
        mean_used = sum(used) / len(used) if used else 0.0
        mean_absent = sum(absent) / len(absent) if absent else 0.0
        out.append(
            SignalContribution(
                name=name,
                n_used=len(used),
                n_absent=len(absent),
                mean_used=mean_used,
                mean_absent=mean_absent,
                lift=mean_used - mean_absent,
                p_value=_mannwhitneyu_p(used, absent),
            )
        )

    out.sort(key=lambda s: s.lift, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Per-verdict hit rate — base-rate block for the Discord eval report
# ---------------------------------------------------------------------------


def compute_verdict_hit_rate(days: int = 30) -> dict[str, list[HitRate]]:
    """Return ``{verdict: [HitRate per horizon]}`` for the last ``days`` of
    ThesisEvaluation rows. Verdicts and horizons follow the design's fixed
    set; missing combinations are present in the result with ``n=0``.
    """
    cutoff = date.today() - timedelta(days=days)
    out: dict[str, list[HitRate]] = {v: [] for v in VERDICTS}

    with get_session() as session:
        # Pull aggregates in a single query — count + win count + sum of
        # returns per (verdict, horizon).
        stmt = (
            select(
                ThesisEvaluation.verdict,
                ThesisEvaluation.horizon,
                func.count().label("n"),
                # CASE-based hit counter so the query is portable across
                # SQLite (used in tests) and Postgres (production). SUM on a
                # raw boolean works on Postgres but fails on SQLite.
                func.sum(
                    case((ThesisEvaluation.hit.is_(True), 1), else_=0)
                ).label("hits"),
                func.avg(ThesisEvaluation.return_pct).label("avg_ret"),
            )
            .where(ThesisEvaluation.scan_date >= cutoff)
            .group_by(ThesisEvaluation.verdict, ThesisEvaluation.horizon)
        )
        rows = session.execute(stmt).all()

    by_key: dict[tuple[str, str], tuple[int, int, float]] = {}
    for verdict, horizon, n, hits, avg_ret in rows:
        # `hits` may come back as Decimal/int depending on backend.
        hit_count = int(hits) if hits is not None else 0
        avg = float(avg_ret) if avg_ret is not None else 0.0
        by_key[(verdict, horizon)] = (int(n), hit_count, avg)

    for v in VERDICTS:
        for h in HORIZONS:
            n, hits, avg = by_key.get((v, h), (0, 0, 0.0))
            win_rate = (hits / n) if n > 0 else 0.0
            out[v].append(
                HitRate(
                    verdict=v,
                    horizon=h,
                    n=n,
                    win_rate=win_rate,
                    avg_return_pct=avg,
                )
            )
    return out
