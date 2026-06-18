"""One-time backfill of historical ``screened_stocks`` pattern trade levels.

PROBLEM
-------
``screened_stocks`` stores the pattern-derived trade levels (entry, stop,
target, reward/risk) per screened name. These power the miss-sweep's level-based
attribution, the R-D chart overlays (entry/stop/target lines), and any later
analysis joining levels to outcomes.

The go-forward screen->persist path has written these levels correctly since the
06-15 cutover. But ~510 patterned rows from scan_date 06-03..06-12 carry NULL
levels: the running scraper only picked up the level-writing code at the 06-15
deploy. ``pattern_type`` IS recorded on those rows; entry/stop/target/rr are NULL.
This is a one-time DATA backfill, not a code fix.

THE FIX (idempotent, dry-run by default)
-----------------------------------------
For each ``close``-session ``screened_stocks`` row in ``[from_date, to_date]``
with ``pattern_type NOT NULL AND entry_price IS NULL``:

    replay the pattern detector AS-OF the row's scan_date over stored prices
      (reuse paper.pattern_replay.replay_pattern_layer — no look-ahead)
    among the as-of ACTIONABLE patterns (in _filter_actionable priority order),
      pick the FIRST whose pattern_type == the row's stored pattern_type — the
      same selection the live screener uses for `best_pattern` (NOT necessarily
      replay's top-ranked `best`, since a matching pattern may rank lower; and
      NOT a confidence re-sort). If NONE matches, leave the row NULL and report
      it in still-NULL — never write wrong-pattern levels.
    coalesce-upsert the four levels via persist_screened_stocks (fills NULL only)

Only ``close``-session rows are reconstructable without look-ahead: the replay
reads the COMPLETED daily ``stock_prices`` bar for scan_date, which equals what
the live screen saw only at the close session (earlier sessions that day did not
yet have the day's final high/low/close). Non-``close`` patterned-NULL rows are
deliberately left NULL — see ``_BACKFILLABLE_SESSION``.

    ASCII flow (one row):

      screened_stocks row (pattern_type=T, levels NULL, scan_date=d)
            │
            ▼
      load stock_prices[symbol] ── window AS-OF d ──► replay_pattern_layer
            │
            ▼
      actionable patterns (priority order) ── FIRST with pattern_type == T
            ──► matched pattern.{entry, stop, target_wave1, rr_ratio}
            │
            ▼  (apply only)
      persist_screened_stocks(coalesce upsert) ── fills the NULL levels

Levels come from the actionable pattern matching the stored type, so they are
faithful by construction (the replay is parity-tested against ``screen_stocks``).

This is a maintenance command (``rainier db backfill-screened-levels``), NOT a
scheduled job — a historical repair.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rainier.core.config import StockScreenerConfig, get_settings
from rainier.core.database import get_session
from rainier.core.models import ScreenedStockRecord
from rainier.core.types import PatternSignal, StockCandidate
from rainier.llm_thesis.persistence import persist_screened_stocks
from rainier.paper.pattern_replay import (
    LIVE_LOOKBACK_MONTHS,
    load_prices,
    replay_pattern_layer,
    window_as_of,
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class BackfillResult:
    """Counts from one backfill run (dry-run or applied).

    ``scanned`` — target rows (``close``-session + patterned + level-NULL + in
        window) that the replay attempted to recover.
    ``recovered`` — rows whose stored pattern_type matched an as-of actionable
        pattern (levels written when ``apply`` is True; would-be-written on dry-run).
    ``still_null`` — target rows where NO actionable pattern matched the stored
        type / the detector raised (left NULL and reported; never errored).
    ``still_null_keys`` — the (symbol, scan_date) of each still-NULL row.
    ``skipped_non_close`` — patterned level-NULL rows in the window from a
        non-``close`` session. These are NOT reconstructable without look-ahead
        (see ``_BACKFILLABLE_SESSION``) so they are excluded from the target set,
        but they ARE still damaged rows — surfaced here so a dry-run cannot look
        falsely complete while non-close NULLs remain.
    """

    scanned: int = 0
    recovered: int = 0
    still_null: int = 0
    still_null_keys: list[tuple[str, date]] = field(default_factory=list)
    skipped_non_close: int = 0


# The replay reconstructs levels from the COMPLETED daily ``stock_prices`` bar
# for ``scan_date``. The live screener runs the SAME ``yf.download(period="6mo")``
# daily fetch for every session (morning/midday/afternoon/close), but only at the
# ``close`` session is the completed EOD bar the one the live screen actually saw.
# For an earlier session that day, the stored EOD bar carries the day's final
# high/low/close that were NOT yet available — replaying it would inject
# look-ahead and write levels the original intraday screen never produced. So we
# only backfill ``close``-session rows (faithful by construction); patterned-NULL
# rows from other sessions are left NULL rather than given look-ahead levels
# (worse-to-write-wrong-than-leave-NULL, per the task's fidelity gate).
_BACKFILLABLE_SESSION = "close"


def _target_rows(
    session: Session, from_date: date, to_date: date
) -> list[ScreenedStockRecord]:
    """``close``-session, patterned, level-NULL rows in ``[from_date, to_date]``.

    Only ``close``-session rows are reconstructable without look-ahead (see
    ``_BACKFILLABLE_SESSION``). Deterministic order (symbol, scan_date) so a
    re-run scans the corpus identically and logs are diff-stable.
    """
    stmt = (
        select(ScreenedStockRecord)
        .where(
            ScreenedStockRecord.scan_date >= from_date,
            ScreenedStockRecord.scan_date <= to_date,
            ScreenedStockRecord.session_name == _BACKFILLABLE_SESSION,
            ScreenedStockRecord.pattern_type.isnot(None),
            ScreenedStockRecord.entry_price.is_(None),
        )
        .order_by(
            ScreenedStockRecord.symbol.asc(),
            ScreenedStockRecord.scan_date.asc(),
        )
    )
    return list(session.execute(stmt).scalars().all())


def _count_non_close_null(
    session: Session, from_date: date, to_date: date
) -> int:
    """Patterned, level-NULL rows in the window from a NON-``close`` session.

    These are excluded from the repair (not reconstructable without look-ahead)
    but are still damaged rows; counting them keeps a dry-run from looking
    complete while non-close NULLs remain.
    """
    stmt = (
        select(func.count())
        .select_from(ScreenedStockRecord)
        .where(
            ScreenedStockRecord.scan_date >= from_date,
            ScreenedStockRecord.scan_date <= to_date,
            ScreenedStockRecord.session_name != _BACKFILLABLE_SESSION,
            ScreenedStockRecord.pattern_type.isnot(None),
            ScreenedStockRecord.entry_price.is_(None),
        )
    )
    return int(session.execute(stmt).scalar_one())


def _match_pattern(
    actionable: list[PatternSignal], stored_type: str
) -> PatternSignal | None:
    """Pick the actionable pattern whose type == ``stored_type``.

    ``actionable`` arrives in ``_filter_actionable`` priority order (the same
    order the live screener uses to pick ``best_pattern = actionable[0]``). When
    several patterns share ``stored_type``, take the FIRST in that order — i.e.
    the one the live screener would have written had that type been the day's
    best. We deliberately do NOT re-sort by confidence: the live path writes
    levels by actionability priority, not confidence (stock_screener.py:92-93,
    "Best = first actionable (sorted by priority), not highest confidence"), so
    re-sorting here could persist a different same-type setup than the original
    screen. Iterating the already-sorted list preserves that priority and is
    reproducible across runs.
    """
    for p in actionable:
        if p.pattern_type == stored_type:
            return p
    return None


def _as_of_idx(df: pd.DataFrame, scan_date: date) -> int | None:
    """Positional index of the last bar on/before ``scan_date``; None if none.

    The screener runs on the bars available as-of the scan, whose latest bar is
    the most recent trading day on or before scan_date.
    """
    cutoff = pd.Timestamp(scan_date)
    if df.index.tz is not None:
        cutoff = cutoff.tz_localize(df.index.tz)
    mask = pd.Index(df.index <= cutoff).to_numpy()
    hits = mask.nonzero()[0]
    if hits.size == 0:
        return None
    # positional index of the last bar on/before scan_date
    return int(hits[-1])


def _match_for_row(
    row: ScreenedStockRecord,
    df: pd.DataFrame | None,
    config: StockScreenerConfig,
) -> PatternSignal | None:
    """Replay the detector as-of ``row.scan_date`` and return the matching pattern.

    Returns None when prices are missing, no bar exists on/before scan_date, no
    as-of actionable pattern has the row's stored ``pattern_type``, OR the
    detector raises on this symbol's window. A per-row detector failure is
    per-symbol noise (one bad price window must not abort a multi-day repair) —
    ``screen_stocks`` treats it the same way (stock_screener.py:86-89). The row
    is left NULL and reported in still-NULL, never errored.
    """
    if df is None or df.empty:
        return None
    t_idx = _as_of_idx(df, row.scan_date)
    if t_idx is None:
        return None
    windowed = window_as_of(df, t_idx)
    if windowed.empty:
        return None
    try:
        actionable, _ = replay_pattern_layer(row.symbol, windowed, config)
    except Exception:
        log.exception(
            "backfill_replay_failed symbol=%s scan_date=%s pattern_type=%s",
            row.symbol,
            row.scan_date,
            row.pattern_type,
        )
        return None
    return _match_pattern(actionable, row.pattern_type)


def _candidate_with_levels(
    row: ScreenedStockRecord, pat: PatternSignal
) -> StockCandidate:
    """Minimal StockCandidate carrying ONLY the four levels to backfill.

    persist_screened_stocks coalesce-upserts the level columns and leaves the
    non-level columns untouched on conflict, so the other fields here are
    placeholders that never reach the existing row.
    """
    return StockCandidate(
        symbol=row.symbol,
        rank=0,
        rank_change=0,
        long_short="",
        capital_flow_direction="N",
        sector=row.sector or "",
        signal_strength=row.composite_score,
        pattern_type=row.pattern_type,
        entry_price=pat.entry_price,
        stop_loss=pat.stop_loss,
        target_price=pat.target_wave1,
        rr_ratio=pat.rr_ratio,
    )


def backfill_screened_levels(
    *,
    from_date: date,
    to_date: date,
    apply: bool = False,
    config: StockScreenerConfig | None = None,
    config_overrides: dict | None = None,
) -> BackfillResult:
    """Replay the pattern detector as-of each historical scan_date and fill levels.

    Dry-run by default (``apply=False``): scans and reports counts, writes nothing.
    With ``apply=True`` it coalesce-upserts the matched levels (fills NULL only,
    never clobbers a set value).

    ``config`` / ``config_overrides`` let a caller (or test) pin the detector
    knobs; otherwise the live ``StockScreenerConfig`` from settings is used.
    """
    if from_date > to_date:
        raise ValueError(f"from_date {from_date} > to_date {to_date}")
    if config is None:
        config = get_settings().stock_screener
    if config_overrides:
        config = config.model_copy(update=config_overrides)

    result = BackfillResult()

    with get_session() as session:
        # Damaged non-close rows are not repairable (look-ahead) but ARE still
        # damaged — count them first so the report surfaces them even when there
        # are zero repairable close rows (a dry-run must never look complete
        # while non-close NULLs remain).
        result.skipped_non_close = _count_non_close_null(session, from_date, to_date)

        rows = _target_rows(session, from_date, to_date)
        result.scanned = len(rows)
        if not rows:
            return result

        # Load prices once for the full symbol set, left-padded by the detector
        # lookback so each as-of window still sees its full ~6-month history.
        symbols = sorted({r.symbol for r in rows})
        earliest = min(r.scan_date for r in rows)
        # Left-pad by 2x the ~6-month detector lookback so the earliest as-of bar
        # still sees its full window even across holiday-heavy stretches. The
        # bound is tz-aware UTC: ``StockPrice.date`` is ``timestamptz``, so a
        # naive bound would be interpreted in the Postgres session's TimeZone
        # GUC (environment-dependent). Mirror pattern_audit's UTC construction.
        start_date = pd.Timestamp(
            (pd.Timestamp(earliest) - pd.DateOffset(months=2 * LIVE_LOOKBACK_MONTHS)).date(),
            tz="UTC",
        )
        prices = load_prices(session, symbols, start_date=start_date)

        candidates_by_key: dict[tuple[date, str], list[StockCandidate]] = {}
        for row in rows:
            matched = _match_for_row(row, prices.get(row.symbol), config)

            if matched is None:
                result.still_null += 1
                result.still_null_keys.append((row.symbol, row.scan_date))
                log.info(
                    "backfill_still_null symbol=%s scan_date=%s pattern_type=%s",
                    row.symbol,
                    row.scan_date,
                    row.pattern_type,
                )
                continue

            result.recovered += 1
            if apply:
                cand = _candidate_with_levels(row, matched)
                candidates_by_key.setdefault(
                    (row.scan_date, row.session_name), []
                ).append(cand)

        if apply and candidates_by_key:
            for (scan_date, session_name), cands in candidates_by_key.items():
                persist_screened_stocks(cands, scan_date, session_name)

    log.info(
        "backfill_screened_levels done scanned=%d recovered=%d still_null=%d apply=%s",
        result.scanned,
        result.recovered,
        result.still_null,
        apply,
    )
    return result
