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
For each ``screened_stocks`` row in ``[from_date, to_date]`` with
``pattern_type NOT NULL AND entry_price IS NULL``:

    replay the pattern detector AS-OF the row's scan_date over stored prices
      (reuse paper.pattern_replay.replay_pattern_layer — no look-ahead)
    among the as-of ACTIONABLE patterns, pick the one whose pattern_type ==
      the row's stored pattern_type  (NOT necessarily replay's top-ranked
      `best` — a matching pattern may rank lower).  If NONE matches, leave the
      row NULL and report it in still-NULL — never write wrong-pattern levels.
    coalesce-upsert the four levels via persist_screened_stocks (fills NULL only)

    ASCII flow (one row):

      screened_stocks row (pattern_type=T, levels NULL, scan_date=d)
            │
            ▼
      load stock_prices[symbol] ── window AS-OF d ──► replay_pattern_layer
            │
            ▼
      actionable patterns ── filter pattern_type == T ── tie-break (conf, then
            entry) ──► matched pattern.{entry, stop, target_wave1, rr_ratio}
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
from sqlalchemy import select
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

    ``scanned`` — target rows (patterned + level-NULL + in window).
    ``recovered`` — rows whose stored pattern_type matched an as-of actionable
        pattern (levels written when ``apply`` is True; would-be-written on dry-run).
    ``still_null`` — target rows where NO actionable pattern matched the stored
        type (left NULL and reported; never errored).
    ``still_null_keys`` — the (symbol, scan_date) of each still-NULL row.
    """

    scanned: int = 0
    recovered: int = 0
    still_null: int = 0
    still_null_keys: list[tuple[str, date]] = field(default_factory=list)


def _target_rows(
    session: Session, from_date: date, to_date: date
) -> list[ScreenedStockRecord]:
    """Patterned + level-NULL rows in ``[from_date, to_date]`` (inclusive).

    Deterministic order (symbol, scan_date, session_name) so a re-run scans the
    corpus identically and logs are diff-stable.
    """
    stmt = (
        select(ScreenedStockRecord)
        .where(
            ScreenedStockRecord.scan_date >= from_date,
            ScreenedStockRecord.scan_date <= to_date,
            ScreenedStockRecord.pattern_type.isnot(None),
            ScreenedStockRecord.entry_price.is_(None),
        )
        .order_by(
            ScreenedStockRecord.symbol.asc(),
            ScreenedStockRecord.scan_date.asc(),
            ScreenedStockRecord.session_name.asc(),
        )
    )
    return list(session.execute(stmt).scalars().all())


def _match_pattern(
    actionable: list[PatternSignal], stored_type: str
) -> PatternSignal | None:
    """Pick the actionable pattern whose type == ``stored_type``.

    If several share the type, tie-break deterministically: highest confidence
    first, then lowest entry_price (a stable numeric key) so the choice is
    reproducible across runs.
    """
    matches = [p for p in actionable if p.pattern_type == stored_type]
    if not matches:
        return None
    matches.sort(key=lambda p: (-p.confidence, p.entry_price))
    return matches[0]


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
        rows = _target_rows(session, from_date, to_date)
        result.scanned = len(rows)
        if not rows:
            return result

        # Load prices once for the full symbol set, left-padded by the detector
        # lookback so each as-of window still sees its full ~6-month history.
        symbols = sorted({r.symbol for r in rows})
        earliest = min(r.scan_date for r in rows)
        # Left-pad by 2x the ~6-month detector lookback so the earliest as-of bar
        # still sees its full window even across holiday-heavy stretches.
        start_date = pd.Timestamp(earliest) - pd.DateOffset(
            months=2 * LIVE_LOOKBACK_MONTHS
        )
        prices = load_prices(session, symbols, start_date=start_date)

        candidates_by_key: dict[tuple[date, str], list[StockCandidate]] = {}
        for row in rows:
            df = prices.get(row.symbol)
            matched: PatternSignal | None = None
            if df is not None and not df.empty:
                t_idx = _as_of_idx(df, row.scan_date)
                if t_idx is not None:
                    windowed = window_as_of(df, t_idx)
                    if not windowed.empty:
                        actionable, _ = replay_pattern_layer(
                            row.symbol, windowed, config
                        )
                        matched = _match_pattern(actionable, row.pattern_type)

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
