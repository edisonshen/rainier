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

PRICE SOURCE — yfinance, not ``stock_prices``
---------------------------------------------
The replay's prices come from a FRESH yfinance daily fetch (the same source the
live screener used), NOT the local ``stock_prices`` table. ``stock_prices`` is
truncated to ~9-33 bars/symbol — far too short for the detector, which needs the
live screen's ~6-month (~124-bar) window to form swings, support/resistance, and
multi-bar shapes. Replaying it over the stub produced ZERO actionable patterns
for 96 of 117 close rows (a 21/510 recovery). Re-pointing to a yfinance 6-month
as-of window recovers ~116/117 close rows. See ``_fetch_history``.

Two fidelity caveats (NEITHER is look-ahead):

* **Corporate-action / vendor restatement.** A fresh ``auto_adjust=True`` pull
  re-applies every split/dividend that happened AFTER scan_date, so absolute OHLC
  — and the entry/stop/target derived from it — can differ slightly from the
  snapshot the original live screen saw. This is restatement of past bars, not
  future data leaking in; the as-of clip still bars post-scan_date BARS from the
  detector (``_as_of_idx`` + ``window_as_of``).
* **~1% residual type-mismatch.** A re-pull is not bit-identical to the historical
  screen's pull, so a tiny set of rows (~1/117, e.g. AMT 06-03) re-detects as a
  NEIGHBORING pattern type. The type-match gate (``_match_pattern``) leaves those
  NULL rather than write a different-type setup — the don't-guess invariant holds.

THE FIX (idempotent, dry-run by default)
-----------------------------------------
For each ``close``-session ``screened_stocks`` row in ``[from_date, to_date]``
with ``pattern_type NOT NULL AND entry_price IS NULL``:

    replay the pattern detector AS-OF the row's scan_date over a fresh yfinance
      6-month window (reuse paper.pattern_replay.replay_pattern_layer — no
      look-ahead)
    among the as-of ACTIONABLE patterns (in _filter_actionable priority order),
      pick the FIRST whose pattern_type == the row's stored pattern_type — the
      same selection the live screener uses for `best_pattern` (NOT necessarily
      replay's top-ranked `best`, since a matching pattern may rank lower; and
      NOT a confidence re-sort). If NONE matches, leave the row NULL and report
      it in still-NULL — never write wrong-pattern levels.
    coalesce-upsert the four levels via persist_screened_stocks (fills NULL only)

Only ``close``-session rows are reconstructable without look-ahead: the replay
reads the COMPLETED daily bar for scan_date (the last bar ``<= scan_date`` of the
fresh 1d pull), which equals what the live screen saw only at the close session
(earlier sessions that day did not yet have the day's final high/low/close).
Non-``close`` patterned-NULL rows are deliberately left NULL — see
``_BACKFILLABLE_SESSION``.

    ASCII flow (one row):

      screened_stocks row (pattern_type=T, levels NULL, scan_date=d)
            │
            ▼
      yfinance 6mo as-of d ── window AS-OF d ──► replay_pattern_layer
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
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from rainier.core.config import StockScreenerConfig, get_settings
from rainier.core.database import get_session
from rainier.core.models import ScreenedStockRecord
from rainier.core.types import PatternSignal, StockCandidate
from rainier.llm_thesis.persistence import persist_screened_stocks
from rainier.paper.pattern_replay import (
    replay_pattern_layer,
    window_as_of,
)

log = logging.getLogger(__name__)

# How far before the window's earliest scan_date to start the yfinance fetch.
# The detector needs ~6 months of bars to form swings / S-R / multi-bar shapes
# as-of each scan_date. period="6mo" ends TODAY, so for an early-June as-of date
# its left edge would be cut short; an explicit ``start`` ~9 months before the
# earliest scan_date guarantees a full ~6-month window even at the left edge
# (the extra ~3 months absorbs holiday-heavy stretches). The fetch runs through
# today; ``_as_of_idx`` + ``window_as_of`` clip to ``<= scan_date`` per row, so
# post-scan_date bars never reach the detector (no look-ahead).
_FETCH_START_LEAD_MONTHS = 9


@dataclass(slots=True, frozen=True)
class _TargetRow:
    """The subset of ``screened_stocks`` columns this repair actually touches.

    We deliberately do NOT load the full ORM ``ScreenedStockRecord``. A full-row
    load emits ``SELECT *`` over every mapped column — including
    ``bearish_invalidation_level`` (added by migration 0012). On a legacy DB
    where 0012 was never applied, that column is missing and the read crashes
    with ``UndefinedColumn`` even though this repair never reads it. Selecting
    only the columns we use makes an unrelated missing column unable to crash the
    command. Columns: the filter/identity set (``scan_date``, ``session_name``,
    ``symbol``, ``entry_price``, ``pattern_type``) plus the two the upsert
    placeholder carries (``sector``, ``composite_score``).
    """

    symbol: str
    scan_date: date
    session_name: str
    pattern_type: str | None
    sector: str | None
    composite_score: float


@dataclass(slots=True)
class BackfillResult:
    """Counts from one backfill run (dry-run or applied).

    ``scanned`` — target rows (``close``-session + patterned + level-NULL + in
        window) that the replay attempted to recover.
    ``recovered`` — rows whose stored pattern_type matched an as-of actionable
        pattern (levels written when ``apply`` is True; would-be-written on dry-run).
    ``still_null`` — target rows where an as-of price window EXISTED but NO
        actionable pattern matched the stored type / the detector raised (left
        NULL and reported; never errored). This is a PERMANENT no-match: a re-run
        produces the same result, so the operator can stop chasing these.
    ``still_null_keys`` — the (symbol, scan_date) of each still-NULL (no-match) row.
    ``no_price_data`` — target rows whose symbol returned NO usable yfinance bars
        even after the per-symbol solo retry (throttle / outage / delisting), so
        the replay could not run AT ALL. Distinct from ``still_null``: this is a
        TRANSIENT/recoverable failure — a re-run once the fetch succeeds may
        recover the row, so the operator should retry rather than treat it as
        permanently unreconstructable. Kept separate so a fetch blip is never
        silently mislabeled as a permanent no-pattern-match.
    ``no_price_data_keys`` — the (symbol, scan_date) of each no-price-data row.
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
    no_price_data: int = 0
    no_price_data_keys: list[tuple[str, date]] = field(default_factory=list)
    skipped_non_close: int = 0


# The replay reconstructs levels from the COMPLETED daily bar for ``scan_date``
# (the last bar ``<= scan_date`` of the fresh yfinance 1d pull). The live screener
# ran the SAME daily fetch for every session (morning/midday/afternoon/close), but
# only at the ``close`` session is that completed EOD bar the one the live screen
# actually saw. For an earlier session that day, the EOD bar carries the day's
# final high/low/close that were NOT yet available — replaying it would inject
# look-ahead and write levels the original intraday screen never produced. So we
# only backfill ``close``-session rows (faithful by construction); patterned-NULL
# rows from other sessions are left NULL rather than given look-ahead levels
# (worse-to-write-wrong-than-leave-NULL, per the task's fidelity gate).
_BACKFILLABLE_SESSION = "close"


def _target_rows(
    session: Session, from_date: date, to_date: date
) -> list[_TargetRow]:
    """``close``-session, patterned, level-NULL rows in ``[from_date, to_date]``.

    Only ``close``-session rows are reconstructable without look-ahead (see
    ``_BACKFILLABLE_SESSION``). Deterministic order (symbol, scan_date) so a
    re-run scans the corpus identically and logs are diff-stable.

    Column-scoped (NOT a full-ORM load): selecting only the columns this repair
    touches keeps an unrelated missing column (e.g. ``bearish_invalidation_level``
    on a DB without migration 0012) from crashing the read. See ``_TargetRow``.
    """
    stmt = (
        select(
            ScreenedStockRecord.symbol,
            ScreenedStockRecord.scan_date,
            ScreenedStockRecord.session_name,
            ScreenedStockRecord.pattern_type,
            ScreenedStockRecord.sector,
            ScreenedStockRecord.composite_score,
        )
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
    return [
        _TargetRow(
            symbol=r.symbol,
            scan_date=r.scan_date,
            session_name=r.session_name,
            pattern_type=r.pattern_type,
            sector=r.sector,
            composite_score=r.composite_score,
        )
        for r in session.execute(stmt).all()
    ]


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


# The level column migration 0005 added (entry_price etc.) plus the 0012 column
# the write path's upsert names. The READ path is column-scoped so a missing
# 0012 column can't crash a dry-run (see ``_TargetRow``); the WRITE path goes
# through ``persist_screened_stocks``, whose ``INSERT ... ON CONFLICT`` references
# ``bearish_invalidation_level`` directly. On a pre-0012 DB that write aborts with
# an opaque ``UndefinedColumn`` mid-repair. ``--apply`` preflights for the column
# so the operator gets a clear "apply migration 0012 first" message instead.
_APPLY_REQUIRED_COLUMN = "bearish_invalidation_level"


def _apply_column_present(session: Session) -> bool:
    """True iff ``screened_stocks.bearish_invalidation_level`` exists on this DB.

    ``to_regclass`` resolves ``screened_stocks`` on the session's search_path, so
    the catalog lookup targets the same table the write path would hit (not a
    same-named table in another schema). ``NOT attisdropped`` excludes a
    logically-dropped column. The write path's upsert references this column; a
    pre-0012 DB lacks it, so ``--apply`` must refuse before attempting the write.
    """
    row = session.execute(
        text(
            "SELECT 1 FROM pg_attribute "
            "WHERE attrelid = to_regclass('screened_stocks') "
            "AND attname = :c AND NOT attisdropped"
        ),
        {"c": _APPLY_REQUIRED_COLUMN},
    ).first()
    return row is not None


def _normalize_symbol_frame(
    df: pd.DataFrame, min_bars: int
) -> pd.DataFrame | None:
    """Mirror ``stock_screener._fetch_stock_data``'s per-symbol normalization.

    Drop rows missing a Close, skip a frame shorter than ``min_bars``, lowercase
    the OHLCV columns. Returns the normalized frame or None when it is empty /
    too short. Kept byte-identical to the live screener so the replay sees the
    same corpus the live screen would have built (modulo vendor restatement —
    see the module docstring's fidelity caveat).
    """
    try:
        df = df.dropna(subset=["Close"]).copy()
    except KeyError:
        return None
    if len(df) < min_bars:
        return None
    df.columns = [c.lower() for c in df.columns]
    return df


def _fetch_history(
    symbols: list[str], *, start: date, end: date, min_bars: int
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV per symbol from yfinance, sliced ``[start, end]``.

    Re-points the backfill's price corpus from the truncated ``stock_prices``
    table (~9-33 bars/symbol — too short for the detector to form a pattern) to a
    FRESH yfinance window, the same source the live screener uses. Returns a
    ``{symbol: DataFrame}`` map with lowercase OHLCV columns and a tz-naive
    DatetimeIndex — the exact shape ``_as_of_idx``/``window_as_of`` consume, so
    everything downstream runs unchanged.

    Mirrors ``stock_screener._fetch_stock_data`` (interval="1d",
    group_by="ticker", per-symbol dropna+min_bars+lowercase) with two
    deliberate differences:

    * Explicit ``start``/``end`` instead of ``period="6mo"``. ``period`` ends
      TODAY, so for an early-June as-of date its left edge would be cut short of
      the ~6-month window the detector needs. The caller passes a ``start`` ~9
      months before the earliest scan_date so even the left-edge row sees a full
      window.
    * ``auto_adjust=True`` pinned (the screener takes the yfinance default). This
      matches how ``stock_prices`` was ingested (``paper/ingest.py``), keeping the
      adjusted basis consistent. A fresh adjusted pull re-applies post-scan_date
      splits/dividends, so absolute OHLC can differ slightly from the original
      screen's snapshot — corporate-action restatement, NOT look-ahead (the as-of
      clip still bars post-scan_date BARS from the detector).

    Per-symbol retry: a single grouped batch download can transiently drop a
    ticker (observed: LLY returned no data in a batch but 202 bars solo) or the
    whole batch can raise. Any symbol missing from / too short in the batch is
    re-fetched solo before being recorded as no-data, so a network blip is never
    mislabeled still-NULL.
    """
    if not symbols:
        return {}

    def _download(syms: list[str]) -> pd.DataFrame | None:
        try:
            return yf.download(
                syms,
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
            )
        except Exception:
            log.exception("yfinance download failed symbols=%s", syms)
            return None

    def _extract(raw: pd.DataFrame | None, sym: str, batch_len: int) -> pd.DataFrame | None:
        if raw is None or raw.empty:
            return None
        try:
            df = raw.copy() if batch_len == 1 else raw[sym].copy()
        except (KeyError, TypeError):
            return None
        return _normalize_symbol_frame(df, min_bars)

    out: dict[str, pd.DataFrame] = {}
    raw = _download(symbols)
    for sym in symbols:
        df = _extract(raw, sym, len(symbols))
        if df is None:
            # Transient batch drop / failure: retry this symbol solo before
            # treating it as no-data (a network blip must not become still-NULL).
            df = _extract(_download([sym]), sym, 1)
        if df is not None:
            out[sym] = df
    return out


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
    """Positional index of the bar EXACTLY ON ``scan_date``; None if absent.

    Fidelity gate: the live close-session screen ran on a daily yfinance pull
    whose LATEST bar is the scan_date bar (the day the screen executed). A
    faithful replay therefore needs that exact bar as the window's last bar — its
    high/low/close drive the reconstructed levels.

    We deliberately do NOT fall back to the most-recent bar ON-OR-BEFORE
    scan_date. If the yfinance pull has a gap (a delisting, a market holiday the
    DateOffset over-shot, or a dropped bar) for the canonical scan_date, an
    on-or-before fallback would build a window ending on the PRIOR trading day and
    replay stale prior-day levels — writing entry/stop/target/rr that the original
    as-of-scan_date screen never produced. That violates "faithful by
    construction": such a row must stay ``still_null`` (reported, never given
    wrong-day levels), not be silently recovered from the wrong bar. So require an
    EXACT scan_date match.

    Compares on the bar's UTC calendar date so the lookup is robust to whatever
    tz the index carries. yfinance daily bars are tz-NAIVE wall-clock dates
    (the common case here), which localize to UTC unchanged; but the same code
    also handled the prior ``stock_prices`` source's tz-aware ``timestamptz``
    index, where taking the LOCAL date under a behind-UTC session would shift a
    midnight-UTC bar onto the prior calendar day and mis-match every row. Convert
    to UTC FIRST either way, mirroring ``paper.ingest.normalize_to_trading_date``
    (the codebase's read/write-symmetric trading-date rule).
    """
    idx = df.index
    # Convert to UTC before flooring to the calendar date. tz-aware → tz_convert;
    # tz-naive (synthetic fixtures) → already a bare wall-clock, localize to UTC so
    # a midnight stamp keeps its own date instead of being read in some other tz.
    idx_utc = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    bar_dates = idx_utc.normalize().date  # ndarray[date], UTC calendar date
    hits = (bar_dates == scan_date).nonzero()[0]
    if hits.size == 0:
        return None
    # Exactly-one bar per trading day; take it (last guards any duplicate ingest).
    return int(hits[-1])


def _match_for_row(
    row: _TargetRow,
    df: pd.DataFrame | None,
    config: StockScreenerConfig,
) -> PatternSignal | None:
    """Replay the detector as-of ``row.scan_date`` and return the matching pattern.

    Returns None when prices are missing, NO bar exists EXACTLY ON scan_date (a
    yfinance gap — never replay the prior day's stale bar; see ``_as_of_idx``), no
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
    row: _TargetRow, pat: PatternSignal
) -> StockCandidate:
    """Minimal StockCandidate carrying the levels persist_screened_stocks fills.

    persist_screened_stocks coalesce-upserts the level columns and leaves the
    non-level columns untouched on conflict, so the other fields here are
    placeholders that never reach the existing row.

    Levels carried: entry/stop/target/rr PLUS ``bearish_invalidation_level`` for an
    exact ``false_breakout`` (the pattern's ``false_high`` key-point). The upsert
    fills that column too, and ``paper.reclaim.detect_and_enqueue_reclaims`` filters
    on ``bearish_invalidation_level IS NOT NULL`` (within its active ~20-day
    window) — so leaving it NULL would silently make every repaired bearish-trap
    row permanently invisible to reclaim, even though the live screener persists
    it. Mirror the live derivation (``stock_screener.py``: false_breakout-only,
    ``key_points['false_high']``, never ``stop_loss`` which sits a buffer above).
    """
    bearish_invalidation_level: float | None = None
    if pat.pattern_type == "false_breakout" and pat.key_points is not None:
        false_high = pat.key_points.get("false_high")
        if false_high is not None:
            bearish_invalidation_level = float(false_high)

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
        bearish_invalidation_level=bearish_invalidation_level,
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
        # Preflight the WRITE path against schema drift BEFORE any scan work.
        # ``persist_screened_stocks`` upserts ``bearish_invalidation_level`` (added
        # by migration 0012); on a pre-0012 DB that write aborts with an opaque
        # ``UndefinedColumn`` AFTER a full replay. A dry-run on the same drifted DB
        # succeeds (column-scoped read) and tells the operator to re-run with
        # ``--apply`` — so without this guard ``--apply`` would crash mid-repair.
        # Fail fast with a clear, actionable message instead. (Dry-run never
        # writes, so it stays runnable on the drifted DB for diagnosis.)
        if apply and not _apply_column_present(session):
            raise RuntimeError(
                "Cannot --apply: screened_stocks is missing the "
                f"{_APPLY_REQUIRED_COLUMN!r} column (migration 0012 not applied). "
                "Apply migration 0012 (migrations/0012_reclaim_queue.sql) first, "
                "then re-run with --apply. Dry-run (without --apply) works without it."
            )

        # Damaged non-close rows are not repairable (look-ahead) but ARE still
        # damaged — count them first so the report surfaces them even when there
        # are zero repairable close rows (a dry-run must never look complete
        # while non-close NULLs remain).
        result.skipped_non_close = _count_non_close_null(session, from_date, to_date)

        rows = _target_rows(session, from_date, to_date)
        result.scanned = len(rows)
        if not rows:
            return result

    # Fetch prices OUTSIDE the DB session: the corpus now comes from yfinance
    # (the source the live screener used), not the truncated ``stock_prices``
    # table. Load once for the full symbol set, started ~9 months before the
    # earliest scan_date so each as-of window still sees its full ~6-month
    # history, then sliced as-of per row by ``_as_of_idx``/``window_as_of``.
    symbols = sorted({r.symbol for r in rows})
    earliest = min(r.scan_date for r in rows)
    fetch_start = (
        pd.Timestamp(earliest) - pd.DateOffset(months=_FETCH_START_LEAD_MONTHS)
    ).date()
    # Pin the fetch's right edge to to_date (the window's last scan_date), NOT
    # date.today(). The repair never needs a bar past to_date — the as-of clip
    # discards everything after each row's scan_date anyway. Pinning to to_date
    # makes the corpus a pure function of [from_date, to_date], so a dry-run and a
    # later --apply re-pull the SAME window (a wall-clock today would shift the
    # right edge and the split/dividend basis between the two runs, so the
    # previewed levels could differ from the applied ones). yfinance ``end`` is
    # EXCLUSIVE, so add a day to keep the to_date bar in-window (mirrors
    # paper.ingest's ``end + timedelta(days=1)``); _as_of_idx still requires the
    # exact scan_date bar, so this only guarantees inclusion, never look-ahead.
    fetch_end = to_date + timedelta(days=1)
    prices = _fetch_history(
        symbols, start=fetch_start, end=fetch_end, min_bars=config.min_daily_bars
    )

    # Replay + persist run with NO open DB session: the prices come from
    # yfinance, and ``persist_screened_stocks`` manages its own session.
    candidates_by_key: dict[tuple[date, str], list[StockCandidate]] = {}
    for row in rows:
        df = prices.get(row.symbol)

        # Distinguish "the fetch returned NO bars for this symbol" (transient /
        # recoverable: throttle, outage, delisting — a re-run may recover it)
        # from "we HAD a price window but no actionable pattern matched the stored
        # type" (permanent no-match — re-running changes nothing). Conflating them
        # would let a fetch blip during the apply run masquerade as a permanent
        # no-match, so the operator marks a recoverable row unreconstructable.
        if df is None or df.empty:
            result.no_price_data += 1
            result.no_price_data_keys.append((row.symbol, row.scan_date))
            log.info(
                "backfill_no_price_data symbol=%s scan_date=%s pattern_type=%s",
                row.symbol,
                row.scan_date,
                row.pattern_type,
            )
            continue

        matched = _match_for_row(row, df, config)

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
        "backfill_screened_levels done scanned=%d recovered=%d still_null=%d "
        "no_price_data=%d apply=%s",
        result.scanned,
        result.recovered,
        result.still_null,
        result.no_price_data,
        apply,
    )
    return result
