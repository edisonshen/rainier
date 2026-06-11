"""Per-(symbol, date) gap-aware daily price ingest (Phase 0, design D9).

Why a fresh path (NOT `qu100_portfolio._save_prices_to_db`, parallel-review P2):
the backtest saver is `ON CONFLICT DO NOTHING` + *symbol-level* missing
detection — a symbol with stale rows is treated as covered and never gets
today's bar (the gap-blindness bug), and a re-fetched split-adjusted bar is
silently dropped (split-no-self-heal bug). This ingest does:

* **per-(symbol, date) gap detection** over a recent trading-day window;
* **canonical daily-bar tz normalization** — every bar maps to the trading-date
  at 00:00 UTC, applied identically on the gap-check read and the upsert write,
  so yfinance timestamps that differ by intraday/tz (incl. a DST-transition day,
  or an instant that lands on the adjacent calendar date) collapse to ONE row;
* **`(symbol, date)` upsert (DO UPDATE)** so adjusted/split values overwrite
  together (o/h/l/c/v as a set), while a partial/NaN re-fetch never NULLs out a
  previously-good value;
* graceful skip of yfinance-missing symbols (persisted absence, not a crash).

The yfinance call is injected (`fetch_fn`) so tests are deterministic.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rainier.core.database import get_session
from rainier.core.models import Stock, StockPrice

from .calendar import DEFAULT_CALENDAR, TradingCalendar

log = logging.getLogger(__name__)

# A single daily bar, normalized: trading-date + OHLCV (any of o/h/l/c/v may be
# None for a NULL/no-data day). `date` is a plain calendar `date` here; the
# upsert canonicalizes it to 00:00 UTC.
Bar = dict[str, Any]

# fetch_fn(symbols, start, end) -> {symbol: [Bar, ...]} (long-form, per symbol).
FetchFn = Callable[[list[str], date, date], dict[str, list[Bar]]]


def canonical_instant(d: date) -> datetime:
    """Map a trading date to its canonical storage instant: 00:00 UTC.

    `StockPrice.date` is `DateTime(timezone=True)` unique on (symbol, date);
    bars must collapse to one deterministic instant regardless of the source
    timestamp's time-of-day or tz (D9 / TEST-SPEC B4).
    """
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def normalize_to_trading_date(ts: Any) -> date:
    """Coerce an arbitrary bar timestamp to its intended trading date.

    Accepts a `date`, a `datetime` (aware or naive), or anything with
    `.year/.month/.day`. An aware datetime is converted to UTC first, then the
    *calendar date in UTC* is taken — applied identically on read and write so
    a naive tz-localize/tz-convert mix can't drift a bar onto the adjacent day.
    """
    if isinstance(ts, datetime):
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc)
        return date(ts.year, ts.month, ts.day)
    if isinstance(ts, date):
        return ts
    # pandas.Timestamp et al.
    return date(ts.year, ts.month, ts.day)


def _recent_window(as_of: date, window_days: int, cal: TradingCalendar) -> list[date]:
    """The last ``window_days`` trading sessions ending at-or-before ``as_of``."""
    sessions: list[date] = []
    cur = as_of
    # Walk back day by day collecting sessions until we have window_days.
    guard = 0
    while len(sessions) < window_days and guard < window_days * 4 + 14:
        if cal.is_session(cur):
            sessions.append(cur)
        cur -= timedelta(days=1)
        guard += 1
    return sorted(sessions)


def _existing_dates(session, symbols: list[str], start: date, end: date) -> set[tuple[str, date]]:
    # A (symbol, date) counts as PRESENT only if it has usable OHLC. A transient
    # NULL/NaN row (yfinance partial) must still register as a GAP so a later
    # ingest re-fetches the real bar (codex) — otherwise pending fills/exits/MTM
    # stay starved on stale empty rows. `close` is the canonical usability probe;
    # a real daily bar always carries all of o/h/l/c, and the COALESCE upsert
    # (B5) never nulls a previously-good value back out.
    rows = session.execute(
        select(StockPrice.symbol, StockPrice.date).where(
            StockPrice.symbol.in_(symbols),
            StockPrice.date >= canonical_instant(start),
            StockPrice.date <= canonical_instant(end),
            StockPrice.open.isnot(None),
            StockPrice.high.isnot(None),
            StockPrice.low.isnot(None),
            StockPrice.close.isnot(None),
        )
    ).all()
    return {(sym, normalize_to_trading_date(d)) for sym, d in rows}


def _upsert_bar(session, symbol: str, bar: Bar) -> None:
    """Upsert one normalized bar. DO UPDATE overwrites o/h/l/c/v together, but a
    NaN/None field uses COALESCE(new, old) so a partial re-fetch never clobbers
    a previously-good value (B5)."""
    trading_date = normalize_to_trading_date(bar["date"])
    instant = canonical_instant(trading_date)
    values = {
        "symbol": symbol,
        "date": instant,
        "open": bar.get("open"),
        "high": bar.get("high"),
        "low": bar.get("low"),
        "close": bar.get("close"),
        "volume": bar.get("volume"),
    }
    stmt = pg_insert(StockPrice).values(**values)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_stock_price_symbol_date",
        set_={
            # COALESCE(EXCLUDED, existing): a non-null re-fetch overwrites; a
            # null re-fetch keeps the old good value (B5).
            "open": func.coalesce(stmt.excluded.open, StockPrice.open),
            "high": func.coalesce(stmt.excluded.high, StockPrice.high),
            "low": func.coalesce(stmt.excluded.low, StockPrice.low),
            "close": func.coalesce(stmt.excluded.close, StockPrice.close),
            "volume": func.coalesce(stmt.excluded.volume, StockPrice.volume),
        },
    )
    session.execute(stmt)


def ingest_prices(
    symbols: list[str],
    *,
    as_of: date,
    fetch_fn: FetchFn,
    window_days: int = 10,
    calendar: TradingCalendar | None = None,
) -> dict[str, int]:
    """Gap-aware ingest for ``symbols`` over the recent ``window_days`` window.

    Returns ``{"upserted": n}``. Idempotent: a re-run with no new bars upserts
    bars that re-confirm existing rows but adds no NEW (symbol, date) pairs.
    """
    cal = calendar or DEFAULT_CALENDAR
    if not symbols:
        return {"upserted": 0}

    symbols = sorted(set(symbols))
    window = _recent_window(as_of, window_days, cal)
    if not window:
        return {"upserted": 0}
    start, end = window[0], window[-1]
    window_set = set(window)

    with get_session() as session:
        have = _existing_dates(session, symbols, start, end)

    # Per-(symbol, date) gaps: a date in the window with no row for this symbol.
    gaps: dict[str, set[date]] = {}
    for sym in symbols:
        missing = {d for d in window_set if (sym, d) not in have}
        if missing:
            gaps[sym] = missing

    if not gaps:
        log.info("ingest_no_gaps symbols=%d", len(symbols))
        return {"upserted": 0}

    # Re-fetch the whole recent window for symbols that have ANY gap, so split
    # adjustments to already-present dates self-heal too.
    fetch_symbols = sorted(gaps)
    fetched = fetch_fn(fetch_symbols, start, end)

    upserted = 0
    with get_session() as session:
        # Ensure parent `stocks` rows exist (FK on stock_prices.symbol).
        existing_stocks = {
            r[0]
            for r in session.execute(
                select(Stock.symbol).where(Stock.symbol.in_(fetch_symbols))
            ).all()
        }
        for sym in fetch_symbols:
            if sym not in existing_stocks:
                session.add(Stock(symbol=sym))
        session.flush()

        for sym in fetch_symbols:
            bars = fetched.get(sym) or []
            if not bars:
                # yfinance returned nothing for this symbol — skip (persisted
                # absence), others still ingest (B7).
                log.warning("ingest_symbol_empty symbol=%s", sym)
                continue
            for bar in bars:
                td = normalize_to_trading_date(bar["date"])
                if td not in window_set:
                    continue
                _upsert_bar(session, sym, bar)
                upserted += 1

    log.info("ingest_done symbols=%d upserted=%d", len(fetch_symbols), upserted)
    return {"upserted": upserted}


# ---------------------------------------------------------------------------
# Universe selectors
# ---------------------------------------------------------------------------


def active_symbols() -> list[str]:
    """`--universe active` = paper_trade rows with status IN ('pending','open')."""
    from rainier.core.models import PaperTrade

    with get_session() as session:
        rows = session.execute(
            select(PaperTrade.symbol)
            .where(PaperTrade.status.in_(("pending", "open")))
            .distinct()
        ).all()
    return sorted({r[0] for r in rows})


def screened_symbols(as_of: date) -> list[str]:
    """`--universe screened` = today's top-50 screened symbols."""
    from rainier.core.models import ScreenedStockRecord

    with get_session() as session:
        rows = session.execute(
            select(ScreenedStockRecord.symbol)
            .where(ScreenedStockRecord.scan_date == as_of)
            .distinct()
        ).all()
    return sorted({r[0] for r in rows})


def get_current_qu100_cohort(as_of: date) -> list[dict[str, Any]]:
    """The current QU100 cohort at-or-before ``as_of`` (shared API — the weekly
    miss-sweep uses it now; PR 5 reuses it for the daily ingest universe).

    Selection (TASK-PLAN qu100-miss-sweep-6b63 acceptance 1): within
    ``ranking_type='top100'`` pick the latest ``data_date <= as_of``, then the
    latest ``captured_at`` within that data_date (the freshest capture batch of
    the day — a backfill that stamped older data_dates with the same
    captured_at can't win because data_date filters first). Live runs pass
    today; ``--regenerate`` passes the report's date and gets the historical
    cohort. Do NOT use the all-history CLI ``qu100`` selector (every symbol
    ever seen) — this is the point-in-time membership.

    Returns ``[{symbol, rank, data_date, captured_at}]`` ordered by rank;
    ``[]`` when no top100 snapshot exists at-or-before ``as_of``.
    """
    from rainier.core.models import MoneyFlowSnapshot

    with get_session() as session:
        max_dd = session.execute(
            select(func.max(MoneyFlowSnapshot.data_date)).where(
                MoneyFlowSnapshot.ranking_type == "top100",
                MoneyFlowSnapshot.data_date <= as_of,
            )
        ).scalar()
        if max_dd is None:
            return []
        max_cap = session.execute(
            select(func.max(MoneyFlowSnapshot.captured_at)).where(
                MoneyFlowSnapshot.ranking_type == "top100",
                MoneyFlowSnapshot.data_date == max_dd,
            )
        ).scalar()
        rows = session.execute(
            select(
                MoneyFlowSnapshot.symbol,
                MoneyFlowSnapshot.rank,
                MoneyFlowSnapshot.data_date,
                MoneyFlowSnapshot.captured_at,
            )
            .where(
                MoneyFlowSnapshot.ranking_type == "top100",
                MoneyFlowSnapshot.data_date == max_dd,
                MoneyFlowSnapshot.captured_at == max_cap,
            )
            .order_by(MoneyFlowSnapshot.rank)
        ).all()
    return [
        {"symbol": r[0], "rank": int(r[1]), "data_date": r[2], "captured_at": r[3]}
        for r in rows
    ]


def _yfinance_fetch_fn(symbols: list[str], start: date, end: date) -> dict[str, list[Bar]]:
    """Production fetch: yfinance → per-symbol long-form bars (NaN → None)."""
    import pandas as pd
    import yfinance as yf

    out: dict[str, list[Bar]] = {}
    batch_size = 20
    total = math.ceil(len(symbols) / batch_size)
    end_buffered = end + timedelta(days=1)
    for bi in range(0, len(symbols), batch_size):
        batch = symbols[bi : bi + batch_size]
        df = yf.download(
            " ".join(batch),
            start=start.isoformat(),
            end=end_buffered.isoformat(),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if df is None or df.empty:
            continue
        if not isinstance(df.columns, pd.MultiIndex) and len(batch) == 1:
            df.columns = pd.MultiIndex.from_product([df.columns, batch])
        for sym in batch:
            try:
                close = df["Close"][sym]
            except (KeyError, TypeError):
                continue
            bars: list[Bar] = []
            for idx in close.index:
                def _v(field: str):
                    try:
                        val = df[field][sym].loc[idx]
                    except (KeyError, TypeError):
                        return None
                    return None if pd.isna(val) else float(val)

                vol = _v("Volume")
                bars.append(
                    {
                        "date": idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx,
                        "open": _v("Open"),
                        "high": _v("High"),
                        "low": _v("Low"),
                        "close": _v("Close"),
                        "volume": int(vol) if vol is not None else None,
                    }
                )
            out[sym] = bars
        log.info("yf_batch_done batch=%d total=%d", bi // batch_size + 1, total)
    return out
