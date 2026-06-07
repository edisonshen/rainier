"""Neon (canonical ``market.*``) data loaders for the dashboard renderers.

WHY THIS EXISTS
---------------
The ETF-ranks and market-breadth dashboards used to render exclusively from
local parquet caches (``data/cache/thematic_features_daily.parquet``,
``data/cache/sp500_breadth_daily.parquet``, a SPY history parquet). Those
caches lag or go missing whenever their parquet-updater cron is disabled or
slips — on 2026-06-05 the ETF dashboard published an *empty* page at 12:40
even though Neon held that day's data. The canonical ``market.*`` store on
Neon is kept fresh by other jobs (incl. PR #131's backfill-spy), so it is the
source of truth.

This module loads the SAME DataFrame shapes the pure render functions already
expect — only the data SOURCE changes (Neon instead of parquet). The pure
``render_*_html`` functions are untouched.

    Neon market.thematic_features_daily ──▶ load_etf_features_neon ──┐
    Neon market.breadth_indicator_daily ──▶ load_breadth_neon ───────┼─▶ pure render_*_html
    Neon market.benchmark_ohlcv (SPY)   ──▶ load_spy_neon ───────────┘

CRITICAL — .env loading
-----------------------
``rainier.db.engine.get_engine()`` reads ``os.environ["DATABASE_URL"]``
directly and does NOT call ``load_dotenv()``. The cron wrappers run
``uv run rainier ...`` from the project dir where ``.env`` lives, but the env
var is only present once ``.env`` is loaded. So the CLI commands MUST call
``ensure_env_loaded()`` (which invokes ``get_settings()`` → ``load_dotenv()``)
BEFORE building the engine, or the Neon path raises "DATABASE_URL not set".

FAIL LOUD
---------
Every loader raises ``EmptyNeonResultError`` when the resolved asof has zero rows.
We never silently fall back to a stale parquet — that is exactly the failure
mode this change is meant to kill. The CLI converts the exception into a
non-zero exit so the cron chain's ``&&`` propagates the failure to the
Discord alert.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

__all__ = [
    "EmptyNeonResultError",
    "ensure_env_loaded",
    "latest_etf_asof",
    "latest_breadth_asof",
    "load_etf_features_neon",
    "load_breadth_neon",
    "load_spy_neon",
]


class EmptyNeonResultError(RuntimeError):
    """Raised when a Neon query returns zero usable rows for the asof.

    Caught by the CLI render commands and turned into a non-zero exit with a
    clear message — never a silent fall-back to a stale parquet.
    """


def ensure_env_loaded() -> None:
    """Load ``.env`` so ``DATABASE_URL`` is present before ``get_engine()``.

    ``get_settings()`` calls ``load_dotenv()`` as a side effect (see
    ``core/config.py``). We rely on that single, already-tested loader rather
    than importing ``dotenv`` here so there is one source of truth for how
    rainier discovers its env.
    """
    from rainier.core.config import get_settings

    get_settings()


# ---------------------------------------------------------------------------
# asof resolution — mirror "latest parquet row wins"
# ---------------------------------------------------------------------------


def latest_etf_asof(engine: Engine) -> date:
    """Return ``max(asof_date)`` from ``market.thematic_features_daily``.

    Raises ``EmptyNeonResultError`` when the table is empty (no asof to render).
    """
    sql = text("SELECT max(asof_date) AS m FROM market.thematic_features_daily")
    with engine.connect() as conn:
        value = conn.execute(sql).scalar_one_or_none()
    if value is None:
        raise EmptyNeonResultError(
            "market.thematic_features_daily is empty — no asof_date to render"
        )
    return _as_date(value)


def latest_breadth_asof(engine: Engine) -> date:
    """Return ``max(asof_date)`` from ``market.breadth_indicator_daily``.

    Raises ``EmptyNeonResultError`` when the table is empty.
    """
    sql = text("SELECT max(asof_date) AS m FROM market.breadth_indicator_daily")
    with engine.connect() as conn:
        value = conn.execute(sql).scalar_one_or_none()
    if value is None:
        raise EmptyNeonResultError(
            "market.breadth_indicator_daily is empty — no asof_date to render"
        )
    return _as_date(value)


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def load_etf_features_neon(engine: Engine, asof: date) -> pd.DataFrame:
    """Load the ETF features frame for ``render_etf_html`` from Neon.

    The pure renderer needs HISTORY (prior ``asof_date`` rows per symbol) for
    its sparklines, so we load every row ``asof_date <= asof`` — not just the
    single asof slice. The frame carries the columns the renderer requires:
    ``asof_date, symbol, sector_id, rank, rank_delta_1d, rank_delta_5d,
    r_5, r_10, r_20, ret_ytd, top15_streak`` (plus the rest of the table,
    which the renderer ignores).

    Raises ``EmptyNeonResultError`` if no row exists exactly at ``asof`` (the
    renderer would otherwise produce a valid-but-empty page).
    """
    sql = text(
        """
        SELECT *
        FROM market.thematic_features_daily
        WHERE asof_date <= :asof
        ORDER BY asof_date, symbol
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"asof": asof.isoformat()})
    if df.empty:
        raise EmptyNeonResultError(
            f"market.thematic_features_daily has no rows on or before {asof.isoformat()}"
        )
    # The renderer filters to == asof; guard here so a missing asof slice fails
    # loud instead of rendering an empty page from the history-only rows.
    asof_str = asof.isoformat()
    asof_col = _asof_as_iso(df["asof_date"])
    if not (asof_col == asof_str).any():
        raise EmptyNeonResultError(
            f"market.thematic_features_daily has no rows for asof={asof_str} "
            "(history present but not the requested day)"
        )
    return df


def load_breadth_neon(engine: Engine, asof: date) -> pd.DataFrame:
    """Load the long-format breadth frame for ``render_breadth_html`` from Neon.

    Returns the FULL history (``asof_date, indicator, value``); the renderer
    applies its own ``asof`` cutoff + trailing-window slice. We do not bound
    by asof here because the breadth charts draw a multi-year trailing series.

    Raises ``EmptyNeonResultError`` when the table is empty or has no row on or
    before ``asof`` (a render against an asof predating all data would be an
    empty page).
    """
    sql = text(
        """
        SELECT asof_date, indicator, value
        FROM market.breadth_indicator_daily
        WHERE asof_date <= :asof
        ORDER BY asof_date, indicator
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"asof": asof.isoformat()})
    if df.empty:
        raise EmptyNeonResultError(
            f"market.breadth_indicator_daily has no rows on or before {asof.isoformat()}"
        )
    return df


def load_spy_neon(engine: Engine, symbol: str = "SPY") -> pd.DataFrame:
    """Load the SPY (benchmark) OHLCV frame for the breadth SPY price pane.

    Schema matches what ``_spy_to_wide`` expects: ``symbol, date, open, high,
    low, close, volume, ...``. The renderer treats an empty frame as "omit the
    SPY pane" (back-compat), so a missing SPY series is NOT fatal — we return
    an empty DataFrame rather than raising. (The breadth indicators themselves
    are the fail-loud gate; SPY is an optional pane.)
    """
    sql = text(
        """
        SELECT symbol, date, open, high, low, close, volume,
               fetched_at, yfinance_version
        FROM market.benchmark_ohlcv
        WHERE symbol = :symbol
        ORDER BY date
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"symbol": symbol})
    return df


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_date(value: object) -> date:
    """Coerce a DB-returned asof value (date / datetime / iso string) to ``date``."""
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.to_datetime(value).date()


def _asof_as_iso(col: pd.Series) -> pd.Series:
    """Return ``asof_date`` as ISO ``YYYY-MM-DD`` strings for comparison."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return col.dt.strftime("%Y-%m-%d")
    # date objects / strings: round-trip through to_datetime for uniformity.
    return pd.to_datetime(col).dt.strftime("%Y-%m-%d")
