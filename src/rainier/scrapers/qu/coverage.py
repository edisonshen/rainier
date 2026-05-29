"""DB-coverage audit for ``money_flow_snapshots`` — qu-money-flow-backfill-ea9d.

Pure-read hygiene check that answers three questions:

  1) **Are we missing any trading days** between the earliest row and the
     latest row in ``money_flow_snapshots``?
  2) **Are per-day row counts in the expected ballpark** (~100 for top100,
     ~100 for bottom100), or did the universe shrink unexpectedly?
  3) **Are non-nullable columns actually populated** (a null-rate audit
     to catch schema-violating rows that slipped through during legacy
     migrations or scraper bugs)?

ASCII flow:

    ┌─────────────────────┐
    │ money_flow_snapshots│  (postgres/timescale OR sqlite-in-memory test)
    └──────────┬──────────┘
               │  read-only query (no INSERT/UPDATE/DELETE)
               ▼
    ┌──────────────────────────────────────┐
    │ _row_counts_by_day, _null_rates,     │
    │ _earliest_latest, _expected_days     │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │ CoverageReport                       │
    │   .missing_dates   .per_day_*        │
    │   .null_rate_by_column   .is_empty   │
    │   .is_clean()   .exit_code()         │
    └──────────┬───────────────────────────┘
               │
               ▼
       stdout (textual report)  +  process exit code
       (0 = OK; 1 = alarm: empty / >1 missing in last 7)

The module exports a small ``compute_report(session, asof, lookback_days)``
function plus an in-memory-sqlite test fixture (mirroring the pattern in
``rainier.research.survivorship``). The CLI wrapper lives in
``rainier.cli`` under the ``qu money-flow-coverage`` subcommand.

Why a dedicated test table (not reflect MoneyFlowSnapshot)?
  - Production model carries TimescaleDB hypertable + JSONB columns that
    sqlite can't reflect.
  - The audit only reads (data_date, ranking_type, rank, symbol, sector,
    industry) — declaring just those columns keeps the test fixture
    cheap and the column-null audit deterministic.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import median
from typing import TYPE_CHECKING, Any, Iterator

from sqlalchemy import (
    Column,
    Date,
    Integer,
    MetaData,
    String,
    Table,
    func,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from rainier.core.config import Settings

# --------------------------------------------------------------------------- #
# Test table — mirrors the production money_flow_snapshots logical shape but
# with only the columns the coverage audit reads. We define it here (not
# from rainier.core.models) because the prod model carries TimescaleDB
# hypertable + JSONB columns that sqlite can't reflect.
#
# Production CLI uses raw SQL against money_flow_snapshots directly (see
# the ``else`` branch in each ``_*`` helper below).
# --------------------------------------------------------------------------- #


_metadata = MetaData()

money_flow_snapshot_test = Table(
    "money_flow_snapshot_test",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("data_date", Date, nullable=False, index=True),
    Column("ranking_type", String(10), nullable=False, server_default="top100"),
    # Non-nullable in production (see core.models.MoneyFlowSnapshot). The
    # test table allows NULL so we can prove the null-rate audit catches
    # schema-violating rows that might appear post-migration.
    Column("rank", Integer, nullable=True),
    Column("symbol", String(10), nullable=False),
    Column("sector", String(100), nullable=True),
    Column("industry", String(200), nullable=True),
)


def bootstrap_sqlite(engine: Engine) -> None:
    """Create the test schema in an in-memory SQLite engine."""
    _metadata.create_all(engine)


def _is_sqlite(session: Session) -> bool:
    return session.bind.dialect.name == "sqlite"


# --------------------------------------------------------------------------- #
# Trading-day calendar — uses NYSE via exchange_calendars (already a
# rainier runtime dep). Falls back to weekday-only if exchange_calendars
# is unavailable.
# --------------------------------------------------------------------------- #


def _trading_days(start: date, end: date) -> list[date]:
    """All NYSE trading sessions in [start, end] inclusive."""
    if start > end:
        return []
    try:
        import exchange_calendars as xcals
    except ImportError:
        out = []
        cur = start
        while cur <= end:
            if cur.weekday() < 5:
                out.append(cur)
            cur += timedelta(days=1)
        return out

    nyse = xcals.get_calendar("XNYS")
    sessions = nyse.sessions_in_range(start.isoformat(), end.isoformat())
    return [s.date() for s in sessions]


# --------------------------------------------------------------------------- #
# Test seeding / mutation helpers — sqlite-only, public so tests can drive
# the audit without reaching into module-private state.
# --------------------------------------------------------------------------- #


def seed_snapshots(
    session: Session,
    *,
    start: date,
    end: date,
    rows_per_day: int = 100,
    ranking_type: str = "top100",
) -> None:
    """Insert ``rows_per_day`` rows for every NYSE trading day in [start, end].

    Symbols are synthetic (``SYM0001..SYM{rows_per_day:04d}``); sector +
    industry populated; ``rank`` is 1..rows_per_day. sqlite-only.
    """
    if not _is_sqlite(session):
        raise RuntimeError("seed_snapshots is for the in-memory sqlite test path only")

    days = _trading_days(start, end)
    rows = []
    for d in days:
        for r in range(1, rows_per_day + 1):
            rows.append({
                "data_date": d,
                "ranking_type": ranking_type,
                "rank": r,
                "symbol": f"SYM{r:04d}",
                "sector": "tech",
                "industry": "software",
            })
    if rows:
        session.execute(money_flow_snapshot_test.insert(), rows)
        session.commit()


def delete_day_for_test(session: Session, *, on_date: date) -> None:
    """Delete every test-table row on ``on_date``. sqlite-only."""
    if not _is_sqlite(session):
        raise RuntimeError("delete_day_for_test is sqlite-only")
    session.execute(
        money_flow_snapshot_test.delete().where(
            money_flow_snapshot_test.c.data_date == on_date
        )
    )
    session.commit()


def shrink_day_for_test(
    session: Session, *, on_date: date, keep_rows: int,
) -> None:
    """Reduce the row count for ``on_date`` to ``keep_rows`` (delete the rest).

    Simulates a universe-shrink day. sqlite-only.
    """
    if not _is_sqlite(session):
        raise RuntimeError("shrink_day_for_test is sqlite-only")
    # Identify the row ids to keep (lowest ranks) and delete the rest.
    keep_ids = [
        r[0]
        for r in session.execute(
            select(money_flow_snapshot_test.c.id)
            .where(money_flow_snapshot_test.c.data_date == on_date)
            .order_by(money_flow_snapshot_test.c.rank)
            .limit(keep_rows)
        ).all()
    ]
    if not keep_ids:
        return
    session.execute(
        money_flow_snapshot_test.delete()
        .where(money_flow_snapshot_test.c.data_date == on_date)
        .where(~money_flow_snapshot_test.c.id.in_(keep_ids))
    )
    session.commit()


def null_field_for_test(session: Session, *, on_date: date, field: str) -> None:
    """Set ``field`` to NULL on every row for ``on_date``. sqlite-only."""
    if not _is_sqlite(session):
        raise RuntimeError("null_field_for_test is sqlite-only")
    if field not in money_flow_snapshot_test.c:
        raise ValueError(f"unknown field {field!r}")
    session.execute(
        money_flow_snapshot_test.update()
        .where(money_flow_snapshot_test.c.data_date == on_date)
        .values({field: None})
    )
    session.commit()


def row_count(session: Session) -> int:
    """Total rows in the table (production OR test). Used by the audit
    + by the pure-read regression test."""
    if _is_sqlite(session):
        return int(
            session.execute(
                select(func.count()).select_from(money_flow_snapshot_test)
            ).scalar_one()
        )
    return int(
        session.execute(
            text("SELECT COUNT(*) FROM money_flow_snapshots")
        ).scalar_one()
    )


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


# Production schema declares these columns NOT NULL — the null-rate
# audit reports them so any nulls (legacy migration drift, scraper bug)
# are loud. Other columns (sector, industry, raw_data, etc.) are
# nullable in prod and intentionally skipped here.
_NOT_NULL_COLUMNS = ("symbol", "rank", "data_date", "ranking_type")


@dataclass
class CoverageReport:
    """Coverage audit result for ``money_flow_snapshots``."""

    asof: date
    lookback_days: int

    # Span over which we audited.
    earliest_date: date | None = None
    latest_date: date | None = None
    trading_days_span: int = 0
    expected_trading_days: int = 0

    # Row stats.
    rows_in_span: int = 0
    per_day_count: dict[date, int] = field(default_factory=dict)
    per_day_min: int = 0
    per_day_max: int = 0
    per_day_median: int = 0

    # Missing-day audit.
    missing_dates: list[date] = field(default_factory=list)
    missing_dates_in_alarm_window: list[date] = field(default_factory=list)

    # Null-rate audit.
    null_rate_by_column: dict[str, float] = field(default_factory=dict)

    # Distinct empty-table alarm flavor.
    is_empty: bool = False

    # Free-form alarm text the CLI prints. None when the report is clean.
    alarm_text: str | None = None
    has_warning: bool = False

    # Threshold — >1 missing in last 7 trading days trips the alarm. Lives
    # on the dataclass so the cron-hook + the unit tests share one source
    # of truth; tests with synthetic data set this to mirror prod.
    alarm_threshold: int = 1  # missing-day count above which exit=1

    def is_clean(self) -> bool:
        """True when no alarm conditions are present (warnings are OK)."""
        return self.alarm_text is None and not self.is_empty

    def exit_code(self) -> int:
        return 0 if self.is_clean() else 1


# --------------------------------------------------------------------------- #
# Core audit — pure-read.
# --------------------------------------------------------------------------- #


def _row_counts_by_day(
    session: Session, start: date, end: date,
) -> dict[date, int]:
    """Return rows-per-day in [start, end]. Counts across all ranking_types."""
    if _is_sqlite(session):
        rows = session.execute(
            select(
                money_flow_snapshot_test.c.data_date,
                func.count(money_flow_snapshot_test.c.id),
            )
            .where(money_flow_snapshot_test.c.data_date >= start)
            .where(money_flow_snapshot_test.c.data_date <= end)
            .group_by(money_flow_snapshot_test.c.data_date)
        ).all()
    else:
        rows = session.execute(
            text(
                "SELECT data_date, COUNT(*) FROM money_flow_snapshots "
                "WHERE data_date BETWEEN :s AND :e GROUP BY data_date"
            ),
            {"s": start, "e": end},
        ).all()
    return {r[0]: int(r[1]) for r in rows}


def _earliest_latest(session: Session) -> tuple[date | None, date | None]:
    """Min / max ``data_date`` across the entire table."""
    if _is_sqlite(session):
        row = session.execute(
            select(
                func.min(money_flow_snapshot_test.c.data_date),
                func.max(money_flow_snapshot_test.c.data_date),
            )
        ).one()
    else:
        row = session.execute(
            text(
                "SELECT MIN(data_date), MAX(data_date) FROM money_flow_snapshots"
            )
        ).one()
    return row[0], row[1]


def _null_rate(session: Session, column: str, start: date, end: date) -> float:
    """Fraction of rows in [start, end] whose ``column`` is NULL.

    Returns 0.0 when the window has no rows (caller treats "empty span" as
    a separate alarm flavor; null rates over an empty span are not signal).
    """
    if _is_sqlite(session):
        total = session.execute(
            select(func.count())
            .select_from(money_flow_snapshot_test)
            .where(money_flow_snapshot_test.c.data_date >= start)
            .where(money_flow_snapshot_test.c.data_date <= end)
        ).scalar_one()
        if not total:
            return 0.0
        null_n = session.execute(
            select(func.count())
            .select_from(money_flow_snapshot_test)
            .where(money_flow_snapshot_test.c.data_date >= start)
            .where(money_flow_snapshot_test.c.data_date <= end)
            .where(money_flow_snapshot_test.c[column].is_(None))
        ).scalar_one()
        return float(null_n) / float(total)
    # Production path — column name is from a hard-coded allow-list
    # (_NOT_NULL_COLUMNS) so the f-string isn't a SQL-injection vector.
    total = session.execute(
        text(
            "SELECT COUNT(*) FROM money_flow_snapshots "
            "WHERE data_date BETWEEN :s AND :e"
        ),
        {"s": start, "e": end},
    ).scalar_one()
    if not total:
        return 0.0
    null_n = session.execute(
        text(
            f"SELECT COUNT(*) FROM money_flow_snapshots "
            f"WHERE data_date BETWEEN :s AND :e AND {column} IS NULL"
        ),
        {"s": start, "e": end},
    ).scalar_one()
    return float(null_n) / float(total)


def compute_report(
    *,
    session: Session,
    asof: date,
    lookback_days: int = 30,
) -> CoverageReport:
    """Run the coverage audit against ``session``.

    ``asof`` anchors the "last 7 trading days" alarm window so tests can
    pass a fixed date (avoiding ``datetime.now()`` inside the module).

    ``lookback_days`` bounds how far back the per-day-count / null-rate
    queries scan. The default of 30 calendar days is plenty for the
    cron-hook (which only needs to see the tail-7).
    """
    report = CoverageReport(asof=asof, lookback_days=lookback_days)

    earliest, latest = _earliest_latest(session)
    report.earliest_date = earliest
    report.latest_date = latest

    if earliest is None or latest is None:
        # Empty table — distinct alarm flavor.
        report.is_empty = True
        report.alarm_text = "table is empty — no money_flow_snapshots rows present"
        return report

    # Audit window: ``lookback_days`` calendar days back from asof.
    window_start = asof - timedelta(days=lookback_days)
    # Clamp window_start to earliest so we don't synthesize "missing days"
    # before the table started getting populated. The reframed scope per
    # DESIGN v2 §3.1: the audit confirms ongoing coverage, NOT historical
    # backfill (that's the investigation doc).
    audit_start = max(window_start, earliest)
    audit_end = min(asof, latest)

    expected_days = _trading_days(audit_start, audit_end)
    report.expected_trading_days = len(expected_days)
    report.trading_days_span = len(_trading_days(earliest, latest))

    per_day = _row_counts_by_day(session, audit_start, audit_end)
    report.per_day_count = per_day
    report.rows_in_span = sum(per_day.values())

    if per_day:
        counts = sorted(per_day.values())
        report.per_day_min = counts[0]
        report.per_day_max = counts[-1]
        # statistics.median returns a float for even-length sequences; we
        # want an integer-ish for the report.
        med = median(counts)
        report.per_day_median = int(med) if med == int(med) else med  # type: ignore[assignment]

    # Missing days = expected NYSE sessions in the audit window not in per_day.
    report.missing_dates = sorted(d for d in expected_days if d not in per_day)

    # Alarm window: the last 7 trading days strictly BEFORE ``asof`` that
    # the table has been operational for. Computed independently of the
    # ``audit_end = min(asof, latest)`` clamp so a scraper outage (latest
    # << asof) still trips the alarm — otherwise `missing_dates` collapses
    # to [] because the audit window itself shrinks to match the data,
    # and an 8-trading-day outage produces exit 0 (regression iter-1).
    #
    # ``asof`` is excluded from the must-exist set when it is itself a
    # trading day, because the same-day scrape may not have run yet at
    # audit time. The alarm targets days that should already be present.
    tail_window_all = _trading_days(asof - timedelta(days=21), asof)
    if tail_window_all and tail_window_all[-1] == asof:
        candidate_tail = tail_window_all[:-1]
    else:
        candidate_tail = tail_window_all
    # Trim to >= earliest so pre-table dates aren't synthesized as missing.
    candidate_tail = [d for d in candidate_tail if d >= earliest]
    must_exist = candidate_tail[-7:]
    if must_exist:
        tail_present = _row_counts_by_day(
            session, must_exist[0], must_exist[-1],
        )
    else:
        tail_present = {}
    report.missing_dates_in_alarm_window = sorted(
        d for d in must_exist if d not in tail_present
    )

    # Null-rate audit on the production NOT-NULL columns.
    for col in _NOT_NULL_COLUMNS:
        report.null_rate_by_column[col] = _null_rate(
            session, col, audit_start, audit_end,
        )

    # Alarm gate. Per plan §Tests-5: >1 missing in last 7 → exit 1;
    # 1 missing → exit 0 with warning; 0 → clean.
    missing_in_tail = len(report.missing_dates_in_alarm_window)
    if missing_in_tail > report.alarm_threshold:
        report.alarm_text = (
            f"{missing_in_tail} trading days missing in the last "
            f"{len(must_exist)} — alarm threshold is "
            f"{report.alarm_threshold}"
        )
    elif missing_in_tail >= 1:
        # 1 missing — warning, not alarm. Exit 0 but flag the warning.
        report.has_warning = True

    return report


# --------------------------------------------------------------------------- #
# Session opener — production path uses rainier.core.database.get_session.
# Indirected through ``_open_session`` so tests can monkey-patch the
# context manager and point the CLI at an in-memory sqlite session.
# --------------------------------------------------------------------------- #


@contextmanager
def _open_session(settings: Settings | None = None) -> Iterator[Session]:
    """Yield a production DB session (postgres / timescale).

    When ``settings`` is provided (the CLI threads ``ctx.obj["settings"]`` here
    so ``--config staging.yaml`` selects the right DB), bind the session to the
    config-derived engine. Without it, fall back to the process-default
    singleton session — preserving prior behaviour for callers that don't pass
    a config-context.
    """
    if settings is None:
        from rainier.core.database import get_session as _get_session

        with _get_session() as s:
            yield s
        return

    from rainier.core.database import get_session_factory

    session = get_session_factory(settings)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Pretty-print
# --------------------------------------------------------------------------- #


def render_text_report(report: CoverageReport) -> str:
    """Render a human-readable, deterministic textual coverage report."""
    lines = [
        "money_flow_snapshots coverage report",
        "--------------------------------------",
    ]
    if report.earliest_date is None or report.latest_date is None:
        lines.append("earliest row:        (none)")
        lines.append("latest row:          (none)")
    else:
        lines.append(f"earliest row:        {report.earliest_date}")
        lines.append(f"latest row:          {report.latest_date}")
    lines.append(f"trading days span:   {report.trading_days_span}")
    lines.append(f"rows in span:        {report.rows_in_span}")
    lines.append(f"expected rows:       {report.expected_trading_days}")
    lines.append(f"asof:                {report.asof}")
    lines.append(f"lookback_days:       {report.lookback_days}")
    lines.append(
        "missing dates:       "
        + (", ".join(d.isoformat() for d in report.missing_dates) or "[]")
    )
    # Stale-table outages have missing_dates=[] (audit window is clamped to
    # latest), but the alarm-window list carries the actual dates that
    # triggered. Surface them so Discord shows operators which days the
    # scraper missed without forcing a manual re-run.
    if report.missing_dates_in_alarm_window and (
        set(report.missing_dates_in_alarm_window) - set(report.missing_dates)
    ):
        lines.append(
            "alarm-window missing: "
            + ", ".join(
                d.isoformat() for d in report.missing_dates_in_alarm_window
            )
        )
    if report.per_day_count:
        lines.append(
            "per-day row count:   "
            f"median={report.per_day_median}, "
            f"min={report.per_day_min}, "
            f"max={report.per_day_max}"
        )
    else:
        lines.append("per-day row count:   (none — window has no rows)")

    if report.null_rate_by_column:
        lines.append("null-rate by column:")
        for col, rate in sorted(report.null_rate_by_column.items()):
            lines.append(f"  {col:<15} {rate:.4%}")

    if report.is_empty:
        lines.append("alarm:               table is empty")
    elif report.alarm_text:
        lines.append(f"alarm:               {report.alarm_text}")
    elif report.has_warning:
        lines.append(
            f"warning:             {len(report.missing_dates_in_alarm_window)} "
            "missing day in last 7 (below alarm threshold)"
        )
    else:
        lines.append("alarm:               no missing days within the last 7 trading days → OK")

    return "\n".join(lines) + "\n"


def report_to_dict(report: CoverageReport) -> dict[str, Any]:
    """JSON-serializable view of the report — for cron logs / structured emit."""
    return {
        "asof": report.asof.isoformat(),
        "lookback_days": report.lookback_days,
        "earliest_date": report.earliest_date.isoformat() if report.earliest_date else None,
        "latest_date": report.latest_date.isoformat() if report.latest_date else None,
        "trading_days_span": report.trading_days_span,
        "expected_trading_days": report.expected_trading_days,
        "rows_in_span": report.rows_in_span,
        "per_day_min": report.per_day_min,
        "per_day_max": report.per_day_max,
        "per_day_median": report.per_day_median,
        "missing_dates": [d.isoformat() for d in report.missing_dates],
        "missing_dates_in_alarm_window": [
            d.isoformat() for d in report.missing_dates_in_alarm_window
        ],
        "null_rate_by_column": dict(report.null_rate_by_column),
        "is_empty": report.is_empty,
        "alarm_text": report.alarm_text,
        "has_warning": report.has_warning,
        "exit_code": report.exit_code(),
    }
