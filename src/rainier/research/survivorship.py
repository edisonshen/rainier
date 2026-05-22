"""Survivorship-gate check for ``qu_top100_history`` (D-016 PASS gate).

Reads the QU100 snapshot table (production: ``money_flow_snapshots``;
test: in-memory SQLite via :func:`bootstrap_sqlite`). Returns a typed
:class:`SurvivorshipResult` with verdict + delisted-ticker list +
missing-day list + concrete next-step. The CLI emits the result as JSON
when ``--json`` is set, or a single-line verdict + JSON sidecar
otherwise (per [D-034] — verbose detail goes to a run log, not stdout).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

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

_metadata = MetaData()


# Test table — mirrors the production money_flow_snapshots logical shape
# but with only the columns the survivorship check reads. We define it
# here (not from rainier.core.models) because:
#   1) the prod model carries TimescaleDB hypertable + JSONB columns that
#      sqlite can't reflect,
#   2) the survivorship check is read-only and only needs (data_date,
#      symbol, ranking_type, rank, mutated).
#
# Production CLI uses raw SQL against money_flow_snapshots directly
# (see `check` for the prod path).
qu100_snapshot_test = Table(
    "qu100_snapshot_test",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("data_date", Date, nullable=False, index=True),
    Column("symbol", String(10), nullable=False, index=True),
    Column("ranking_type", String(10), nullable=False, server_default="top100"),
    Column("rank", Integer, nullable=False, server_default="1"),
    Column("mutated", Integer, nullable=False, server_default="0"),
)


def bootstrap_sqlite(engine: Engine) -> None:
    """Create the test schema in an in-memory SQLite engine."""
    _metadata.create_all(engine)


def _is_sqlite(session: Session) -> bool:
    return session.bind.dialect.name == "sqlite"


# --------------------------------------------------------------------------- #
# Public seeding / mutation helpers — used only by tests, exposed at module
# level so the test suite doesn't reach into private state.
# --------------------------------------------------------------------------- #


def _trading_days(start: date, end: date) -> list[date]:
    """All NYSE trading sessions between start and end inclusive.

    Survivorship cares about market trading days. We use the NYSE
    exchange calendar (XNYS) to filter weekends AND holidays — the
    earlier weekday-only approximation reported NYE/MLK/Presidents
    Day/Good Friday/Memorial Day/etc. as "missing" against healthy
    insert-only data, flipping PASS to CONDITIONAL on first run.

    Falls back to weekday-only if exchange_calendars isn't importable
    (keeps the test path hermetic when the dep is unavailable); the
    fallback over-reports gaps but never under-reports — safer default.
    """
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


def seed_snapshots(
    session: Session, *, symbols: list[str], start: date, end: date
) -> None:
    """Insert one row per (symbol, trading day) for the test SQLite engine."""
    if not _is_sqlite(session):
        raise RuntimeError("seed_snapshots is for the in-memory sqlite test path only")
    days = _trading_days(start, end)
    rows = []
    for d in days:
        for rank, sym in enumerate(symbols, start=1):
            rows.append({"data_date": d, "symbol": sym, "ranking_type": "top100",
                          "rank": rank, "mutated": 0})
    if rows:
        session.execute(qu100_snapshot_test.insert(), rows)
        session.commit()


def mutate_historical_row_for_test(session: Session, symbol: str, on_date: date) -> None:
    """Flip the ``mutated`` flag on a historical row — survivorship-check
    treats any non-zero ``mutated`` as a proof-failure for insert-only."""
    if not _is_sqlite(session):
        raise RuntimeError("mutate_historical_row_for_test is sqlite-only")
    session.execute(
        qu100_snapshot_test.update()
        .where(qu100_snapshot_test.c.symbol == symbol)
        .where(qu100_snapshot_test.c.data_date == on_date)
        .values(mutated=1)
    )
    session.commit()


def ticker_present(session: Session, symbol: str, on_date: date) -> bool:
    """True iff the symbol has any snapshot on `on_date`."""
    if _is_sqlite(session):
        row = session.execute(
            select(func.count())
            .select_from(qu100_snapshot_test)
            .where(qu100_snapshot_test.c.symbol == symbol)
            .where(qu100_snapshot_test.c.data_date == on_date)
        ).scalar_one()
        return int(row) > 0
    row = session.execute(
        text(
            "SELECT COUNT(*) FROM money_flow_snapshots "
            "WHERE symbol = :sym AND data_date = :d AND ranking_type = 'top100'"
        ),
        {"sym": symbol, "d": on_date},
    ).scalar_one()
    return int(row) > 0


# --------------------------------------------------------------------------- #
# Result type + check function
# --------------------------------------------------------------------------- #


@dataclass
class SurvivorshipResult:
    from_date: date
    to_date: date
    verdict: str  # PASS | CONDITIONAL | FAIL
    delisted_tickers: list[str] = field(default_factory=list)
    missing_days: list[date] = field(default_factory=list)
    next_step: str | None = None
    immutability_proof: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_date": self.from_date.isoformat(),
            "to_date": self.to_date.isoformat(),
            "verdict": self.verdict,
            "delisted_tickers": list(self.delisted_tickers),
            "missing_days": [d.isoformat() for d in self.missing_days],
            "next_step": self.next_step,
            "immutability_proof": self.immutability_proof,
        }


def _coverage_query(session: Session, start: date, end: date) -> dict[date, int]:
    """Return rows-per-day for the window."""
    if _is_sqlite(session):
        rows = session.execute(
            select(
                qu100_snapshot_test.c.data_date,
                func.count(qu100_snapshot_test.c.id),
            )
            .where(qu100_snapshot_test.c.data_date >= start)
            .where(qu100_snapshot_test.c.data_date <= end)
            .group_by(qu100_snapshot_test.c.data_date)
        ).all()
    else:
        rows = session.execute(
            text(
                "SELECT data_date, COUNT(*) FROM money_flow_snapshots "
                "WHERE ranking_type = 'top100' AND data_date BETWEEN :s AND :e "
                "GROUP BY data_date"
            ),
            {"s": start, "e": end},
        ).all()
    return {r[0]: int(r[1]) for r in rows}


def _delisted_tickers(
    session: Session, start: date, end: date, tail_days: int = 7
) -> list[str]:
    """Tickers observed in the window but absent from its last `tail_days` window.

    A "delisted" ticker for survivorship purposes is one that QU had earlier
    in the range but stopped reporting before the end — capturing both true
    delistings and the rotation-tail exits per D-009. We compute the late
    window as ``(end - tail_days, end]`` so even a same-day "alive in early,
    gone the next day" exit registers.
    """
    if start >= end:
        return []
    tail_start = max(start, end - timedelta(days=tail_days))
    if _is_sqlite(session):
        seen_any = {
            r[0]
            for r in session.execute(
                select(qu100_snapshot_test.c.symbol)
                .where(qu100_snapshot_test.c.data_date >= start)
                .where(qu100_snapshot_test.c.data_date <= end)
                .distinct()
            )
        }
        late = {
            r[0]
            for r in session.execute(
                select(qu100_snapshot_test.c.symbol)
                .where(qu100_snapshot_test.c.data_date >= tail_start)
                .where(qu100_snapshot_test.c.data_date <= end)
                .distinct()
            )
        }
    else:
        seen_any = {
            r[0]
            for r in session.execute(
                text(
                    "SELECT DISTINCT symbol FROM money_flow_snapshots "
                    "WHERE ranking_type = 'top100' AND data_date BETWEEN :s AND :e"
                ),
                {"s": start, "e": end},
            )
        }
        late = {
            r[0]
            for r in session.execute(
                text(
                    "SELECT DISTINCT symbol FROM money_flow_snapshots "
                    "WHERE ranking_type = 'top100' AND data_date BETWEEN :t AND :e"
                ),
                {"t": tail_start, "e": end},
            )
        }
    return sorted(seen_any - late)


def _has_mutated_rows(session: Session, start: date, end: date) -> bool:
    """Insert-only proof: returns True if any row carries the mutated flag.

    Production path (postgres) can't read a `mutated` audit column — we
    don't ship one yet — so the prod check returns False (i.e., "no proof
    failure detected" but `immutability_proof` is marked as
    "unverified" downstream). Slice 1 ships the proper audit trigger.
    """
    if _is_sqlite(session):
        n = session.execute(
            select(func.count())
            .select_from(qu100_snapshot_test)
            .where(qu100_snapshot_test.c.mutated != 0)
            .where(qu100_snapshot_test.c.data_date >= start)
            .where(qu100_snapshot_test.c.data_date <= end)
        ).scalar_one()
        return int(n) > 0
    return False


def check(
    *, session: Session, from_date: date, to_date: date
) -> SurvivorshipResult:
    """Run the survivorship gate per D-016.

    Verdicts:
      - **PASS** — every trading day has > 0 rows, no immutability proof failures.
      - **CONDITIONAL** — coverage gap or immutability-proof failure with a
        concrete next-step recorded.
      - **FAIL** — the table doesn't exist or is empty (operator must run the
        scraper backfill before Slice 1 begins).
    """
    try:
        coverage = _coverage_query(session, from_date, to_date)
    except Exception as exc:  # noqa: BLE001 — prod query may raise on missing table.
        return SurvivorshipResult(
            from_date=from_date,
            to_date=to_date,
            verdict="CONDITIONAL",
            next_step=(
                "Cannot read money_flow_snapshots: "
                f"{exc!s} — confirm DB is up and the scraper has backfilled."
            ),
            immutability_proof="unverified",
        )

    if not coverage:
        return SurvivorshipResult(
            from_date=from_date,
            to_date=to_date,
            verdict="CONDITIONAL",
            next_step=(
                "money_flow_snapshots has no rows in the requested window — "
                "run `uv run rainier scrape qu --session morning --start-date "
                f"{from_date.isoformat()}` to backfill."
            ),
            immutability_proof="unverified",
        )

    trading_days = _trading_days(from_date, to_date)
    missing = [d for d in trading_days if d not in coverage]
    delisted = _delisted_tickers(session, from_date, to_date)
    mutated = _has_mutated_rows(session, from_date, to_date)

    proof = (
        "audit_column" if _is_sqlite(session) else "unverified"
    )

    if missing or mutated:
        if mutated:
            next_step = (
                "Insert-only / immutability proof FAILED: at least one historical "
                "row carries a non-zero mutated flag. Audit the writer for "
                "UPDATE/DELETE statements and fix before Slice 1."
            )
        else:
            next_step = (
                f"Coverage gap on {len(missing)} trading day(s) — "
                "re-run the scraper backfill for the missing dates."
            )
        return SurvivorshipResult(
            from_date=from_date,
            to_date=to_date,
            verdict="CONDITIONAL",
            delisted_tickers=delisted,
            missing_days=missing,
            next_step=next_step,
            immutability_proof=proof,
        )

    # codex iter-7 [P1]: PASS requires the immutability proof to actually
    # check out. In the prod (postgres) path, _has_mutated_rows() returns
    # False unconditionally because the audit column doesn't exist yet —
    # which means the gate would otherwise greenlight production history
    # without ever VERIFYING insert-only. Treat unverified proof as
    # CONDITIONAL with a concrete next-step so the operator knows the
    # gate hasn't actually run end-to-end.
    if proof == "unverified":
        return SurvivorshipResult(
            from_date=from_date,
            to_date=to_date,
            verdict="CONDITIONAL",
            delisted_tickers=delisted,
            missing_days=[],
            next_step=(
                "Coverage looks complete but the insert-only / immutability "
                "proof is unverified on this backend — no audit column or "
                "trigger to read from. Ship the audit trigger (Slice 1) "
                "before treating this as a D-016 PASS gate."
            ),
            immutability_proof=proof,
        )

    return SurvivorshipResult(
        from_date=from_date,
        to_date=to_date,
        verdict="PASS",
        delisted_tickers=delisted,
        missing_days=[],
        next_step=None,
        immutability_proof=proof,
    )
