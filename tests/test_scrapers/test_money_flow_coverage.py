"""Tests for ``rainier.scrapers.qu.coverage`` — money_flow_snapshots DB hygiene.

The coverage module is a DB-read-only audit over ``money_flow_snapshots`` that
answers: "are we missing trading days?" + "are per-day row counts in the
expected ballpark?" + "are non-nullable columns actually populated?".

Production CLI reads from postgres via the same session contract; here we
exercise it against an in-memory sqlite engine so the tests stay hermetic.
The pattern mirrors ``tests/research/test_survivorship.py`` — define a test
table that carries the columns the check reads, seed it via helpers in the
production module, run ``compute_report`` against the session.
"""

from __future__ import annotations

from datetime import date

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rainier.scrapers.qu import coverage

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def in_memory_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    coverage.bootstrap_sqlite(engine)
    with Session(engine, future=True) as s:
        yield s


def _seed_clean_window(
    session: Session, *, start: date, end: date, rows_per_day: int = 100,
    ranking_type: str = "top100",
) -> None:
    """Insert ``rows_per_day`` snapshots for every NYSE trading day in [start, end]."""
    coverage.seed_snapshots(
        session,
        start=start,
        end=end,
        rows_per_day=rows_per_day,
        ranking_type=ranking_type,
    )


# --------------------------------------------------------------------------- #
# 1. Clean window → exit 0, missing_dates: []
# --------------------------------------------------------------------------- #


def test_coverage_reports_clean_when_all_days_present(in_memory_session):
    # 9 NYSE trading days ending 2026-05-25 (Mon). 2026-05-25 is Memorial Day
    # holiday — pick a window that avoids it for the "all present" case.
    # 2026-05-11..2026-05-21 = Mon May 11 → Thu May 21 = 9 trading days.
    start = date(2026, 5, 11)
    end = date(2026, 5, 21)
    _seed_clean_window(in_memory_session, start=start, end=end, rows_per_day=100)

    report = coverage.compute_report(
        session=in_memory_session,
        asof=end,
        lookback_days=11,  # cover the whole window
    )

    assert report.is_clean()
    assert report.missing_dates == []
    assert report.earliest_date == start
    assert report.latest_date == end
    assert report.trading_days_span == 9
    assert report.rows_in_span == 9 * 100
    assert report.expected_trading_days == 9
    assert report.exit_code() == 0


# --------------------------------------------------------------------------- #
# 2. Missing recent day → exit 1, missing day listed
# --------------------------------------------------------------------------- #


def test_coverage_detects_missing_recent_day(in_memory_session):
    """Plan §Tests #2 + §Acceptance gate: missing days in tail-7 trip alarm.

    The plan's §Acceptance line specifies "exit 1 when >1 trading day
    missing in the last 7". To trip the alarm, we delete 2 recent days
    and confirm both land in ``missing_dates``. (The 1-missing-only case
    is covered by ``test_coverage_alarm_threshold_one_missing_is_warning``
    below.)
    """
    start = date(2026, 5, 11)
    end = date(2026, 5, 21)
    _seed_clean_window(in_memory_session, start=start, end=end, rows_per_day=100)
    # Delete 2 recent days so the alarm threshold (>1 missing) trips.
    coverage.delete_day_for_test(in_memory_session, on_date=date(2026, 5, 19))
    coverage.delete_day_for_test(in_memory_session, on_date=date(2026, 5, 20))

    report = coverage.compute_report(
        session=in_memory_session,
        asof=end,
        lookback_days=11,
    )

    assert not report.is_clean()
    assert date(2026, 5, 19) in report.missing_dates
    assert date(2026, 5, 20) in report.missing_dates
    assert report.exit_code() == 1


# --------------------------------------------------------------------------- #
# 3. Weekend days are NOT flagged as missing
# --------------------------------------------------------------------------- #


def test_coverage_ignores_weekends(in_memory_session):
    # Friday 2026-05-15 → Tuesday 2026-05-19. Weekend in middle.
    start = date(2026, 5, 15)  # Fri
    end = date(2026, 5, 19)  # Tue
    _seed_clean_window(in_memory_session, start=start, end=end, rows_per_day=100)

    report = coverage.compute_report(
        session=in_memory_session,
        asof=end,
        lookback_days=10,
    )

    assert report.is_clean()
    # Sat May 16, Sun May 17 — must NOT appear as missing.
    assert date(2026, 5, 16) not in report.missing_dates
    assert date(2026, 5, 17) not in report.missing_dates


# --------------------------------------------------------------------------- #
# 4. Market holidays (NYSE) are NOT flagged as missing
# --------------------------------------------------------------------------- #


def test_coverage_ignores_market_holidays(in_memory_session):
    # MLK Day: 2024-01-15 (Mon). Window: Fri 2024-01-12 → Wed 2024-01-17.
    # Trading days: Fri Jan 12, Tue Jan 16, Wed Jan 17 (Mon Jan 15 is MLK).
    start = date(2024, 1, 12)
    end = date(2024, 1, 17)
    _seed_clean_window(in_memory_session, start=start, end=end, rows_per_day=100)

    report = coverage.compute_report(
        session=in_memory_session,
        asof=end,
        lookback_days=10,
    )

    assert report.is_clean()
    # MLK Day must NOT appear as missing even though it's Mon-Fri.
    assert date(2024, 1, 15) not in report.missing_dates


# --------------------------------------------------------------------------- #
# 5. Alarm threshold: >1 missing in last 7 trading days
# --------------------------------------------------------------------------- #


def test_coverage_alarm_threshold(in_memory_session):
    """Plan spec: >1 trading day missing in last 7 → exit 1.

    1 missing → exit 0 with warning text.
    0 missing → exit 0 clean.
    """
    # 10-day window, asof = Thu 2026-05-21.
    asof = date(2026, 5, 21)
    start = date(2026, 5, 7)

    # Case A: 2 missing in last 7 → exit 1.
    _seed_clean_window(in_memory_session, start=start, end=asof, rows_per_day=50)
    coverage.delete_day_for_test(in_memory_session, on_date=date(2026, 5, 19))
    coverage.delete_day_for_test(in_memory_session, on_date=date(2026, 5, 20))
    report = coverage.compute_report(
        session=in_memory_session, asof=asof, lookback_days=14,
    )
    assert report.exit_code() == 1
    assert len(report.missing_dates_in_alarm_window) >= 2


def test_coverage_alarm_threshold_one_missing_is_warning(in_memory_session):
    """1 missing in last 7 → exit 0 but the warning text is present."""
    asof = date(2026, 5, 21)
    start = date(2026, 5, 7)
    _seed_clean_window(in_memory_session, start=start, end=asof, rows_per_day=50)
    coverage.delete_day_for_test(in_memory_session, on_date=date(2026, 5, 20))

    report = coverage.compute_report(
        session=in_memory_session, asof=asof, lookback_days=14,
    )

    # 1 missing in tail-7 → exit 0 (warning, not alarm).
    assert report.exit_code() == 0
    assert len(report.missing_dates_in_alarm_window) == 1
    assert report.has_warning


# --------------------------------------------------------------------------- #
# 6. Per-day rowcount stats (universe shrink not an alarm)
# --------------------------------------------------------------------------- #


def test_coverage_per_day_rowcount(in_memory_session):
    """Universe-shrink day (50 rows vs 100 elsewhere) is reported in stats,
    but does NOT trip the alarm gate.
    """
    start = date(2026, 5, 11)
    end = date(2026, 5, 21)
    _seed_clean_window(in_memory_session, start=start, end=end, rows_per_day=100)
    # Shrink 2026-05-15 down to 50 rows.
    coverage.shrink_day_for_test(
        in_memory_session, on_date=date(2026, 5, 15), keep_rows=50,
    )

    report = coverage.compute_report(
        session=in_memory_session, asof=end, lookback_days=11,
    )

    # Per-day stats reflect the shrink.
    assert report.per_day_min == 50
    assert report.per_day_max == 100
    # Median = 100 (one shrink day vs eight full days).
    assert report.per_day_median == 100
    # Not an alarm — exit code stays 0.
    assert report.exit_code() == 0


# --------------------------------------------------------------------------- #
# 7. NULLs in non-nullable cols → reported in null-rate table
# --------------------------------------------------------------------------- #


def test_coverage_null_rates(in_memory_session):
    """If `rank` is null on any row, null-rate table records non-zero rate.

    Production schema declares `rank` NOT NULL, but the test fixture allows
    a NULL so we can prove the audit catches schema-violating rows that
    might slip through TimescaleDB / postgres in legacy/migrated data.
    """
    start = date(2026, 5, 18)
    end = date(2026, 5, 21)
    _seed_clean_window(in_memory_session, start=start, end=end, rows_per_day=10)
    coverage.null_field_for_test(
        in_memory_session, on_date=date(2026, 5, 19), field="rank",
    )

    report = coverage.compute_report(
        session=in_memory_session, asof=end, lookback_days=10,
    )

    null_rates = report.null_rate_by_column
    assert "rank" in null_rates
    assert null_rates["rank"] > 0


# --------------------------------------------------------------------------- #
# 8. Coverage check performs zero writes
# --------------------------------------------------------------------------- #


def test_coverage_is_pure_db_read(in_memory_session):
    """compute_report must not INSERT/UPDATE/DELETE anything.

    We snapshot the row count + a sentinel field before and after the call.
    """
    start = date(2026, 5, 11)
    end = date(2026, 5, 21)
    _seed_clean_window(in_memory_session, start=start, end=end, rows_per_day=20)

    before_count = coverage.row_count(in_memory_session)
    _ = coverage.compute_report(
        session=in_memory_session, asof=end, lookback_days=11,
    )
    after_count = coverage.row_count(in_memory_session)

    assert before_count == after_count
    # And nothing dirty in the session — compute_report must not mutate.
    assert not in_memory_session.dirty
    assert not in_memory_session.new
    assert not in_memory_session.deleted


# --------------------------------------------------------------------------- #
# 9. Stale table — scraper stopped days ago → alarm fires.
#
# Regression for review iter-1: previously the `audit_end = min(asof, latest)`
# clamp silently restricted `missing_dates` to the populated window, so an
# 8-trading-day scraper outage produced exit 0 ("no missing days in tail-7").
# The whole point of the alarm is to catch that.
# --------------------------------------------------------------------------- #


def test_coverage_detects_stale_table_no_recent_data(in_memory_session):
    """Scraper outage: latest row in DB is 8 trading days before asof.

    The alarm must fire because most of the last 7 trading days BEFORE
    asof have no rows. Otherwise the cron-hook's whole reason for being
    (catching scrape outages) is defeated.
    """
    # Seed through Mon 2026-05-11; pretend the scraper stopped that day.
    start = date(2026, 4, 27)
    end_seeded = date(2026, 5, 11)
    _seed_clean_window(
        in_memory_session, start=start, end=end_seeded, rows_per_day=100,
    )
    # asof = Thu 2026-05-21 → 8 trading days have passed since latest=5/11.
    asof = date(2026, 5, 21)

    report = coverage.compute_report(
        session=in_memory_session, asof=asof, lookback_days=30,
    )

    assert report.exit_code() == 1, (
        f"stale-table outage must trigger alarm; latest={report.latest_date} "
        f"asof={asof} missing_in_tail={report.missing_dates_in_alarm_window}"
    )
    # Concrete missing days that should be flagged.
    assert date(2026, 5, 12) in report.missing_dates_in_alarm_window
    assert date(2026, 5, 20) in report.missing_dates_in_alarm_window
    # asof itself (5/21) should NOT be flagged — today's scrape may not have
    # run yet at audit time. The alarm is for days strictly before asof.
    assert date(2026, 5, 21) not in report.missing_dates_in_alarm_window

    # codex iter-1 [P3]: the rendered Discord-bound report MUST surface the
    # alarm-window dates so operators see which days are missing. Without
    # this, `missing dates: []` shows up in the alert (because the main
    # audit window is clamped to `latest`) even as the alarm fires.
    rendered = coverage.render_text_report(report)
    assert "alarm-window missing:" in rendered, rendered
    assert "2026-05-12" in rendered
    assert "2026-05-20" in rendered


# --------------------------------------------------------------------------- #
# 10. Empty table → exit 1 with distinct "table is empty" alarm
# --------------------------------------------------------------------------- #


def test_coverage_returns_alarm_when_table_empty(in_memory_session):
    # No seeding — table is empty.
    report = coverage.compute_report(
        session=in_memory_session, asof=date(2026, 5, 21), lookback_days=7,
    )

    assert report.exit_code() == 1
    assert report.is_empty
    # Distinct alarm flavor from "missing days".
    assert "empty" in (report.alarm_text or "").lower()


# --------------------------------------------------------------------------- #
# CLI smoke: --asof produces deterministic output, exits non-zero on alarm
# --------------------------------------------------------------------------- #


def test_cli_money_flow_coverage_exit_code(in_memory_session, monkeypatch):
    """Hook the CLI to the in-memory session via a monkeypatched session
    factory, then call ``rainier qu money-flow-coverage --asof 2026-05-21``
    twice — once on a clean DB, once after deleting a recent day — and
    confirm exit codes flip.
    """
    from rainier import cli as cli_module

    asof = date(2026, 5, 21)
    start = date(2026, 5, 7)
    _seed_clean_window(in_memory_session, start=start, end=asof, rows_per_day=20)

    # Monkey-patch coverage._open_session to yield the test session.
    from contextlib import contextmanager

    @contextmanager
    def _fake_session(settings=None):
        yield in_memory_session

    monkeypatch.setattr(coverage, "_open_session", _fake_session)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli, ["qu", "money-flow-coverage", "--asof", asof.isoformat()],
    )
    assert result.exit_code == 0, result.output
    assert "money_flow_snapshots coverage report" in result.output

    # Now delete 2 trading days in the tail-7 → exit 1.
    coverage.delete_day_for_test(in_memory_session, on_date=date(2026, 5, 19))
    coverage.delete_day_for_test(in_memory_session, on_date=date(2026, 5, 20))

    result = runner.invoke(
        cli_module.cli, ["qu", "money-flow-coverage", "--asof", asof.isoformat()],
    )
    assert result.exit_code == 1
    assert "missing" in result.output.lower()


# --------------------------------------------------------------------------- #
# Regression: --config threads config-derived settings into the coverage session
# (codex iter-2 of PR #92, deferred [P2]). Before the fix, the coverage command
# opened the session via the no-arg singleton get_session(), silently ignoring
# `--config staging.yaml` and reading the default DB.
# --------------------------------------------------------------------------- #


def test_cli_money_flow_coverage_honors_config(in_memory_session, tmp_path, monkeypatch):
    """`rainier --config <yaml> qu money-flow-coverage` must open the session
    with the settings resolved from that YAML — not the process default.

    Deterministic: spy on ``coverage._open_session`` to capture the settings it
    receives, and use a distinguishing ``database.pool_size`` (7) in the temp
    config that the default (5) would never produce.
    """
    from contextlib import contextmanager

    from rainier import cli as cli_module

    asof = date(2026, 5, 21)
    start = date(2026, 5, 7)
    _seed_clean_window(in_memory_session, start=start, end=asof, rows_per_day=20)

    # A config whose database.pool_size (7) is distinct from the default (5),
    # so we can prove the --config-resolved settings reached _open_session.
    config_path = tmp_path / "staging.yaml"
    config_path.write_text("database:\n  pool_size: 7\n")

    captured: dict[str, object] = {}

    @contextmanager
    def _spy_open_session(settings=None):
        captured["settings"] = settings
        yield in_memory_session

    monkeypatch.setattr(coverage, "_open_session", _spy_open_session)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        ["--config", str(config_path), "qu", "money-flow-coverage",
         "--asof", asof.isoformat()],
    )

    assert result.exit_code == 0, result.output
    settings = captured.get("settings")
    assert settings is not None, "_open_session was called without config-derived settings"
    assert settings.database.pool_size == 7, (
        "coverage session did not receive --config-resolved settings "
        f"(got pool_size={settings.database.pool_size!r})"
    )


def test_open_session_bypasses_engine_singleton(tmp_path, monkeypatch):
    """``_open_session(settings)`` must bind to an engine built from ``settings``
    even when ``core.database``'s global ``_engine`` singleton was already
    initialized to a *different* DB earlier in the process.

    Regression for codex iter-1 [P2]: ``get_session_factory(settings)`` routed
    through ``get_engine``, which returns the cached singleton and ignores its
    ``settings`` arg once set — so a prior DB-using command in the same process
    silently pinned ``--config staging.yaml`` to the wrong DB. The fix builds a
    dedicated engine directly from ``settings`` instead.

    Deterministic: pre-seed the singleton with a sentinel ``default.sqlite``
    engine, then assert the session yielded for a ``staging.sqlite`` settings
    binds to staging, not the sentinel.
    """
    from rainier.core import database
    from rainier.core.config import Settings

    default_db = tmp_path / "default.sqlite"
    staging_db = tmp_path / "staging.sqlite"

    # Pre-initialize the global singleton against the default DB, as any earlier
    # DB-using command in the same process would.
    sentinel = create_engine(f"sqlite:///{default_db}")
    monkeypatch.setattr(database, "_engine", sentinel)
    monkeypatch.setattr(database, "_session_factory", None)

    staging_settings = Settings(database_url=f"sqlite:///{staging_db}")

    with coverage._open_session(staging_settings) as session:
        bound_url = str(session.get_bind().url)

    assert bound_url == f"sqlite:///{staging_db}", (
        "config-scoped session bound to the wrong DB — singleton leaked through "
        f"(got {bound_url!r})"
    )
    # Sentinel singleton remains untouched (we never mutated it).
    assert database._engine is sentinel
