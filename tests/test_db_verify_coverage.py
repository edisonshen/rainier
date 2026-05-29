"""Integration tests for Phase 3 ``rainier db verify-coverage`` (task plan §5).

verify-coverage proves parquet and Postgres agree per ``(asof_date, table)`` —
row count + an order-independent content checksum. We assert:

  * after a clean backfill the report is all-match and exits 0;
  * an injected missing row -> nonzero exit naming the offending (date, table);
  * an injected checksum drift (mutated value) -> nonzero exit naming it;
  * float round-trip parity does NOT false-positive (parquet float64 vs PG
    DOUBLE_PRECISION/REAL is within the documented rounding tolerance).

PG-backed tests gated on ``requires_postgres``; the URL-unset CLI test runs
always. Fixtures + the synthetic cache builder are shared with
tests/test_db_backfill.py.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine, text

from tests.test_db_backfill import _write_cache

# ---------------------------------------------------------------------------
# Postgres fixtures (mirror tests/test_db_dual_write.py resolution)
# ---------------------------------------------------------------------------


try:
    from pytest_postgresql import factories as _pg_factories

    _HAS_PYTEST_POSTGRESQL = True
except ImportError:  # pragma: no cover
    _HAS_PYTEST_POSTGRESQL = False


def _local_pg_binary_available() -> bool:
    import shutil

    return shutil.which("pg_config") is not None and shutil.which("initdb") is not None


if _HAS_PYTEST_POSTGRESQL and _local_pg_binary_available():
    postgresql_proc = _pg_factories.postgresql_proc(port=None, unixsocketdir="/tmp")
    postgresql = _pg_factories.postgresql("postgresql_proc")


@pytest.fixture
def database_url(request, monkeypatch):
    env_url = os.environ.get("RAINIER_TEST_DATABASE_URL")
    if env_url:
        monkeypatch.setenv("DATABASE_URL", env_url)
        yield env_url
        return

    if not _HAS_PYTEST_POSTGRESQL:
        pytest.skip("pytest-postgresql not installed")
    if not _local_pg_binary_available():
        pytest.skip("pg_config / initdb not on PATH; set RAINIER_TEST_DATABASE_URL")

    pg = request.getfixturevalue("postgresql")
    url = (
        f"postgresql+psycopg://{pg.info.user}@{pg.info.host}:{pg.info.port}"
        f"/{pg.info.dbname}"
    )
    monkeypatch.setenv("DATABASE_URL", url)
    yield url


@pytest.fixture
def migrated_engine(database_url):
    from alembic import command
    from alembic.config import Config

    repo_root = Path(__file__).resolve().parents[1]
    cfg = Config(str(repo_root / "db" / "alembic.ini"))
    cfg.set_main_option("script_location", str(repo_root / "db" / "alembic"))
    command.upgrade(cfg, "head")

    eng = create_engine(database_url)
    try:
        yield eng
    finally:
        with eng.begin() as conn:
            conn.exec_driver_sql("DROP SCHEMA IF EXISTS market CASCADE")
            conn.exec_driver_sql("DROP TABLE IF EXISTS public.alembic_version")
        eng.dispose()


# ---------------------------------------------------------------------------
# verify_coverage (module API): clean backfill -> all-match
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_verify_clean_backfill_all_match(migrated_engine, tmp_path, database_url):
    from rainier.db.backfill import backfill_from_parquet
    from rainier.db.verify import verify_coverage

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB", "CCC"], n_days=10)
    backfill_from_parquet(migrated_engine, cache)

    report = verify_coverage(migrated_engine, cache)
    assert report.ok, f"expected all-match, drift: {report.drift}"
    assert report.drift == []


@pytest.mark.requires_postgres
def test_verify_detects_missing_row(migrated_engine, tmp_path, database_url):
    """Delete a PG row -> verify reports drift naming the (date, table)."""
    from rainier.db.backfill import backfill_from_parquet
    from rainier.db.verify import verify_coverage

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB"], n_days=8)
    backfill_from_parquet(migrated_engine, cache)

    # Pick a known asof_date and drop one feature row for it.
    with migrated_engine.begin() as conn:
        victim = conn.execute(
            text(
                "SELECT asof_date FROM market.thematic_features_daily "
                "ORDER BY asof_date LIMIT 1"
            )
        ).scalar_one()
        conn.execute(
            text(
                "DELETE FROM market.thematic_features_daily "
                "WHERE asof_date=:d AND symbol='AAA'"
            ),
            {"d": victim},
        )

    report = verify_coverage(migrated_engine, cache)
    assert not report.ok
    offenders = {(d.table, d.asof_date) for d in report.drift}
    assert ("thematic_features_daily", victim) in offenders, report.drift


@pytest.mark.requires_postgres
def test_verify_detects_checksum_drift(migrated_engine, tmp_path, database_url):
    """Mutate a PG value (same row count) -> checksum mismatch is caught."""
    from rainier.db.backfill import backfill_from_parquet
    from rainier.db.verify import verify_coverage

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB"], n_days=8)
    backfill_from_parquet(migrated_engine, cache)

    with migrated_engine.begin() as conn:
        victim = conn.execute(
            text(
                "SELECT asof_date FROM market.thematic_labels_daily "
                "ORDER BY asof_date LIMIT 1"
            )
        ).scalar_one()
        # Change a value without changing the row count.
        conn.execute(
            text(
                "UPDATE market.thematic_labels_daily SET fwd_3d_ret = 99.0 "
                "WHERE asof_date=:d AND symbol='AAA'"
            ),
            {"d": victim},
        )

    report = verify_coverage(migrated_engine, cache)
    assert not report.ok
    offenders = {(d.table, d.asof_date) for d in report.drift}
    assert ("thematic_labels_daily", victim) in offenders, report.drift


@pytest.mark.requires_postgres
def test_verify_float_roundtrip_no_false_positive(migrated_engine, tmp_path, database_url):
    """Parquet float64 vs PG REAL/DOUBLE_PRECISION round-trip must not trip the
    checksum. A clean backfill of float-heavy feature/label tables verifies
    clean — the tolerance absorbs the storage-precision delta."""
    from rainier.db.backfill import backfill_from_parquet
    from rainier.db.verify import verify_coverage

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB", "CCC", "DDD"], n_days=15)
    backfill_from_parquet(migrated_engine, cache)

    report = verify_coverage(migrated_engine, cache)
    feat_drift = [d for d in report.drift if d.table == "thematic_features_daily"]
    label_drift = [d for d in report.drift if d.table == "thematic_labels_daily"]
    assert feat_drift == [], f"float features falsely flagged: {feat_drift}"
    assert label_drift == [], f"float labels falsely flagged: {label_drift}"
    assert report.ok


# ---------------------------------------------------------------------------
# CLI surface: exit codes
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_cli_verify_exit_zero_on_match(migrated_engine, tmp_path, database_url):
    from rainier.cli import cli
    from rainier.db.backfill import backfill_from_parquet

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB"], n_days=8)
    backfill_from_parquet(migrated_engine, cache)

    res = CliRunner().invoke(
        cli, ["db", "verify-coverage", "--cache-dir", str(cache)]
    )
    assert res.exit_code == 0, res.output


@pytest.mark.requires_postgres
def test_cli_verify_nonzero_on_drift(migrated_engine, tmp_path, database_url):
    from rainier.cli import cli
    from rainier.db.backfill import backfill_from_parquet

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB"], n_days=8)
    backfill_from_parquet(migrated_engine, cache)

    with migrated_engine.begin() as conn:
        victim = conn.execute(
            text("SELECT asof_date FROM market.thematic_features_daily LIMIT 1")
        ).scalar_one()
        conn.execute(
            text(
                "DELETE FROM market.thematic_features_daily "
                "WHERE asof_date=:d AND symbol='AAA'"
            ),
            {"d": victim},
        )

    res = CliRunner().invoke(
        cli, ["db", "verify-coverage", "--cache-dir", str(cache)]
    )
    assert res.exit_code != 0
    assert "thematic_features_daily" in res.output
    assert str(victim) in res.output


def test_cli_verify_requires_database_url(tmp_path, monkeypatch):
    """DATABASE_URL unset -> a clean ClickException, not a raw traceback."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from rainier.cli import cli

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA"], n_days=3)

    res = CliRunner().invoke(
        cli, ["db", "verify-coverage", "--cache-dir", str(cache)]
    )
    assert res.exit_code != 0
    assert res.exception is None or isinstance(res.exception, SystemExit), (
        f"expected a clean ClickException, got {res.exception!r}"
    )
    assert "DATABASE_URL" in res.output


def test_cli_verify_registered():
    from rainier.cli import cli

    res = CliRunner().invoke(cli, ["db", "--help"])
    assert res.exit_code == 0, res.output
    assert "verify-coverage" in res.output
