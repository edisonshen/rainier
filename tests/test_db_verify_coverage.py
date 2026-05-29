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


@pytest.mark.requires_postgres
def test_verify_real_column_float32_boundary_no_false_positive(
    migrated_engine, tmp_path, database_url
):
    """Regression (real-data smoke): a parquet float64 in a PG REAL column whose
    float32 round-trip straddles a significant-figure rounding boundary must NOT
    be flagged. Earlier sig-fig rounding gave parquet 0.07677094638... -> 0.0767709
    but PG's float32 0.07677095 -> 0.076771, a false positive. Casting both sides
    through float32 makes them bit-identical. Also exercises NaN -> NULL parity
    (recent asof dates have incomplete forward returns)."""
    import math

    import numpy as np
    import pandas as pd
    from sqlalchemy import text

    from rainier.db import schema
    from rainier.db.rows import frame_to_pg_rows
    from rainier.db.upsert import market_upsert
    from rainier.db.verify import verify_coverage

    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    asof = date(2026, 5, 8)
    # Values picked from the real-data smoke that broke sig-fig rounding, plus a
    # NaN forward-return row (incomplete horizon).
    boundary = [
        0.07677094638347626,
        0.007865045219659805,
        0.0001843344944063574,
        0.022905850782990456,
        -0.03227955102920532,
    ]
    rows = []
    for i, sym in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
        v = boundary[i]
        rows.append(
            {
                "asof_date": asof, "symbol": sym,
                "fwd_3d_ret": v, "fwd_5d_ret": v * 1.5, "fwd_10d_ret": v * 2.0,
                "fwd_20d_ret": float("nan"), "fwd_30d_ret": float("nan"),
                "fwd_5d_excess_ret": v * 0.5, "fwd_10d_excess_ret": v * 0.25,
                "fwd_20d_excess_ret": float("nan"),
                "fwd_10d_max_drawdown": abs(v), "fwd_10d_max_runup": abs(v) * 0.3,
                "label_complete_through": None,
            }
        )
    label_df = pd.DataFrame(rows)
    label_df.to_parquet(cache / "thematic_labels_daily.parquet")

    # Backfill ONLY labels (write directly; no FK on labels).
    label_cols = list(schema.thematic_labels_daily.columns.keys())
    market_upsert(
        migrated_engine, schema.thematic_labels_daily,
        frame_to_pg_rows(label_df, label_cols), ["asof_date", "symbol"],
    )

    # Sanity: the parquet float64 and the PG REAL round-trip really do differ in
    # the low bits (so the test would catch a naive repr/hash regression).
    with migrated_engine.connect() as conn:
        pg_v = conn.execute(
            text(
                "SELECT fwd_3d_ret FROM market.thematic_labels_daily "
                "WHERE asof_date=:a AND symbol='AAA'"
            ),
            {"a": asof},
        ).scalar_one()
    assert pg_v != boundary[0], "PG REAL must lose precision vs parquet float64"
    assert math.isclose(float(np.float32(boundary[0])), float(np.float32(pg_v)))

    report = verify_coverage(migrated_engine, cache)
    drift = [d for d in report.drift if d.table == "thematic_labels_daily"]
    assert drift == [], f"float32-boundary values falsely flagged drift: {drift}"
    assert report.ok


# ---------------------------------------------------------------------------
# Timezone-aware datetime parity (TIMESTAMPTZ columns)
# ---------------------------------------------------------------------------


def test_checksum_tz_aware_same_instant_matches():
    """Same instant in different tz offsets must hash identically.

    PG returns TIMESTAMPTZ (fetched_at/computed_at) in the session timezone, so
    a clean backfill can yield ``08:30-08:00`` on the PG side vs ``16:30+00:00``
    in parquet — the SAME instant. The checksum must normalize to UTC so this
    is NOT flagged as drift. (Regression: a session TZ != UTC would otherwise
    false-positive a clean parity run.)"""
    import datetime as _dt

    from rainier.db.verify import _checksum

    cols = ["symbol", "fetched_at"]
    pk = ("symbol",)
    utc = _dt.datetime(2024, 11, 8, 16, 30, tzinfo=_dt.timezone.utc)
    pacific = utc.astimezone(_dt.timezone(_dt.timedelta(hours=-8)))
    assert utc != pacific or str(utc) != str(pacific)  # reprs differ pre-norm
    parquet_rows = [{"symbol": "AAA", "fetched_at": utc}]
    pg_rows = [{"symbol": "AAA", "fetched_at": pacific}]
    assert _checksum(parquet_rows, cols, pk, frozenset()) == _checksum(
        pg_rows, cols, pk, frozenset()
    ), "same instant in a different tz offset must hash equal"


@pytest.mark.requires_postgres
def test_verify_clean_under_non_utc_session_tz(migrated_engine, tmp_path, database_url):
    """A clean backfill verifies clean even when the PG session timezone is not
    UTC (TIMESTAMPTZ columns then render in that zone). End-to-end guard for the
    tz-normalization fix."""
    from sqlalchemy import event

    from rainier.db.backfill import backfill_from_parquet
    from rainier.db.verify import verify_coverage

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB"], n_days=6)
    backfill_from_parquet(migrated_engine, cache)

    # Force a non-UTC timezone on EVERY connection verify_coverage opens (it
    # opens its own connection in _read_pg, so a one-off SET on a separate
    # connection would not reach it). With this, TIMESTAMPTZ columns render in
    # US/Pacific while parquet stays UTC — the case the fix must absorb.
    pac_engine = create_engine(database_url)

    @event.listens_for(pac_engine, "connect")
    def _set_pacific(dbapi_conn, _rec):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("SET TIME ZONE 'America/Los_Angeles'")
        cur.close()

    try:
        # Sanity: a TIMESTAMPTZ really renders in the session zone now.
        with pac_engine.connect() as conn:
            rendered = conn.exec_driver_sql(
                "SELECT fetched_at::text FROM market.thematic_ohlcv LIMIT 1"
            ).scalar_one()
        assert "+00" not in rendered, f"expected non-UTC render, got {rendered!r}"

        report = verify_coverage(pac_engine, cache)
        assert report.ok, f"non-UTC session must not false-positive drift: {report.drift}"
    finally:
        pac_engine.dispose()


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
