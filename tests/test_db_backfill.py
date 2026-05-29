"""Integration tests for Phase 3 ``rainier db backfill-from-parquet`` (task plan §5).

The backfill reads the five parquet caches and UPSERTs their full history into
the matching ``market.*`` tables (registries first for FK order). We assert:

  * every table is populated and PG row counts match the parquet frames;
  * a second run is idempotent (UPSERT — no duplicate rows);
  * ``--dry-run`` reports counts but mutates nothing;
  * ``--asof-start/--asof-end`` windows the date-keyed tables;
  * DATABASE_URL unset -> a clean ClickException (not a raw traceback).

PG-backed tests gated on ``requires_postgres``; the URL-unset test runs always.
The fixtures mirror tests/test_db_dual_write.py so they share the docker-PG
resolution (RAINIER_TEST_DATABASE_URL -> pytest-postgresql -> skip).
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine, text

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
# Synthetic parquet cache builder (matches the five Phase 2 table schemas)
# ---------------------------------------------------------------------------


def _trading_days(n: int, start: date = date(2024, 10, 1)) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d = d + timedelta(days=1)
    return out


def _write_cache(cache: Path, symbols: list[str], n_days: int) -> dict[str, pd.DataFrame]:
    """Write the five parquet caches; return the in-memory frames for assertions."""
    cache.mkdir(parents=True, exist_ok=True)
    days = _trading_days(n_days)

    # Registries.
    ticker_rows = [
        {"ticker_id": i + 1, "symbol": s, "first_seen": days[0]}
        for i, s in enumerate(symbols)
    ]
    ticker_df = pd.DataFrame(ticker_rows)
    sector_df = pd.DataFrame(
        [{"sector_id": 1, "sector_name": "test_sector", "first_seen": days[0]}]
    )
    ticker_df.to_parquet(cache / "ticker_registry.parquet")
    sector_df.to_parquet(cache / "sector_registry.parquet")

    sym_to_id = {s: i + 1 for i, s in enumerate(symbols)}

    # OHLCV (thematic_universe).
    ohlcv_rows = []
    feat_rows = []
    label_rows = []
    for s in symbols:
        close = 100.0
        for i, day in enumerate(days):
            close = close * (1.0 + 0.003 * (sym_to_id[s] - 1))
            ohlcv_rows.append(
                {
                    "symbol": s, "date": day, "open": close, "high": close * 1.01,
                    "low": close * 0.99, "close": close, "volume": 1_000_000 + i,
                    "fetched_at": pd.Timestamp("2024-11-08T16:30:00Z"),
                    "yfinance_version": "1.2.0",
                }
            )
            feat_rows.append(
                {
                    "asof_date": day, "trading_day_ordinal": i, "symbol": s,
                    "ticker_id": sym_to_id[s], "sector_id": 1, "close": close,
                    "ret_1d": 0.003, "rel_5": 0.01, "rel_10": 0.02, "rel_20": 0.03,
                    "ret_ytd": 0.1, "relvol_5": 1.1, "relvol_10": 1.2, "relvol_20": 1.3,
                    "r_5": 1, "r_10": 2, "r_20": 3, "rank": sym_to_id[s],
                    "rank_delta_1d": 0, "rank_delta_5d": 0, "top15_streak": i,
                    "rank_minus_sector_mean": 0.5, "universe_size": len(symbols),
                    "universe_yaml_sha": "deadbeef",
                    "computed_at": pd.Timestamp("2024-11-08T16:30:00Z"),
                }
            )
            label_rows.append(
                {
                    "asof_date": day, "symbol": s, "fwd_3d_ret": 0.01,
                    "fwd_5d_ret": 0.02, "fwd_10d_ret": 0.03, "fwd_20d_ret": 0.04,
                    "fwd_30d_ret": 0.05, "fwd_5d_excess_ret": 0.001,
                    "fwd_10d_excess_ret": 0.002, "fwd_20d_excess_ret": 0.003,
                    "fwd_10d_max_drawdown": -0.02, "fwd_10d_max_runup": 0.06,
                    "label_complete_through": days[-1],
                }
            )
    ohlcv_df = pd.DataFrame(ohlcv_rows)
    feat_df = pd.DataFrame(feat_rows)
    label_df = pd.DataFrame(label_rows)
    ohlcv_df.to_parquet(cache / "thematic_universe.parquet")
    feat_df.to_parquet(cache / "thematic_features_daily.parquet")
    label_df.to_parquet(cache / "thematic_labels_daily.parquet")

    return {
        "tickers": ticker_df, "sectors": sector_df, "thematic_ohlcv": ohlcv_df,
        "thematic_features_daily": feat_df, "thematic_labels_daily": label_df,
    }


def _count(eng, table: str) -> int:
    with eng.connect() as conn:
        return conn.execute(text(f"SELECT count(*) FROM market.{table}")).scalar_one()


# ---------------------------------------------------------------------------
# backfill_from_parquet (module API)
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_backfill_populates_all_tables(migrated_engine, tmp_path, database_url):
    from rainier.db.backfill import backfill_from_parquet

    cache = tmp_path / "cache"
    frames = _write_cache(cache, ["AAA", "BBB", "CCC"], n_days=12)

    counts = backfill_from_parquet(migrated_engine, cache)

    for table, frame in frames.items():
        assert _count(migrated_engine, table) == len(frame), f"{table} row count"
        assert counts[table] == len(frame), f"{table} reported count"


@pytest.mark.requires_postgres
def test_backfill_is_idempotent(migrated_engine, tmp_path, database_url):
    from rainier.db.backfill import backfill_from_parquet

    cache = tmp_path / "cache"
    frames = _write_cache(cache, ["AAA", "BBB"], n_days=10)

    backfill_from_parquet(migrated_engine, cache)
    first = {t: _count(migrated_engine, t) for t in frames}
    backfill_from_parquet(migrated_engine, cache)
    second = {t: _count(migrated_engine, t) for t in frames}
    assert first == second, "re-run must not change row counts (UPSERT)"


@pytest.mark.requires_postgres
def test_backfill_respects_fk_order(migrated_engine, tmp_path, database_url):
    """Registries load before features, so the feature FK never violates."""
    from rainier.db.backfill import backfill_from_parquet

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA", "BBB", "CCC"], n_days=8)
    # Should not raise an IntegrityError on the FK.
    backfill_from_parquet(migrated_engine, cache)
    assert _count(migrated_engine, "thematic_features_daily") > 0


@pytest.mark.requires_postgres
def test_backfill_dry_run_mutates_nothing(migrated_engine, tmp_path, database_url):
    from rainier.db.backfill import backfill_from_parquet

    cache = tmp_path / "cache"
    frames = _write_cache(cache, ["AAA", "BBB"], n_days=6)

    counts = backfill_from_parquet(migrated_engine, cache, dry_run=True)
    # Counts are reported...
    for table, frame in frames.items():
        assert counts[table] == len(frame)
    # ...but nothing is written.
    for table in frames:
        assert _count(migrated_engine, table) == 0, f"{table} must stay empty on dry-run"


@pytest.mark.requires_postgres
def test_backfill_asof_window(migrated_engine, tmp_path, database_url):
    """--asof-start/--end window the date-keyed tables (ohlcv/features/labels);
    registries always load fully (FK parents)."""
    from rainier.db.backfill import backfill_from_parquet

    cache = tmp_path / "cache"
    frames = _write_cache(cache, ["AAA", "BBB"], n_days=12)
    feat = frames["thematic_features_daily"]
    all_dates = sorted(feat["asof_date"].unique())
    # Window to the middle three dates.
    start, end = all_dates[3], all_dates[5]

    backfill_from_parquet(migrated_engine, cache, asof_start=start, asof_end=end)

    expected = feat[(feat["asof_date"] >= start) & (feat["asof_date"] <= end)]
    assert _count(migrated_engine, "thematic_features_daily") == len(expected)
    # Registries are always loaded fully (parents).
    assert _count(migrated_engine, "tickers") == len(frames["tickers"])


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_cli_backfill_from_parquet(migrated_engine, tmp_path, database_url):
    from rainier.cli import cli

    cache = tmp_path / "cache"
    frames = _write_cache(cache, ["AAA", "BBB"], n_days=8)

    res = CliRunner().invoke(
        cli, ["db", "backfill-from-parquet", "--cache-dir", str(cache)]
    )
    assert res.exit_code == 0, res.output
    for table, frame in frames.items():
        assert _count(migrated_engine, table) == len(frame)


def test_cli_backfill_requires_database_url(tmp_path, monkeypatch):
    """DATABASE_URL unset -> a clean ClickException, not a raw traceback."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from rainier.cli import cli

    cache = tmp_path / "cache"
    _write_cache(cache, ["AAA"], n_days=3)

    res = CliRunner().invoke(
        cli, ["db", "backfill-from-parquet", "--cache-dir", str(cache)]
    )
    assert res.exit_code != 0
    assert res.exception is None or isinstance(res.exception, SystemExit), (
        f"expected a clean ClickException, got {res.exception!r}"
    )
    assert "DATABASE_URL" in res.output


def test_cli_backfill_rejects_reversed_window(tmp_path, monkeypatch):
    """--asof-start > --asof-end -> clean ClickException, never a silent
    registries-only run. A reversed window filters every date-keyed row out, so
    without this guard backfill would 'succeed' having written nothing but the
    FK parents. Validation runs before the DB engine is required, so no
    DATABASE_URL is needed."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from rainier.cli import cli

    res = CliRunner().invoke(
        cli,
        [
            "db", "backfill-from-parquet", "--cache-dir", str(tmp_path / "cache"),
            "--asof-start", "2026-05-10", "--asof-end", "2026-05-01",
        ],
    )
    assert res.exit_code != 0
    assert res.exception is None or isinstance(res.exception, SystemExit), (
        f"expected a clean ClickException, got {res.exception!r}"
    )
    assert "asof-start" in res.output and "asof-end" in res.output


def test_cli_backfill_registered():
    """`rainier db --help` lists backfill-from-parquet alongside the legacy cmds."""
    from rainier.cli import cli

    res = CliRunner().invoke(cli, ["db", "--help"])
    assert res.exit_code == 0, res.output
    assert "backfill-from-parquet" in res.output
    # Regression guard: the legacy + Phase 1 commands must still be present.
    for sub in ("init", "ping", "migrate"):
        assert sub in res.output
