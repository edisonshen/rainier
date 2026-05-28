"""Integration tests for the Phase 2 dual-write writers (task plan §5/§6).

Each writer is driven through its real entry point (the backfill ``backfill()``
function and the ``thematic`` CLI commands via ``CliRunner``) against a temp
Postgres migrated to head, and we assert:

  * PG row counts match the parquet frame the writer produced;
  * a sample row's columns mirror parquet;
  * a same-asof_date re-run is idempotent (UPSERT — no duplicate rows);
  * the FK-ordered registries+ohlcv land without violation.

A separate test exercises the DATABASE_URL-unset skip path with NO Postgres
needed: the writer logs a warning, skips PG, still writes parquet, exits 0.

PG-backed tests gated on ``requires_postgres``; the skip-path test runs always.
"""

from __future__ import annotations

import datetime as dt
import importlib
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine, text


# ---------------------------------------------------------------------------
# Shared synthetic OHLCV builder (matches backfill_thematic_universe schema)
# ---------------------------------------------------------------------------


def _build_ohlcv_panel(symbols: list[str], n_days: int) -> pd.DataFrame:
    dates = []
    d = date(2024, 10, 1)
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d = d + timedelta(days=1)

    rows = []
    for s_idx, sym in enumerate(symbols):
        close = 100.0
        for i, day in enumerate(dates):
            if i > 0:
                close = close * (1.0 + 0.005 * (s_idx - 2))
            rows.append(
                {
                    "symbol": sym,
                    "date": day,
                    "open": close,
                    "high": close * 1.005,
                    "low": close * 0.995,
                    "close": close,
                    "volume": 1_000_000,
                    "fetched_at": pd.Timestamp("2024-11-08T16:30:00Z"),
                    "yfinance_version": "1.2.0",
                }
            )
    return pd.DataFrame(rows)


def _write_universe_yaml(path: Path, symbols: list[str]) -> None:
    body = (
        "version: 1\n"
        "schema: thematic_universe.v1\n"
        "asof_seeded: 2024-10-01\n"
        "universe:\n"
        "  test_sector:\n"
    )
    body += "".join(f"    - {s}\n" for s in symbols)
    path.write_text(body)


# ---------------------------------------------------------------------------
# Postgres fixtures (mirror tests/test_db_migrations.py resolution)
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
        command.downgrade(cfg, "base")
        eng.dispose()


def _count(eng, table) -> int:
    with eng.connect() as conn:
        return conn.execute(text(f"SELECT count(*) FROM market.{table}")).scalar_one()


def _import_backfill():
    """Import scripts/backfill_thematic_universe.py as a module."""
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    mod = importlib.import_module("backfill_thematic_universe")
    return importlib.reload(mod)


# ---------------------------------------------------------------------------
# backfill_thematic_universe -> market.tickers / sectors / thematic_ohlcv
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_backfill_dual_writes_ohlcv_and_registries(migrated_engine, tmp_path, database_url):
    backfill = _import_backfill()
    symbols = ["AAA", "BBB", "CCC"]
    panel = _build_ohlcv_panel(symbols, n_days=10)

    def fake_fetch(syms, start, end):
        out = {}
        for s in syms:
            sub = panel.loc[panel["symbol"] == s, ["date", "open", "high", "low", "close", "volume"]]
            out[s] = sub.reset_index(drop=True)
        return out

    yaml_path = tmp_path / "universe.yaml"
    _write_universe_yaml(yaml_path, symbols)
    out_path = tmp_path / "thematic_universe.parquet"

    written = backfill.backfill(
        symbols=symbols,
        start="2024-10-01",
        end="2024-10-31",
        out_path=out_path,
        fetch_fn=fake_fetch,
        yaml_path=yaml_path,
    )
    written = Path(written)
    assert written.exists(), "parquet must still be written"
    parquet_df = pd.read_parquet(written)

    # OHLCV rows mirror parquet.
    assert _count(migrated_engine, "thematic_ohlcv") == len(parquet_df)
    # Registries: 3 tickers, 1 sector.
    assert _count(migrated_engine, "tickers") == 3
    assert _count(migrated_engine, "sectors") == 1

    # Sample row mirrors parquet.
    with migrated_engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT close, volume FROM market.thematic_ohlcv "
                "WHERE symbol='AAA' ORDER BY date LIMIT 1"
            )
        ).one()
    p_first = parquet_df.loc[parquet_df["symbol"] == "AAA"].sort_values("date").iloc[0]
    assert row.close == pytest.approx(float(p_first["close"]))
    assert row.volume == int(p_first["volume"])


@pytest.mark.requires_postgres
def test_backfill_dual_write_idempotent(migrated_engine, tmp_path, database_url):
    backfill = _import_backfill()
    symbols = ["AAA", "BBB"]
    panel = _build_ohlcv_panel(symbols, n_days=8)

    def fake_fetch(syms, start, end):
        return {
            s: panel.loc[
                panel["symbol"] == s,
                ["date", "open", "high", "low", "close", "volume"],
            ].reset_index(drop=True)
            for s in syms
        }

    yaml_path = tmp_path / "universe.yaml"
    _write_universe_yaml(yaml_path, symbols)

    backfill.backfill(
        symbols=symbols, start="2024-10-01", end="2024-10-31",
        out_path=tmp_path / "u1.parquet", fetch_fn=fake_fetch, yaml_path=yaml_path,
    )
    n_ohlcv = _count(migrated_engine, "thematic_ohlcv")
    n_tick = _count(migrated_engine, "tickers")

    # Second run, same data -> idempotent UPSERT, no duplicate rows.
    backfill.backfill(
        symbols=symbols, start="2024-10-01", end="2024-10-31",
        out_path=tmp_path / "u2.parquet", fetch_fn=fake_fetch, yaml_path=yaml_path,
    )
    assert _count(migrated_engine, "thematic_ohlcv") == n_ohlcv
    assert _count(migrated_engine, "tickers") == n_tick


# ---------------------------------------------------------------------------
# thematic compute -> market.thematic_features_daily
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_thematic_compute_dual_writes_features(migrated_engine, tmp_path, database_url):
    from rainier.cli import cli

    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    panel = _build_ohlcv_panel(symbols, n_days=40)
    cache = tmp_path / "cache"
    cache.mkdir()
    panel_path = cache / "thematic_universe.parquet"
    panel.to_parquet(panel_path)
    yaml_path = tmp_path / "universe.yaml"
    _write_universe_yaml(yaml_path, symbols)
    out_path = cache / "features.parquet"
    asof = date(2024, 11, 8)

    runner = CliRunner()
    args = [
        "thematic", "compute", "--asof", asof.isoformat(),
        "--ohlcv", str(panel_path), "--yaml", str(yaml_path), "--out", str(out_path),
        "--ticker-registry", str(cache / "tr.parquet"),
        "--sector-registry", str(cache / "sr.parquet"),
    ]
    res = runner.invoke(cli, args)
    assert res.exit_code == 0, res.output
    feat_df = pd.read_parquet(out_path)
    assert _count(migrated_engine, "thematic_features_daily") == len(feat_df)

    # trading_day_ordinal made it into PG.
    with migrated_engine.connect() as conn:
        ordn = conn.execute(
            text(
                "SELECT trading_day_ordinal FROM market.thematic_features_daily "
                "WHERE symbol='AAA' AND asof_date=:a"
            ),
            {"a": asof},
        ).scalar_one()
    assert ordn is not None

    # Re-run --force: idempotent (same asof_date -> UPSERT, no duplicate).
    res2 = runner.invoke(cli, args + ["--force"])
    assert res2.exit_code == 0, res2.output
    assert _count(migrated_engine, "thematic_features_daily") == len(feat_df)


# ---------------------------------------------------------------------------
# thematic backfill-labels -> market.thematic_labels_daily
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_thematic_backfill_labels_dual_writes(migrated_engine, tmp_path, database_url):
    from rainier.cli import cli

    symbols = ["AAA", "BBB", "CCC"]
    panel = _build_ohlcv_panel(symbols, n_days=50)
    cache = tmp_path / "cache"
    cache.mkdir()
    panel_path = cache / "thematic_universe.parquet"
    panel.to_parquet(panel_path)
    out_path = cache / "labels.parquet"

    runner = CliRunner()
    args = ["thematic", "backfill-labels", "--ohlcv", str(panel_path), "--out", str(out_path)]
    res = runner.invoke(cli, args)
    assert res.exit_code == 0, res.output
    lbl_df = pd.read_parquet(out_path)
    assert _count(migrated_engine, "thematic_labels_daily") == len(lbl_df)

    # Idempotent re-run.
    res2 = runner.invoke(cli, args)
    assert res2.exit_code == 0, res2.output
    assert _count(migrated_engine, "thematic_labels_daily") == len(lbl_df)


@pytest.mark.requires_postgres
def test_labels_dual_write_handles_null_complete_through(
    migrated_engine, tmp_path, database_url
):
    """A short panel (<=30 trading days) yields label_complete_through=None for
    valid rows — the relaxed-NOT-NULL seam (migration 0002) must accept them."""
    from rainier.cli import cli

    symbols = ["AAA", "BBB"]
    panel = _build_ohlcv_panel(symbols, n_days=10)  # < 30 -> None completion
    cache = tmp_path / "cache"
    cache.mkdir()
    panel_path = cache / "thematic_universe.parquet"
    panel.to_parquet(panel_path)
    out_path = cache / "labels.parquet"

    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["thematic", "backfill-labels", "--ohlcv", str(panel_path), "--out", str(out_path)],
    )
    assert res.exit_code == 0, res.output
    lbl_df = pd.read_parquet(out_path)
    assert _count(migrated_engine, "thematic_labels_daily") == len(lbl_df)
    with migrated_engine.connect() as conn:
        n_null = conn.execute(
            text(
                "SELECT count(*) FROM market.thematic_labels_daily "
                "WHERE label_complete_through IS NULL"
            )
        ).scalar_one()
    assert n_null == len(lbl_df), "all rows should have NULL completion on a short panel"


# ---------------------------------------------------------------------------
# DATABASE_URL-unset skip path — NO Postgres needed
# ---------------------------------------------------------------------------


def test_backfill_skips_pg_when_database_url_unset(tmp_path, monkeypatch, capsys):
    """DATABASE_URL unset -> warn + skip PG + parquet still writes + exit 0."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    backfill = _import_backfill()
    symbols = ["AAA", "BBB"]
    panel = _build_ohlcv_panel(symbols, n_days=6)

    def fake_fetch(syms, start, end):
        return {
            s: panel.loc[
                panel["symbol"] == s,
                ["date", "open", "high", "low", "close", "volume"],
            ].reset_index(drop=True)
            for s in syms
        }

    yaml_path = tmp_path / "universe.yaml"
    _write_universe_yaml(yaml_path, symbols)
    out_path = tmp_path / "thematic_universe.parquet"

    written = backfill.backfill(
        symbols=symbols, start="2024-10-01", end="2024-10-31",
        out_path=out_path, fetch_fn=fake_fetch, yaml_path=yaml_path,
    )
    written = Path(written)
    assert written.exists(), "parquet must be written even with PG unset"
    df = pd.read_parquet(written)
    assert len(df) == len(panel)
    # A warning is surfaced (stderr/stdout) mentioning DATABASE_URL.
    captured = capsys.readouterr()
    combined = (captured.out + captured.err).lower()
    assert "database_url" in combined


def test_compute_skips_pg_when_database_url_unset(tmp_path, monkeypatch):
    """thematic compute with DATABASE_URL unset still writes parquet, exit 0."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from rainier.cli import cli

    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    panel = _build_ohlcv_panel(symbols, n_days=40)
    cache = tmp_path / "cache"
    cache.mkdir()
    panel_path = cache / "thematic_universe.parquet"
    panel.to_parquet(panel_path)
    yaml_path = tmp_path / "universe.yaml"
    _write_universe_yaml(yaml_path, symbols)
    out_path = cache / "features.parquet"

    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "thematic", "compute", "--asof", "2024-11-08",
            "--ohlcv", str(panel_path), "--yaml", str(yaml_path), "--out", str(out_path),
            "--ticker-registry", str(cache / "tr.parquet"),
            "--sector-registry", str(cache / "sr.parquet"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert len(df) == 5
