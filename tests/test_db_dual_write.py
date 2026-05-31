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
        # Drop the schema directly rather than `downgrade base`: these tests
        # write data (incl. NULL label_complete_through rows), and the 0002
        # downgrade deliberately refuses to re-tighten NOT NULL over NULL rows.
        # Teardown wants an unconditional reset, so we DROP CASCADE and clear
        # alembic's bookkeeping so the next test's `upgrade head` starts clean.
        with eng.begin() as conn:
            conn.exec_driver_sql("DROP SCHEMA IF EXISTS market CASCADE")
            conn.exec_driver_sql("DROP TABLE IF EXISTS public.alembic_version")
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
        min_coverage=0.0,
        ticker_registry_path=tmp_path / "tr.parquet",
        sector_registry_path=tmp_path / "sr.parquet",
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
        min_coverage=0.0,
        ticker_registry_path=tmp_path / "tr.parquet",
        sector_registry_path=tmp_path / "sr.parquet",
    )
    n_ohlcv = _count(migrated_engine, "thematic_ohlcv")
    n_tick = _count(migrated_engine, "tickers")

    # Second run, same data -> idempotent UPSERT, no duplicate rows.
    backfill.backfill(
        symbols=symbols, start="2024-10-01", end="2024-10-31",
        out_path=tmp_path / "u2.parquet", fetch_fn=fake_fetch, yaml_path=yaml_path,
        min_coverage=0.0,
        ticker_registry_path=tmp_path / "tr.parquet",
        sector_registry_path=tmp_path / "sr.parquet",
    )
    assert _count(migrated_engine, "thematic_ohlcv") == n_ohlcv
    assert _count(migrated_engine, "tickers") == n_tick


@pytest.mark.requires_postgres
def test_backfill_and_compute_share_stable_ids_across_yaml_reorder(
    migrated_engine, tmp_path, database_url
):
    """Regression: backfill must use the SAME persistent registry as compute so
    IDs stay consistent even when the YAML is reordered between runs. A local
    counter would re-number by YAML order and collide with the compute path's
    persistent IDs on the market.tickers.symbol UNIQUE constraint."""
    from rainier.cli import cli

    backfill = _import_backfill()
    symbols = ["AAA", "BBB", "CCC"]
    panel = _build_ohlcv_panel(symbols, n_days=40)

    def fake_fetch(syms, start, end):
        return {
            s: panel.loc[
                panel["symbol"] == s,
                ["date", "open", "high", "low", "close", "volume"],
            ].reset_index(drop=True)
            for s in syms
        }

    cache = tmp_path / "cache"
    cache.mkdir()
    tr = cache / "tr.parquet"
    sr = cache / "sr.parquet"
    panel_path = cache / "thematic_universe.parquet"
    panel.to_parquet(panel_path)

    # Backfill seeds the persistent registry in YAML order AAA,BBB,CCC.
    yaml1 = tmp_path / "u1.yaml"
    _write_universe_yaml(yaml1, ["AAA", "BBB", "CCC"])
    backfill.backfill(
        symbols=symbols, start="2024-10-01", end="2024-10-31",
        out_path=cache / "u.parquet", fetch_fn=fake_fetch, yaml_path=yaml1,
        min_coverage=0.0, ticker_registry_path=tr, sector_registry_path=sr,
    )
    # Capture the id the backfill assigned to AAA.
    with migrated_engine.connect() as conn:
        aaa_id_after_backfill = conn.execute(
            text("SELECT ticker_id FROM market.tickers WHERE symbol='AAA'")
        ).scalar_one()

    # Now compute with a REORDERED YAML (CCC first). A local counter would give
    # AAA a different id here; the shared persistent registry keeps it stable.
    yaml2 = tmp_path / "u2.yaml"
    _write_universe_yaml(yaml2, ["CCC", "BBB", "AAA"])
    res = CliRunner().invoke(
        cli,
        [
            "thematic", "compute", "--asof", "2024-11-08",
            "--ohlcv", str(panel_path), "--yaml", str(yaml2),
            "--out", str(cache / "features.parquet"),
            "--ticker-registry", str(tr), "--sector-registry", str(sr),
        ],
    )
    assert res.exit_code == 0, res.output  # no UNIQUE(symbol) collision

    with migrated_engine.connect() as conn:
        # Exactly one row per symbol — no duplicate AAA under two ids.
        n_aaa = conn.execute(
            text("SELECT count(*) FROM market.tickers WHERE symbol='AAA'")
        ).scalar_one()
        aaa_id_after_compute = conn.execute(
            text("SELECT ticker_id FROM market.tickers WHERE symbol='AAA'")
        ).scalar_one()
    assert n_aaa == 1, "symbol must not be duplicated across backfill+compute"
    assert aaa_id_after_compute == aaa_id_after_backfill, "ticker_id must stay stable"


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


@pytest.mark.requires_postgres
def test_frame_to_pg_rows_omits_absent_optional_column(migrated_engine):
    """Regression (codex P2): a frame lacking trading_day_ordinal must NOT
    null out an existing ordinal on re-upsert. _frame_to_pg_rows omits the
    absent column so market_upsert leaves the prior PG value intact."""
    from rainier.cli import _frame_to_pg_rows
    from rainier.db import schema
    from rainier.db.upsert import market_upsert

    # Parents for the FK.
    market_upsert(
        migrated_engine, schema.tickers,
        [{"ticker_id": 1, "symbol": "AAA", "first_seen": date(2024, 1, 1)}],
        ["ticker_id"], immutable_cols=["first_seen"],
    )
    market_upsert(
        migrated_engine, schema.sectors,
        [{"sector_id": 1, "sector_name": "tech", "first_seen": date(2024, 1, 1)}],
        ["sector_id"], immutable_cols=["first_seen"],
    )

    feature_cols = list(schema.thematic_features_daily.columns.keys())
    base = {
        "asof_date": date(2024, 11, 8), "trading_day_ordinal": 42, "symbol": "AAA",
        "ticker_id": 1, "sector_id": 1, "close": 100.0, "rank": 1,
        "rank_delta_1d": 0, "rank_delta_5d": 0, "top15_streak": 0,
        "universe_size": 1, "universe_yaml_sha": "deadbeef",
        "computed_at": pd.Timestamp("2024-11-08T16:30:00Z"),
    }
    # First write WITH the ordinal.
    df_with = pd.DataFrame([base])
    market_upsert(
        migrated_engine, schema.thematic_features_daily,
        _frame_to_pg_rows(df_with, feature_cols), ["asof_date", "symbol"],
    )
    # Re-write the SAME key from a frame that LACKS trading_day_ordinal.
    base_no_ord = {k: v for k, v in base.items() if k != "trading_day_ordinal"}
    df_without = pd.DataFrame([base_no_ord])
    rows = _frame_to_pg_rows(df_without, feature_cols)
    assert "trading_day_ordinal" not in rows[0], "absent column must be omitted, not None"
    market_upsert(
        migrated_engine, schema.thematic_features_daily, rows, ["asof_date", "symbol"],
    )
    with migrated_engine.connect() as conn:
        ordn = conn.execute(
            text(
                "SELECT trading_day_ordinal FROM market.thematic_features_daily "
                "WHERE symbol='AAA' AND asof_date=:a"
            ),
            {"a": date(2024, 11, 8)},
        ).scalar_one()
    assert ordn == 42, "existing ordinal must survive a re-upsert that omits the column"


# ---------------------------------------------------------------------------
# thematic backfill-labels -> market.thematic_labels_daily
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
def test_compute_mirror_uses_registry_first_seen_not_asof(
    migrated_engine, tmp_path, database_url
):
    """Regression (codex P2): when the registry parquet already records an
    older first_seen, the PG ticker/sector rows must carry that date, not the
    current compute asof_dt (first_seen is insert-only, so the first PG write
    is the only chance to get provenance right)."""
    from rainier.breadth import registry as _reg
    from rainier.cli import cli

    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    panel = _build_ohlcv_panel(symbols, n_days=40)
    cache = tmp_path / "cache"
    cache.mkdir()
    panel_path = cache / "thematic_universe.parquet"
    panel.to_parquet(panel_path)
    yaml_path = tmp_path / "universe.yaml"
    _write_universe_yaml(yaml_path, symbols)
    tr = cache / "tr.parquet"
    sr = cache / "sr.parquet"

    # Pre-seed the registry with an OLD first_seen (PG not yet enabled then).
    universe = {"test_sector": symbols}
    _reg.seed_registries_from_universe(
        universe, asof=date(2024, 1, 1),
        ticker_registry_path=tr, sector_registry_path=sr,
    )

    # Now compute with a much later asof; PG enabled. first_seen must be the
    # registry's 2024-01-01, not 2024-11-08.
    res = CliRunner().invoke(
        cli,
        [
            "thematic", "compute", "--asof", "2024-11-08",
            "--ohlcv", str(panel_path), "--yaml", str(yaml_path),
            "--out", str(cache / "features.parquet"),
            "--ticker-registry", str(tr), "--sector-registry", str(sr),
        ],
    )
    assert res.exit_code == 0, res.output
    with migrated_engine.connect() as conn:
        fs = conn.execute(
            text("SELECT first_seen FROM market.tickers WHERE symbol='AAA'")
        ).scalar_one()
    assert fs == date(2024, 1, 1), "PG first_seen must mirror the registry, not asof"


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
# thematic backfill --incremental (CLI surface) -> market.thematic_ohlcv MUST
# advance. Regression for the daily-cron mirror gap: the CLI subcommand
# `thematic backfill --incremental` (cli.py:thematic_backfill) must forward
# yaml_path into module.backfill(...) so _dual_write_pg mirrors the freshly
# fetched OHLCV into Neon. Before the fix the CLI omitted yaml_path, so the
# cron-wired `thematic backfill --incremental` advanced ONLY the parquet and
# left market.thematic_ohlcv STALE (live catch-up 2026-05-30: features/labels
# reached today but thematic_ohlcv was stuck days behind). The worker's
# module-level dual-write tests pass yaml_path explicitly, so they could not
# catch this — the broken seam was the CLI command, which this test drives.
# ---------------------------------------------------------------------------


def _patch_yfinance(monkeypatch, panel: pd.DataFrame) -> None:
    """Make the real ``_yfinance_fetch`` offline by stubbing ``yf.download``.

    The CLI ``thematic backfill`` command resolves its fetcher internally
    (no fetch_fn injection point), so to exercise the real CLI seam we stub
    the network call one layer down. ``_yfinance_fetch`` bumps ``end`` by +1
    day (exclusive end) and slices [start, end); we honor that window here so
    the incremental path's recent-window resolution is exercised end to end.
    """
    import types

    def fake_download(sym, start=None, end=None, **kwargs):
        start_d = pd.to_datetime(start).date()
        end_d = pd.to_datetime(end).date()  # already +1 (exclusive) from caller
        sub = panel.loc[
            (panel["symbol"] == sym)
            & (panel["date"] >= start_d)
            & (panel["date"] < end_d),
            ["date", "open", "high", "low", "close", "volume"],
        ].copy()
        if sub.empty:
            return pd.DataFrame()
        # _yfinance_fetch expects yfinance's capitalized columns + a DatetimeIndex.
        sub = sub.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        sub = sub.set_index("Date")
        return sub

    fake_yf = types.SimpleNamespace(download=fake_download, __version__="stub-test")
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)


@pytest.mark.requires_postgres
def test_cli_thematic_backfill_incremental_advances_pg_ohlcv(
    migrated_engine, tmp_path, database_url, monkeypatch
):
    """`thematic backfill --incremental` (CLI) must ADVANCE market.thematic_ohlcv
    in Postgres, not just the parquet — proves the daily cron keeps Neon's raw
    OHLCV fresh. This is the load-bearing assertion for the daily-cron wire.
    """
    from rainier.cli import cli

    symbols = ["AAA", "BBB", "CCC"]
    # Derive `today` from the REAL clock. The CLI loads the backfill script as a
    # FRESH module instance (importlib.util.module_from_spec), so a monkeypatch
    # of any pre-imported module's date.today() would NOT reach it — the CLI
    # always uses the real date.today(). Building the synthetic recent window
    # relative to the real today keeps this test clock-independent: whatever
    # incremental window backfill() computes (today-5..today), our stub panel
    # covers it. (codex iter-2: don't freeze a module the CLI never reuses.)
    today = date.today()

    # An existing parquet cache whose high-water mark is INSIDE the incremental
    # window (gap guard satisfied), with dates distinct from the recent fetch
    # window. The dual-write mirrors only the freshly fetched frame, so the
    # seed rows never reach PG — PG advances by exactly the recent window.
    seed_dates = [today - timedelta(days=d) for d in (5, 4, 3)]
    seed_rows = [
        {
            "symbol": s, "date": d, "open": 50.0, "high": 50.5, "low": 49.5,
            "close": 50.0, "volume": 1_000_000,
            "fetched_at": pd.Timestamp.now(tz="UTC"), "yfinance_version": "seed",
        }
        for s in symbols
        for d in seed_dates
    ]
    cache = tmp_path / "cache"
    cache.mkdir()
    panel_path = cache / "thematic_universe.parquet"
    pd.DataFrame(seed_rows).to_parquet(panel_path)
    tr = cache / "tr.parquet"
    sr = cache / "sr.parquet"
    yaml_path = tmp_path / "universe.yaml"
    _write_universe_yaml(yaml_path, symbols)

    # Recent-window rows the incremental fetch should pick up. All within
    # backfill()'s window (today-INCREMENTAL_WINDOW_DAYS .. today), so the
    # windowed stub returns them regardless of what real date the CLI sees.
    recent_dates = [today - timedelta(days=2), today - timedelta(days=1), today]
    recent_rows = []
    for s_idx, sym in enumerate(symbols):
        for j, d in enumerate(recent_dates):
            close = 200.0 + s_idx * 10 + j
            recent_rows.append(
                {
                    "symbol": sym, "date": d, "open": close, "high": close * 1.01,
                    "low": close * 0.99, "close": close, "volume": 2_000_000,
                    "fetched_at": pd.Timestamp.now(tz="UTC"),
                    "yfinance_version": "stub-test",
                }
            )
    recent_panel = pd.DataFrame(recent_rows)
    _patch_yfinance(monkeypatch, recent_panel)

    # PG starts with zero OHLCV rows.
    assert _count(migrated_engine, "thematic_ohlcv") == 0

    res = CliRunner().invoke(
        cli,
        [
            "thematic", "backfill", "--incremental",
            "--yaml", str(yaml_path),
            "--out", str(panel_path),
            "--ticker-registry", str(tr),
            "--sector-registry", str(sr),
        ],
    )
    assert res.exit_code == 0, res.output

    # The recent window landed in PG (3 symbols x 3 recent dates = 9 rows).
    pg_after = _count(migrated_engine, "thematic_ohlcv")
    assert pg_after == len(symbols) * len(recent_dates), (
        f"market.thematic_ohlcv must ADVANCE on the incremental CLI path; "
        f"got {pg_after} rows, expected {len(symbols) * len(recent_dates)}. "
        f"output:\n{res.output}"
    )
    # Registries mirrored too (the daily cron keeps the FK parents fresh).
    assert _count(migrated_engine, "tickers") == len(symbols)
    assert _count(migrated_engine, "sectors") == 1

    # max(date) advanced to today -> run-daily's stale-OHLCV guard passes.
    with migrated_engine.connect() as conn:
        max_date = conn.execute(
            text("SELECT max(date) FROM market.thematic_ohlcv")
        ).scalar_one()
        sample_close = conn.execute(
            text(
                "SELECT close FROM market.thematic_ohlcv "
                "WHERE symbol='AAA' AND date=:d"
            ),
            {"d": today},
        ).scalar_one()
    assert max_date == today, "thematic_ohlcv must reach today's date"
    assert sample_close == pytest.approx(202.0), "today's AAA close mirrored to PG"

    # Idempotent: a second incremental run over the same window adds no rows.
    res2 = CliRunner().invoke(
        cli,
        [
            "thematic", "backfill", "--incremental",
            "--yaml", str(yaml_path),
            "--out", str(panel_path),
            "--ticker-registry", str(tr),
            "--sector-registry", str(sr),
        ],
    )
    assert res2.exit_code == 0, res2.output
    assert _count(migrated_engine, "thematic_ohlcv") == pg_after, (
        "re-running incremental must be idempotent (UPSERT, no duplicate rows)"
    )


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
        min_coverage=0.0,
    )
    written = Path(written)
    assert written.exists(), "parquet must be written even with PG unset"
    df = pd.read_parquet(written)
    assert len(df) == len(panel)
    # A warning is surfaced (stderr/stdout) mentioning DATABASE_URL.
    captured = capsys.readouterr()
    combined = (captured.out + captured.err).lower()
    assert "database_url" in combined


def test_mirror_guard_swallows_sqlalchemy_error(monkeypatch, capsys):
    """DATABASE_URL set but PG unreachable -> the SQLAlchemyError raised inside
    the mirror body is caught + warned, NOT propagated (parquet load-bearing)."""
    from sqlalchemy.exc import SQLAlchemyError

    from rainier.db.dualwrite import mirror_guard

    # Point at a host that cannot connect; the engine object is created lazily,
    # so begin()/execute() inside the body is what raises.
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://nouser@127.0.0.1:1/nodb"
    )
    with mirror_guard("unit-test-writer") as eng:
        assert eng is not None, "engine object should be returned when URL is set"
        # Simulate the upsert raising an operational error.
        with eng.begin():  # pragma: no cover - the connect itself raises
            pass
    # No exception escaped; a warning naming the writer was emitted.
    combined = (capsys.readouterr().err).lower()
    assert "unit-test-writer" in combined
    assert "dual-write failed" in combined

    # Sanity: a non-SQLAlchemy error inside the body still propagates.
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://nouser@127.0.0.1:1/nodb")
    with pytest.raises(ValueError):
        with mirror_guard("unit-test-writer"):
            raise ValueError("programmer bug must not be swallowed")
    # Guard against the SQLAlchemyError import being unused if the body changes.
    assert issubclass(SQLAlchemyError, Exception)


# ---------------------------------------------------------------------------
# mirror_guard loud-on-failure diagnostic (design DESIGN-mirror-guard-loud-on-
# failure.md §4, items 1-10). All assertions target STDERR specifically.
# NO Postgres needed — these drive the guard + its helpers directly.
# ---------------------------------------------------------------------------

_SENTINEL = "PG-MIRROR-FAILURE"


def test_mirror_guard_loud_on_body_failure(monkeypatch, capsys):
    """§4.1 — DATABASE_URL set, body raises SQLAlchemyError -> loud diagnostic on
    stderr with sentinel + redacted host + error class; no exception escapes;
    password substring absent."""
    from sqlalchemy.exc import OperationalError

    from rainier.db.dualwrite import mirror_guard

    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://bob:hunter2@db.example.com:5432/mirror"
    )
    with mirror_guard("unit-writer") as eng:
        assert eng is not None, "engine object returned when URL is set"
        raise OperationalError("boom", None, Exception("conn refused"))

    err = capsys.readouterr().err
    assert _SENTINEL in err, "loud sentinel must be on stderr"
    assert "unit-writer" in err
    assert "OperationalError" in err, "error class named"
    assert "db.example.com" in err, "redacted host present"
    assert "hunter2" not in err, "password must not leak"
    assert "bob" not in err, "username must not leak"


def test_mirror_guard_before_yield_failure_is_non_fatal(monkeypatch, capsys):
    """§4.2 — engine creation raises SQLAlchemyError BEFORE the yield. Today this
    escapes as `RuntimeError: generator didn't yield` and aborts the caller. The
    fix must emit the loud diagnostic AND still yield None so the body runs."""
    from sqlalchemy.exc import ArgumentError

    import rainier.db.dualwrite as dw

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")

    def boom(_writer):
        raise ArgumentError("malformed URL")

    monkeypatch.setattr(dw, "pg_engine_or_skip", boom)

    body_ran_with_none = False
    # Must NOT raise RuntimeError: generator didn't yield.
    with dw.mirror_guard("before-yield-writer") as eng:
        body_ran_with_none = eng is None

    assert body_ran_with_none, "body must run with eng is None after engine-creation failure"
    err = capsys.readouterr().err
    assert _SENTINEL in err, "before-yield failure must still emit loud diagnostic"
    assert "ArgumentError" in err


def test_mirror_guard_malformed_port_valueerror_is_non_fatal(monkeypatch, capsys):
    """codex iter-2 — a malformed DATABASE_URL with a non-numeric port makes
    make_url/create_engine raise a bare ``ValueError`` (int('notaport')) BEFORE
    the yield, not a SQLAlchemyError. That must still be non-fatal: emit the loud
    diagnostic and yield None, not abort the caller (design §3.1 — any
    engine-creation failure on a SET url is non-fatal)."""
    from rainier.db.dualwrite import mirror_guard

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host:notaport/db")

    body_ran_with_none = False
    # Must NOT escape as ValueError nor RuntimeError: generator didn't yield.
    with mirror_guard("malformed-port-writer") as eng:
        body_ran_with_none = eng is None

    assert body_ran_with_none, "malformed-port URL must yield None, not abort"
    err = capsys.readouterr().err
    assert _SENTINEL in err, "malformed-port failure must emit loud diagnostic"
    assert "ValueError" in err, "the bare ValueError class must be named"


def test_mirror_guard_unset_stays_quiet(monkeypatch, capsys):
    """§4.3 — DATABASE_URL unset -> the benign skip warning, NO sentinel."""
    from rainier.db.dualwrite import mirror_guard

    monkeypatch.delenv("DATABASE_URL", raising=False)
    with mirror_guard("quiet-writer") as eng:
        assert eng is None
    err = capsys.readouterr().err
    assert _SENTINEL not in err, "unset case must not emit the loud sentinel"
    assert "database_url" in err.lower(), "benign skip warning preserved"


def test_mirror_guard_success_is_silent(monkeypatch, capsys):
    """§4.4 — set + body succeeds -> no sentinel."""
    import rainier.db.dualwrite as dw

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")

    class _FakeEngine:
        def dispose(self):
            pass

    monkeypatch.setattr(dw, "pg_engine_or_skip", lambda _w: _FakeEngine())

    with dw.mirror_guard("ok-writer") as eng:
        assert eng is not None  # body succeeds, no raise

    err = capsys.readouterr().err
    assert _SENTINEL not in err, "successful mirror must be silent"


def test_mirror_guard_non_sqlalchemy_error_propagates(monkeypatch):
    """§4.5 — body raises ValueError (programmer bug) -> it escapes."""
    from rainier.db.dualwrite import mirror_guard

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://nouser@127.0.0.1:1/nodb")
    with pytest.raises(ValueError):
        with mirror_guard("propagate-writer"):
            raise ValueError("programmer bug must not be swallowed")


@pytest.mark.parametrize(
    ("url", "must_absent", "must_present"),
    [
        # credentialed (user:pass)
        ("postgresql://bob:hunter2@db.example.com:5432/mirror", ["bob", "hunter2"], ["db.example.com"]),
        # password-less userinfo — username must NOT survive
        ("postgresql://bob@db.example.com/mirror", ["bob"], ["db.example.com"]),
        # percent-encoded password
        ("postgresql://bob:p%40ss%21@db.example.com/mirror", ["bob", "p@ss", "p%40ss"], ["db.example.com"]),
        # IPv6 host
        ("postgresql://bob:hunter2@[2001:db8::1]:5432/mirror", ["bob", "hunter2"], ["2001:db8::1"]),
        # query-string options
        ("postgresql://bob:hunter2@db.example.com/mirror?sslmode=require", ["bob", "hunter2"], ["db.example.com"]),
        # driver-prefixed
        ("postgresql+psycopg://bob:hunter2@db.example.com/mirror", ["bob", "hunter2"], ["db.example.com"]),
    ],
)
def test_redact_host_never_leaks_creds(url, must_absent, must_present):
    """§4.6 — _redact_host renders host/port/db only; neither username nor
    password survives; never raises."""
    from rainier.db.dualwrite import _redact_host

    out = _redact_host(url)
    for s in must_absent:
        assert s not in out, f"{s!r} must not survive in {out!r}"
    for s in must_present:
        assert s in out, f"{s!r} should be in {out!r}"


def test_redact_host_malformed_returns_unparseable():
    """§4.6 — malformed URL -> '<unparseable>'; never raises."""
    from rainier.db.dualwrite import _redact_host

    for bad in ["not a url at all", "://://", "", "@@@", "postgres://:::"]:
        out = _redact_host(bad)
        assert out == "<unparseable>" or "<unparseable>" in out
        # never echoes raw garbage credentials
    assert _redact_host("postgresql://u:secret@") == "<unparseable>" or "secret" not in _redact_host(
        "postgresql://u:secret@"
    )


def test_redact_host_unescaped_at_in_password_does_not_leak():
    """codex iter-5 — an unescaped '@' in the password (e.g.
    postgresql://u:pa@ss@db/prod) makes make_url parse host as `ss@db`, leaking
    the password fragment `ss`. _redact_host must treat an '@'-bearing host as
    unparseable, never rendering it."""
    from rainier.db.dualwrite import _redact_host

    for url in [
        "postgresql://u:pa@ss@db/prod",
        "postgresql://user:p@ssw0rd@host:5432/db",
        "postgresql+psycopg://u:a@b@c/d",
    ]:
        out = _redact_host(url)
        assert out == "<unparseable>", f"@-in-password host must be unparseable, got {out!r}"
        assert "ss" not in out and "ssw0rd" not in out


def test_scrub_credentials_url_and_keyvalue_forms():
    """§4.7 — _scrub_credentials drops URL userinfo (with/without password) and
    password=/pwd= key-value forms; credential-free text unchanged; never
    raises."""
    from rainier.db.dualwrite import _scrub_credentials

    # URL with user:pass embedded in arbitrary error text
    t1 = "could not connect: postgresql://user:secret@host/db (timeout)"
    s1 = _scrub_credentials(t1)
    assert "secret" not in s1
    assert "user" not in s1.replace("could not", "")  # the username 'user' is scrubbed
    assert "postgresql://@host/db" in s1

    # password-less userinfo — username must not survive
    t2 = "url=postgresql://user@host/db here"
    s2 = _scrub_credentials(t2)
    assert "postgresql://@host/db" in s2

    # unescaped '@' INSIDE the password — the WHOLE userinfo (both @s) must go,
    # not just up to the first '@' (codex iter-6 regression: an old class that
    # stopped at the first '@' left the `ss@db` password fragment behind).
    t3 = "could not connect: postgresql://u:pa@ss@db/prod (timeout)"
    s3 = _scrub_credentials(t3)
    assert "pa" not in s3, f"password fragment leaked: {s3!r}"
    assert "ss" not in s3, f"password fragment leaked: {s3!r}"
    assert "postgresql://@db/prod" in s3

    # key/value forms
    assert "secret" not in _scrub_credentials("password=secret extra")
    assert "secret" not in _scrub_credentials("'password': 'secret'")
    assert "secret" not in _scrub_credentials("pwd=secret;host=x")
    assert "secret" not in _scrub_credentials('"password": "secret"')

    # quoted value with embedded whitespace / separators must be consumed WHOLE
    # (codex iter-1 regression: the value class used to stop at the first space,
    # leaving the secret tail behind as `password=***'foo bar'`).
    for whitespace_form in [
        "password='foo bar'",
        'password="foo bar"',
        "password='foo,bar;baz'",
        "'password': 'foo bar baz'",
    ]:
        scrubbed = _scrub_credentials(whitespace_form)
        assert "foo" not in scrubbed, f"quoted secret leaked: {scrubbed!r}"
        assert "bar" not in scrubbed, f"quoted secret leaked: {scrubbed!r}"
        assert "baz" not in scrubbed, f"quoted secret leaked: {scrubbed!r}"
        assert "***" in scrubbed

    # quoted value containing the OPPOSITE quote char must stop only at the
    # matching closing quote (codex iter-3 regression: an either-quote class let
    # `password="foo'bar"` leak the tail after ***).
    for embedded_quote_form in [
        "password=\"foo'bar\"",
        "password='foo\"bar'",
        "password='foo\"bar baz'",
    ]:
        scrubbed = _scrub_credentials(embedded_quote_form)
        assert "foo" not in scrubbed, f"embedded-quote secret leaked: {scrubbed!r}"
        assert "bar" not in scrubbed, f"embedded-quote secret leaked: {scrubbed!r}"
        assert "***" in scrubbed

    # backslash-ESCAPED quote inside the value (JSON-serialized connect args) must
    # be consumed as part of the secret (codex iter-4 regression: a quoted branch
    # that stopped at the escaped quote leaked the tail `bar`).
    for escaped_quote_form in [
        '"password": "foo\\"bar"',
        "'password': 'foo\\'bar'",
        '"password": "foo\\"bar baz"',
    ]:
        scrubbed = _scrub_credentials(escaped_quote_form)
        assert "foo" not in scrubbed, f"escaped-quote secret leaked: {scrubbed!r}"
        assert "bar" not in scrubbed, f"escaped-quote secret leaked: {scrubbed!r}"
        assert "baz" not in scrubbed, f"escaped-quote secret leaked: {scrubbed!r}"
        assert "***" in scrubbed

    # credential-free / malformed unchanged + no raise
    for clean in ["just an error message", "", "host=db port=5432", "no creds here"]:
        assert _scrub_credentials(clean) == clean


def test_diagnostic_scrubs_credential_in_error_message(monkeypatch, capsys):
    """§4.7 end-to-end — when the raised SQLAlchemyError's MESSAGE itself carries
    a credentialed URL, the printed diagnostic must be scrubbed."""
    from sqlalchemy.exc import OperationalError

    from rainier.db.dualwrite import mirror_guard

    monkeypatch.setenv("DATABASE_URL", "postgresql://safe@host/db")
    with mirror_guard("scrub-writer"):
        raise OperationalError(
            "FATAL: password authentication failed for postgresql://admin:topsecret@host/db",
            None,
            Exception("auth"),
        )
    err = capsys.readouterr().err
    assert _SENTINEL in err
    assert "topsecret" not in err, "credential in error message must be scrubbed"
    assert "admin" not in err, "username in error message must be scrubbed"


def test_mirror_guard_disposes_engine_on_caught_error(monkeypatch, capsys):
    """§4.8 — dispose() runs even when the body raises SQLAlchemyError."""
    from sqlalchemy.exc import OperationalError

    import rainier.db.dualwrite as dw

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")

    disposed = {"called": False}

    class _FakeEngine:
        def dispose(self):
            disposed["called"] = True

    monkeypatch.setattr(dw, "pg_engine_or_skip", lambda _w: _FakeEngine())

    with dw.mirror_guard("dispose-writer"):
        raise OperationalError("boom", None, Exception("x"))

    assert disposed["called"], "engine must be disposed on the caught-error path"
    assert _SENTINEL in capsys.readouterr().err


def test_mirror_guard_empty_database_url_stays_quiet(monkeypatch, capsys):
    """§4.9 — DATABASE_URL='' (empty) is treated as unset -> quiet skip, no
    sentinel (pins the empty=unset decision)."""
    from rainier.db.dualwrite import mirror_guard

    monkeypatch.setenv("DATABASE_URL", "")
    with mirror_guard("empty-writer") as eng:
        assert eng is None
    err = capsys.readouterr().err
    assert _SENTINEL not in err, "empty DATABASE_URL must stay quiet"


def test_redact_host_socket_renders_meaningfully():
    """§4.10 — a host-less / socket URL must render the socket path (and db when
    available), not a bare db name, and leak no creds."""
    from rainier.db.dualwrite import _redact_host

    url = "postgresql://bob:hunter2@/mirror?host=/var/run/postgresql"
    out = _redact_host(url)
    assert "/var/run/postgresql" in out, f"socket path must appear in {out!r}"
    assert out != "mirror", "bare db name is not a meaningful host"
    assert "hunter2" not in out
    assert "bob" not in out


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
