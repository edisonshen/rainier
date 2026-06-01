"""End-to-end CLI smoke test for `rainier thematic run-daily`.

Drives the full path: stub OHLCV → compute Layer A → compute Layer B → render HTML.
Re-running with same inputs must short-circuit (idempotent).

Design ref: docs/DESIGN-thematic-ranks-dashboard.md §7 ([D-004]).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner


def _build_ohlcv_panel(symbols: list[str], n_days: int) -> pd.DataFrame:
    """Synth OHLCV panel matching the schema produced by backfill_thematic_universe."""
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
                close = close * (1.0 + 0.005 * (s_idx - 2))  # spread of returns
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


@pytest.fixture
def fake_cache(tmp_path: Path) -> dict[str, Path]:
    """Stage thematic_universe.parquet, registries, and a YAML universe.

    Returns a dict of paths that the run-daily CLI consumes.
    """
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    panel = _build_ohlcv_panel(symbols, n_days=60)

    cache_dir = tmp_path / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    panel_path = cache_dir / "thematic_universe.parquet"
    panel.to_parquet(panel_path)

    # Minimal universe YAML grouping tickers by single sector.
    yaml_path = tmp_path / "thematic_universe.yaml"
    yaml_path.write_text(
        "version: 1\n"
        "schema: thematic_universe.v1\n"
        "asof_seeded: 2024-10-01\n"
        "universe:\n"
        "  test_sector:\n"
        "    - AAA\n"
        "    - BBB\n"
        "    - CCC\n"
        "    - DDD\n"
        "    - EEE\n"
    )

    return {
        "panel": panel_path,
        "yaml": yaml_path,
        "features_out": cache_dir / "thematic_features_daily.parquet",
        "labels_out": cache_dir / "thematic_labels_daily.parquet",
        "log_out": cache_dir / "thematic_universe_log.parquet",
        "ticker_registry": cache_dir / "ticker_registry.parquet",
        "sector_registry": cache_dir / "sector_registry.parquet",
        "html_out": docs_dir / "thematic-ranks-latest.html",
        "docs_dir": docs_dir,
    }


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


def test_thematic_compute_writes_features_parquet(fake_cache):
    """`thematic compute --asof <date>` populates thematic_features_daily.parquet."""
    from rainier.cli import cli

    runner = CliRunner()
    asof = date(2024, 11, 8)
    result = runner.invoke(
        cli,
        [
            "thematic",
            "compute",
            "--asof",
            asof.isoformat(),
            "--ohlcv",
            str(fake_cache["panel"]),
            "--yaml",
            str(fake_cache["yaml"]),
            "--out",
            str(fake_cache["features_out"]),
            "--ticker-registry",
            str(fake_cache["ticker_registry"]),
            "--sector-registry",
            str(fake_cache["sector_registry"]),
        ],
    )
    assert result.exit_code == 0, (
        f"compute failed: exit={result.exit_code} output={result.output}"
    )
    assert fake_cache["features_out"].exists()
    df = pd.read_parquet(fake_cache["features_out"])
    assert len(df) == 5  # 5 tickers, one asof_date
    assert (df["asof_date"] == asof).all()


def test_thematic_backfill_labels_writes_labels_parquet(fake_cache):
    """`thematic backfill-labels` populates thematic_labels_daily.parquet."""
    from rainier.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "thematic",
            "backfill-labels",
            "--ohlcv",
            str(fake_cache["panel"]),
            "--out",
            str(fake_cache["labels_out"]),
        ],
    )
    assert result.exit_code == 0, (
        f"backfill-labels failed: exit={result.exit_code} output={result.output}"
    )
    assert fake_cache["labels_out"].exists()


def test_thematic_render_writes_html(fake_cache):
    """`thematic render` writes a self-contained HTML file."""
    from rainier.cli import cli

    runner = CliRunner()
    # First need a features parquet
    runner.invoke(
        cli,
        [
            "thematic",
            "compute",
            "--asof",
            "2024-11-08",
            "--ohlcv",
            str(fake_cache["panel"]),
            "--yaml",
            str(fake_cache["yaml"]),
            "--out",
            str(fake_cache["features_out"]),
            "--ticker-registry",
            str(fake_cache["ticker_registry"]),
            "--sector-registry",
            str(fake_cache["sector_registry"]),
        ],
    )

    result = runner.invoke(
        cli,
        [
            "thematic",
            "render",
            "--asof",
            "2024-11-08",
            "--features",
            str(fake_cache["features_out"]),
            "--yaml",
            str(fake_cache["yaml"]),
            "--out",
            str(fake_cache["html_out"]),
        ],
    )
    assert result.exit_code == 0, (
        f"render failed: exit={result.exit_code} output={result.output}"
    )
    assert fake_cache["html_out"].exists()
    html = fake_cache["html_out"].read_text()
    assert "AAA" in html and "EEE" in html


def test_thematic_run_daily_writes_all_artifacts(fake_cache):
    """`thematic run-daily` writes features parquet + labels parquet + HTML."""
    from rainier.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "thematic",
            "run-daily",
            "--asof",
            "2024-11-08",
            "--ohlcv",
            str(fake_cache["panel"]),
            "--yaml",
            str(fake_cache["yaml"]),
            "--features-out",
            str(fake_cache["features_out"]),
            "--labels-out",
            str(fake_cache["labels_out"]),
            "--ticker-registry",
            str(fake_cache["ticker_registry"]),
            "--sector-registry",
            str(fake_cache["sector_registry"]),
            "--html-out",
            str(fake_cache["html_out"]),
        ],
    )
    assert result.exit_code == 0, (
        f"run-daily failed: exit={result.exit_code} output={result.output}"
    )
    assert fake_cache["features_out"].exists()
    assert fake_cache["labels_out"].exists()
    assert fake_cache["html_out"].exists()


def test_thematic_run_daily_is_idempotent(fake_cache):
    """Re-running run-daily for same asof leaves parquets unchanged."""
    from rainier.cli import cli

    runner = CliRunner()
    args = [
        "thematic",
        "run-daily",
        "--asof",
        "2024-11-08",
        "--ohlcv",
        str(fake_cache["panel"]),
        "--yaml",
        str(fake_cache["yaml"]),
        "--features-out",
        str(fake_cache["features_out"]),
        "--labels-out",
        str(fake_cache["labels_out"]),
        "--ticker-registry",
        str(fake_cache["ticker_registry"]),
        "--sector-registry",
        str(fake_cache["sector_registry"]),
        "--html-out",
        str(fake_cache["html_out"]),
    ]
    r1 = runner.invoke(cli, args)
    assert r1.exit_code == 0

    # Snapshot features parquet
    df1 = pd.read_parquet(fake_cache["features_out"])
    rows_1 = len(df1)

    # Run again with same args.
    r2 = runner.invoke(cli, args)
    assert r2.exit_code == 0
    df2 = pd.read_parquet(fake_cache["features_out"])
    rows_2 = len(df2)

    # Idempotent: same number of rows (no duplicate insert)
    assert rows_1 == rows_2, (
        f"non-idempotent: row count grew {rows_1} -> {rows_2}"
    )


def test_incremental_backfill_satisfies_stale_guard(fake_cache):
    """A stale cache that trips the run-daily guard becomes fresh after a
    `thematic backfill --incremental` refresh — the cron chain
    (`backfill --incremental && run-daily`) no longer fails the stale-OHLCV
    guard for `asof=today`.

    Mirrors the breadth cron's `backfill-ohlcv --incremental && compute`
    ordering: the incremental refresh brings max(date) up to today, so the
    freshness guard in run-daily passes on the very next step.
    """
    import importlib.util

    from rainier.cli import cli

    # Rebuild the cache with DEEP history (>= 21 days) ending a few days before
    # asof — a realistic "missed the last day or two" gap that the incremental
    # window (today-5..today) is designed to bridge. The old fixture ended in
    # 2024 (a ~580-day gap), which the gap guard now (correctly) rejects as
    # un-bridgeable; that is the gap-too-large path, not the refresh path.
    panel_path = fake_cache["panel"]
    pre = pd.read_parquet(panel_path)
    symbols = sorted(pre["symbol"].unique().tolist())

    asof = date(2026, 5, 29)
    # 30 trading days ending at asof-3 (within the 5-day incremental window).
    deep_dates: list[date] = []
    d = asof - timedelta(days=3)
    while len(deep_dates) < 30:
        if d.weekday() < 5:
            deep_dates.append(d)
        d = d - timedelta(days=1)
    deep_dates.sort()
    deep_rows = [
        {
            "symbol": s, "date": dd, "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0 + i * 0.1, "volume": 1_000_000,
            "fetched_at": pd.Timestamp.now(tz="UTC"), "yfinance_version": "seed",
        }
        for s in symbols
        for i, dd in enumerate(deep_dates)
    ]
    pd.DataFrame(deep_rows).to_parquet(panel_path)
    pre = pd.read_parquet(panel_path)
    pre["date"] = pd.to_datetime(pre["date"]).dt.date
    assert pre["date"].max() < asof, "cache must start stale (but within window)"

    # Load the backfill script module + drive its incremental path with a
    # stubbed fetch (offline) to refresh the EXISTING parquet in place.
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "backfill_thematic_universe.py"
    )
    spec = importlib.util.spec_from_file_location("backfill_thematic_universe", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    recent = [asof - timedelta(days=2), asof - timedelta(days=1), asof]

    def _stub(syms, start, end):
        start_d = pd.to_datetime(start).date()
        end_d = pd.to_datetime(end).date()
        out = {}
        for s in syms:
            rows = [
                {
                    "date": d,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1_000_000,
                }
                for d in recent
                if start_d <= d <= end_d
            ]
            out[s] = pd.DataFrame(
                rows, columns=["date", "open", "high", "low", "close", "volume"]
            )
        return out

    result_path = mod.backfill(
        symbols=symbols,
        start="2024-10-01",
        end="2024-10-08",
        out_path=panel_path,
        incremental=True,
        fetch_fn=_stub,
        today=asof,
        min_coverage=0.0,
    )
    assert result_path == panel_path, "incremental must refresh in place"

    refreshed = pd.read_parquet(panel_path)
    refreshed["date"] = pd.to_datetime(refreshed["date"]).dt.date
    assert refreshed["date"].max() == asof, "incremental did not advance max(date)"

    # Now run-daily for asof=today must NOT trip the stale-OHLCV guard.
    runner = CliRunner()
    r = runner.invoke(
        cli,
        [
            "thematic",
            "run-daily",
            "--asof",
            asof.isoformat(),
            "--ohlcv",
            str(panel_path),
            "--yaml",
            str(fake_cache["yaml"]),
            "--features-out",
            str(fake_cache["features_out"]),
            "--labels-out",
            str(fake_cache["labels_out"]),
            "--ticker-registry",
            str(fake_cache["ticker_registry"]),
            "--sector-registry",
            str(fake_cache["sector_registry"]),
            "--html-out",
            str(fake_cache["html_out"]),
        ],
    )
    assert r.exit_code == 0, (
        f"run-daily tripped after incremental refresh: "
        f"exit={r.exit_code} output={r.output}"
    )
    assert "stale" not in r.output.lower(), (
        f"stale guard should be satisfied; got: {r.output!r}"
    )


# ---------------------------------------------------------------------------
# Stale-OHLCV diagnostic — surface-don't-silo (review iter-1 / codex iter-1)
# ---------------------------------------------------------------------------


def test_run_daily_stale_ohlcv_surfaces_diagnostic(fake_cache):
    """When the OHLCV cache's max date is before --asof, run-daily must fail
    fast with a diagnostic that names the next-step backfill command rather
    than silently producing an empty Layer A and missing render.

    Per DESIGN-thematic-ranks-dashboard.md §7 + memory feedback_surface_dont_silo.
    """
    from rainier.cli import cli

    runner = CliRunner()
    # fake_cache panel ends at the synthesized last weekday in the 60-day
    # window; asof set well past that.
    asof_stale = date(2030, 1, 2)
    result = runner.invoke(
        cli,
        [
            "thematic",
            "run-daily",
            "--asof",
            asof_stale.isoformat(),
            "--ohlcv",
            str(fake_cache["panel"]),
            "--yaml",
            str(fake_cache["yaml"]),
            "--features-out",
            str(fake_cache["features_out"]),
            "--labels-out",
            str(fake_cache["labels_out"]),
            "--ticker-registry",
            str(fake_cache["ticker_registry"]),
            "--sector-registry",
            str(fake_cache["sector_registry"]),
            "--html-out",
            str(fake_cache["html_out"]),
        ],
    )
    # ClickException returns exit_code=1 with a clean diagnostic.
    assert result.exit_code != 0, "stale OHLCV should fail fast"
    assert "stale" in result.output.lower(), (
        f"diagnostic should mention 'stale'; got: {result.output!r}"
    )
    assert "thematic backfill" in result.output, (
        f"diagnostic should name the next-step backfill command; got: {result.output!r}"
    )
    # The recovery flow now points at the sanctioned `--force --adopt` bridge
    # (atomic in-place canonical replace + Neon mirror) instead of the old
    # manual `mv cohort -> canonical` swap (revision-immutability preserved by
    # the atomic os.replace inside backfill()).
    assert "--force --adopt" in result.output, (
        f"diagnostic must mention the --force --adopt bridge; "
        f"got: {result.output!r}"
    )
    assert " mv " not in result.output, (
        f"diagnostic must NOT tell the operator to manually mv (unsanctioned "
        f"cache mutation); got: {result.output!r}"
    )


# ---------------------------------------------------------------------------
# Shallow-history guard — a fresh-host / deleted-cache incremental run writes
# only the 5-day window; run-daily must FAIL LOUD rather than emit sentinel
# ranks over <20 trading days of history (codex iter-3 [P1]).
# ---------------------------------------------------------------------------


def _make_universe_yaml(path: Path, symbols: list[str]) -> None:
    body = (
        "version: 1\n"
        "schema: thematic_universe.v1\n"
        "asof_seeded: 2026-05-01\n"
        "universe:\n"
        "  test_sector:\n"
    )
    body += "".join(f"    - {s}\n" for s in symbols)
    path.write_text(body)


def test_check_freshness_rejects_shallow_history():
    """`_check_ohlcv_freshness` raises when the cache has < 20 trading days
    of history at/before asof — Layer A's rel_20/vol_20 windows would all be
    the no-data sentinel, so the job must fail loud instead of exiting 0 with
    unusable ranks. Direct unit test on the shared guard (no DB / no network).
    """
    from types import SimpleNamespace

    from rainier.cli import _check_ohlcv_freshness

    symbols = ["AAA", "BBB", "CCC"]
    # Only 5 trading days ending at asof — exactly what an incremental-only
    # refresh on a fresh host produces.
    asof = date(2026, 5, 29)
    dates = []
    d = asof
    while len(dates) < 5:
        if d.weekday() < 5:
            dates.append(d)
        d = d - timedelta(days=1)
    rows = [
        {"symbol": s, "date": dd, "close": 100.0}
        for s in symbols
        for dd in dates
    ]
    panel = pd.DataFrame(rows)
    spec = SimpleNamespace(sectors={"test_sector": symbols})

    import click

    with pytest.raises(click.ClickException) as exc:
        _check_ohlcv_freshness(panel, asof, "data/cache/thematic_universe.parquet", spec)
    msg = str(exc.value)
    assert "shallow" in msg.lower(), f"diagnostic must say 'shallow'; got: {msg!r}"
    assert "thematic backfill --force --adopt" in msg, (
        f"diagnostic must name the sanctioned full-history adopt bridge; "
        f"got: {msg!r}"
    )
    assert " mv " not in msg, (
        f"diagnostic must NOT tell the operator to manually mv; got: {msg!r}"
    )


def test_check_freshness_shallow_boundary_off_by_one():
    """Boundary: exactly 20 distinct dates still fails (asof_idx=19 ->
    prior_idx=-1 for rel_20 -> sentinel), 21 passes. Locks the off-by-one
    (codex iter-5): the guard requires >= 21 observations, not 20.
    """
    from types import SimpleNamespace

    import click

    from rainier.cli import _check_ohlcv_freshness

    symbols = ["AAA", "BBB"]
    spec = SimpleNamespace(sectors={"test_sector": symbols})

    # Exactly 20 trading days -> must still REJECT.
    panel20 = _build_ohlcv_panel(symbols, n_days=20)
    panel20["date"] = pd.to_datetime(panel20["date"]).dt.date
    asof20 = panel20["date"].max()
    with pytest.raises(click.ClickException, match="shallow"):
        _check_ohlcv_freshness(panel20, asof20, "data/cache/thematic_universe.parquet", spec)

    # 21 trading days -> must ACCEPT (boundary just clears).
    panel21 = _build_ohlcv_panel(symbols, n_days=21)
    panel21["date"] = pd.to_datetime(panel21["date"]).dt.date
    asof21 = panel21["date"].max()
    _check_ohlcv_freshness(panel21, asof21, "data/cache/thematic_universe.parquet", spec)


def test_check_freshness_accepts_deep_history():
    """A cache with >= 21 trading days at/before asof passes the guard."""
    from types import SimpleNamespace

    from rainier.cli import _check_ohlcv_freshness

    symbols = ["AAA", "BBB"]
    panel = _build_ohlcv_panel(symbols, n_days=25)
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    asof = panel["date"].max()
    spec = SimpleNamespace(sectors={"test_sector": symbols})

    # Must NOT raise: deep enough history, fresh, full coverage.
    _check_ohlcv_freshness(panel, asof, "data/cache/thematic_universe.parquet", spec)


def test_run_daily_rejects_incremental_only_thin_cache(tmp_path):
    """End-to-end: a thin (5-day) incremental-only cache makes run-daily fail
    loud with the shallow-history diagnostic — the fresh-host footgun codex
    flagged. The incremental refresh is a refresh, not a substitute for the
    full-history seed.
    """
    import importlib.util

    from rainier.cli import cli

    symbols = ["AAA", "BBB", "CCC"]
    asof = date(2026, 5, 29)
    panel_path = tmp_path / "thematic_universe.parquet"
    yaml_path = tmp_path / "universe.yaml"
    _make_universe_yaml(yaml_path, symbols)

    # Simulate a fresh-host incremental run: no existing cache, only the 5-day
    # window gets written. Drive the real backfill script's incremental path.
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "backfill_thematic_universe.py"
    )
    spec = importlib.util.spec_from_file_location("backfill_thematic_universe", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    recent = [asof - timedelta(days=k) for k in range(5)]

    def _stub(syms, start, end):
        start_d = pd.to_datetime(start).date()
        end_d = pd.to_datetime(end).date()
        return {
            s: pd.DataFrame(
                [
                    {
                        "date": dd, "open": 100.0, "high": 101.0,
                        "low": 99.0, "close": 100.5, "volume": 1_000_000,
                    }
                    for dd in recent
                    if start_d <= dd <= end_d
                ],
                columns=["date", "open", "high", "low", "close", "volume"],
            )
            for s in syms
        }

    mod.backfill(
        symbols=symbols,
        start="2024-10-01",
        end="2024-10-08",
        out_path=panel_path,
        incremental=True,
        fetch_fn=_stub,
        today=asof,
        min_coverage=0.0,
    )
    # The cache exists but is shallow (< 20 trading days).
    assert pd.read_parquet(panel_path)["date"].nunique() < 20

    runner = CliRunner()
    r = runner.invoke(
        cli,
        [
            "thematic", "run-daily", "--asof", asof.isoformat(),
            "--ohlcv", str(panel_path), "--yaml", str(yaml_path),
            "--features-out", str(tmp_path / "features.parquet"),
            "--labels-out", str(tmp_path / "labels.parquet"),
            "--ticker-registry", str(tmp_path / "tr.parquet"),
            "--sector-registry", str(tmp_path / "sr.parquet"),
            "--html-out", str(tmp_path / "out.html"),
        ],
    )
    assert r.exit_code != 0, f"thin cache must fail fast; output={r.output!r}"
    assert "shallow" in r.output.lower(), (
        f"diagnostic must mention 'shallow' history; got: {r.output!r}"
    )


def test_thematic_compute_partial_universe_coverage_fails(fake_cache, tmp_path):
    """Regression — codex iter-8 [P2]: the direct `thematic compute` path
    must apply the same partial-coverage gate as `run-daily`, otherwise an
    operator running compute directly with a partial cache would silently
    produce shrunken ranks.
    """
    from rainier.cli import cli

    panel = pd.read_parquet(fake_cache["panel"])
    asof_dt = date(2024, 11, 8)
    drop_mask = (panel["date"] == asof_dt) & panel["symbol"].isin(
        ["AAA", "BBB", "CCC"]
    )
    partial = panel.loc[~drop_mask].reset_index(drop=True)
    partial_path = tmp_path / "partial_universe.parquet"
    partial.to_parquet(partial_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "thematic",
            "compute",
            "--asof",
            asof_dt.isoformat(),
            "--ohlcv",
            str(partial_path),
            "--yaml",
            str(fake_cache["yaml"]),
            "--out",
            str(fake_cache["features_out"]),
            "--ticker-registry",
            str(fake_cache["ticker_registry"]),
            "--sector-registry",
            str(fake_cache["sector_registry"]),
        ],
    )
    assert result.exit_code != 0, "direct compute must also fail on partial coverage"
    assert "partial coverage" in result.output.lower(), (
        f"diagnostic should mention partial coverage; got: {result.output!r}"
    )


def test_run_daily_partial_universe_coverage_fails(fake_cache, tmp_path):
    """Regression — codex iter-6 [P1]: ranks are cross-sectional, so if a
    partial backfill leaves the panel without close rows for >25% of the
    YAML universe on asof, run-daily must surface that gap rather than
    silently rendering a shrunken dashboard.
    """
    from rainier.cli import cli

    # Read the fake panel and drop 3 of 5 tickers on the chosen asof to
    # simulate a partial yfinance run. 3/5 = 60% missing -> well above 25%.
    panel = pd.read_parquet(fake_cache["panel"])
    asof_dt = date(2024, 11, 8)
    drop_mask = (panel["date"] == asof_dt) & panel["symbol"].isin(
        ["AAA", "BBB", "CCC"]
    )
    partial = panel.loc[~drop_mask].reset_index(drop=True)
    partial_path = tmp_path / "partial_universe.parquet"
    partial.to_parquet(partial_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "thematic",
            "run-daily",
            "--asof",
            asof_dt.isoformat(),
            "--ohlcv",
            str(partial_path),
            "--yaml",
            str(fake_cache["yaml"]),
            "--features-out",
            str(fake_cache["features_out"]),
            "--labels-out",
            str(fake_cache["labels_out"]),
            "--ticker-registry",
            str(fake_cache["ticker_registry"]),
            "--sector-registry",
            str(fake_cache["sector_registry"]),
            "--html-out",
            str(fake_cache["html_out"]),
        ],
    )
    assert result.exit_code != 0, "partial-coverage backfill should fail fast"
    assert "partial coverage" in result.output.lower(), (
        f"diagnostic should mention partial coverage; got: {result.output!r}"
    )
