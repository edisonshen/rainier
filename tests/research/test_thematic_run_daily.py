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
    assert "backfill_thematic_universe" in result.output, (
        f"diagnostic should name the next-step backfill command; got: {result.output!r}"
    )
    # codex iter-3 [P1]: the recovery flow must account for backfill's
    # revision-immutability (--force writes a sibling cohort, not in-place).
    # Diagnostic must include the cohort-swap step (--force then mv).
    assert "--force" in result.output, (
        f"diagnostic must mention --force (sibling cohort flow); "
        f"got: {result.output!r}"
    )
    assert " mv " in result.output, (
        f"diagnostic must include cohort-swap mv step; got: {result.output!r}"
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
