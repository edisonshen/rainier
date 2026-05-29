"""CLI decision-logic tests for `rainier thematic backfill-names --refresh-stale`.

Bug under test: when a new ticker is added to ``config/thematic_universe.yaml``,
the ``--refresh-stale`` 30-day gate blocked its name fetch for up to 30 days
because ``is_stale`` short-circuited the whole run on a fresh parquet — leaving
the new ticker rendering as a symbol fallback.

Fix: in ``--refresh-stale`` mode, even when the parquet is fresh (<30d), any
current-universe symbol absent from the cache is force-refreshed; existing
cached symbols keep the 30-day cadence (NOT re-fetched).

These tests are deterministic — ``backfill_names`` is monkeypatched to capture
the exact symbol set, never hitting yfinance.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yaml
from click.testing import CliRunner

from rainier.cli import cli


def _write_universe_yaml(path, tickers: list[str]) -> None:
    """Minimal thematic_universe.v1 yaml flattening to ``tickers``."""
    doc = {
        "version": 1,
        "schema": "thematic_universe.v1",
        "universe": {"bucket": list(tickers)},
    }
    path.write_text(yaml.safe_dump(doc))


def _write_fresh_cache(path, symbols: list[str]) -> None:
    """Write an etf_names parquet dated *now* (fresh, <30d) for ``symbols``."""
    now = datetime.now(tz=timezone.utc)
    df = pd.DataFrame(
        {
            "symbol": symbols,
            "long_name": [f"{s} Long" for s in symbols],
            "short_name": symbols,
            "fetched_at": [now] * len(symbols),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def test_refresh_stale_force_fetches_only_new_universe_symbol(tmp_path, monkeypatch):
    """Fresh parquet + a new universe ticker → fetch ONLY the new ticker.

    Regression guard: EFA + SPY are already cached and fresh; QQQ is new.
    The 30-day gate must NOT block QQQ, and must NOT re-fetch EFA/SPY.
    """
    from rainier.dashboard import etf_names_backfill

    universe = tmp_path / "thematic_universe.yaml"
    _write_universe_yaml(universe, ["EFA", "SPY", "QQQ"])

    cache = tmp_path / "etf_names.parquet"
    _write_fresh_cache(cache, ["EFA", "SPY"])

    captured: dict[str, list[str]] = {}

    def _spy_backfill(symbols, out_path, **kwargs):
        captured["symbols"] = list(symbols)
        return out_path

    monkeypatch.setattr(etf_names_backfill, "backfill_names", _spy_backfill)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "thematic",
            "backfill-names",
            "--universe-yaml",
            str(universe),
            "--output",
            str(cache),
            "--refresh-stale",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("symbols") == ["QQQ"], (
        f"expected only the new ticker QQQ to be fetched, got "
        f"{captured.get('symbols')!r}"
    )


def test_refresh_stale_skips_when_fresh_and_no_new_symbols(tmp_path, monkeypatch):
    """Fresh parquet + every universe symbol already cached → no fetch at all."""
    from rainier.dashboard import etf_names_backfill

    universe = tmp_path / "thematic_universe.yaml"
    _write_universe_yaml(universe, ["EFA", "SPY"])

    cache = tmp_path / "etf_names.parquet"
    _write_fresh_cache(cache, ["EFA", "SPY"])

    called = {"n": 0}

    def _spy_backfill(symbols, out_path, **kwargs):
        called["n"] += 1
        return out_path

    monkeypatch.setattr(etf_names_backfill, "backfill_names", _spy_backfill)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "thematic",
            "backfill-names",
            "--universe-yaml",
            str(universe),
            "--output",
            str(cache),
            "--refresh-stale",
        ],
    )

    assert result.exit_code == 0, result.output
    assert called["n"] == 0, "no fetch should happen when fresh + no new symbols"
    assert "fresh" in result.output.lower()


def test_refresh_stale_full_backfill_when_parquet_stale(tmp_path, monkeypatch):
    """Stale parquet (missing) → full-universe backfill (existing behavior intact)."""
    from rainier.dashboard import etf_names_backfill

    universe = tmp_path / "thematic_universe.yaml"
    _write_universe_yaml(universe, ["EFA", "SPY", "QQQ"])

    cache = tmp_path / "etf_names.parquet"  # does NOT exist → stale

    captured: dict[str, list[str]] = {}

    def _spy_backfill(symbols, out_path, **kwargs):
        captured["symbols"] = list(symbols)
        return out_path

    monkeypatch.setattr(etf_names_backfill, "backfill_names", _spy_backfill)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "thematic",
            "backfill-names",
            "--universe-yaml",
            str(universe),
            "--output",
            str(cache),
            "--refresh-stale",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("symbols") == ["EFA", "SPY", "QQQ"], (
        f"stale parquet should trigger full-universe backfill, got "
        f"{captured.get('symbols')!r}"
    )


def test_no_refresh_stale_flag_always_full_backfill(tmp_path, monkeypatch):
    """Without --refresh-stale, a fresh parquet still triggers full backfill."""
    from rainier.dashboard import etf_names_backfill

    universe = tmp_path / "thematic_universe.yaml"
    _write_universe_yaml(universe, ["EFA", "SPY"])

    cache = tmp_path / "etf_names.parquet"
    _write_fresh_cache(cache, ["EFA", "SPY"])

    captured: dict[str, list[str]] = {}

    def _spy_backfill(symbols, out_path, **kwargs):
        captured["symbols"] = list(symbols)
        return out_path

    monkeypatch.setattr(etf_names_backfill, "backfill_names", _spy_backfill)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "thematic",
            "backfill-names",
            "--universe-yaml",
            str(universe),
            "--output",
            str(cache),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("symbols") == ["EFA", "SPY"]