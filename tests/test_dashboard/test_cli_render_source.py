"""CLI tests for the `--source {neon,parquet}` switch on the two dashboard
render commands (`dashboard render-etf-html`, `market-breadth render-html`).

Covers:
  - parquet back-compat path still renders a non-empty page (explicit
    --source parquet AND the implicit-by---features/--input path).
  - neon is the DEFAULT source and routes through the Neon loaders +
    ensure_env_loaded (we monkeypatch the loaders so no DB is needed).
  - a zero-row Neon result fails loud (non-zero exit), never a silent
    empty page.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from rainier.cli import cli

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ETF_FEATURES = FIXTURES / "etf_features_small.parquet"
ETF_REGISTRY = FIXTURES / "etf_sector_registry_small.parquet"
BREADTH = FIXTURES / "sp500_breadth_small.parquet"
ETF_ASOF = "2026-05-25"


# ---------------------------------------------------------------------------
# parquet back-compat
# ---------------------------------------------------------------------------


def test_etf_parquet_explicit_source(tmp_path):
    out = tmp_path / "etf.html"
    res = CliRunner().invoke(
        cli,
        [
            "dashboard", "render-etf-html",
            "--source", "parquet",
            "--features", str(ETF_FEATURES),
            "--registry", str(ETF_REGISTRY),
            "--asof", ETF_ASOF,
            "--rendered-at-pt", "12:40",
            "--output", str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    html = out.read_text()
    assert "Universe: 0" not in html


def test_etf_features_path_implies_parquet(tmp_path, monkeypatch):
    """Passing --features alone (no --source) must imply parquet — it must NOT
    try to hit Neon. We make the Neon path explode if reached."""
    import rainier.dashboard.neon_source as ns

    def boom(*_a, **_k):
        raise AssertionError("neon path reached despite explicit --features")

    monkeypatch.setattr(ns, "ensure_env_loaded", boom)

    out = tmp_path / "etf.html"
    res = CliRunner().invoke(
        cli,
        [
            "dashboard", "render-etf-html",
            "--features", str(ETF_FEATURES),
            "--registry", str(ETF_REGISTRY),
            "--asof", ETF_ASOF,
            "--rendered-at-pt", "12:40",
            "--output", str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Universe: 0" not in out.read_text()


def test_breadth_input_implies_parquet(tmp_path, monkeypatch):
    import rainier.dashboard.neon_source as ns

    def boom(*_a, **_k):
        raise AssertionError("neon path reached despite explicit --input")

    monkeypatch.setattr(ns, "ensure_env_loaded", boom)

    out = tmp_path / "breadth.html"
    res = CliRunner().invoke(
        cli,
        [
            "market-breadth", "render-html",
            "--input", str(BREADTH),
            "--rendered-at-pt", "12:40",
            "--output", str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "No breadth data available" not in out.read_text()


# ---------------------------------------------------------------------------
# neon is the default
# ---------------------------------------------------------------------------


def test_etf_neon_is_default(tmp_path, monkeypatch):
    """No --source / --features → neon path: ensure_env_loaded called, loaders
    invoked, engine disposed."""
    import rainier.dashboard.neon_source as ns

    calls = {"env": 0, "asof": 0, "load": 0}

    class FakeEngine:
        def dispose(self):
            calls["disposed"] = True

    monkeypatch.setattr(ns, "ensure_env_loaded", lambda: calls.__setitem__("env", 1))
    monkeypatch.setattr("rainier.db.engine.get_engine", lambda: FakeEngine())
    monkeypatch.setattr(
        ns, "latest_etf_asof",
        lambda _e: (calls.__setitem__("asof", 1), date(2026, 5, 25))[1],
    )

    feats = pd.read_parquet(ETF_FEATURES)

    def fake_load(_e, _asof):
        calls["load"] = 1
        return feats

    monkeypatch.setattr(ns, "load_etf_features_neon", fake_load)

    out = tmp_path / "etf.html"
    res = CliRunner().invoke(
        cli,
        [
            "dashboard", "render-etf-html",
            "--registry", str(ETF_REGISTRY),
            "--rendered-at-pt", "12:40",
            "--output", str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert calls["env"] == 1 and calls["asof"] == 1 and calls["load"] == 1
    assert calls.get("disposed") is True, "engine must be disposed"
    assert "Universe: 0" not in out.read_text()


def test_etf_neon_fails_loud_on_empty(tmp_path, monkeypatch):
    """A zero-row Neon result must exit non-zero — never a silent empty page."""
    import rainier.dashboard.neon_source as ns

    class FakeEngine:
        def dispose(self):
            pass

    monkeypatch.setattr(ns, "ensure_env_loaded", lambda: None)
    monkeypatch.setattr("rainier.db.engine.get_engine", lambda: FakeEngine())

    def raise_empty(_e):
        raise ns.EmptyNeonResultError("no asof")

    monkeypatch.setattr(ns, "latest_etf_asof", raise_empty)

    out = tmp_path / "etf.html"
    res = CliRunner().invoke(
        cli,
        [
            "dashboard", "render-etf-html",
            "--registry", str(ETF_REGISTRY),
            "--rendered-at-pt", "12:40",
            "--output", str(out),
        ],
    )
    assert res.exit_code != 0, res.output
    assert "error" in res.output.lower()
    assert not out.exists(), "must not write an output file on empty Neon"


def test_breadth_neon_is_default(tmp_path, monkeypatch):
    import rainier.dashboard.neon_source as ns

    calls = {}

    class FakeEngine:
        def dispose(self):
            calls["disposed"] = True

    monkeypatch.setattr(ns, "ensure_env_loaded", lambda: calls.__setitem__("env", 1))
    monkeypatch.setattr("rainier.db.engine.get_engine", lambda: FakeEngine())
    monkeypatch.setattr(ns, "latest_breadth_asof", lambda _e: date(2026, 5, 25))

    breadth = pd.read_parquet(BREADTH)
    monkeypatch.setattr(ns, "load_breadth_neon", lambda _e, _a: breadth)
    monkeypatch.setattr(ns, "load_spy_neon", lambda _e: pd.DataFrame())

    out = tmp_path / "breadth.html"
    res = CliRunner().invoke(
        cli,
        [
            "market-breadth", "render-html",
            "--rendered-at-pt", "12:40",
            "--output", str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert calls.get("env") == 1 and calls.get("disposed") is True
    assert "No breadth data available" not in out.read_text()


def test_breadth_neon_fails_loud_on_empty(tmp_path, monkeypatch):
    import rainier.dashboard.neon_source as ns

    class FakeEngine:
        def dispose(self):
            pass

    monkeypatch.setattr(ns, "ensure_env_loaded", lambda: None)
    monkeypatch.setattr("rainier.db.engine.get_engine", lambda: FakeEngine())
    monkeypatch.setattr(
        ns, "latest_breadth_asof",
        lambda _e: (_ for _ in ()).throw(ns.EmptyNeonResultError("empty")),
    )

    out = tmp_path / "breadth.html"
    res = CliRunner().invoke(
        cli,
        [
            "market-breadth", "render-html",
            "--rendered-at-pt", "12:40",
            "--output", str(out),
        ],
    )
    assert res.exit_code != 0, res.output
    assert "error" in res.output.lower()
    assert not out.exists()
