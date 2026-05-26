"""Unit tests for scripts/seed_sp500_universe.py.

The seed script is operator-run, not CI-run. It hits Wikipedia when invoked
from the CLI. Tests inject a frozen HTML fixture so CI stays offline.

Acceptance pinned by docs/TASK-PLAN-sp500-universe-yaml-94a6.md §Tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "seed_sp500_universe.py"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "wikipedia_sp500_table.html"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "seed_sp500_universe", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture
def seed_mod():
    return _load_module()


@pytest.fixture
def fixture_html() -> str:
    return FIXTURE_PATH.read_text()


# ---------------------------------------------------------------------------
# Test 1 — parse fixture table into the expected YAML schema + symbol count.
# ---------------------------------------------------------------------------


def test_seed_parses_fixture_table_to_expected_yaml(
    seed_mod, fixture_html, tmp_path
):
    out = tmp_path / "sp500_universe.yaml"
    n = seed_mod.seed_universe(
        html=fixture_html,
        out_path=out,
        asof="2026-05-25",
        source_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    )

    assert out.exists()
    # Fixture is the full frozen snapshot — 503 rows.
    assert 495 <= n <= 510, f"symbol count {n} outside acceptance range"

    yaml = YAML()
    parsed = yaml.load(out.read_text())
    assert parsed["version"] == 1
    assert parsed["schema"] == "sp500_universe.v1"
    assert parsed["asof_seeded"] == "2026-05-25"
    assert parsed["source"] == (
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    assert parsed["source_table_row_count"] == n
    universe = parsed["universe"]
    assert len(universe) == n
    # Each entry has exactly {symbol, sector}.
    for entry in universe:
        assert set(entry.keys()) == {"symbol", "sector"}
        assert entry["symbol"].isupper() or "." in entry["symbol"] or "-" in entry["symbol"]
        # Sector is lowercase snake_case — letters / digits / underscores only.
        assert entry["sector"].replace("_", "").isalnum()
        assert entry["sector"] == entry["sector"].lower()
    # Known anchors from the frozen snapshot must be present.
    symbols = {e["symbol"] for e in universe}
    assert "AAPL" in symbols
    assert "MSFT" in symbols
    assert "NVDA" in symbols


# ---------------------------------------------------------------------------
# Test 2 — skip rows missing the symbol cell; count reflects skip.
# ---------------------------------------------------------------------------


MALFORMED_HTML = """
<!DOCTYPE html>
<html><body>
<table id="constituents">
<tr>
  <th>Symbol</th><th>Security</th><th>GICS Sector</th>
  <th>GICS Sub-Industry</th><th>HQ</th><th>Date</th><th>CIK</th><th>Founded</th>
</tr>
<tr>
  <td>AAPL</td><td>Apple</td><td>Information Technology</td>
  <td>Tech Hw</td><td>Cupertino</td><td>1982-11-30</td><td>0000320193</td><td>1976</td>
</tr>
<tr>
  <td></td><td>Anon Co</td><td>Industrials</td>
  <td>Conglom</td><td>Nowhere</td><td>-</td><td>-</td><td>-</td>
</tr>
<tr>
  <td>MSFT</td><td>Microsoft</td><td>Information Technology</td>
  <td>Sys Sw</td><td>Redmond</td><td>1994-06-01</td><td>0000789019</td><td>1975</td>
</tr>
</table>
</body></html>
"""


def test_seed_skips_malformed_rows(seed_mod, tmp_path, caplog):
    out = tmp_path / "sp500_universe.yaml"
    import logging
    caplog.set_level(logging.WARNING)
    n = seed_mod.seed_universe(
        html=MALFORMED_HTML,
        out_path=out,
        asof="2026-05-25",
        source_url="http://example.test",
    )
    assert n == 2  # AAPL + MSFT; empty-symbol row dropped.
    yaml = YAML()
    parsed = yaml.load(out.read_text())
    syms = [e["symbol"] for e in parsed["universe"]]
    assert syms == ["AAPL", "MSFT"]
    # Skip is logged for audit.
    skip_logged = any("skip" in r.message.lower() for r in caplog.records)
    assert skip_logged, "malformed-row skip should be logged"


# ---------------------------------------------------------------------------
# Test 3 — sector name normalization: lowercase snake_case.
# ---------------------------------------------------------------------------


SECTOR_NORMALIZE_HTML = """
<!DOCTYPE html>
<html><body>
<table id="constituents">
<tr>
  <th>Symbol</th><th>Security</th><th>GICS Sector</th>
  <th>GICS Sub-Industry</th><th>HQ</th><th>Date</th><th>CIK</th><th>Founded</th>
</tr>
<tr><td>AAPL</td><td>A</td><td>Information Technology</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>XOM</td><td>E</td><td>Energy</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>JNJ</td><td>J</td><td>Health Care</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>VZ</td><td>V</td><td>Communication Services</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>JPM</td><td>J</td><td>Financials</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
</table>
</body></html>
"""


def test_seed_normalizes_sector_names(seed_mod, tmp_path):
    out = tmp_path / "sp500_universe.yaml"
    seed_mod.seed_universe(
        html=SECTOR_NORMALIZE_HTML,
        out_path=out,
        asof="2026-05-25",
        source_url="http://example.test",
    )
    yaml = YAML()
    parsed = yaml.load(out.read_text())
    by_sym = {e["symbol"]: e["sector"] for e in parsed["universe"]}
    assert by_sym["AAPL"] == "information_technology"
    assert by_sym["XOM"] == "energy"
    assert by_sym["JNJ"] == "health_care"
    assert by_sym["VZ"] == "communication_services"
    assert by_sym["JPM"] == "financials"


# ---------------------------------------------------------------------------
# Test 4 — idempotent: running twice on the same fixture produces byte-identical YAML.
# ---------------------------------------------------------------------------


def test_yaml_is_idempotent(seed_mod, fixture_html, tmp_path):
    out1 = tmp_path / "sp500_universe.yaml"
    out2 = tmp_path / "sp500_universe_again.yaml"

    seed_mod.seed_universe(
        html=fixture_html,
        out_path=out1,
        asof="2026-05-25",
        source_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    )
    seed_mod.seed_universe(
        html=fixture_html,
        out_path=out2,
        asof="2026-05-25",
        source_url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    )

    assert out1.read_bytes() == out2.read_bytes(), (
        "seed script must produce byte-identical YAML on repeat runs"
    )
