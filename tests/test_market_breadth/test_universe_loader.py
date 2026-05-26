"""Unit tests for src/rainier/market_breadth/universe_loader.py.

The S&P 500 universe YAML uses a list-of-dicts shape:

    universe:
      - {symbol: 'AAPL', sector: information_technology}
      - {symbol: 'BRK.B', sector: financials}

`load_sp500_universe(yaml_path)` returns a deterministic list of
``(symbol, sector)`` tuples preserving YAML order. The loader is the
thin adapter the breadth backfill calls; symbol dot-vs-dash translation
happens at fetch time, NOT here — the loader returns Wikipedia spelling
verbatim so the YAML and the output parquet share the same join key.

Design refs:
    docs/DESIGN-market-breadth-webpage.md §3.2
    docs/TASK-PLAN-sp500-ohlcv-backfill-47b8.md §Tests
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rainier.market_breadth import universe_loader as ul

SAMPLE_YAML = """\
version: 1
schema: sp500_universe.v1
asof_seeded: '2026-05-25'
source: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
source_table_row_count: 4
universe:
  - {symbol: 'AAPL', sector: information_technology}
  - {symbol: 'BRK.B', sector: financials}
  - {symbol: 'JPM',  sector: financials}
  - {symbol: 'XOM',  sector: energy}
"""


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_universe_loader_reads_yaml(tmp_path):
    """YAML list-of-dicts → list[(symbol, sector)], YAML order preserved.

    Wikipedia spelling preserved (`BRK.B` stays with the dot; translation to
    yfinance's `BRK-B` happens at fetch time in ohlcv_backfill.py, NOT here).
    """
    yaml_path = _write_yaml(tmp_path / "sp500_universe.yaml", SAMPLE_YAML)
    entries = ul.load_sp500_universe(yaml_path)

    assert entries == [
        ("AAPL", "information_technology"),
        ("BRK.B", "financials"),
        ("JPM", "financials"),
        ("XOM", "energy"),
    ]


# ---------------------------------------------------------------------------
# Malformed YAML — docstring promises ValueError, not AttributeError
# ---------------------------------------------------------------------------


def test_universe_loader_rejects_list_at_root(tmp_path):
    """Root-level list (missing the `universe:` wrapper) raises ValueError.

    Regression for /review iter-2: `(parsed or {}).get(...)` blew up with
    AttributeError when the YAML root was a list, violating the docstring
    contract that promises ValueError for malformed input.
    """
    p = _write_yaml(tmp_path / "bad.yaml", "- {symbol: AAPL, sector: tech}\n")
    with pytest.raises(ValueError, match="expected top-level `universe:`"):
        ul.load_sp500_universe(p)


def test_universe_loader_rejects_scalar_root(tmp_path):
    """Scalar root (e.g. a bare string) raises ValueError, not AttributeError."""
    p = _write_yaml(tmp_path / "scalar.yaml", "just a string\n")
    with pytest.raises(ValueError, match="expected top-level `universe:`"):
        ul.load_sp500_universe(p)


def test_universe_loader_rejects_empty_yaml(tmp_path):
    """Empty YAML file raises ValueError (parsed is None → ``universe`` missing)."""
    p = _write_yaml(tmp_path / "empty.yaml", "")
    with pytest.raises(ValueError, match="expected top-level `universe:`"):
        ul.load_sp500_universe(p)
