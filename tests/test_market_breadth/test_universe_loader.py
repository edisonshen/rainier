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
