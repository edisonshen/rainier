"""Unit tests for QU parsers — no browser or DB needed."""

from __future__ import annotations

import pytest

from rainier.scrapers.qu.parsers import (
    parse_capital_flow_rows,
    parse_daily_change,
    parse_qu100_rows,
    parse_rank_fraction,
)


# ---------------------------------------------------------------------------
# parse_daily_change
# ---------------------------------------------------------------------------


class TestParseDailyChange:
    def test_arrow_up(self):
        assert parse_daily_change("▲ 9") == 9

    def test_arrow_up_no_space(self):
        assert parse_daily_change("▲9") == 9

    def test_arrow_down(self):
        assert parse_daily_change("▼ 3") == -3

    def test_arrow_down_no_space(self):
        assert parse_daily_change("▼3") == -3

    def test_zero(self):
        assert parse_daily_change("0") == 0

    def test_new_entry(self):
        assert parse_daily_change("new") == 0

    def test_new_uppercase(self):
        assert parse_daily_change("NEW") == 0

    def test_empty_string(self):
        assert parse_daily_change("") == 0

    def test_whitespace(self):
        assert parse_daily_change("  ▲ 14  ") == 14

    def test_plain_positive(self):
        assert parse_daily_change("5") == 5

    def test_plain_negative(self):
        assert parse_daily_change("-7") == -7

    def test_plus_sign(self):
        assert parse_daily_change("+12") == 12

    def test_garbage_returns_zero(self):
        assert parse_daily_change("abc") == 0

    def test_triangle_up(self):
        assert parse_daily_change("△ 22") == 22

    def test_triangle_down(self):
        assert parse_daily_change("▽ 4") == -4


# ---------------------------------------------------------------------------
# parse_rank_fraction
# ---------------------------------------------------------------------------


class TestParseRankFraction:
    def test_normal(self):
        assert parse_rank_fraction("1/1672") == (1, 1672)

    def test_high_rank(self):
        assert parse_rank_fraction("1719/1729") == (1719, 1729)

    def test_whitespace(self):
        assert parse_rank_fraction("  10/100  ") == (10, 100)

    def test_invalid_no_slash(self):
        assert parse_rank_fraction("100") == (0, 0)

    def test_invalid_non_numeric(self):
        assert parse_rank_fraction("a/b") == (0, 0)

    def test_empty(self):
        assert parse_rank_fraction("") == (0, 0)


# ---------------------------------------------------------------------------
# parse_qu100_rows
# ---------------------------------------------------------------------------


class TestParseQU100Rows:
    def test_single_row(self):
        raw = [
            {
                "rank": "1",
                "symbol": "TSLA",
                "daily_change": "▲ 9",
                "sector": "Consumer Cyclical",
                "industry": "Autos",
                "long_short": "No dominance",
            }
        ]
        result = parse_qu100_rows(raw)
        assert len(result) == 1
        assert result[0].symbol == "TSLA"
        assert result[0].rank == 1
        assert result[0].daily_change == 9
        assert result[0].sector == "Consumer Cyclical"
        assert result[0].long_short == "No dominance"
        assert result[0].raw is raw[0]

    def test_multiple_rows(self):
        raw = [
            {"rank": "1", "symbol": "TSLA", "daily_change": "▲ 9", "sector": "", "industry": "", "long_short": ""},
            {"rank": "2", "symbol": "NVDA", "daily_change": "0", "sector": "Technology", "industry": "Semiconductors", "long_short": "Long in"},
            {"rank": "3", "symbol": "XOM", "daily_change": "▼ 5", "sector": "Energy", "industry": "Oil", "long_short": ""},
        ]
        result = parse_qu100_rows(raw)
        assert len(result) == 3
        assert result[1].symbol == "NVDA"
        assert result[1].daily_change == 0
        assert result[2].daily_change == -5

    def test_skips_empty_symbol(self):
        raw = [
            {"rank": "1", "symbol": "", "daily_change": "0", "sector": "", "industry": "", "long_short": ""},
            {"rank": "2", "symbol": "AAPL", "daily_change": "0", "sector": "", "industry": "", "long_short": ""},
        ]
        result = parse_qu100_rows(raw)
        assert len(result) == 1
        assert result[0].symbol == "AAPL"

    def test_lowercased_symbol_uppercased(self):
        raw = [{"rank": "1", "symbol": "aapl", "daily_change": "0", "sector": "", "industry": "", "long_short": ""}]
        result = parse_qu100_rows(raw)
        assert result[0].symbol == "AAPL"

    def test_empty_input(self):
        assert parse_qu100_rows([]) == []

    def test_real_site_data(self):
        """Test with actual data format from QuantUnicorn site."""
        raw = [
            {"rank": "1", "symbol": "MRK", "daily_change": "▲92", "sector": "Healthcare", "industry": "Drug Manufacturers", "long_short": "Long in"},
            {"rank": "8", "symbol": "USO", "daily_change": "▼2", "sector": "Financial Services", "industry": "ETF", "long_short": "No dominance"},
            {"rank": "51", "symbol": "FSLY", "daily_change": "0", "sector": "Technology", "industry": "Application Software", "long_short": "Long in"},
            {"rank": "99", "symbol": "CTRN", "daily_change": "New", "sector": "", "industry": "", "long_short": "Long in"},
        ]
        result = parse_qu100_rows(raw)
        assert len(result) == 4
        assert result[0].symbol == "MRK"
        assert result[0].daily_change == 92
        assert result[0].long_short == "Long in"
        assert result[1].daily_change == -2
        assert result[2].daily_change == 0
        assert result[3].daily_change == 0  # "New" -> 0
        assert result[3].symbol == "CTRN"


# ---------------------------------------------------------------------------
# parse_capital_flow_rows
# ---------------------------------------------------------------------------


class TestParseCapitalFlowRows:
    def test_daily_rows(self):
        raw = [
            {"date": "2026-02-09", "direction": "+", "long_short": "No dominance", "rank": "1/1672"},
            {"date": "2026-02-06", "direction": "+", "long_short": "No dominance", "rank": "10/1742"},
        ]
        result = parse_capital_flow_rows(raw, "daily")
        assert len(result) == 2
        assert result[0].flow_date == "2026-02-09"
        assert result[0].rank == 1
        assert result[0].rank_total == 1672
        assert result[0].period_type == "daily"
        assert result[0].week_start is None

    def test_weekly_rows(self):
        raw = [
            {"date": "2026-02-03 ~ 2026-02-07", "direction": "-", "long_short": "Short in", "rank": "500/1700"},
        ]
        result = parse_capital_flow_rows(raw, "weekly")
        assert len(result) == 1
        assert result[0].flow_date == "2026-02-03"
        assert result[0].week_start == "2026-02-03"
        assert result[0].week_end == "2026-02-07"
        assert result[0].period_type == "weekly"
        assert result[0].rank == 500

    def test_empty_input(self):
        assert parse_capital_flow_rows([], "daily") == []
