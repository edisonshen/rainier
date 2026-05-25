"""Unit tests for QU parsers — no browser or DB needed."""

from __future__ import annotations

from rainier.scrapers.qu.parsers import (
    api_rows_to_qu100_rows,
    parse_capital_flow_rows,
    parse_daily_change,
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
# api_rows_to_qu100_rows (D-4 adapter for the in-page-fetch /api/v3/mf100 path)
# ---------------------------------------------------------------------------


class TestApiRowsToQU100Rows:
    """Adapter that maps the /api/v3/mf100 response shape onto QU100Row.

    API shape (per spike capture):
        {"rank": 1, "ticker": "MU", "change": "0",
         "industry": "Semiconductors", "long_short": "No dominance",
         "sector": "Technology"}

    The adapter renames `ticker -> symbol` and runs `change` through
    `parse_daily_change` to land on `daily_change`. No other transformations.
    """

    def test_happy_path(self):
        api = [
            {
                "rank": 1,
                "ticker": "MU",
                "change": "0",
                "industry": "Semiconductors",
                "long_short": "No dominance",
                "sector": "Technology",
            }
        ]
        rows = api_rows_to_qu100_rows(api)
        assert len(rows) == 1
        assert rows[0].rank == 1
        assert rows[0].symbol == "MU"
        assert rows[0].daily_change == 0
        assert rows[0].sector == "Technology"
        assert rows[0].industry == "Semiconductors"
        assert rows[0].long_short == "No dominance"
        # raw preserves the original API dict so downstream JSONB storage
        # keeps the source-of-truth payload.
        assert rows[0].raw is api[0]

    def test_change_variants(self):
        api = [
            {"rank": 1, "ticker": "AAA", "change": "0",
             "industry": "", "long_short": "", "sector": ""},
            {"rank": 2, "ticker": "BBB", "change": "-3",
             "industry": "", "long_short": "", "sector": ""},
            {"rank": 3, "ticker": "CCC", "change": "1604",
             "industry": "", "long_short": "", "sector": ""},
            {"rank": 4, "ticker": "DDD", "change": "new",
             "industry": "", "long_short": "", "sector": ""},
            # change missing entirely — should not raise, falls through to 0
            {"rank": 5, "ticker": "EEE",
             "industry": "", "long_short": "", "sector": ""},
        ]
        rows = api_rows_to_qu100_rows(api)
        assert [r.daily_change for r in rows] == [0, -3, 1604, 0, 0]

    def test_field_rename_ticker_to_symbol(self):
        # The API field is named `ticker`. The QU100Row field is `symbol`.
        # The adapter must never leak `ticker` onto QU100Row.
        api = [{"rank": 1, "ticker": "msft", "change": "0",
                "industry": "", "long_short": "", "sector": ""}]
        rows = api_rows_to_qu100_rows(api)
        assert rows[0].symbol == "MSFT"  # uppercased per existing convention

    def test_empty_data(self):
        assert api_rows_to_qu100_rows([]) == []

    def test_skips_empty_ticker(self):
        # Defensive: an empty ticker shouldn't produce a junk row.
        api = [
            {"rank": 1, "ticker": "", "change": "0",
             "industry": "", "long_short": "", "sector": ""},
            {"rank": 2, "ticker": "AAPL", "change": "5",
             "industry": "", "long_short": "", "sector": ""},
        ]
        rows = api_rows_to_qu100_rows(api)
        assert len(rows) == 1
        assert rows[0].symbol == "AAPL"


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
