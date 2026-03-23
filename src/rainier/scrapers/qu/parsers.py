"""Pure parsing functions for QU data.

No Playwright dependency — testable with dict fixtures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QU100Row:
    """Parsed row from the QU100 ranking table."""

    rank: int
    symbol: str
    daily_change: int
    sector: str
    industry: str
    long_short: str
    raw: dict


@dataclass
class CapitalFlowRow:
    """Parsed row from a daily/weekly rank table on the detail page."""

    flow_date: str  # "2026-02-09" or first date of week range
    direction: str  # "+", "-", "N"
    long_short: str
    rank: int
    rank_total: int
    period_type: str  # "daily" or "weekly"
    week_start: str | None  # for weekly rows
    week_end: str | None  # for weekly rows
    raw: dict


@dataclass
class BarChartData:
    """Parsed bar from a capital flow chart."""

    label: str  # time label from the chart axis
    bar_type: str  # "hourly", "daily", "weekly"
    total_flow: float
    near_term_flow: float
    raw: dict


def parse_daily_change(text: str) -> int:
    """
    Parse daily change text from QU100 table.

    Examples:
        "▲ 9"  -> +9
        "▼ 3"  -> -3
        "0"    -> 0
        "new"  -> 0
    """
    text = text.strip()
    if not text or text == "0" or text.lower() == "new":
        return 0

    match = re.match(r"[▲△↑+]\s*(\d+)", text)
    if match:
        return int(match.group(1))

    match = re.match(r"[▼▽↓-]\s*(\d+)", text)
    if match:
        return -int(match.group(1))

    # Fallback: try parsing as plain integer
    try:
        return int(text)
    except ValueError:
        return 0


def parse_rank_fraction(text: str) -> tuple[int, int]:
    """
    Parse "1/1672" into (rank=1, total=1672).

    Returns (0, 0) if parsing fails.
    """
    text = text.strip()
    parts = text.split("/")
    if len(parts) != 2:
        return 0, 0
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return 0, 0


def parse_qu100_rows(raw_rows: list[dict]) -> list[QU100Row]:
    """
    Parse raw dicts extracted via page.evaluate() into QU100Row objects.

    Expected raw_row keys: rank, symbol, daily_change, sector, industry, long_short
    """
    results = []
    for row in raw_rows:
        symbol = row.get("symbol", "").strip().upper()
        if not symbol:
            continue
        results.append(
            QU100Row(
                rank=int(row.get("rank", 0) or 0),
                symbol=symbol,
                daily_change=parse_daily_change(str(row.get("daily_change", "0"))),
                sector=row.get("sector", "").strip(),
                industry=row.get("industry", "").strip(),
                long_short=row.get("long_short", "").strip(),
                raw=row,
            )
        )
    return results


def parse_capital_flow_rows(
    raw_rows: list[dict], period_type: str
) -> list[CapitalFlowRow]:
    """Parse raw dicts from a daily or weekly rank table."""
    results = []
    for row in raw_rows:
        date_text = row.get("date", "")
        rank_text = row.get("rank", "0/0")
        rank, rank_total = parse_rank_fraction(rank_text)

        week_start = None
        week_end = None
        flow_date = date_text.strip()

        if period_type == "weekly" and "~" in date_text:
            parts = date_text.split("~")
            week_start = parts[0].strip()
            week_end = parts[1].strip()
            flow_date = week_start

        results.append(
            CapitalFlowRow(
                flow_date=flow_date,
                direction=row.get("direction", "N").strip(),
                long_short=row.get("long_short", "").strip(),
                rank=rank,
                rank_total=rank_total,
                period_type=period_type,
                week_start=week_start,
                week_end=week_end,
                raw=row,
            )
        )
    return results
