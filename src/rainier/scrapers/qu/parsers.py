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


def _api_rows_to_qu100_rows(api_rows: list[dict]) -> list[QU100Row]:
    """Adapt /api/v3/mf100 response rows to QU100Row (DESIGN D-4).

    The API payload (per the spike capture) ships rows shaped like::

        {"rank": 1, "ticker": "MU", "change": "0",
         "industry": "Semiconductors", "long_short": "No dominance",
         "sector": "Technology"}

    QU100Row uses `symbol` instead of `ticker` and `daily_change: int`
    instead of `change: str`. The only transformations are the rename
    and feeding `change` through `parse_daily_change` (which already
    handles "0", signed integers, "new", and arrow-prefixed inputs).

    Empty / missing tickers are skipped defensively.
    """
    results: list[QU100Row] = []
    for row in api_rows:
        symbol = str(row.get("ticker", "")).strip().upper()
        if not symbol:
            continue
        results.append(
            QU100Row(
                rank=int(row.get("rank", 0) or 0),
                symbol=symbol,
                daily_change=parse_daily_change(str(row.get("change", "0"))),
                # ``or ""`` (not ``get(key, "")``): the in-page sanitizer turns
                # QU's bare ``NaN`` string field into JSON ``null`` -> Python
                # ``None``. ``get("sector", "")`` only defaults a *missing* key,
                # so a present-but-null value would otherwise stringify to the
                # literal ``"None"``. ``or ""`` collapses both to empty string.
                # All three string fields get the same coalescing so a bare
                # ``NaN`` in any of them never persists as the dominance/sector
                # category literal ``"None"``.
                sector=str(row.get("sector") or "").strip(),
                industry=str(row.get("industry") or "").strip(),
                long_short=str(row.get("long_short") or "").strip(),
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


@dataclass(frozen=True)
class TickerRankRow:
    """One daily/weekly row of the capitalflow page's rank summary (mfsummary)."""

    label: str  # "2026-10-16" or "2026-10-12 to 2026-10-16"
    long_short: str
    rank: int
    total: int  # how many tickers QU ranked that period
    sentiment: str  # "+", "-", "N"


@dataclass(frozen=True)
class TickerFlowBar:
    """One near-term/total money-flow bar from the capitalflow chart (mfplot)."""

    label: str
    near: float
    total: float


@dataclass(frozen=True)
class TickerFlow:
    """Everything the QU capitalflow page shows for one ticker."""

    symbol: str
    updated: str  # QU's "Last updated on ..." stamp
    daily_ranks: list[TickerRankRow]
    weekly_ranks: list[TickerRankRow]
    daily_bars: list[TickerFlowBar]
    weekly_bars: list[TickerFlowBar]

    @property
    def empty(self) -> bool:
        return not (self.daily_ranks or self.weekly_ranks or self.daily_bars)

    def prompt_section(self) -> str:
        """Render as a thesis-prompt section (newest rows first, as QU shows them)."""
        lines = [f"--- QU capital flow ({self.updated}) ---"]
        if self.empty:
            lines.append("(QU has no capital-flow data for this ticker)")
            return "\n".join(lines)
        for title, rows in (("Daily", self.daily_ranks), ("Weekly", self.weekly_ranks)):
            lines.append(f"{title} rank (rank / tickers ranked, long/short, sentiment):")
            lines += [
                f"- {r.label}: {r.rank}/{r.total}, {r.long_short}, {r.sentiment}" for r in rows
            ]
        for title, bars in (("Daily", self.daily_bars), ("Weekly", self.weekly_bars)):
            lines.append(f"{title} money flow (near-term / total):")
            lines += [f"- {b.label}: {b.near:+.2f} / {b.total:+.2f}" for b in bars]
        return "\n".join(lines)


def _int_or_zero(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_ticker_ranks(rows: list[dict] | None) -> list[TickerRankRow]:
    """Parse one ``mfsummary`` list (``data.daily`` or ``data.weekly``)."""
    return [
        TickerRankRow(
            label=str(r.get("label", "")),
            long_short=str(r.get("long_short") or ""),
            rank=_int_or_zero(r.get("rank")),
            total=_int_or_zero(r.get("total")),
            sentiment=str(r.get("sentiment") or "N"),
        )
        for r in rows or []
    ]


def parse_ticker_bars(rows: list[dict] | None) -> list[TickerFlowBar]:
    """Parse an ``mfplot`` ``data`` list; rows missing a number are dropped."""
    bars: list[TickerFlowBar] = []
    for r in rows or []:
        near, total = r.get("near"), r.get("total")
        if near is None or total is None:
            continue
        bars.append(
            TickerFlowBar(label=str(r.get("xlabel", "")), near=float(near), total=float(total))
        )
    return bars
