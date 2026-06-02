"""Daily paper-book report: compute → persist snapshot → render → Discord (D11).

The report is rendered from a **persisted snapshot** (`paper_report_snapshot`),
never fired-and-forgotten:

* `compute_daily_payload(as_of)` reads the raw inputs (`paper_trade` +
  `stock_prices`) and builds the pinned payload (H1):
    - counts by status;
    - **realized $P&L (closed only) AND mark-to-market-including-open** (both —
      realization-asymmetry disclosure, never realized-only);
    - open MTM valued at the as-of close (H1b: last-available close + a
      staleness flag when the symbol has no as-of bar);
    - today's exits keyed by `exit_date == as_of`;
    - win-rate over **closed** trades only;
    - `same_bar_ambiguous_exits` count;
    - total residual cash.
* `persist_daily_snapshot(as_of, payload)` upserts one row per
  `(report_type='daily', as_of_date)` (H2).
* `render_payload(payload)` → Discord text (the plain `report` path re-renders
  from the stored snapshot; H3).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rainier.core.database import get_session
from rainier.core.models import PaperReportSnapshot, PaperTrade, StockPrice

log = logging.getLogger(__name__)

REPORT_TYPE_DAILY = "daily"
REPORT_TYPE_WEEKLY = "weekly"


def _as_date(d: Any) -> date:
    from datetime import datetime

    return d.date() if isinstance(d, datetime) else d


def _latest_close_on_or_before(session, symbol: str, as_of: date) -> tuple[float | None, date | None]:
    """The most recent close at-or-before as_of (H1b: tolerate a missing as-of
    bar). Returns (close, bar_date) or (None, None)."""
    from rainier.paper.ingest import canonical_instant

    row = session.execute(
        select(StockPrice.close, StockPrice.date)
        .where(
            StockPrice.symbol == symbol,
            StockPrice.date <= canonical_instant(as_of),
            StockPrice.close.isnot(None),
        )
        .order_by(StockPrice.date.desc())
        .limit(1)
    ).first()
    if row is None:
        return None, None
    return float(row[0]), _as_date(row[1])


def compute_daily_payload(as_of: date) -> dict[str, Any]:
    """Build the pinned daily payload from raw inputs (paper_trade + prices)."""
    with get_session() as session:
        positions = session.execute(select(PaperTrade)).scalars().all()

        counts = {"pending": 0, "open": 0, "closed": 0, "expired": 0}
        realized_pnl = 0.0
        open_mtm_pnl = 0.0
        residual_cash = 0.0
        closed_wins = 0
        closed_total = 0
        same_bar = 0
        todays_exits: list[dict[str, Any]] = []
        stale_mtm: list[str] = []

        for p in positions:
            counts[p.status] = counts.get(p.status, 0) + 1
            if p.residual_cash is not None:
                residual_cash += float(p.residual_cash)

            if p.status == "closed":
                closed_total += 1
                if p.pnl is not None:
                    realized_pnl += float(p.pnl)
                if p.return_pct is not None and p.return_pct > 0:
                    closed_wins += 1
                if p.exit_date is not None and _as_date(p.exit_date) == as_of:
                    todays_exits.append(
                        {
                            "symbol": p.symbol,
                            "exit_reason": p.exit_reason,
                            "exit_price": p.exit_price,
                            "return_pct": p.return_pct,
                            "pnl": p.pnl,
                        }
                    )

            elif p.status == "open":
                # MTM at the as-of close (last-available close if as-of is a
                # holiday/halt/gap — H1b), valued on the booked shares/entry.
                close_px, bar_date = _latest_close_on_or_before(
                    session, p.symbol, as_of
                )
                if close_px is None or p.entry_price is None or p.shares is None:
                    continue
                if bar_date is not None and bar_date != as_of:
                    stale_mtm.append(p.symbol)
                open_mtm_pnl += p.shares * (close_px - float(p.entry_price))

        same_bar = _count_same_bar_exits(session, as_of)

    realized_and_open_mtm = realized_pnl + open_mtm_pnl
    win_rate = (closed_wins / closed_total) if closed_total else None

    return {
        "report_type": REPORT_TYPE_DAILY,
        "as_of_date": as_of.isoformat(),
        "counts_by_status": counts,
        "realized_pnl": round(realized_pnl, 4),
        "mtm_including_open_pnl": round(realized_and_open_mtm, 4),
        "open_mtm_pnl": round(open_mtm_pnl, 4),
        "todays_exits": todays_exits,
        "win_rate_closed": win_rate,
        "closed_trades": closed_total,
        "same_bar_ambiguous_exits": same_bar,
        "total_residual_cash": round(residual_cash, 4),
        "stale_mtm_symbols": sorted(set(stale_mtm)),
        # Methodological disclosure (parallel-review): cumulative figure is a sum
        # of independent $10k experiments, not account equity.
        "disclosure": (
            "Cumulative P&L is the sum of independent $10k experiments, not "
            "account equity (no portfolio cap, no fees/slippage). Realized is "
            "closed-only; MTM-including-open corrects realization asymmetry. "
            "same_bar_ambiguous_exits counts the SL-first downward bias."
        ),
    }


def _count_same_bar_exits(session, as_of: date) -> int:
    """Count closed positions whose exit bar had BOTH low<=stop AND high>=target
    (the SL-first conservative convention's known downward bias)."""
    from rainier.paper.ingest import canonical_instant

    positions = session.execute(
        select(PaperTrade).where(PaperTrade.status == "closed")
    ).scalars().all()
    n = 0
    for p in positions:
        if p.exit_date is None:
            continue
        bar = session.execute(
            select(StockPrice.high, StockPrice.low).where(
                StockPrice.symbol == p.symbol,
                StockPrice.date == canonical_instant(_as_date(p.exit_date)),
            )
        ).first()
        if bar is None or bar[0] is None or bar[1] is None:
            continue
        high, low = float(bar[0]), float(bar[1])
        if low <= p.stop_loss and high >= p.target_price:
            n += 1
    return n


def persist_daily_snapshot(as_of: date, payload: dict[str, Any]) -> None:
    """Upsert one (daily, as_of_date) snapshot row (H2/H4)."""
    with get_session() as session:
        stmt = (
            pg_insert(PaperReportSnapshot)
            .values(
                report_type=REPORT_TYPE_DAILY, as_of_date=as_of, payload=payload
            )
            .on_conflict_do_update(
                constraint="uq_paper_report_snapshot_type_date",
                set_={"payload": payload},
            )
        )
        session.execute(stmt)


def load_snapshot(report_type: str, as_of: date) -> dict[str, Any] | None:
    with get_session() as session:
        row = session.execute(
            select(PaperReportSnapshot.payload).where(
                PaperReportSnapshot.report_type == report_type,
                PaperReportSnapshot.as_of_date == as_of,
            )
        ).first()
    return dict(row[0]) if row is not None else None


def render_payload(payload: dict[str, Any]) -> str:
    """Render a snapshot payload into Discord-ready text (snapshot-only path)."""
    c = payload.get("counts_by_status", {})
    lines = [
        f"**Paper book — {payload.get('as_of_date')}**",
        f"positions: {c.get('pending', 0)} pending · {c.get('open', 0)} open · "
        f"{c.get('closed', 0)} closed · {c.get('expired', 0)} expired",
        f"realized P&L (closed): ${payload.get('realized_pnl', 0):,.2f}",
        f"MTM incl. open: ${payload.get('mtm_including_open_pnl', 0):,.2f}",
    ]
    wr = payload.get("win_rate_closed")
    if wr is not None:
        lines.append(
            f"win-rate (closed, n={payload.get('closed_trades', 0)}): {wr:.0%}"
        )
    exits = payload.get("todays_exits", [])
    if exits:
        lines.append("today's exits:")
        for e in exits:
            lines.append(
                f"  {e['symbol']} {e['exit_reason']} @ {e['exit_price']} "
                f"(${e.get('pnl', 0):,.2f})"
            )
    sba = payload.get("same_bar_ambiguous_exits", 0)
    if sba:
        lines.append(f"same-bar ambiguous exits (SL-first bias): {sba}")
    lines.append(f"residual cash: ${payload.get('total_residual_cash', 0):,.2f}")
    lines.append("")
    lines.append(payload.get("disclosure", ""))
    return "\n".join(lines)


def send_daily_paper_report(payload: dict[str, Any], discord_config: Any) -> bool:
    """Push the rendered report to Discord (D11). Non-fatal on failure / no
    webhook configured (H6): logs + returns False, never raises."""
    text = render_payload(payload)
    try:
        from rainier.alerts.discord import send_daily_report

        webhook = getattr(discord_config, "webhook_url", None) if discord_config else None
        if not discord_config or not getattr(discord_config, "enabled", False) or not webhook:
            log.info("paper_report_discord_skipped reason=no_webhook")
            return False
        send_daily_report(text, discord_config)
        return True
    except Exception:
        log.exception("paper_report_discord_failed")
        return False
