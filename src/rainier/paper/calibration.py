"""Calibration block for the LLM thesis prompt (design D7a).

WHAT THIS IS
------------
Each day, after the paper-book report runs, we compute a small "here is how
your prior calls actually graded out" block and feed it back into tomorrow's
thesis prompt. The goal is to let the LLM self-correct (e.g. "my high-confidence
setup_long calls only hit 40% at 5d — tighten up").

THE TRAP THIS AVOIDS (realization asymmetry, D7a2)
--------------------------------------------------
The paper book holds winners indefinitely but realizes losers fast (stop-loss
fires early). So a "win-rate over *closed* paper trades" is survivorship-skewed
upward — the open winners aren't in the denominator yet. Using that as the
headline would teach the LLM the wrong lesson.

So the HEADLINE is the UNBIASED `ThesisEvaluation` fixed-horizon stats: every
graded thesis is held to a fixed 1d/5d/10d close-to-close horizon regardless of
the paper-book's hold policy. Every thesis matures; nothing survives by staying
open. The realized paper record (last K=10 closed + mark-to-market INCLUDING
open) is carried only as clearly-labeled *supplementary* context.

  ┌─────────────────────────────────────────────────────────────┐
  │ HEADLINE  (unbiased — ThesisEvaluation fixed-horizon)         │
  │   win-rate + avg return by verdict, by confidence bucket,     │
  │   for each of 1d / 5d / 10d                                   │
  ├─────────────────────────────────────────────────────────────┤
  │ SUPPLEMENTARY  (labeled realized-closed — survivorship-prone) │
  │   last K=10 closed paper trades (symbol, reason, return)      │
  │   + mark-to-market INCLUDING open positions                   │
  └─────────────────────────────────────────────────────────────┘

The compute reads from the LEGACY local-TimescaleDB engine only
(`core.database.get_session`): `thesis_evaluations` + `paper_trade`.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select

from rainier.core.database import get_session
from rainier.core.models import PaperCalibration, PaperTrade, ThesisEvaluation

log = logging.getLogger(__name__)

# Horizons the headline reports, in display order.
HORIZONS: tuple[str, ...] = ("1d", "5d", "10d")
# Rolling window for the unbiased headline aggregation.
HEADLINE_WINDOW_DAYS = 60
# Number of recent closed paper trades shown as supplementary context.
CLOSED_SAMPLE_K = 10
# Confidence buckets (llm_confidence 1-10) for the headline breakdown.
_CONF_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("low (1-4)", 1, 4),
    ("mid (5-7)", 5, 7),
    ("high (8-10)", 8, 10),
)


def _as_date_local(d: Any) -> date:
    """Coerce a datetime/date column value to a plain date for comparison."""
    if isinstance(d, datetime):
        return d.date()
    return d


def _hit(verdict: str, return_pct: float) -> bool:
    """Direction-aware hit: setup_long hits on a gain; watch/no_setup hit on a
    flat-or-loss (the LLM was right NOT to buy). Mirrors eval/research."""
    if verdict == "setup_long":
        return return_pct > 0
    if verdict in ("watch", "no_setup"):
        return return_pct <= 0
    return False


def _agg(returns: list[float], hits: list[bool]) -> dict[str, Any]:
    n = len(returns)
    return {
        "n": n,
        "win_rate": (sum(1 for h in hits if h) / n) if n else None,
        "avg_return_pct": (sum(returns) / n) if n else None,
    }


def _conf_bucket(conf: int | None) -> str | None:
    if conf is None:
        return None
    for label, lo, hi in _CONF_BUCKETS:
        if lo <= conf <= hi:
            return label
    return None


def _headline_for_horizon(rows: list[tuple[str, int | None, float]]) -> dict[str, Any]:
    """rows = (verdict, llm_confidence, return_pct) for one horizon."""
    by_verdict: dict[str, tuple[list[float], list[bool]]] = {}
    by_bucket: dict[str, tuple[list[float], list[bool]]] = {}
    all_returns: list[float] = []
    all_hits: list[bool] = []
    for verdict, conf, ret in rows:
        hit = _hit(verdict, ret)
        all_returns.append(ret)
        all_hits.append(hit)
        vr, vh = by_verdict.setdefault(verdict, ([], []))
        vr.append(ret)
        vh.append(hit)
        bucket = _conf_bucket(conf)
        if bucket is not None:
            br, bh = by_bucket.setdefault(bucket, ([], []))
            br.append(ret)
            bh.append(hit)
    return {
        "overall": _agg(all_returns, all_hits),
        "by_verdict": {
            v: _agg(r, h) for v, (r, h) in sorted(by_verdict.items())
        },
        "by_confidence_bucket": {
            b: _agg(r, h) for b, (r, h) in by_bucket.items()
        },
    }


def compute_calibration_payload(
    as_of: date, *, window_days: int = HEADLINE_WINDOW_DAYS, k: int = CLOSED_SAMPLE_K
) -> dict[str, Any]:
    """Build the bounded calibration payload (design D7a).

    HEADLINE (unbiased): per-horizon win-rate + avg return, broken down by
    verdict and by confidence bucket, over the rolling `window_days` of
    `thesis_evaluations` (scan_date <= as_of).

    SUPPLEMENTARY (labeled realized): the last `k` closed paper trades and the
    mark-to-market-including-open dollar figure.
    """
    cutoff = as_of - timedelta(days=window_days)

    # Per-horizon maturity ceiling (codex iter-2 [P2]): a thesis only counts in
    # the headline once its fixed horizon has actually elapsed by `as_of`. In
    # forward operation that's automatic, but a historical replay/backfill
    # (run_daily_eval --eval-date <past>) runs after later-dated rows already
    # exist, so a `scan_date <= as_of` filter alone would pull in e.g. a 10d row
    # for a thesis scanned the day before `as_of` — leaking future outcomes into
    # the calibration row. A row at `horizon` is mature iff its scan_date is on
    # or before the date `_HORIZON_DAYS[horizon]` trading days before `as_of`.
    from rainier.llm_thesis.eval import _HORIZON_DAYS, _trading_days_back

    horizon_ceiling: dict[str, date] = {
        h: _trading_days_back(as_of, _HORIZON_DAYS[h])
        for h in HORIZONS
        if h in _HORIZON_DAYS
    }

    with get_session() as session:
        # --- HEADLINE: unbiased fixed-horizon thesis evaluations. ---
        eval_rows = session.execute(
            select(
                ThesisEvaluation.horizon,
                ThesisEvaluation.verdict,
                ThesisEvaluation.llm_confidence,
                ThesisEvaluation.return_pct,
                ThesisEvaluation.scan_date,
            ).where(
                ThesisEvaluation.scan_date >= cutoff,
                ThesisEvaluation.scan_date <= as_of,
            )
        ).all()

        # --- SUPPLEMENTARY: last K closed paper trades + MTM-including-open.
        # Select scalar columns (not full ORM rows) so nothing is read after the
        # session closes — matches the codebase's materialize-inside-the-block
        # convention (report.py / positions.py).
        closed = session.execute(
            select(
                PaperTrade.symbol,
                PaperTrade.exit_reason,
                PaperTrade.return_pct,
                PaperTrade.pnl,
            )
            .where(
                PaperTrade.status == "closed",
                PaperTrade.exit_date.isnot(None),
                PaperTrade.exit_date <= as_of,
                PaperTrade.shadow.is_(False),  # WS A isolation: live read only.
            )
            .order_by(PaperTrade.exit_date.desc(), PaperTrade.id.desc())
            .limit(k)
        ).all()

    by_horizon: dict[str, list[tuple[str, int | None, float]]] = {}
    for horizon, verdict, conf, ret, row_scan in eval_rows:
        h = str(horizon)
        ceiling = horizon_ceiling.get(h)
        # Drop rows whose horizon hadn't matured by as_of (replay hindsight guard).
        if ceiling is not None and _as_date_local(row_scan) > ceiling:
            continue
        by_horizon.setdefault(h, []).append(
            (str(verdict or ""), conf, float(ret))
        )
    headline = {
        h: _headline_for_horizon(by_horizon.get(h, []))
        for h in HORIZONS
        if h in by_horizon
    }

    # NOTE: this sum is over the last-K SAMPLE only (the query is .limit(k)).
    # It is NOT the book's total realized P&L — that comes from the daily
    # payload below. Surfaced as a clearly sample-scoped field so the renderer
    # never pairs it with all-book MTM (codex iter-1 [P2]).
    sample_realized_pnl = 0.0
    closed_sample: list[dict[str, Any]] = []
    closed_wins = 0
    for symbol, exit_reason, return_pct, pnl in closed:
        if pnl is not None:
            sample_realized_pnl += float(pnl)
        if return_pct is not None and return_pct > 0:
            closed_wins += 1
        closed_sample.append(
            {
                "symbol": symbol,
                "outcome": "win" if (return_pct or 0) > 0 else "loss",
                "exit_reason": exit_reason,
                "return_pct": (
                    round(float(return_pct), 4) if return_pct is not None else None
                ),
            }
        )

    # MTM-including-open comes from the daily report payload (realized + open
    # MTM), the canonical realization-asymmetry-corrected figure. Imported
    # lazily to avoid a module-load cycle (report imports nothing from here).
    from rainier.paper.report import compute_daily_payload

    daily = compute_daily_payload(as_of)

    return {
        "as_of_date": as_of.isoformat(),
        "window_days": window_days,
        # HEADLINE — unbiased, survivorship-free.
        "headline_fixed_horizon": headline,
        # SUPPLEMENTARY — realized paper record, explicitly labeled survivorship-
        # prone so neither a renderer nor a reader mistakes it for the headline.
        "supplementary_realized": {
            "label": "realized-closed (survivorship-prone — NOT the headline)",
            "n_closed_sample": len(closed_sample),
            "closed_win_rate": (
                (closed_wins / len(closed_sample)) if closed_sample else None
            ),
            "recent_closed": closed_sample,
            # P&L summed over the last-K SAMPLE shown above — NOT the book total.
            "sample_realized_pnl_usd": round(sample_realized_pnl, 2),
            # The BOOK-TOTAL realized P&L (all closed, as-of capped) from the
            # daily payload — the figure that correctly pairs with MTM-incl-open.
            "realized_pnl_usd": daily.get("realized_pnl"),
            "mtm_including_open_pnl_usd": daily.get("mtm_including_open_pnl"),
        },
    }


def persist_calibration(as_of: date, payload: dict[str, Any]) -> None:
    """Upsert one (as_of_date) calibration row (idempotent per day)."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    with get_session() as session:
        stmt = (
            pg_insert(PaperCalibration)
            .values(as_of_date=as_of, payload=payload)
            .on_conflict_do_update(
                constraint="uq_paper_calibration_as_of_date",
                set_={"payload": payload},
            )
        )
        session.execute(stmt)


def load_latest_calibration(
    as_of: date | None = None, *, strict_before: bool = False
) -> dict[str, Any] | None:
    """Return the most recent calibration payload relative to ``as_of``.

    With ``strict_before=False`` (default, generic loader) the bound is
    at-or-before ``as_of``. With ``strict_before=True`` it is STRICTLY before
    ``as_of`` — the no-hindsight contract for the prompt-injection path
    (codex iter-3 [P2]).

    The thesis prompt for a scan on day D must only ever see calibration
    computed on a PRIOR day: day D's own ``paper_calibration`` row is written
    by ``run_daily_eval`` at end-of-day D (after that day's eval + paper-book
    outcomes), so injecting it into a same-day scan rerun / historical replay
    would leak same-day hindsight into the prompt. ``generate_thesis`` therefore
    passes ``strict_before=True``.

    Returns None when no eligible row exists yet (Phase-0 days / fresh DB) — the
    prompt then renders no section.
    """
    with get_session() as session:
        q = select(PaperCalibration.payload).order_by(
            PaperCalibration.as_of_date.desc()
        )
        if as_of is not None:
            if strict_before:
                q = q.where(PaperCalibration.as_of_date < as_of)
            else:
                q = q.where(PaperCalibration.as_of_date <= as_of)
        row = session.execute(q.limit(1)).first()
    return dict(row[0]) if row is not None else None


def _fmt_pct(v: float | None) -> str:
    return f"{v:+.2%}" if v is not None else "n/a"


def _fmt_rate(v: float | None) -> str:
    return f"{v:.0%}" if v is not None else "n/a"


def render_calibration_section(payload: dict[str, Any] | None) -> str:
    """Render the calibration payload into a bounded prompt section.

    Returns "" when there's nothing to show, so the caller can append
    unconditionally. The HEADLINE is the unbiased fixed-horizon stats; the
    realized paper record is rendered under an explicit "supplementary /
    survivorship-prone" label so the LLM never reads it as the primary signal.
    """
    if not payload:
        return ""
    headline = payload.get("headline_fixed_horizon") or {}
    supp = payload.get("supplementary_realized") or {}
    if not headline and not supp:
        return ""

    lines: list[str] = [
        "--- Calibration (how prior calls graded out) ---",
        "HEADLINE = unbiased fixed-horizon outcomes (every thesis matures at a "
        "fixed 1d/5d/10d close — survivorship-free). Treat this as primary.",
    ]
    for horizon in HORIZONS:
        block = headline.get(horizon)
        if not block:
            continue
        overall = block.get("overall", {})
        lines.append(
            f"[{horizon}] overall: win-rate {_fmt_rate(overall.get('win_rate'))}, "
            f"avg {_fmt_pct(overall.get('avg_return_pct'))} (n={overall.get('n', 0)})"
        )
        by_verdict = block.get("by_verdict", {})
        for verdict, agg in by_verdict.items():
            lines.append(
                f"    {verdict}: win-rate {_fmt_rate(agg.get('win_rate'))}, "
                f"avg {_fmt_pct(agg.get('avg_return_pct'))} (n={agg.get('n', 0)})"
            )
        by_bucket = block.get("by_confidence_bucket", {})
        for bucket, agg in by_bucket.items():
            lines.append(
                f"    conf {bucket}: win-rate {_fmt_rate(agg.get('win_rate'))}, "
                f"avg {_fmt_pct(agg.get('avg_return_pct'))} (n={agg.get('n', 0)})"
            )

    if supp:
        lines.append(
            f"SUPPLEMENTARY ({supp.get('label', 'realized-closed')}) — context "
            "only, do NOT treat as the headline:"
        )
        recent = supp.get("recent_closed", [])
        if recent:
            rendered = ", ".join(
                f"{r['symbol']} {r['outcome']}({r.get('exit_reason') or '?'} "
                f"{_fmt_pct(r.get('return_pct'))})"
                for r in recent
            )
            lines.append(f"    last {len(recent)} closed: {rendered}")
        lines.append(
            f"    realized $P&L (all closed): "
            f"${supp.get('realized_pnl_usd') or 0:,.2f} · MTM incl. open: "
            f"${supp.get('mtm_including_open_pnl_usd') or 0:,.2f}"
        )
    return "\n".join(lines)
