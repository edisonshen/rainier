"""Weekly missed-winner sweep (Phase 3, design §5(4) / D8).

COVERAGE DIAGNOSTIC ONLY: a flagged "missed winner" is a trailing
point-to-point return with no entry-timing/stop/target — NOT comparable to the
path-dependent paper trades, and never an input to weight-tuning/calibration.
It answers one question: are big movers concentrated outside our funnel?

    Friday 09:00 PT run (as_of = Friday)
        │
        ▼
    window_end   = last completed session BEFORE the run  (Thursday)
    window_start = 10 trading days before window_end
    return       = close(window_end) / close(window_start) − 1
                   (11 closes, 10 intervals)
        │
        ▼
    cohort = get_current_qu100_cohort(as_of)      ← point-in-time membership
    ingest_prices(cohort ∪ declined, as_of=window_end, window_days=11)
        │                              └── anchored at window_end so the
        ▼                                  in-progress Friday bar is NEVER
    flag: return ≥ +10% (inclusive) AND not held   written (a partial upsert
        │                                          would later read complete)
        ▼
    attribute (highest funnel stage reached ANYWHERE in the window):
      (i)   declined thesis (watch/no_setup)  → verdict_watch_or_no_setup
      (ii)  screened, pattern_type NOT NULL   → not_in_top5
      (iii) screened, pattern_type NULL       → no_pattern
      (iv)  else                              → rank_too_low
        │
        ▼
    snapshot (paper_report_snapshot, weekly) ── durable record
    ResearchInsight kind=missed_winner       ── mutable action queue
    Discord push                             ── non-fatal
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from sqlalchemy import func, select

from rainier.alerts.discord import _http_status, send_daily_report
from rainier.core.database import get_session
from rainier.core.models import (
    LLMAnalysisRecord,
    PaperTrade,
    ScreenedStockRecord,
    StockPrice,
)

from .calendar import DEFAULT_CALENDAR, TradingCalendar
from .ingest import FetchFn, canonical_instant, get_current_qu100_cohort, ingest_prices
from .report import REPORT_TYPE_WEEKLY, persist_snapshot

log = logging.getLogger(__name__)

# 10 trading-day intervals (operator 2026-06-02, design D8).
WINDOW_SESSIONS = 10
# The ingest window must cover the anchor close too: 11 sessions, because the
# default window_days=10 drops close(window_start) (acceptance 3). +2 buffer
# sessions so a market holiday AT window_start (DEFAULT_CALENDAR is Mon-Fri
# only — no holiday table) still gets a real close fetched just before it for
# the at-or-before endpoint lookup (review iter-2).
INGEST_WINDOW_SESSIONS = WINDOW_SESSIONS + 1 + 2
# Endpoint closes use an at-or-before lookup capped at this many calendar days
# back, so an unpriced endpoint (e.g. window_end = Thanksgiving, which the
# no-holiday default calendar happily returns) degrades to the last completed
# PRICED day — per the plan's "last completed priced trading day" — instead of
# wiping the whole cohort into missing_price_symbols (review iter-2). 5 days
# covers a holiday cluster + weekend; anything staler is genuinely missing.
_CLOSE_LOOKBACK_DAYS = 5
# Forward-return thresholds — both INCLUSIVE (acceptance 2 / 6).
MISS_THRESHOLD = 0.10
DODGE_THRESHOLD = -0.10
# Float guard so an exact ±10% close pair (e.g. 90/100 → −0.09999999999999998)
# still lands on the inclusive side of the threshold.
_EPS = 1e-9

# Attribution buckets, ordered highest funnel stage first (acceptance 4).
BUCKET_VERDICT = "verdict_watch_or_no_setup"
BUCKET_NOT_TOP5 = "not_in_top5"
BUCKET_NO_PATTERN = "no_pattern"
BUCKET_RANK_TOO_LOW = "rank_too_low"
BUCKETS = (BUCKET_VERDICT, BUCKET_NOT_TOP5, BUCKET_NO_PATTERN, BUCKET_RANK_TOO_LOW)

# One tuning hypothesis per dominant bucket (acceptance 7 — the design's
# outcome→action mapping). Hypotheses, not actions: this metric never tunes.
_HYPOTHESES = {
    BUCKET_VERDICT: (
        "The LLM saw these names and declined them. Hypothesis: the verdict "
        "criteria are too conservative on this regime — review the declined "
        "theses' rationale against what actually moved before touching the "
        "prompt."
    ),
    BUCKET_NOT_TOP5: (
        "Winners were screened WITH a tradable pattern but ranked below the "
        "thesis cut. Hypothesis: the composite ranking under-weights whatever "
        "carried these names — compare their component scores to the top-5's."
    ),
    BUCKET_NO_PATTERN: (
        "Winners were screened but no tradable pattern fired. Hypothesis: the "
        "pattern detector set misses this move shape — inspect their charts "
        "for a recurring, codifiable structure."
    ),
    BUCKET_RANK_TOO_LOW: (
        "Most missed winners never entered the top-50 screen — expected, the "
        "screen covers a subset of QU100 by design. Hypothesis: widening "
        "screen depth would capture them, at proportional scan cost; treat as "
        "coverage information, not a tuning signal."
    ),
}

_DISCLOSURE = (
    "Coverage diagnostic ONLY: trailing point-to-point returns with no "
    "entry-timing/stop/target — not comparable to paper trades; never feeds "
    "weight-tuning or calibration. rank_too_low dominance is EXPECTED (the "
    "screen covers a subset of QU100). The current-cohort sweep has "
    "survivorship bias: symbols that crashed out of the index don't appear."
)


def compute_window(
    as_of: date, calendar: TradingCalendar | None = None
) -> tuple[date, date]:
    """The prior completed 10-trading-day window for a run on ``as_of``.

    ``window_end`` = last completed priced trading day strictly BEFORE the run
    (Thursday for a Friday 09:00 run — never the in-progress session);
    ``window_start`` = 10 trading days before ``window_end``. Exactly 11
    sessions inclusive (11 closes, 10 intervals) — acceptance 2.
    """
    cal = calendar or DEFAULT_CALENDAR
    window_end = cal.prev_session(as_of)
    window_start = cal.sub_sessions(window_end, WINDOW_SESSIONS)
    return window_start, window_end


# ---------------------------------------------------------------------------
# Funnel / held / pricing queries (raw inputs only — never ResearchInsight)
# ---------------------------------------------------------------------------


def _declined_symbols(session, window_start: date, window_end: date) -> set[str]:
    """Symbols with a thesis row in-window whose verdict is watch/no_setup —
    funnel tier (i). In-window = the thesis's screened scan_date."""
    rows = session.execute(
        select(ScreenedStockRecord.symbol)
        .join(
            LLMAnalysisRecord,
            ScreenedStockRecord.thesis_id == LLMAnalysisRecord.id,
        )
        .where(
            ScreenedStockRecord.scan_date >= window_start,
            ScreenedStockRecord.scan_date <= window_end,
            LLMAnalysisRecord.recommendation.in_(("watch", "no_setup")),
        )
        .distinct()
    ).all()
    return {r[0] for r in rows}


def _screened_sets(
    session, window_start: date, window_end: date
) -> tuple[set[str], set[str]]:
    """(symbols screened with a pattern, all screened symbols) in-window."""
    rows = session.execute(
        select(ScreenedStockRecord.symbol, ScreenedStockRecord.pattern_type).where(
            ScreenedStockRecord.scan_date >= window_start,
            ScreenedStockRecord.scan_date <= window_end,
        )
    ).all()
    with_pattern = {sym for sym, pat in rows if pat is not None}
    screened = {sym for sym, _pat in rows}
    return with_pattern, screened


def _held_symbols(session, window_start: date, window_end: date) -> set[str]:
    """Symbols we actually HELD during the window (acceptance 5): filled rows
    only (entry_date set, status open/closed) whose
    [entry_date, COALESCE(exit_date, window_end)] overlaps the window.
    Pending / never-filled expired rows are NOT held."""
    rows = session.execute(
        select(PaperTrade.symbol)
        .where(
            PaperTrade.entry_date.isnot(None),
            PaperTrade.status.in_(("open", "closed")),
            PaperTrade.entry_date <= window_end,
            func.coalesce(PaperTrade.exit_date, window_end) >= window_start,
            PaperTrade.shadow.is_(False),  # WS A isolation: live read only.
        )
        .distinct()
    ).all()
    return {r[0] for r in rows}


def _closes_at_or_before(
    session, symbols: set[str], d: date
) -> dict[str, float]:
    """Per-symbol close on ``d``, or the latest close within
    ``_CLOSE_LOOKBACK_DAYS`` before it. An exact-date lookup would zero out the
    entire report whenever an endpoint lands on a market holiday (~10
    Fridays/yr for window_start; every Thanksgiving for window_end) — the
    no-holiday DEFAULT_CALENDAR can't avoid picking one (review iter-2)."""
    if not symbols:
        return {}
    floor = canonical_instant(d - timedelta(days=_CLOSE_LOOKBACK_DAYS))
    rows = session.execute(
        select(StockPrice.symbol, StockPrice.close, StockPrice.date).where(
            StockPrice.symbol.in_(sorted(symbols)),
            StockPrice.date <= canonical_instant(d),
            StockPrice.date >= floor,
            StockPrice.close.isnot(None),
        )
    ).all()
    best: dict[str, Any] = {}
    out: dict[str, float] = {}
    for sym, px, dt in rows:
        if sym not in best or dt > best[sym]:
            best[sym] = dt
            out[sym] = float(px)
    return out


def _attribute(
    symbol: str,
    declined: set[str],
    with_pattern: set[str],
    screened: set[str],
) -> str:
    """Bucket = the HIGHEST funnel stage the symbol reached at any point inside
    the window (acceptance 4) — a day-2 declined thesis beats a day-9
    unscreened miss."""
    if symbol in declined:
        return BUCKET_VERDICT
    if symbol in with_pattern:
        return BUCKET_NOT_TOP5
    if symbol in screened:
        return BUCKET_NO_PATTERN
    return BUCKET_RANK_TOO_LOW


# ---------------------------------------------------------------------------
# Payload compute (raw inputs; shared by the live sweep and --regenerate)
# ---------------------------------------------------------------------------


def compute_weekly_payload(
    as_of: date,
    calendar: TradingCalendar | None = None,
    *,
    cohort: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the weekly missed-winner payload from raw inputs only:
    money_flow_snapshots (cohort), stock_prices, screened_stocks,
    analysis_results, paper_trade — NEVER from ResearchInsight (the insight
    queue is mutable; the snapshot is the durable record — acceptance 7).

    ``cohort`` lets the live sweep pass the membership it already priced —
    re-reading here would race the QU midday scrape (the Fri 09:00 PT run is
    12:00 ET; the morning slot often slips past it), swapping in a cohort the
    ingest never covered (review iter-2). ``--regenerate`` passes None and
    fetches the historical cohort itself.

    JSON-native throughout so the persisted snapshot re-renders identically.
    """
    window_start, window_end = compute_window(as_of, calendar)
    if cohort is None:
        cohort = get_current_qu100_cohort(as_of)
    cohort_symbols = sorted({c["symbol"] for c in cohort})
    rank_by_symbol: dict[str, int] = {}
    for c in cohort:
        rank_by_symbol.setdefault(c["symbol"], c["rank"])

    with get_session() as session:
        declined = _declined_symbols(session, window_start, window_end)
        with_pattern, screened = _screened_sets(session, window_start, window_end)
        held = _held_symbols(session, window_start, window_end)
        price_universe = set(cohort_symbols) | declined
        start_closes = _closes_at_or_before(session, price_universe, window_start)
        end_closes = _closes_at_or_before(session, price_universe, window_end)

    def _ret(symbol: str) -> float | None:
        start_px = start_closes.get(symbol)
        end_px = end_closes.get(symbol)
        if start_px is None or end_px is None or start_px <= 0:
            return None
        return end_px / start_px - 1.0

    missed: list[dict[str, Any]] = []
    missing_prices: list[str] = []
    held_winner_count = 0
    for symbol in cohort_symbols:
        ret = _ret(symbol)
        if ret is None:
            missing_prices.append(symbol)
            continue
        if ret < MISS_THRESHOLD - _EPS:
            continue
        if symbol in held:
            held_winner_count += 1
            continue
        missed.append(
            {
                "symbol": symbol,
                "rank": rank_by_symbol.get(symbol),
                "return_pct": round(ret, 6),
                "bucket": _attribute(symbol, declined, with_pattern, screened),
            }
        )
    missed.sort(key=lambda w: (-w["return_pct"], w["symbol"]))

    bucket_counts = {b: 0 for b in BUCKETS}
    for w in missed:
        bucket_counts[w["bucket"]] += 1
    dominant = None
    if missed:
        # Ties resolve to the highest funnel stage (BUCKETS order).
        dominant = max(BUCKETS, key=lambda b: (bucket_counts[b], -BUCKETS.index(b)))

    # R-B: dodged losers — declined in-window, forward return ≤ −10%
    # (inclusive), and NOT held (a loser we held wasn't dodged — acceptance 6).
    # A declined name with no price visibility is tracked in
    # missing_price_symbols too: an off-cohort decliner that can't price
    # (delisted after crashing — exactly the dodged shape) must not vanish
    # silently from the count it exists to inform (review iter-2).
    dodged: list[dict[str, Any]] = []
    for symbol in sorted(declined):
        ret = _ret(symbol)
        if ret is None:
            if symbol not in missing_prices:
                missing_prices.append(symbol)
            continue
        if symbol in held:
            continue
        if ret <= DODGE_THRESHOLD + _EPS:
            dodged.append({"symbol": symbol, "return_pct": round(ret, 6)})
    dodged.sort(key=lambda d: (d["return_pct"], d["symbol"]))
    missing_prices.sort()

    first = cohort[0] if cohort else None
    return {
        "report_type": REPORT_TYPE_WEEKLY,
        "as_of_date": as_of.isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "threshold_pct": MISS_THRESHOLD,
        "cohort": {
            "size": len(cohort_symbols),
            "data_date": first["data_date"].isoformat() if first else None,
            "captured_at": first["captured_at"].isoformat() if first else None,
        },
        "missed_winners": missed,
        "bucket_counts": bucket_counts,
        "dominant_bucket": dominant,
        "tuning_hypothesis": _HYPOTHESES[dominant] if dominant else None,
        "held_winner_count": held_winner_count,
        "dodged_losers": {"count": len(dodged), "names": dodged},
        "missing_price_symbols": missing_prices,
        "disclosure": _DISCLOSURE,
    }


def persist_weekly_snapshot(as_of: date, payload: dict[str, Any]) -> None:
    """Upsert one (weekly, as_of_date) snapshot row — the durable record."""
    persist_snapshot(REPORT_TYPE_WEEKLY, as_of, payload)


def render_weekly_payload(payload: dict[str, Any]) -> str:
    """Render a weekly snapshot payload into Discord-ready text."""
    cohort = payload.get("cohort", {})
    lines = [
        f"**QU100 missed-winner sweep — {payload.get('as_of_date')}**",
        f"window: {payload.get('window_start')} → {payload.get('window_end')} "
        f"(10 trading days)",
        f"cohort: {cohort.get('size', 0)} names "
        f"(data_date {cohort.get('data_date')})",
    ]
    winners = payload.get("missed_winners", [])
    if winners:
        lines.append(f"missed winners (≥ +10%, not held): {len(winners)}")
        for w in winners[:15]:
            lines.append(
                f"  {w['symbol']} {w['return_pct']:+.1%} — {w['bucket']}"
            )
        if len(winners) > 15:
            lines.append(f"  … +{len(winners) - 15} more")
        counts = payload.get("bucket_counts", {})
        lines.append(
            "buckets: " + " · ".join(f"{b}:{counts.get(b, 0)}" for b in BUCKETS)
        )
        lines.append(f"dominant bucket: {payload.get('dominant_bucket')}")
        lines.append(f"hypothesis: {payload.get('tuning_hypothesis')}")
    else:
        lines.append("missed winners (≥ +10%, not held): none")
    dodged = payload.get("dodged_losers", {})
    lines.append(
        f"R-B dodged losers (declined, ≤ −10%): {dodged.get('count', 0)}"
    )
    for d in dodged.get("names", [])[:10]:
        lines.append(f"  {d['symbol']} {d['return_pct']:+.1%}")
    missing = payload.get("missing_price_symbols", [])
    if missing:
        shown = ", ".join(missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        lines.append(f"missing prices: {shown}{more}")
    lines.append("")
    lines.append(payload.get("disclosure", ""))
    return "\n".join(lines)


def send_weekly_paper_report(payload: dict[str, Any], discord_config: Any) -> bool:
    """Push the weekly report to Discord. Non-fatal on failure / no webhook:
    logs + returns False, never raises (mirrors the daily path).

    ``_http_status``/``send_daily_report`` are module-top imports and the
    render runs INSIDE the try, so the except path can always execute — the
    never-raises guarantee is what keeps Discord exceptions away from any
    caller that might stringify them (review iter-1, security specialist).
    """
    try:
        text = render_weekly_payload(payload)
        webhook = (
            getattr(discord_config, "webhook_url", None) if discord_config else None
        )
        if (
            not discord_config
            or not getattr(discord_config, "enabled", False)
            or not webhook
        ):
            log.info("weekly_sweep_discord_skipped reason=no_webhook")
            return False
        send_daily_report(text, discord_config)
        return True
    except Exception as exc:
        # Never log str(exc) or the traceback: httpx.HTTPStatusError.__str__
        # embeds the request URL, and the webhook URL carries its secret token
        # (codex [P1] 2026-06-09, commit 96fbd13). Status + class only.
        log.error(
            "weekly_sweep_discord_failed status=%s error_type=%s",
            _http_status(exc),
            type(exc).__name__,
        )
        return False


# ---------------------------------------------------------------------------
# ResearchInsight emission (acceptance 8)
# ---------------------------------------------------------------------------


def _week_subject(as_of: date) -> str:
    iso = as_of.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def emit_missed_winner_insight(as_of: date, payload: dict[str, Any]) -> None:
    """One `missed_winner` insight per week (subject = the as-of ISO week),
    severity=info, action=noop. Evidence = the flagged names with bucket +
    return. The durable record remains the snapshot — insights are the mutable
    queue (a re-run UPSERTs the same pending row)."""
    from rainier.llm_thesis.research import emit_insight

    winners = payload["missed_winners"]
    dominant = payload.get("dominant_bucket")
    evidence = {
        "window_start": payload["window_start"],
        "window_end": payload["window_end"],
        "winners": [
            {"symbol": w["symbol"], "bucket": w["bucket"],
             "return_pct": w["return_pct"]}
            for w in winners
        ],
        "bucket_counts": payload["bucket_counts"],
        "dominant_bucket": dominant,
        "dodged_losers_count": payload["dodged_losers"]["count"],
    }
    if winners:
        rationale = (
            f"Missed-winner sweep {payload['window_start']}→"
            f"{payload['window_end']}: {len(winners)} QU100 names ≥ +10% not "
            f"held; dominant bucket {dominant}. "
            f"{payload['tuning_hypothesis']} R-B dodged losers: "
            f"{payload['dodged_losers']['count']}. Coverage diagnostic only — "
            "never a tuning input."
        )
    else:
        rationale = (
            f"Missed-winner sweep {payload['window_start']}→"
            f"{payload['window_end']}: no QU100 name ≥ +10% was missed. "
            f"R-B dodged losers: {payload['dodged_losers']['count']}."
        )
    subject = _week_subject(as_of)
    emit_insight(
        kind="missed_winner",
        subject=subject,
        severity="info",
        evidence=evidence,
        action={"kind": "noop", "target": subject, "params": {}},
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Orchestrator — the Fri 09:00 PT research-job step (acceptance 2/3)
# ---------------------------------------------------------------------------


def sweep_missed_winners(
    *,
    as_of: date,
    fetch_fn: FetchFn,
    calendar: TradingCalendar | None = None,
    discord_config: Any = None,
) -> dict[str, Any]:
    """Run the weekly sweep: cohort pricing → forward returns → flag → attribute
    → snapshot + insight + Discord. Returns the persisted payload.

    Sweep-owned pricing (acceptance 3): the daily ingest covers only
    active ∪ top-50 until PR 5 lands, so the sweep ingests the full cohort
    itself with ``as_of = window_end`` (never the in-progress session — a
    partial Friday bar upserted mid-session would later read as a complete
    day) and an 11-session window so close(window_start) is fetched too.
    Declined names are included so R-B (acceptance 6) prices even when a name
    has dropped out of the current cohort.
    """
    cal = calendar or DEFAULT_CALENDAR
    window_start, window_end = compute_window(as_of, cal)

    cohort = get_current_qu100_cohort(as_of)
    symbols = {c["symbol"] for c in cohort}
    with get_session() as session:
        symbols |= _declined_symbols(session, window_start, window_end)

    if symbols:
        ingest_prices(
            sorted(symbols),
            as_of=window_end,
            fetch_fn=fetch_fn,
            window_days=INGEST_WINDOW_SESSIONS,
            calendar=cal,
        )
    else:
        log.warning("weekly_sweep_empty_cohort as_of=%s", as_of.isoformat())

    # Pass the cohort we actually priced — a second read here would race the
    # QU midday scrape and could swap in a never-ingested membership
    # (review iter-2).
    payload = compute_weekly_payload(as_of, cal, cohort=cohort)
    persist_weekly_snapshot(as_of, payload)
    emit_missed_winner_insight(as_of, payload)
    send_weekly_paper_report(payload, discord_config)
    log.info(
        "weekly_sweep_done as_of=%s window=%s..%s cohort=%d missed=%d "
        "dominant=%s dodged=%d missing_prices=%d",
        as_of.isoformat(),
        payload["window_start"],
        payload["window_end"],
        payload["cohort"]["size"],
        len(payload["missed_winners"]),
        payload["dominant_bucket"],
        payload["dodged_losers"]["count"],
        len(payload["missing_price_symbols"]),
    )
    return payload
