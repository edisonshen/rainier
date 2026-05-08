"""Discord webhook notifications for trade signals and stock candidates."""

from __future__ import annotations

import json
import logging
from datetime import date as date_cls
from datetime import datetime

import httpx

from rainier.core.config import DiscordConfig
from rainier.core.types import Direction, Signal, StockCandidate

log = logging.getLogger(__name__)

# Pattern type → display label (Caisen methodology)
PATTERN_LABELS: dict[str, str] = {
    "w_bottom": "W Bottom",
    "m_top": "M Top",
    "false_breakdown": "False Breakdown",
    "false_breakout": "False Breakout",
    "false_breakdown_w": "False Breakdown W",
    "false_breakout_m": "False Breakout M",
    "bull_flag": "Bull Flag",
    "bear_flag": "Bear Flag",
    "hs_bottom": "H&S Bottom",
    "hs_top": "H&S Top",
    "sym_triangle_bottom": "Sym Tri Bottom",
    "sym_triangle_top": "Sym Tri Top",
}


def send_signal_alert(signal: Signal, config: DiscordConfig, chart_path: str | None = None):
    """Send a signal alert to Discord via webhook."""
    if not config.enabled or not config.webhook_url:
        return

    direction_emoji = "\U0001f7e2" if signal.direction == Direction.LONG else "\U0001f534"
    side = "BUY" if signal.direction == Direction.LONG else "SELL"

    embed = {
        "title": f"{direction_emoji} {signal.symbol} {side} Signal",
        "color": 0x00E676 if signal.direction == Direction.LONG else 0xFF1744,
        "fields": [
            {"name": "Timeframe", "value": signal.timeframe.value, "inline": True},
            {"name": "Entry", "value": f"{signal.entry_price:.2f}", "inline": True},
            {"name": "Stop Loss", "value": f"{signal.stop_loss:.2f}", "inline": True},
            {"name": "Take Profit", "value": f"{signal.take_profit:.2f}", "inline": True},
            {"name": "R:R", "value": f"{signal.rr_ratio:.1f}", "inline": True},
            {"name": "Confidence", "value": f"{signal.confidence:.0%}", "inline": True},
        ],
    }

    payload = {"embeds": [embed]}

    response = httpx.post(config.webhook_url, json=payload, timeout=10)
    response.raise_for_status()


def send_daily_report(report_text: str, config: DiscordConfig):
    """Send daily report to Discord."""
    if not config.enabled or not config.webhook_url:
        return

    # Discord message limit is 2000 chars; split if needed
    chunks = [report_text[i : i + 1990] for i in range(0, len(report_text), 1990)]

    for chunk in chunks:
        payload = {"content": chunk}
        response = httpx.post(config.webhook_url, json=payload, timeout=10)
        response.raise_for_status()


# ---------------------------------------------------------------------------
# Stock candidate alerts (QU100 screener → Discord)
# ---------------------------------------------------------------------------


def _resolve_webhook_url(config: DiscordConfig) -> str | None:
    """Get the webhook URL for stock alerts, falling back to main webhook."""
    return config.stock_webhook_url or config.webhook_url or None


# Session → display label
SESSION_LABELS: dict[str, str] = {
    "morning": "Morning (ET 11:30)",
    "midday": "Midday (ET 13:30)",
    "afternoon": "Afternoon (ET 15:30)",
    "close": "Close / After Hours (ET 17:30)",
}


def _format_summary_embed(
    candidates: list[StockCandidate], session: str | None = None,
) -> dict:
    """Format a summary table embed for all candidates."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    session_label = SESSION_LABELS.get(session or "", session or "")
    header = (
        f"{'#':>3} {'Sym':<6} {'Now':>8} "
        f"{'Pattern':<10} {'Status':<8} {'Entry':>8} {'Dist':>6}"
    )
    lines = [header]
    lines.append("-" * 58)

    for i, c in enumerate(candidates, 1):
        pattern = PATTERN_LABELS.get(
            c.pattern_type or "", c.pattern_type or "-"
        )
        if len(pattern) > 9:
            pattern = pattern[:9]
        status = c.pattern_status or "-"
        price = f"${c.current_price:.0f}" if c.current_price else "-"
        entry = f"${c.entry_price:.0f}" if c.entry_price else "-"
        if c.distance_to_entry_pct is not None:
            dist = f"{c.distance_to_entry_pct:+.1f}%"
        else:
            dist = "-"
        lines.append(
            f"{i:>3} {c.symbol:<6} {price:>8} "
            f"{pattern:<10} {status:<8} {entry:>8} {dist:>6}"
        )

    table = "\n".join(lines)

    title = f"\U0001f4ca QU100 Actionable Setups ({len(candidates)})"
    if session_label:
        title += f" — {session_label}"

    return {
        "title": title,
        "description": (
            f"**{now} PT** | Dist = current vs entry\n"
            f"```\n{table}\n```"
        ),
        "color": 0x2196F3,
    }


def _format_candidate_embed(candidate: StockCandidate) -> dict:
    """Format a detail embed for a single candidate with pattern data."""
    pattern_label = PATTERN_LABELS.get(
        candidate.pattern_type or "", candidate.pattern_type or "Unknown"
    )
    is_bullish = candidate.pattern_direction == "bullish"
    color = 0x00E676 if is_bullish else 0xFF1744
    direction_emoji = "\U0001f7e2" if is_bullish else "\U0001f534"
    vol_icon = "\u2705" if candidate.volume_confirmed else "\u274c"

    # Current price context
    price_str = f"${candidate.current_price:.2f}" if candidate.current_price else "-"
    if candidate.distance_to_entry_pct is not None:
        dist = candidate.distance_to_entry_pct
        if abs(dist) < 1.0:
            dist_label = f"AT ENTRY ({dist:+.1f}%)"
        elif dist > 0:
            dist_label = f"{dist:+.1f}% above entry"
        else:
            dist_label = f"{abs(dist):.1f}% below entry"
    else:
        dist_label = "-"

    freshness = ""
    if candidate.bars_since_breakout is not None:
        if candidate.bars_since_breakout == 0:
            freshness = "TODAY"
        elif candidate.bars_since_breakout == 1:
            freshness = "yesterday"
        else:
            freshness = f"{candidate.bars_since_breakout}d ago"

    fields = [
        {"name": "Pattern", "value": pattern_label, "inline": True},
        {
            "name": "Status",
            "value": (candidate.pattern_status or "-")
            + (f" ({freshness})" if freshness else ""),
            "inline": True,
        },
        {"name": "Volume", "value": vol_icon, "inline": True},
        {"name": "Now", "value": price_str, "inline": True},
    ]

    if candidate.entry_price is not None:
        fields.append(
            {"name": "Entry", "value": f"${candidate.entry_price:.2f}", "inline": True}
        )
    fields.append({"name": "Dist", "value": dist_label, "inline": True})

    if candidate.stop_loss is not None:
        fields.append(
            {"name": "Stop Loss", "value": f"${candidate.stop_loss:.2f}", "inline": True}
        )
    if candidate.target_price is not None:
        fields.append(
            {"name": "Target", "value": f"${candidate.target_price:.2f}", "inline": True}
        )
    if candidate.rr_ratio is not None:
        fields.append(
            {"name": "R:R", "value": f"{candidate.rr_ratio:.1f}", "inline": True}
        )
    if candidate.pattern_confidence is not None:
        fields.append(
            {"name": "Confidence", "value": f"{candidate.pattern_confidence:.0%}", "inline": True}
        )

    fields.append({"name": "Sector", "value": candidate.sector, "inline": True})
    fields.append(
        {"name": "Rank", "value": f"#{candidate.rank} ({candidate.long_short})", "inline": True}
    )

    return {
        "title": f"{direction_emoji} {candidate.symbol}",
        "color": color,
        "fields": fields,
    }


def _build_payloads(
    candidates: list[StockCandidate], session: str | None = None,
) -> list[dict]:
    """Build webhook payloads, splitting across messages to respect Discord limits.

    Discord limits: 10 embeds per message, 6000 chars total per message.
    """
    summary = _format_summary_embed(candidates, session=session)
    detail_embeds = [
        _format_candidate_embed(c)
        for c in candidates
        if c.pattern_type is not None
    ]

    # Group into payloads of max 10 embeds each (summary counts as 1)
    payloads: list[dict] = []
    all_embeds = [summary] + detail_embeds
    for i in range(0, len(all_embeds), 10):
        batch = all_embeds[i : i + 10]
        payloads.append({"embeds": batch})

    return payloads


def _truncate(text: str, limit: int) -> str:
    """Single-place truncation with an ellipsis to respect Discord caps."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_thesis_message(
    candidate: StockCandidate,
    thesis: dict,
) -> str:
    """Render one rich thesis message for Discord.

    Each message is capped at <=1800 chars (Discord 2000-char limit, 200 buffer).
    """
    verdict = thesis.get("verdict", "?")
    setup_q = thesis.get("setup_quality")
    confidence = thesis.get("llm_confidence")
    radar = thesis.get("paragraph_radar") or ""
    evidence = thesis.get("paragraph_evidence") or ""
    invalidation = thesis.get("paragraph_invalidation") or ""
    risks = thesis.get("risks") or []
    watch = thesis.get("watch_items") or []
    patterns = thesis.get("patterns_in_chart_not_in_indicators") or []

    header = f"**{candidate.symbol}** — verdict: `{verdict}`"
    if setup_q is not None and confidence is not None:
        header += f" · quality {setup_q}/10 · confidence {confidence}/10"

    parts = [header]
    parts.append(f"*Radar:* {radar}")
    parts.append(f"*Evidence:* {evidence}")
    parts.append(f"*Invalidation:* {invalidation}")
    if risks:
        parts.append("*Risks:* " + "; ".join(str(r) for r in risks))
    if watch:
        parts.append("*Watch:* " + "; ".join(str(w) for w in watch))
    if isinstance(patterns, list) and patterns:
        parts.append("*Chart patterns (not in indicators):* " + "; ".join(patterns))
    elif patterns == "none":
        pass

    body = "\n".join(parts)
    return _truncate(body, 1800)


def send_stock_candidates(
    candidates: list[StockCandidate],
    config: DiscordConfig,
    theses: dict[str, dict] | None = None,
) -> None:
    """Send QU100 stock candidate alerts to Discord.

    Args:
        candidates: Top N screened stock candidates, sorted by pattern match quality.
        config: Discord configuration with webhook URL and enabled flag.
        theses: Optional `{symbol: thesis_dict}` map. When provided, the standard
            top-20 summary is followed by a rich thesis message for each of the
            top-5 candidates whose symbol is present in `theses`. Default `None`
            preserves existing top-20-only behavior — the regression contract.
    """
    if not candidates:
        return
    if not config.enabled:
        log.debug("discord_alerts_disabled")
        return

    webhook_url = _resolve_webhook_url(config)
    if not webhook_url:
        log.warning("discord_no_webhook_url")
        return

    payloads = _build_payloads(candidates)

    for payload in payloads:
        try:
            response = httpx.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception:
            log.exception("discord_send_failed")

    if not theses:
        return

    # Per-ticker rich thesis messages — top 5 only, in candidate order.
    for candidate in candidates[:5]:
        thesis = theses.get(candidate.symbol)
        if not thesis:
            continue
        try:
            content = _format_thesis_message(candidate, thesis)
            response = httpx.post(
                webhook_url, json={"content": content}, timeout=10
            )
            response.raise_for_status()
        except Exception:
            log.exception("discord_thesis_send_failed symbol=%s", candidate.symbol)


# ---------------------------------------------------------------------------
# Daily eval report (PR2)
# ---------------------------------------------------------------------------


def _format_eval_message(
    *,
    eval_date: date_cls,
    yesterday_rows: list[dict],
    base_rates: dict[str, list],   # {verdict: [HitRate per horizon]}
    signal_contribs: list,         # [SignalContribution]
    p_threshold: float = 0.05,
) -> str:
    """Render the Discord eval-report body. <=1800 chars per message."""
    lines: list[str] = []
    lines.append(f"**Eval report — {eval_date.isoformat()}**")
    if yesterday_rows:
        scan_date = yesterday_rows[0].get("scan_date") or "(prior scan)"
        session = yesterday_rows[0].get("session_name") or "afternoon"
        lines.append(
            f"Yesterday's {session} scan ({scan_date}):"
        )
        for row in yesterday_rows:
            mark = "[HIT]" if row.get("hit") else "[miss]"
            verdict = row.get("verdict") or "?"
            confidence = row.get("llm_confidence")
            sym = row.get("symbol") or "?"
            ret = float(row.get("return_pct") or 0.0)
            conf_part = f"({confidence}/10)" if confidence is not None else ""
            lines.append(
                f"  {mark} {sym:<6} {verdict:<10} {conf_part:<7} -> {ret:+.2%}"
            )
    else:
        lines.append("Yesterday's scan: no graded picks.")

    lines.append("")
    lines.append("30-day base rates (rolling, 5d horizon):")
    for verdict in ("setup_long", "watch", "no_setup"):
        rates = base_rates.get(verdict, [])
        five = next((r for r in rates if getattr(r, "horizon", None) == "5d"), None)
        if five is None or five.n == 0:
            lines.append(f"  {verdict:<12} (no data)")
            continue
        lines.append(
            f"  {verdict:<12} win-rate {five.win_rate:.0%}   "
            f"avg {five.avg_return_pct:+.2%}  n={five.n}"
        )

    lines.append("")
    sigs_with_p = [c for c in signal_contribs if c.p_value is not None]
    if sigs_with_p:
        lines.append(
            f"Signal contribution (rolling 30d, p<{p_threshold:.2f} only):"
        )
        any_significant = False
        for c in sigs_with_p:
            if c.p_value is None or c.p_value > p_threshold:
                continue
            any_significant = True
            tag = "[+]" if c.lift > 0 else "[-]"
            lines.append(
                f"  {tag} {c.name:<22} {c.lift:+.2%} lift   "
                f"n={c.n_used}   p={c.p_value:.3f}"
            )
        if not any_significant:
            lines.append("  (no signals with p<%.2f this window)" % p_threshold)
    else:
        lines.append("Signal contribution: insufficient data.")

    body = "\n".join(lines)
    return _truncate(body, 1800)


def send_eval_report(
    *,
    eval_date: date_cls,
    yesterday_rows: list[dict],
    base_rates: dict[str, list],
    signal_contribs: list,
    config: DiscordConfig,
) -> None:
    """Post the daily eval report to Discord (stock channel, then webhook)."""
    if not config.enabled:
        log.debug("discord_alerts_disabled")
        return
    webhook_url = _resolve_webhook_url(config)
    if not webhook_url:
        log.warning("discord_no_webhook_url_eval")
        return

    message = _format_eval_message(
        eval_date=eval_date,
        yesterday_rows=yesterday_rows,
        base_rates=base_rates,
        signal_contribs=signal_contribs,
    )
    try:
        response = httpx.post(webhook_url, json={"content": message}, timeout=10)
        response.raise_for_status()
    except Exception:
        log.exception("discord_eval_send_failed")


def format_stock_candidates_json(candidates: list[StockCandidate]) -> str:
    """Format candidates as JSON string for dry-run / debugging."""
    if not candidates:
        return "[]"
    payloads = _build_payloads(candidates)
    return json.dumps(payloads, indent=2, ensure_ascii=False)
