"""Discord webhook notifications for trade signals and stock candidates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime
from typing import Any

import httpx
import structlog

from rainier.core.config import DiscordConfig
from rainier.core.types import Direction, Signal, StockCandidate

# structlog (NOT stdlib logging) so every Discord operation lands in the same
# structured stream as the scraper + pipeline (``data/qu-scrape.log``). The
# module previously used ``logging.getLogger`` whose records never reached that
# file, so a failed webhook POST logged ``discord_send_failed`` into the void
# while the pipeline still printed ``discord_sent=20`` — the channel stayed
# empty and the logs looked healthy. See ``send_stock_candidates``.
log = structlog.get_logger()


def _http_status(exc: Exception) -> int | None:
    """Extract the HTTP status code from an httpx error, else None.

    ``raise_for_status()`` raises ``HTTPStatusError`` (has ``.response``);
    connection/timeout errors have no status. Used so per-operation failure
    logs carry the actual code (404 deleted webhook, 401 rotated token, 429
    rate-limited) instead of a bare "it failed".
    """
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None) if resp is not None else None


@dataclass(frozen=True)
class DiscordSendResult:
    """Truthful per-operation outcome of a candidate-report send.

    ``len(candidates)`` is what we *tried* to report; these are what actually
    reached Discord. The pipeline logs ``discord_sent`` from ``fully_ok`` so a
    silent webhook failure no longer reads as success.
    """

    candidates_reported: int          # len(candidates) attempted
    candidate_payloads_ok: int        # webhook POSTs that returned 2xx
    candidate_payloads_failed: int    # webhook POSTs that errored
    thesis_ok: int                    # per-ticker thesis embeds delivered
    thesis_failed: int                # per-ticker thesis embeds that errored

    @property
    def fully_ok(self) -> bool:
        """True iff every candidate-report payload landed (no failures)."""
        return (
            self.candidate_payloads_failed == 0
            and self.candidate_payloads_ok > 0
        )

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


def _resolve_llm_webhook_url(config: DiscordConfig) -> str | None:
    """Get the webhook URL for per-ticker LLM thesis embeds.

    Routing precedence: ``llm_webhook_url`` (dedicated QU100-LLM channel) →
    ``stock_webhook_url`` (general QU100 channel) → ``webhook_url`` (catch-all)
    → ``None``. The dedicated LLM channel is opt-in; leaving it empty keeps
    legacy behavior where rich thesis embeds and the top-20 screener post
    share the stock channel.
    """
    return (
        config.llm_webhook_url
        or config.stock_webhook_url
        or config.webhook_url
        or None
    )


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


# ---------------------------------------------------------------------------
# PR5 — Discord embed redesign + chart PNG attach + dashboard deep-link
# ---------------------------------------------------------------------------
#
# The PR1/2 renderer emitted six wall-of-text paragraphs per ticker × 5 tickers
# per scan. Live feedback after the first run was the same: too dense to scan,
# decision-critical info (verdict, score, entry/SL/TP) buried in prose. PR5
# replaces it with a Discord *embed* — color-coded bar by verdict, headline
# `verdict · symbol · score`, chip line of decision-critical signals, monospace
# trade-levels block, top risks, single most-actionable watch item, optional
# short LLM-noticed observation, optional clickable link to the dashboard.
#
# Flow:
#                                      ChartImage row (BYTEA)
#                                              │
#         ┌──────────┐    payload_json        ▼ files[0]   ┌─────────┐
# thesis ▶│ format_  │──┐               ┌───────────────┐  │ Discord │
#         │  thesis_ │  ├──multipart────▶ {embed, file} ├─▶│ webhook │
# chart ─▶│  embed   │──┘               └───────────────┘  └─────────┘
#         └──────────┘
#
# One ticker = one webhook POST = one Discord message with embed + attached
# image. Webhooks don't support interactive components (buttons), so the
# dashboard deep-link is rendered as a plain URL on the embed title.

# Verdict → embed color bar. Matches the design review:
#   setup_long  → green   (clear long setup)
#   watch       → yellow  (on radar but not a buy)
#   no_setup    → gray    (no edge here)
_VERDICT_COLORS: dict[str, int] = {
    "setup_long": 0x2ECC71,
    "watch": 0xF1C40F,
    "no_setup": 0x95A5A6,
}

# Discord embed-field limits we respect (webhook spec).
_EMBED_TITLE_MAX = 256
_EMBED_DESC_MAX = 4096
_EMBED_FIELD_NAME_MAX = 256
_EMBED_FIELD_VALUE_MAX = 1024
_EMBED_FOOTER_MAX = 2048

# Chip-line cap. Kept well under _EMBED_DESC_MAX so we never have to ellipsize
# in the middle of a chip.
_CHIP_LINE_MAX = 200


def _truncate_at_word(text: str, max_chars: int) -> str:
    """Word-boundary safe truncation. Appends ``…`` when shortened.

    The PR1 renderer used naive ``text[:N-3] + '...'`` which produced
    mid-word cuts like ``"breaking thru exte..."`` (live feedback iter-1).
    PR5 backs up to the previous whitespace before the cap so the trailing
    ellipsis lands at a real word break.

    Returns the input unchanged when ``len(text) <= max_chars``. Returns an
    empty string for ``max_chars <= 0`` (defensive — callers should never
    pass a non-positive cap, but we don't want to raise from a renderer).
    """
    if max_chars <= 0:
        return ""
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    # Reserve one char for the ellipsis. We use the single-codepoint U+2026
    # so length-checks are predictable.
    cap = max_chars - 1
    if cap <= 0:
        return "…"
    cut = text[:cap]
    space = cut.rfind(" ")
    if space > 0:
        # Trim trailing whitespace so the ellipsis sits flush with the last word.
        cut = cut[:space].rstrip()
    return cut + "…"


def _format_pct(value: float | None, *, with_sign: bool = True) -> str:
    """Format a fractional value as a percentage string.

    ``0.087`` -> ``"+8.7%"`` when ``with_sign`` else ``"8.7%"``.
    Returns ``"-"`` for ``None``.
    """
    if value is None:
        return "-"
    if with_sign:
        return f"{value * 100:+.1f}%"
    return f"{value * 100:.1f}%"


def _build_chip_line(
    candidate: StockCandidate,
    thesis: dict,
    pattern_type: str | None,
) -> str:
    """Compose the embed description — a `·`-separated chip line.

    Picks the most decision-critical 3–5 signals from the candidate +
    thesis: pattern type + confidence, rank trajectory delta, sector
    momentum delta, volume confirmation. Caps the total line at
    ``_CHIP_LINE_MAX`` chars (200) so the user can read the whole line at
    a glance — anything longer defeats the "scannable in 5s" goal.
    """
    chips: list[str] = []

    # 1) Pattern type (e.g. w_bottom) + confidence when present.
    if pattern_type:
        label = pattern_type
        conf = candidate.pattern_confidence
        if conf is not None:
            chips.append(f"{label} {conf:.2f}")
        else:
            chips.append(label)

    # 2) Rank trajectory if the thesis carries it under signals payload.
    rank_chip = _rank_chip(candidate, thesis)
    if rank_chip:
        chips.append(rank_chip)

    # 3) Sector momentum delta when the signal payload carries it.
    sector_chip = _sector_chip(thesis)
    if sector_chip:
        chips.append(sector_chip)

    # 4) Volume confirmation flag.
    if candidate.volume_confirmed:
        chips.append("vol✓")

    line = " · ".join(c for c in chips if c)
    return _truncate_at_word(line, _CHIP_LINE_MAX)


def _rank_chip(candidate: StockCandidate, thesis: dict) -> str | None:
    """Build the rank-trajectory chip from the rank_trajectory signal payload.

    Falls back to ``candidate.rank_change`` (single-day delta) when the
    signal didn't run. Format: ``"rank #85→#14 in 4d ↗"`` or ``"rank +3"``.
    """
    signals = thesis.get("signals") if isinstance(thesis, dict) else None
    if isinstance(signals, dict):
        rt = signals.get("rank_trajectory")
        if isinstance(rt, dict):
            points = rt.get("points") or []
            delta = rt.get("delta_10d")
            trend = rt.get("trend")
            if isinstance(points, list) and len(points) >= 2:
                try:
                    first = points[0]
                    last = points[-1]
                    # Each point may be (date, rank) or {"date":..,"rank":..}.
                    first_rank = first[1] if isinstance(first, (list, tuple)) else first.get("rank")
                    last_rank = last[1] if isinstance(last, (list, tuple)) else last.get("rank")
                    arrow = (
                        "↗" if trend == "rising"
                        else ("↘" if trend == "falling" else "→")
                    )
                    days = len(points) - 1
                    return f"rank #{int(first_rank)}→#{int(last_rank)} in {days}d {arrow}"
                except (TypeError, ValueError, IndexError):
                    pass
            if isinstance(delta, (int, float)):
                return f"rank delta {int(delta):+d}"
    if candidate.rank_change:
        return f"rank {candidate.rank_change:+d}"
    return None


def _sector_chip(thesis: dict) -> str | None:
    """Sector momentum chip from the sector_momentum signal payload."""
    signals = thesis.get("signals") if isinstance(thesis, dict) else None
    if not isinstance(signals, dict):
        return None
    sm = signals.get("sector_momentum")
    if not isinstance(sm, dict):
        return None
    delta = sm.get("delta")
    if not isinstance(delta, (int, float)):
        return None
    arrow = "↗" if delta > 0 else ("↘" if delta < 0 else "→")
    return f"sector {delta:+.2f} {arrow}"


def _why_bullets(thesis: dict) -> list[str]:
    """Top 3-4 short bullets pulled from the thesis prose.

    Strategy: split paragraph_radar + paragraph_evidence on sentence
    boundaries, drop blanks, keep the first 4, cap each at 60 chars
    (word-boundary safe).

    PR5 review iter-1: LLM-generated paragraphs flow into Discord text;
    scrub ``@everyone`` / ``@here`` / backticks before emitting so an
    adversarial completion can't trigger a mass-mention or break the
    surrounding embed formatting (mirrors the same defense already in
    place on the eval/research renderers).
    """
    chunks: list[str] = []
    for key in ("paragraph_radar", "paragraph_evidence"):
        text = _scrub_discord_text((thesis.get(key) or "").strip())
        if not text:
            continue
        # Split on sentence-ish boundaries; we don't need linguistic
        # accuracy, just visual chunks.
        for piece in re.split(r"(?<=[.!?;])\s+", text):
            piece = piece.strip().rstrip(".;")
            if piece:
                chunks.append(piece)
    bullets = [_truncate_at_word(c, 60) for c in chunks[:4] if c]
    return bullets


def _levels_block(candidate: StockCandidate) -> str:
    """Monospace block of Entry / Stop / Target / Now with %deltas + R/R.

    Wrapped in triple backticks so Discord renders the field in a fixed
    monospace font. Right-aligned dollar amounts so the eye scans the
    decimal column.
    """
    lines: list[str] = []
    entry = candidate.entry_price
    stop = candidate.stop_loss
    target = candidate.target_price
    now = candidate.current_price
    rr = candidate.rr_ratio

    def _row(label: str, price: float | None, suffix: str = "") -> str:
        if price is None:
            return f"{label:<7} -"
        body = f"${price:>10.2f}"
        if suffix:
            body = f"{body}  {suffix}"
        return f"{label:<7} {body}"

    lines.append(_row("Entry", entry))
    if stop is not None and entry is not None:
        pct = (stop - entry) / entry if entry else None
        lines.append(_row("Stop", stop, f"({_format_pct(pct)})"))
    else:
        lines.append(_row("Stop", stop))
    if target is not None and entry is not None:
        pct = (target - entry) / entry if entry else None
        suffix = f"({_format_pct(pct)}"
        if rr is not None:
            suffix += f", {rr:.1f}R"
        suffix += ")"
        lines.append(_row("Target", target, suffix))
    else:
        lines.append(_row("Target", target))
    if now is not None:
        if entry is not None:
            pct = (now - entry) / entry if entry else None
            lines.append(_row("Now", now, f"({_format_pct(pct)})"))
        else:
            lines.append(_row("Now", now))

    body = "\n".join(lines)
    return f"```\n{body}\n```"


def _llm_noticed(thesis: dict) -> str | None:
    """Compose the optional 'LLM noticed' field.

    The schema's ``patterns_in_chart_not_in_indicators`` is either a list
    (max 5) of named patterns OR the literal string ``"none"``. We emit
    the field only when there's signal — joining the list into one short
    sentence (or showing the first item if the list is long). PR5 review
    iter-1: scrubs LLM-controlled text via ``_scrub_discord_text``.
    """
    raw = thesis.get("patterns_in_chart_not_in_indicators")
    if raw == "none" or raw is None:
        return None
    if isinstance(raw, list):
        if not raw:
            return None
        items = [_scrub_discord_text(str(x).strip()) for x in raw if str(x).strip()]
        items = [i for i in items if i]
        if not items:
            return None
        text = "; ".join(items)
        return _truncate_at_word(text, 200)
    if isinstance(raw, str):
        text = _scrub_discord_text(raw.strip())
        if not text:
            return None
        return _truncate_at_word(text, 200)
    return None


def _risks_lines(thesis: dict) -> list[str]:
    """Top-3 risks, each <80 chars, word-boundary safe.

    PR5 review iter-1: scrubs LLM-controlled text so a malicious
    ``risks=["@everyone bad risk"]`` can't trigger a mass mention.
    """
    risks = thesis.get("risks") or []
    if not isinstance(risks, list):
        return []
    out: list[str] = []
    for r in risks[:3]:
        text = _scrub_discord_text(str(r).strip())
        if not text:
            continue
        out.append(_truncate_at_word(text, 80))
    return out


def _watch_line(thesis: dict) -> str | None:
    """Single most-actionable watch item, <120 chars, word-boundary safe.

    PR5 review iter-1: scrubs LLM-controlled text.
    """
    items = thesis.get("watch_items") or []
    if not isinstance(items, list) or not items:
        return None
    first = _scrub_discord_text(str(items[0]).strip())
    if not first:
        return None
    return _truncate_at_word(first, 120)


def format_thesis_embed(
    thesis: dict,
    candidate: StockCandidate,
    *,
    dashboard_base_url: str | None = None,
    thesis_id: int | None = None,
    chart_filename: str | None = None,
) -> dict[str, Any]:
    """Build the Discord embed dict for a single thesis.

    The returned dict is ready to drop into the ``embeds`` list of a
    Discord webhook payload. When ``chart_filename`` is set, the embed
    references it via ``attachment://<filename>`` so the multipart
    file part renders inline at the bottom of the embed.

    When ``dashboard_base_url`` and ``thesis_id`` are both set the embed
    title becomes a clickable link to ``<base>?thesis_id=<id>``. Either
    being None (the default) renders no link.
    """
    verdict = str(thesis.get("verdict") or "?")
    confidence = thesis.get("llm_confidence")
    setup_q = thesis.get("setup_quality")

    color = _VERDICT_COLORS.get(verdict, _VERDICT_COLORS["no_setup"])
    title = f"{verdict} · {candidate.symbol}"
    if confidence is not None:
        title += f" · {confidence}/10"
    title = _truncate_at_word(title, _EMBED_TITLE_MAX)

    description = _build_chip_line(candidate, thesis, candidate.pattern_type)

    embed: dict[str, Any] = {
        "color": color,
        "title": title,
    }
    if description:
        embed["description"] = description

    # Deep-link on the title.
    if dashboard_base_url and thesis_id is not None:
        sep = "&" if "?" in dashboard_base_url else "?"
        embed["url"] = f"{dashboard_base_url}{sep}thesis_id={int(thesis_id)}"

    # Inline image attachment by filename.
    if chart_filename:
        embed["image"] = {"url": f"attachment://{chart_filename}"}

    fields: list[dict[str, Any]] = []

    why_bullets = _why_bullets(thesis)
    if why_bullets:
        why_value = "\n".join(f"• {b}" for b in why_bullets)
        fields.append({
            "name": "WHY",
            "value": _truncate_at_word(why_value, _EMBED_FIELD_VALUE_MAX),
            "inline": False,
        })

    levels = _levels_block(candidate)
    fields.append({
        "name": "LEVELS",
        "value": levels[:_EMBED_FIELD_VALUE_MAX],
        "inline": False,
    })

    risks = _risks_lines(thesis)
    if risks:
        risks_value = "\n".join(f"• {r}" for r in risks)
        fields.append({
            "name": "RISKS",
            "value": _truncate_at_word(risks_value, _EMBED_FIELD_VALUE_MAX),
            "inline": False,
        })

    watch = _watch_line(thesis)
    if watch:
        fields.append({
            "name": "WATCH",
            "value": _truncate_at_word(watch, _EMBED_FIELD_VALUE_MAX),
            "inline": False,
        })

    noticed = _llm_noticed(thesis)
    if noticed:
        fields.append({
            "name": "LLM noticed",
            "value": _truncate_at_word(noticed, _EMBED_FIELD_VALUE_MAX),
            "inline": False,
        })

    embed["fields"] = fields

    # Footer: setup_quality + signals_used (concise, small font in Discord).
    signals_used = thesis.get("signals_used") or []
    if isinstance(signals_used, list):
        sigs = ", ".join(str(s) for s in signals_used[:6])
    else:
        sigs = ""
    footer_parts: list[str] = []
    if setup_q is not None:
        footer_parts.append(f"setup_quality {setup_q}/10")
    if sigs:
        footer_parts.append(f"signals: {sigs}")
    if footer_parts:
        embed["footer"] = {
            "text": _truncate_at_word(" · ".join(footer_parts), _EMBED_FOOTER_MAX)
        }

    return embed


def _post_thesis_embed(
    *,
    webhook_url: str,
    embed: dict[str, Any],
    chart_bytes: bytes | None,
    chart_filename: str,
) -> None:
    """POST one Discord webhook message: embed (+ optional file attachment).

    When ``chart_bytes`` is set we POST as multipart/form-data per Discord's
    webhook spec — the JSON envelope rides in ``payload_json`` and the file
    rides in ``files[0]``. The embed's ``image.url`` references
    ``attachment://<chart_filename>`` so Discord renders the file inline at
    the bottom of the embed.

    When ``chart_bytes`` is None we drop the embed's ``image`` field (if
    set) and POST plain JSON.
    """
    payload_obj = {"embeds": [embed]}

    if chart_bytes:
        files = {
            "files[0]": (chart_filename, chart_bytes, "image/png"),
        }
        data = {"payload_json": json.dumps(payload_obj)}
        response = httpx.post(
            webhook_url, files=files, data=data, timeout=10
        )
    else:
        # Strip stale image reference if there's no actual file attached.
        embed_no_image = dict(embed)
        embed_no_image.pop("image", None)
        payload_obj = {"embeds": [embed_no_image]}
        response = httpx.post(webhook_url, json=payload_obj, timeout=10)
    response.raise_for_status()


def _load_chart_bytes_for_thesis(thesis: dict) -> bytes | None:
    """Read the first ChartImage row referenced by the thesis (PR5).

    PR5 promise: Discord pulls the EXACT bytes that went to the LLM from
    the DB — never re-renders kaleido. Returns None when no chart is
    associated; the caller posts the embed without an image attachment.
    """
    raw_id = thesis.get("_thesis_id")
    if raw_id is None:
        return None
    try:
        # Lazy import: keeps the alerts module from pulling SQLAlchemy
        # in tests that only exercise pure formatting.
        from rainier.dashboard.data import load_thesis_chart
        return load_thesis_chart(int(raw_id))
    except Exception:
        log.exception("discord_load_chart_failed", thesis_id=raw_id)
        return None


def _resolve_thesis_id(thesis: dict) -> int | None:
    """Pull the LLMAnalysisRecord id stamped on the thesis dict.

    The scheduler/CLI persists the row id via ``out[symbol]["_thesis_id"]
    = record_id`` so the renderer can build the dashboard deep-link.
    Returns None when the field is missing (older callsites).
    """
    raw = thesis.get("_thesis_id") if isinstance(thesis, dict) else None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return None


def send_stock_candidates(
    candidates: list[StockCandidate],
    config: DiscordConfig,
    theses: dict[str, dict] | None = None,
    *,
    dashboard_base_url: str | None = None,
    session: str | None = None,
) -> DiscordSendResult:
    """Send QU100 stock candidate alerts to Discord.

    Args:
        candidates: Top N screened stock candidates, sorted by pattern match quality.
        config: Discord configuration with webhook URL and enabled flag.
        theses: Optional ``{symbol: thesis_dict}`` map. When provided, the standard
            top-20 summary is followed by a Discord *embed* per top-5 ticker
            with verdict color bar, chip line, monospace levels, attached
            chart PNG, and (optionally) a clickable dashboard link. Default
            ``None`` preserves existing top-20-only behavior — the regression
            contract.
        dashboard_base_url: Public URL of the Streamlit dashboard. When set
            and the thesis dict carries ``_thesis_id``, the embed title
            links to ``<base>?thesis_id=<id>``. None disables the link.
        session: Scrape session label ("morning" | "midday" | "afternoon" |
            "close"). When set, the summary embed title gets a `—
            Morning/Midday/Afternoon/Close` suffix so operators can tell
            multiple same-day scans apart in the stock channel. ``None``
            preserves the legacy session-less title.
    """
    reported = len(candidates)
    if not candidates:
        return DiscordSendResult(0, 0, 0, 0, 0)
    if not config.enabled:
        log.info("discord_alerts_disabled", candidates=reported, session=session)
        return DiscordSendResult(reported, 0, 0, 0, 0)

    webhook_url = _resolve_webhook_url(config)
    if not webhook_url:
        log.warning(
            "discord_no_webhook_url", channel="stock",
            candidates=reported, session=session,
        )
        return DiscordSendResult(reported, 0, 0, 0, 0)

    payloads = _build_payloads(candidates, session=session)
    log.info(
        "discord_candidates_send_starting",
        channel="stock", session=session,
        candidates=reported, payloads=len(payloads),
    )

    payloads_ok = 0
    payloads_failed = 0
    for idx, payload in enumerate(payloads):
        n_embeds = len(payload.get("embeds", []))
        try:
            response = httpx.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            payloads_ok += 1
            log.info(
                "discord_payload_ok",
                idx=idx, status=response.status_code, embeds=n_embeds,
            )
        except Exception as exc:
            payloads_failed += 1
            log.error(
                "discord_payload_failed",
                idx=idx, status=_http_status(exc),
                error=str(exc)[:200], embeds=n_embeds,
            )
    log.info(
        "discord_candidates_send_done",
        channel="stock", session=session,
        payloads_ok=payloads_ok, payloads_failed=payloads_failed,
    )

    if not theses:
        return DiscordSendResult(reported, payloads_ok, payloads_failed, 0, 0)

    # Per-ticker rich thesis embeds route to the LLM-dedicated channel when
    # configured (DISCORD_LLM_WEBHOOK_URL), otherwise fall back to the stock
    # channel. Note the regular top-20 screener payload above always stays on
    # the stock channel (`webhook_url`) regardless of llm_webhook_url state.
    llm_webhook_url = _resolve_llm_webhook_url(config)
    if not llm_webhook_url:
        log.warning("discord_no_webhook_url_llm")
        return DiscordSendResult(reported, payloads_ok, payloads_failed, 0, 0)

    # Per-ticker rich thesis embeds — top 5 only, in candidate order.
    thesis_ok = 0
    thesis_failed = 0
    for candidate in candidates[:5]:
        thesis = theses.get(candidate.symbol)
        if not thesis:
            continue
        try:
            thesis_id = _resolve_thesis_id(thesis)
            chart_bytes = _load_chart_bytes_for_thesis(thesis)
            chart_filename = f"chart_{candidate.symbol}.png"
            embed = format_thesis_embed(
                thesis,
                candidate,
                dashboard_base_url=dashboard_base_url,
                thesis_id=thesis_id,
                chart_filename=chart_filename if chart_bytes else None,
            )
            _post_thesis_embed(
                webhook_url=llm_webhook_url,
                embed=embed,
                chart_bytes=chart_bytes,
                chart_filename=chart_filename,
            )
            thesis_ok += 1
            log.info("discord_thesis_ok", symbol=candidate.symbol)
        except Exception as exc:
            thesis_failed += 1
            log.error(
                "discord_thesis_send_failed",
                symbol=candidate.symbol, status=_http_status(exc),
                error=str(exc)[:200],
            )
    log.info(
        "discord_thesis_send_done",
        thesis_ok=thesis_ok, thesis_failed=thesis_failed,
    )
    return DiscordSendResult(
        reported, payloads_ok, payloads_failed, thesis_ok, thesis_failed,
    )


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
    """Render the Discord eval-report body. <=1800 chars per message.

    PR3 carry-over [P2]: when ``yesterday_rows`` mixes ``afternoon`` and
    ``close`` sessions, render a separate header per session so the operator
    can distinguish "afternoon scan picks" from "close scan picks". The old
    renderer used the first row's session as a single global heading and
    interleaved both sets of picks underneath, hiding which scan produced
    each pick.
    """
    lines: list[str] = []
    lines.append(f"**Eval report — {eval_date.isoformat()}**")
    if yesterday_rows:
        # Group rows by session_name preserving session insertion order. Use a
        # plain list-of-(key, list) so the rendering stays deterministic and
        # we don't need an OrderedDict import. Rows of unknown session collapse
        # under the "(unknown)" bucket.
        by_session: list[tuple[str, list[dict]]] = []
        index: dict[str, int] = {}
        for row in yesterday_rows:
            sess = row.get("session_name") or "(unknown)"
            idx = index.get(sess)
            if idx is None:
                index[sess] = len(by_session)
                by_session.append((sess, [row]))
            else:
                by_session[idx][1].append(row)

        for sess, rows in by_session:
            scan_date = rows[0].get("scan_date") or "(prior scan)"
            lines.append(f"Yesterday's {sess} scan ({scan_date}):")
            for row in rows:
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
        log.info("discord_eval_send_ok", status=response.status_code)
    except Exception as exc:
        log.error(
            "discord_eval_send_failed",
            status=_http_status(exc), error=str(exc)[:200],
        )


# ---------------------------------------------------------------------------
# Weekly research report (PR3) — auto-research insight summary
# ---------------------------------------------------------------------------


_INSIGHT_SEVERITY_TAGS: dict[str, str] = {
    "info": "[info]",
    "warn": "[WARN]",
    "critical": "[CRIT]",
}


def _scrub_discord_text(text: str) -> str:
    """Strip Discord-meaningful characters from LLM/user text rendered into
    a webhook message.

    `subject` (e.g. `new_pattern_discovered`) and `rationale` text can include
    LLM-generated strings — an adversarial LLM could embed `@everyone`,
    backticks, or markdown that breaks the surrounding code block.

    We replace `@` with a fullwidth variant (still readable, no mention),
    drop backticks, and collapse newlines to spaces. Cheap, no escape
    arms-race.
    """
    if not text:
        return ""
    return (
        text.replace("@", "＠")  # FULLWIDTH COMMERCIAL AT
        .replace("`", "'")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _format_research_message(
    *,
    eval_date: date_cls,
    insights: list,           # [ResearchInsight]
    severity_filter: tuple[str, ...] = ("warn", "critical"),
) -> str:
    """Render the weekly research report. <=1800 chars per message.

    Lists pending `warn`/`critical` insights with the structured action and
    rationale. `info` severities are excluded by default (they're suggestions,
    not alerts) but can be opted in via the filter.

    Each line includes the insight id so the operator can quickly run
    `rainier thesis research insights accept ID` from the CLI.
    """
    lines: list[str] = []
    lines.append(f"**Research report — {eval_date.isoformat()}**")
    if not insights:
        lines.append("No new pending warn/critical insights this week.")
        body = "\n".join(lines)
        return _truncate(body, 1800)

    bucket: list = [
        ins for ins in insights
        if getattr(ins, "severity", None) in severity_filter
        and getattr(ins, "status", "pending") == "pending"
    ]
    if not bucket:
        lines.append(
            "No new pending warn/critical insights this week."
        )
        body = "\n".join(lines)
        return _truncate(body, 1800)

    # Sort: critical first, then warn; within each, by recurrence_count desc
    # so the loudest signals show first.
    severity_order = {"critical": 0, "warn": 1, "info": 2}
    bucket.sort(
        key=lambda i: (
            severity_order.get(i.severity, 99),
            -(getattr(i, "recurrence_count", 1) or 1),
        )
    )

    lines.append(f"{len(bucket)} pending insight(s) this week:")
    for ins in bucket:
        tag = _INSIGHT_SEVERITY_TAGS.get(ins.severity, ins.severity)
        kind = ins.kind
        # Review iter-1 [P2]: scrub LLM-generated text (subject + rationale)
        # before piping to Discord so an adversarial pattern name like
        # "@everyone bad pattern" can't trigger a server-wide mention.
        subj = _scrub_discord_text((ins.subject or "")[:30])
        recur = getattr(ins, "recurrence_count", 1) or 1
        action_kind = "noop"
        if isinstance(ins.action, dict):
            action_kind = str(ins.action.get("kind", "?"))
        rationale = _scrub_discord_text((ins.rationale or "").strip())
        if len(rationale) > 200:
            rationale = rationale[:197] + "..."
        recur_part = f" (x{recur})" if recur > 1 else ""
        lines.append(
            f"  {tag} #{ins.id} {kind} <{subj}>{recur_part} -> action={action_kind}"
        )
        lines.append(f"      {rationale}")

    lines.append("")
    lines.append("Review with `rainier thesis research insights list`")
    body = "\n".join(lines)
    return _truncate(body, 1800)


def send_research_report(
    *,
    eval_date: date_cls,
    insights: list,
    config: DiscordConfig,
) -> None:
    """Post the weekly research report to Discord (stock channel, then webhook)."""
    if not config.enabled:
        log.debug("discord_alerts_disabled")
        return
    webhook_url = _resolve_webhook_url(config)
    if not webhook_url:
        log.warning("discord_no_webhook_url_research")
        return

    message = _format_research_message(
        eval_date=eval_date, insights=insights,
    )
    try:
        response = httpx.post(webhook_url, json={"content": message}, timeout=10)
        response.raise_for_status()
        log.info("discord_research_send_ok", status=response.status_code)
    except Exception as exc:
        log.error(
            "discord_research_send_failed",
            status=_http_status(exc), error=str(exc)[:200],
        )


def format_stock_candidates_json(candidates: list[StockCandidate]) -> str:
    """Format candidates as JSON string for dry-run / debugging."""
    if not candidates:
        return "[]"
    payloads = _build_payloads(candidates)
    return json.dumps(payloads, indent=2, ensure_ascii=False)
