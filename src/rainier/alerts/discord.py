"""Discord webhook notifications for trade signals."""

from __future__ import annotations

import httpx

from rainier.core.config import DiscordConfig
from rainier.core.types import Direction, Signal


def send_signal_alert(signal: Signal, config: DiscordConfig, chart_path: str | None = None):
    """Send a signal alert to Discord via webhook."""
    if not config.enabled or not config.webhook_url:
        return

    direction_emoji = "\U0001f7e2" if signal.direction == Direction.LONG else "\U0001f534"  # green/red circle
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
