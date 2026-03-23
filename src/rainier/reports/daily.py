"""Daily review + next-day outlook report."""

from __future__ import annotations

from datetime import datetime

from rainier.core.types import AnalysisResult, Direction, Signal, SRType


def generate_daily_report(
    results: dict[str, AnalysisResult],
    signals: dict[str, list[Signal]],
    date: datetime | None = None,
) -> str:
    """Generate a daily report with today's review and tomorrow's outlook.

    Args:
        results: symbol → AnalysisResult (from daily/4H analysis)
        signals: symbol → list of signals generated today
        date: report date (defaults to today)
    """
    if date is None:
        date = datetime.now()

    lines: list[str] = []
    lines.append(f"# Daily Report — {date.strftime('%Y-%m-%d')}")
    lines.append("")

    # === Today's Review ===
    lines.append("## Today's Review")
    lines.append("")

    total_signals = sum(len(sigs) for sigs in signals.values())
    lines.append(f"**Signals generated:** {total_signals}")
    lines.append("")

    for symbol, sigs in signals.items():
        if not sigs:
            continue
        lines.append(f"### {symbol}")
        for sig in sigs:
            side = "BUY" if sig.direction == Direction.LONG else "SELL"
            lines.append(
                f"- {side} @ {sig.entry_price:.2f} | "
                f"SL {sig.stop_loss:.2f} | TP {sig.take_profit:.2f} | "
                f"R:R {sig.rr_ratio:.1f} | Conf {sig.confidence:.0%} | "
                f"Status: {sig.status.value}"
            )
        lines.append("")

    # === Tomorrow's Outlook ===
    lines.append("## Tomorrow's Outlook")
    lines.append("")

    for symbol, result in results.items():
        lines.append(f"### {symbol}")

        # Bias
        if result.bias:
            bias_str = "Bullish" if result.bias == Direction.LONG else "Bearish"
        else:
            bias_str = "Neutral / No clear bias"
        lines.append(f"**Bias:** {bias_str}")

        # Key S/R levels
        h_levels = [l for l in result.sr_levels if l.sr_type == SRType.HORIZONTAL]
        top_levels = sorted(h_levels, key=lambda l: l.strength, reverse=True)[:5]

        if top_levels:
            lines.append("**Key levels to watch:**")
            for level in top_levels:
                role = "Support" if level.role.value == "support" else "Resistance"
                lines.append(
                    f"- {role} @ {level.price:.2f} "
                    f"(strength {level.strength:.0%}, {level.touches} touches)"
                )

        # Inside bars (range compression building)
        if result.inside_bars:
            recent_ib = [ib for ib in result.inside_bars if ib.index >= len(result.candles) - 10]
            if recent_ib:
                lines.append(
                    f"**Range compression:** {len(recent_ib)} inside bar(s) in last 10 candles — "
                    "watch for breakout"
                )

        lines.append("")

    return "\n".join(lines)
