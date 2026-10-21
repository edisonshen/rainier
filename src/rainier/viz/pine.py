"""Export S/R zones as a TradingView Pine v5 indicator (paste into the Pine Editor)."""

from __future__ import annotations

from datetime import datetime

from rainier.core.types import SRLevel, SRRole


def zones_to_pine(symbol: str, levels: list[SRLevel], asof: datetime) -> str:
    lines = [
        "//@version=5",
        f'indicator("Rainier S/R {symbol} {asof:%Y-%m-%d %H:%M}", overlay=true, max_lines_count=100,'
        " max_boxes_count=100, max_labels_count=100)",
        "if barstate.islastconfirmedhistory",
    ]
    for lv in sorted(levels, key=lambda l: -l.price):
        color = "color.red" if lv.role == SRRole.RESISTANCE else "color.teal"
        tfs = "+".join(t.value for t in lv.timeframes) or (lv.source_tf.value if lv.source_tf else "")
        label = f"{tfs} {'R' if lv.role == SRRole.RESISTANCE else 'S'} {lv.price:.2f}"
        lo, hi = lv.zone_low, lv.zone_high
        if lo is not None and hi is not None and hi > lo:
            lines.append(
                f"    box.new(bar_index - 300, {hi:.2f}, bar_index + 20, {lo:.2f},"
                f" border_color=color.new({color}, 100), bgcolor=color.new({color}, 85))")
        width = 1 + min(len(set(lv.timeframes)), 3)
        lines.append(
            f"    line.new(bar_index - 300, {lv.price:.2f}, bar_index + 20, {lv.price:.2f},"
            f" color={color}, width={width})")
        lines.append(
            f'    label.new(bar_index + 20, {lv.price:.2f}, "{label}", style=label.style_none,'
            f" textcolor={color}, size=size.small)")
    return "\n".join(lines) + "\n"
