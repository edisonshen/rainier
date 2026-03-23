"""Export signals to TraderSync-compatible CSV format."""

from __future__ import annotations

import csv
from pathlib import Path

from rainier.core.types import Direction, Signal


def export_tradersync_csv(signals: list[Signal], output_path: Path) -> Path:
    """Write signals to a CSV file compatible with TraderSync import.

    TraderSync expected columns:
    Date, Symbol, Side, Quantity, Entry Price, Exit Price, Commission, Notes
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Date",
            "Symbol",
            "Side",
            "Quantity",
            "Entry Price",
            "Exit Price",
            "Commission",
            "Notes",
        ])

        for signal in signals:
            side = "BUY" if signal.direction == Direction.LONG else "SELL"
            notes = (
                f"Confidence: {signal.confidence:.2f} | "
                f"R:R: {signal.rr_ratio:.1f} | "
                f"SL: {signal.stop_loss:.2f} | "
                f"TP: {signal.take_profit:.2f}"
            )
            writer.writerow([
                signal.timestamp.strftime("%Y-%m-%d %H:%M"),
                signal.symbol,
                side,
                1,  # default quantity
                f"{signal.entry_price:.2f}",
                "",  # exit price filled later
                "",  # commission
                notes,
            ])

    return output_path
