"""Label generation for ML training from backtest results.

Converts BacktestTrade outcomes into binary labels:
- take_profit hit → 1 (profitable)
- stop_loss hit → 0 (unprofitable)
- end_of_data → excluded by default (ambiguous)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rainier.backtest.engine import BacktestTrade


@dataclass(frozen=True, slots=True)
class LabelPolicy:
    """Controls how backtest outcomes are converted to training labels."""

    exclude_end_of_data: bool = True
    """If True, trades closed at end-of-data are excluded from training.
    If False, they're labeled by PnL sign (positive → 1, negative → 0)
    and marked with is_soft_label=True."""


class LabelGenerator:
    """Generates ML training labels from backtest trade results."""

    def __init__(self, policy: LabelPolicy | None = None):
        self.policy = policy or LabelPolicy()

    def generate(self, trades: list[BacktestTrade]) -> pd.DataFrame:
        """Convert backtest trades into a labeled DataFrame.

        Returns a DataFrame with columns:
        - entry_bar: int — bar index where the trade was entered
        - exit_bar: int — bar index where the trade exited
        - direction: str — "LONG" or "SHORT"
        - entry_price: float
        - exit_price: float
        - pnl: float
        - exit_reason: str
        - label: int — 1 (profitable) or 0 (unprofitable)
        - is_soft_label: bool — True if label is derived from PnL sign
          (end-of-data trades when not excluded)
        """
        rows: list[dict] = []

        for trade in trades:
            if trade.exit_reason == "end_of_data" and self.policy.exclude_end_of_data:
                continue

            if trade.exit_reason == "take_profit":
                label = 1
                is_soft = False
            elif trade.exit_reason == "stop_loss":
                label = 0
                is_soft = False
            elif trade.exit_reason == "end_of_data":
                # Included (policy says don't exclude): label by PnL sign
                label = 1 if trade.pnl > 0 else 0
                is_soft = True
            else:
                # Unknown exit reason — skip
                continue

            rows.append({
                "entry_bar": trade.entry_bar,
                "exit_bar": trade.exit_bar,
                "direction": trade.signal.direction.value,
                "entry_price": trade.signal.entry_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "exit_reason": trade.exit_reason,
                "label": label,
                "is_soft_label": is_soft,
            })

        return pd.DataFrame(rows)

    def summary(self, labels_df: pd.DataFrame) -> dict:
        """Return label distribution summary for validation."""
        if labels_df.empty:
            return {"total": 0, "positive": 0, "negative": 0, "soft": 0}

        return {
            "total": len(labels_df),
            "positive": int((labels_df["label"] == 1).sum()),
            "negative": int((labels_df["label"] == 0).sum()),
            "soft": int(labels_df["is_soft_label"].sum()),
            "positive_rate": float((labels_df["label"] == 1).mean()),
        }
