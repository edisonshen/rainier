"""ResonanceScorer — power-weighted panel agreement → score ∈ [0,1].

The resonance score is the **power-weighted fraction of members risk-on**:

    r[t] = Σ wᵢ · signalᵢ[t]  /  Σ wᵢ        ∈ [0, 1]

Weighting modes (design §5.3, A/B'd in §6.3):
  * ``equal``            — every member weight 1 (the simplest baseline).
  * ``category_balanced`` — the DEFAULT prior. The 6 categories are weighted
    equally; each member's weight = 1 / (#members in its category). So the many
    trend members don't make the score secretly mean "trend ×4". Categories
    with zero present members (e.g. breadth absent) are dropped and the
    remaining categories keep equal total mass.
  * ``custom``           — caller supplies per-member power weights (a
    swept/tuned config; capped per §6.4).

Only members actually present in the score (``available_members`` at compute
time) participate — an absent breadth/SPY member changes the denominator but
not the [0,1] range.

The scorer does NOT shift; it scores ``close[t]`` decision series.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from rainier.signals.panel import SignalPanel


def category_balanced_weights(
    panel: SignalPanel, member_names: list[str]
) -> dict[str, float]:
    """Per-member weights so each *present* category gets equal total mass.

    weight(member) = 1 / (#present members in member's category). The category
    totals are then equal (each sums to 1 across its members), so the score is
    a uniform average over present categories.
    """
    by_cat: dict[str, list[str]] = defaultdict(list)
    for name in member_names:
        by_cat[panel.category_of(name)].append(name)
    weights: dict[str, float] = {}
    for _cat, names in by_cat.items():
        share = 1.0 / len(names)
        for name in names:
            weights[name] = share
    return weights


class ResonanceScorer:
    """Combine panel member series into a daily resonance score ∈ [0,1]."""

    def __init__(
        self,
        panel: SignalPanel,
        mode: str = "category_balanced",
        custom_weights: dict[str, float] | None = None,
    ) -> None:
        if mode not in ("equal", "category_balanced", "custom"):
            raise ValueError(f"unknown weighting mode: {mode!r}")
        if mode == "custom" and not custom_weights:
            raise ValueError("mode='custom' requires custom_weights")
        self.panel = panel
        self.mode = mode
        self.custom_weights = custom_weights or {}

    def resolve_weights(self, member_names: list[str]) -> dict[str, float]:
        """Per-member power weights for the members actually present."""
        if self.mode == "equal":
            return {name: 1.0 for name in member_names}
        if self.mode == "category_balanced":
            return category_balanced_weights(self.panel, member_names)
        # custom: keep only weights for present members; missing → 0 (excluded).
        w = {name: float(self.custom_weights.get(name, 0.0)) for name in member_names}
        if all(v <= 0 for v in w.values()):
            raise ValueError("custom_weights resolved to all-zero over present members")
        return w

    def score(self, panel_series: dict[str, np.ndarray]) -> np.ndarray:
        """Power-weighted fraction risk-on per day, ∈ [0,1].

        ``panel_series`` is the ``SignalPanel.compute`` output (name → {0,1}).
        """
        names = list(panel_series.keys())
        if not names:
            raise ValueError("empty panel — nothing to score")
        weights = self.resolve_weights(names)
        n = len(next(iter(panel_series.values())))
        num = np.zeros(n, dtype=float)
        wsum = 0.0
        for name in names:
            w = weights.get(name, 0.0)
            if w <= 0:
                continue
            num += w * panel_series[name]
            wsum += w
        if wsum <= 0:
            raise ValueError("total weight is zero")
        return num / wsum
