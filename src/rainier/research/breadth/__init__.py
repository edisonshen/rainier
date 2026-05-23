"""Breadth / thematic-ranks support package.

Phase A ships the data substrate: universe loader, ticker / sector
registries, universe-state log writer.

Phase B (this PR) adds Layer A point-in-time features + Layer B forward-
return labels in `ranks.py`, the read-side loaders in `loaders.py`, and
the ML/RL panel builders in `ml_builders.py`. The dashboard renderer
lives at `rainier.viz.thematic_dashboard`.

Design ref: docs/DESIGN-thematic-ranks-dashboard.md
"""

from __future__ import annotations

from rainier.research.breadth import (
    loaders,
    ml_builders,
    ranks,
    registry,
    universe_loader,
)

__all__ = [
    "loaders",
    "ml_builders",
    "ranks",
    "registry",
    "universe_loader",
]
