"""Breadth / thematic-ranks support package.

Phase A (this PR) ships the data substrate: universe loader, ticker /
sector registries, universe-state log writer.

Phase B (`thematic-ranks-dashboard-d245`) adds `ranks.py`, `loaders.py`,
`ml_builders.py`, and the dashboard renderer on top.

Design ref: docs/DESIGN-thematic-ranks-dashboard.md
"""

from __future__ import annotations

from rainier.research.breadth import registry, universe_loader

__all__ = ["registry", "universe_loader"]
