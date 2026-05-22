"""Trade simulator — STUB. Full implementation lands in Slice 1.

Slice 1 ships:
  - entry-TTL gating (limit doesn't fill within N bars → unfilled_ttl)
  - pessimistic fill (cap at the worst observed slippage in the entry bar)
  - time-cap exit (force-close at max_hold_days; outcome=time_cap_exit)

Slice 0 ships the module + a module-level marker so the package import
graph is honest about what's coming.
"""

from __future__ import annotations

SLICE_THIS_LANDS_IN: int = 1
