"""CHART_RENDER_VERSION — SHA pinning the chart-render code path.

Bumping the SHA invalidates every cached LLM output that depended on a
previous chart render. This is intentional: a render-code change means the
LLM saw a different image, so the cache must miss.

Per docs/DESIGN-qu100-llm-backtest.md [D-025] and
docs/TASK-PLAN-chart-export-determinism-3c8e.md §5, the SHA is computed once
at module-import time over:

  1. the bytes of every Python file in the render path
     (this module + chart_export.py + viz/charts.py),
  2. the installed `plotly` version string,
  3. the installed `kaleido` version string.

A change to any of those inputs shifts the SHA, which propagates through the
L3 evaluator's compute_input_hash and forces a cache miss on the next LLM
call — exactly the semantics required for the byte-determinism contract.

Public surface:
  * `CHART_RENDER_VERSION` — `str`, 64 hex chars.
  * `_RENDER_FILES` (private but stable) — the list of `Path` objects the SHA
    is computed over. Tests monkeypatch this when verifying that mutation of
    any tracked file shifts the SHA.
  * `_compute()` — recomputes the SHA on demand (used only by tests; callers
    must read the module-level constant).
"""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# ASCII map of the render path that this SHA covers:
#
#   chart_render_version.py  <-- this file, also part of the SHA
#       │
#       ├── reads ── chart_export.py    (renders Plotly fig → PNG via kaleido)
#       │                │
#       │                └── calls ── viz/charts.create_static_stock_chart
#       │
#       └── reads ── viz/charts.py      (figure assembly: candles, lines, font)
#
# Any change to the bytes of any of these three files bumps CHART_RENDER_VERSION
# and busts the LLM cache. plotly + kaleido version strings are mixed in below
# so a dependency upgrade also forces a cache bust — a kaleido upgrade can
# legitimately shift PNG bytes.
_RENDER_FILES: list[Path] = [
    Path(__file__),
    Path(__file__).parent / "chart_export.py",
    Path(__file__).parent.parent / "viz" / "charts.py",
]


def _safe_version(pkg: str) -> str:
    """Return the installed package version, or '<unknown>' if not installed.

    Returning a sentinel rather than raising keeps the SHA computable in any
    environment — a missing package is itself a meaningful state change that
    correctly busts the cache when the package is later installed.
    """
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "<unknown>"


def _compute() -> str:
    """Compute the SHA-256 over the render path. Used at import time + by tests."""
    h = hashlib.sha256()
    # Sort by string path so the SHA is invariant under list reordering.
    for p in sorted(_RENDER_FILES, key=lambda x: str(x)):
        h.update(p.read_bytes())
    # Separator byte so a future contributor can't sneak in a file whose tail
    # bytes look like a version string.
    h.update(b"\x00plotly=")
    h.update(_safe_version("plotly").encode())
    h.update(b"\x00kaleido=")
    h.update(_safe_version("kaleido").encode())
    return h.hexdigest()


CHART_RENDER_VERSION: str = _compute()
"""SHA-256 hex string pinning the chart-render code path. 64 hex chars."""

__all__ = ["CHART_RENDER_VERSION"]
