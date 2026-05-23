"""Plotly → static PNG via kaleido (multimodal LLM input).

This module is part of the byte-deterministic chart-render contract — see
docs/DESIGN-qu100-llm-backtest.md [D-025] and
docs/TASK-PLAN-chart-export-determinism-3c8e.md. The PNG bytes returned by
`render_chart_png(symbol, df)` MUST be byte-identical across processes for
the same (symbol, df) so the LLM-research engine's `compute_input_hash` can
treat the chart hash as a stable cache key.

Determinism pinning, in order of execution:

  1. `create_static_stock_chart` (in viz/charts.py) assembles the Plotly
     figure with an explicit font family and a deterministic annotation
     order — no wall-clock, no random colors, no system-font cascade.
  2. `fig.to_image(...)` here is called with `engine='kaleido'`, `scale=1`,
     and explicit width/height — never relying on defaults that could drift
     across kaleido versions.
  3. `_strip_png_metadata` post-processes the PNG bytes to scrub `tIME`,
     `tEXt`, `iTXt`, and `zTXt` chunks. Today's kaleido 0.2.1 does not emit
     these on macOS/Linux, but a future upgrade could re-introduce a
     `tIME` timestamp; this scrub is the belt-and-suspenders defense.
  4. `CHART_RENDER_VERSION` (re-exported from chart_render_version) is the
     cache-bust SHA — bumping any render-path source byte or any pinned
     dep version shifts the SHA. The constant is exposed here for the
     follow-on Slice-1 cache-key contract (TASK-PLAN §followups, line 280);
     wiring it into `compute_input_hash` lives with the L3 evaluator
     harness and is intentionally NOT done in this task.

Raises on kaleido / Plotly export failures so the caller can choose to skip
the ticker (per the per-ticker try/except in service.compute_theses_and_persist).
"""

from __future__ import annotations

import hashlib
import logging

import pandas as pd

from rainier.core.types import PatternSignal
from rainier.llm_thesis.chart_render_version import CHART_RENDER_VERSION
from rainier.viz.charts import create_static_stock_chart

log = logging.getLogger(__name__)

# Pinned render dimensions for the LLM-research path. Exposed as kwargs on
# `render_chart_png` for back-compat, but callers that consume the
# byte-determinism contract MUST pass these defaults — overriding shifts the
# canvas and the PNG bytes.
RENDER_WIDTH = 1280
RENDER_HEIGHT = 800
RENDER_SCALE = 1

# Non-critical PNG chunks whose payload can leak wall-clock / build metadata.
# We scrub these post-render so the cache key is stable across machines.
# (PNG chunk-type bytes are case-sensitive ASCII per RFC 2083.)
_METADATA_CHUNK_TYPES: frozenset[bytes] = frozenset({
    b"tIME",  # last-modification timestamp — wall-clock leak
    b"tEXt",  # Latin-1 text (often "Software=Kaleido X.Y") — build leak
    b"iTXt",  # UTF-8 text — same hazard
    b"zTXt",  # compressed text — same hazard
})

# PNG signature: 8 fixed bytes that every valid PNG starts with.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _strip_png_metadata(png: bytes) -> bytes:
    """Remove non-critical metadata chunks from a PNG byte string.

    PNG chunk layout (per RFC 2083):
        [4-byte length][4-byte type][length bytes data][4-byte CRC]

    The critical chunks (IHDR, IDAT, IEND, PLTE, ...) are preserved
    verbatim. The chunks listed in `_METADATA_CHUNK_TYPES` are dropped.
    We never recompute the CRC of the survivors — each chunk carries its
    own CRC computed over (type + data) only, so removing a different chunk
    cannot invalidate it.

    Returns the original bytes unchanged if no metadata chunks are present
    (the fast path for current kaleido).
    """
    if not png.startswith(_PNG_SIGNATURE):
        # Not a PNG — caller can decide what to do. We never raise here so
        # the contract with the MagicMock-based unit tests stays simple.
        return png

    out = bytearray(_PNG_SIGNATURE)
    i = len(_PNG_SIGNATURE)
    n = len(png)
    while i + 8 <= n:
        length = int.from_bytes(png[i:i + 4], "big")
        chunk_type = png[i + 4:i + 8]
        chunk_end = i + 12 + length  # 4 len + 4 type + length data + 4 CRC
        if chunk_end > n:
            # Truncated PNG — bail out and return what we have so the caller
            # surfaces the error via the SHA mismatch on the next render.
            return bytes(png)
        if chunk_type not in _METADATA_CHUNK_TYPES:
            out.extend(png[i:chunk_end])
        i = chunk_end
        if chunk_type == b"IEND":
            break

    return bytes(out)


def render_chart_png(
    symbol: str,
    df: pd.DataFrame,
    pattern: PatternSignal | None = None,
    width: int = RENDER_WIDTH,
    height: int = RENDER_HEIGHT,
) -> tuple[bytes, str]:
    """Render the static stock chart to PNG bytes + sha256 digest.

    The PNG bytes are post-processed to strip non-critical metadata chunks so
    the SHA-256 is a pure function of (symbol, df, pattern) — see module
    docstring for the full determinism contract.

    Args:
        symbol: ticker label drawn in the chart title and watermark.
        df: OHLCV dataframe; the renderer tails the last 60 bars.
        pattern: optional Caisen pattern signal; draws entry / SL / target
            horizontal lines when present.
        width, height: pinned canvas dimensions. Overriding these breaks the
            byte-determinism contract for downstream cache lookups — callers
            in the LLM-research path MUST keep the defaults.

    Returns:
        `(png_bytes, sha256_hex)` — the digest is computed over the
        metadata-scrubbed bytes, which is what callers should hash into
        their input-hash for cache lookups.
    """
    fig = create_static_stock_chart(symbol, df, pattern=pattern)
    raw_png = fig.to_image(
        format="png",
        width=width,
        height=height,
        scale=RENDER_SCALE,
        engine="kaleido",
    )
    png = _strip_png_metadata(raw_png)
    digest = hashlib.sha256(png).hexdigest()
    return png, digest


__all__ = ["render_chart_png", "CHART_RENDER_VERSION"]
