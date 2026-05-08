"""Plotly → static PNG via kaleido (multimodal LLM input)."""

from __future__ import annotations

import hashlib
import logging

import pandas as pd

from rainier.core.types import PatternSignal
from rainier.viz.charts import create_static_stock_chart

log = logging.getLogger(__name__)


def render_chart_png(
    symbol: str,
    df: pd.DataFrame,
    pattern: PatternSignal | None = None,
    width: int = 1280,
    height: int = 800,
) -> tuple[bytes, str]:
    """Render the static stock chart to PNG bytes + sha256 digest.

    Raises on kaleido / Plotly export failures so the caller can choose to skip
    the ticker (per the per-ticker try/except in service.compute_theses_and_persist).
    """
    fig = create_static_stock_chart(symbol, df, pattern=pattern)
    png = fig.to_image(format="png", width=width, height=height, engine="kaleido")
    digest = hashlib.sha256(png).hexdigest()
    return png, digest
