"""Plotly charts: candlesticks with S/R lines, pin bars, signals, and review overlays."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from rainier.core.types import (
    PatternSignal,
)

# ---------------------------------------------------------------------------
# Static stock chart for LLM consumption (PR1)
# ---------------------------------------------------------------------------

# Pinned font family for the LLM-research static chart path. Used by
# `create_static_stock_chart` only — the interactive futures chart in
# `_build_figure` keeps its system-font cascade for terminal display.
#
# Why "DejaVu Sans, sans-serif": kaleido 0.2.1 ships its own headless Chromium
# whose default font installation includes DejaVu Sans and Liberation as
# bundled fallbacks. Requesting "DejaVu Sans" explicitly hits the bundled
# font on every host (macOS, Linux container, Linux desktop) — there is no
# system-font lookup that could drift. Earlier iterations of this constant
# used "Arial, sans-serif"; while empirical two-process tests on macOS arm64
# with kaleido 0.2.1 show kaleido's bundled Chromium ignores host fonts,
# requesting a font that IS in the bundle removes the theoretical drift hazard
# entirely (a host with Arial installed plus a future kaleido that DOES honor
# system fonts would render differently from a Linux container without Arial).
# Pinning a bundled family makes the cross-host invariance explicit, not
# inferred. See TASK-PLAN §4.3.
_LLM_CHART_FONT = "DejaVu Sans, sans-serif"


def create_static_stock_chart(
    symbol: str,
    df: pd.DataFrame,
    pattern: PatternSignal | None = None,
    bars: int = 60,
) -> go.Figure:
    """Render a static daily candlestick chart for LLM multimodal input.

    60 daily candles + horizontal lines for entry/stop_loss/target if a 蔡森
    pattern fired + pattern label annotation. No tabs, no interactivity. Stock
    side.

    Determinism contract (see docs/TASK-PLAN-chart-export-determinism-3c8e.md):
    the returned figure's `to_json()` is identical across processes for the
    same `(symbol, df, pattern)`. Callers in the LLM-research path consume
    `render_chart_png` (in `llm_thesis/chart_export.py`), which feeds this
    figure to kaleido and post-strips PNG metadata — together they yield
    byte-identical PNG output.
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"{symbol} — no data",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(family=_LLM_CHART_FONT, size=20, color="#888"),
        )
        fig.update_layout(
            template="plotly_dark", height=800, width=1280,
            paper_bgcolor="#000000", plot_bgcolor="#000000",
            font=dict(family=_LLM_CHART_FONT),
        )
        return fig

    # Tail to last `bars` (default 60); columns may be lower or upper case.
    tail = df.tail(bars).copy()
    cols = {c.lower(): c for c in tail.columns}
    o_col = cols.get("open", "open")
    h_col = cols.get("high", "high")
    l_col = cols.get("low", "low")
    c_col = cols.get("close", "close")
    x_idx = list(range(len(tail)))

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x_idx,
                open=tail[o_col],
                high=tail[h_col],
                low=tail[l_col],
                close=tail[c_col],
                increasing_line_color="#26a69a",
                increasing_fillcolor="#26a69a",
                decreasing_line_color="#ef5350",
                decreasing_fillcolor="#ef5350",
                showlegend=False,
                name=symbol,
            )
        ]
    )

    if pattern is not None:
        x0, x1 = 0, len(tail) - 1
        # Entry = green, SL = red, target = blue (deterministic, easy to OCR).
        levels = [
            (pattern.entry_price, "#26a69a", "Entry"),
            (pattern.stop_loss, "#ef5350", "SL"),
            (pattern.target_wave1, "#42A5F5", "Target"),
        ]
        for price, color, label in levels:
            if price is None or price <= 0:
                continue
            fig.add_shape(
                type="line",
                x0=x0, x1=x1, y0=price, y1=price,
                line=dict(color=color, width=1.5, dash="dot"),
            )
            fig.add_annotation(
                x=x1, y=price, text=f"{label} {price:.2f}",
                showarrow=False, xanchor="right", xshift=-4,
                font=dict(family=_LLM_CHART_FONT, size=10, color=color),
            )
        fig.add_annotation(
            x=0, y=1.04, xref="x", yref="paper",
            text=f"Pattern: {pattern.pattern_type} ({pattern.direction})",
            showarrow=False, xanchor="left",
            font=dict(family=_LLM_CHART_FONT, size=12, color="#FFD54F"),
        )

    fig.update_layout(
        title=dict(text=symbol, x=0.02, y=0.98, xanchor="left",
                   font=dict(family=_LLM_CHART_FONT, size=18, color="#fff")),
        template="plotly_dark",
        xaxis=dict(rangeslider=dict(visible=False), showgrid=False),
        yaxis=dict(side="right", gridcolor="rgba(255,255,255,0.1)"),
        margin=dict(l=10, r=70, t=60, b=30),
        height=800, width=1280,
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        showlegend=False,
        # Pin font family for byte-determinism — see _LLM_CHART_FONT.
        font=dict(family=_LLM_CHART_FONT),
    )
    return fig
