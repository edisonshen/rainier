"""Plotly charts: candlesticks with S/R lines, pin bars, and signal markers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from quant.core.types import AnalysisResult, Direction, Signal, SRType, Timeframe


def create_chart(
    df: pd.DataFrame,
    result: AnalysisResult,
    signals: list[Signal] | None = None,
    output_path: Path | None = None,
) -> go.Figure:
    """Create an interactive plotly chart with analysis overlays.

    Layers:
    1. Candlestick chart
    2. Horizontal S/R lines (solid = strong, dashed = weak)
    3. Diagonal trendlines
    4. Pin bar markers
    5. Signal entry/SL/TP markers
    """
    fig = go.Figure()

    timestamps = df["timestamp"]

    # 1. Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=timestamps,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    # 2. Horizontal S/R lines — primary, bold and clear
    #    Higher TF lines are thicker (1D > 4H > 1H > current)
    for level in result.sr_levels:
        if level.sr_type != SRType.HORIZONTAL:
            continue
        color = "#FF6D00"  # orange for all pin bar lines (like the reference images)
        # Width by source TF: daily=3, 4H=2, 1H=1.5, same TF=1
        tf = level.source_tf
        if tf in (Timeframe.D1, Timeframe.W1):
            width = 3
        elif tf == Timeframe.H4:
            width = 2
        elif tf == Timeframe.H1:
            width = 1.5
        else:
            width = 1

        tf_label = tf.value if tf else ""
        fig.add_hline(
            y=level.price,
            line_dash="solid",
            line_color=color,
            line_width=width,
            opacity=0.85,
            annotation_text=f"{level.price:.2f} {tf_label}",
            annotation_position="right",
            annotation_font_size=9,
            annotation_font_color=color,
        )

    # 3. Diagonal trendlines — secondary, thinner and dimmer
    for level in result.sr_levels:
        if level.sr_type != SRType.DIAGONAL:
            continue
        color = "#66BB6A" if level.role.value == "support" else "#EF5350"
        start_idx = max(0, level.anchor_index - 50)
        end_idx = len(df) - 1
        if start_idx >= len(df) or end_idx < 0:
            continue

        x_vals = [timestamps.iloc[start_idx], timestamps.iloc[end_idx]]
        y_vals = [level.price_at(start_idx), level.price_at(end_idx)]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                name=f"Trend {'S' if level.role.value == 'support' else 'R'}",
                showlegend=False,
                opacity=0.5,
            )
        )

    # 4. Signal markers only — no pin bar markers (too noisy on 5m charts)
    if signals:
        for sig in signals:
            color = "#00E676" if sig.direction == Direction.LONG else "#FF1744"
            # Entry
            fig.add_trace(
                go.Scatter(
                    x=[sig.timestamp],
                    y=[sig.entry_price],
                    mode="markers+text",
                    marker=dict(symbol="diamond", size=10, color=color),
                    text=[f"{sig.direction.value} {sig.confidence:.0%}"],
                    textposition="top center",
                    textfont=dict(size=9, color=color),
                    name=f"Signal {sig.direction.value}",
                    showlegend=False,
                )
            )
            # SL line
            fig.add_shape(
                type="line",
                x0=sig.timestamp,
                x1=sig.timestamp,
                y0=sig.entry_price,
                y1=sig.stop_loss,
                line=dict(color="red", width=1, dash="dot"),
            )
            # TP line
            fig.add_shape(
                type="line",
                x0=sig.timestamp,
                x1=sig.timestamp,
                y0=sig.entry_price,
                y1=sig.take_profit,
                line=dict(color="green", width=1, dash="dot"),
            )

    # Layout
    fig.update_layout(
        title=f"{result.symbol} {result.timeframe.value} — S/R + Pin Bars",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=700,
        margin=dict(l=60, r=20, t=50, b=40),
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

    return fig
