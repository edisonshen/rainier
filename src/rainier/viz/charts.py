"""Plotly charts: candlesticks with S/R lines, pin bars, signals, and review overlays."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from rainier.core.types import AnalysisResult, Direction, Signal, SRType, Timeframe
from rainier.signals.scorer import (
    _multi_tf_confluence_score,
    _sr_strength_score,
    _trend_alignment_score,
    _volume_spike_score,
    _wick_ratio_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tick_labels(
    timestamps: pd.Series, timeframe: Timeframe | None = None,
) -> tuple[list[int], list[str]]:
    """Build x-axis tick positions and labels, adapted to timeframe."""
    n = len(timestamps)

    if timeframe == Timeframe.M5:
        # Fixed session times: 4:00, 6:30, 9:30, 12:30, 15:00, 23:00
        target_times = [(4, 0), (6, 30), (9, 30), (12, 30), (15, 0), (23, 0)]
        positions = []
        labels = []
        prev_date: str = ""
        for i in range(n):
            ts = timestamps.iloc[i]
            hm = (ts.hour, ts.minute)
            if hm in target_times:
                date_str = f"{ts.month}/{ts.day}"
                if date_str != prev_date:
                    labels.append(date_str)
                    prev_date = date_str
                else:
                    labels.append(ts.strftime("%-H:%M"))
                positions.append(i)
        return positions, labels

    # Aim for ~25-30 labels max, regardless of timeframe
    step = max(1, n // 25)

    positions = list(range(0, n, step))
    labels = []
    prev_date: str = ""
    for i in positions:
        ts = timestamps.iloc[i]
        date_str = f"{ts.month}/{ts.day}"
        if date_str != prev_date:
            labels.append(date_str)
            prev_date = date_str
        elif timeframe in (Timeframe.H1, Timeframe.H4):
            labels.append(ts.strftime("%-H:%M"))
        else:
            labels.append("")
    return positions, labels


def _timestamp_to_index(timestamps: pd.Series, ts) -> int:
    diffs = (timestamps - pd.Timestamp(ts)).abs()
    return int(diffs.idxmin())


def _build_figure(df: pd.DataFrame, result: AnalysisResult, signals: list[Signal] | None = None):
    """Build a single plotly figure with all layers. Returns (fig, pinbar_trace_indices)."""
    fig = go.Figure()
    timestamps = df["timestamp"]
    n = len(df)

    # Per-TF visible bar limits
    tf_limits = {
        Timeframe.D1: 120,
        Timeframe.H4: 120,   # ~2 weeks of 4H bars
        Timeframe.H1: 7 * 24,  # ~2 weeks of 1H bars (trading hours vary)
        Timeframe.M5: 500,
    }
    max_visible = tf_limits.get(result.timeframe, 200)
    vis_start = max(0, n - max_visible) if n > max_visible else 0
    vis_df = df.iloc[vis_start:]
    vis_x = list(range(vis_start, n))
    vis_hover = vis_df["timestamp"].apply(lambda t: t.strftime("%m/%d %H:%M"))

    # 1. Candlesticks (visible window only — S/R lines still span full range)
    fig.add_trace(go.Candlestick(
        x=vis_x, open=vis_df["open"], high=vis_df["high"],
        low=vis_df["low"], close=vis_df["close"],
        name="Price", showlegend=False,
        increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
        decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
        text=vis_hover, hoverinfo="text+y",
    ))

    # 2. Horizontal S/R lines (clipped to visible window)
    for level in result.sr_levels:
        if level.sr_type != SRType.HORIZONTAL:
            continue
        tf = level.source_tf
        width = 3 if tf in (Timeframe.D1, Timeframe.W1) else 2 if tf == Timeframe.H4 else 1.5 if tf == Timeframe.H1 else 1
        tf_label = tf.value if tf else ""
        first = level.first_seen.strftime("%m/%d") if level.first_seen else "?"
        last = level.last_tested.strftime("%m/%d") if level.last_tested else "?"
        sr_hover = (f"{level.price:.2f} {tf_label} | touches={level.touches} "
                    f"str={level.strength:.2f} | {first}→{last}")
        sr_end = n - 8  # End line before the TF label
        fig.add_trace(go.Scatter(
            x=[vis_start, sr_end], y=[level.price, level.price],
            mode="lines", line=dict(color="#00BCD4", width=width),
            opacity=0.85, text=[sr_hover, sr_hover], hoverinfo="text",
            showlegend=False,
        ))
        fig.add_annotation(
            x=n - 1, y=level.price, text=tf_label,
            showarrow=False, xanchor="right", xshift=-8,
            font=dict(size=8, color="#00BCD4"),
        )

    # 3. Diagonal trendlines (clipped to visible window)
    for level in result.sr_levels:
        if level.sr_type != SRType.DIAGONAL:
            continue
        color = "#66BB6A" if level.role.value == "support" else "#EF5350"
        si = max(vis_start, level.anchor_index - 50)
        ei = n - 1
        if si >= n:
            continue
        fig.add_trace(go.Scatter(
            x=[si, ei], y=[level.price_at(si), level.price_at(ei)],
            mode="lines", line=dict(color=color, width=1, dash="dot"),
            showlegend=False, opacity=0.5,
        ))

    # 4. Signal markers with score breakdown + entry/SL/TP boxes (hidden by default)
    signal_indices = []
    signal_shape_start = len(fig.layout.shapes) if fig.layout.shapes else 0
    for sig in (signals or []):
        is_long = sig.direction == Direction.LONG
        color = "#26a69a" if is_long else "#ef5350"
        sig_idx = _timestamp_to_index(timestamps, sig.timestamp)
        vis_count = n - vis_start
        box_half = max(3, vis_count // 80)

        sub_scores = ""
        if sig.pin_bar:
            pb = sig.pin_bar
            sub_scores = (f"SR={_sr_strength_score(pb):.2f} Wick={_wick_ratio_score(pb):.2f} "
                          f"Vol={_volume_spike_score(pb, df):.2f} "
                          f"Trend={_trend_alignment_score(pb, result.bias):.2f} "
                          f"Conf={_multi_tf_confluence_score(pb, result.sr_levels):.2f}")

        sig_hover = (f"{sig.timestamp.strftime('%m/%d %H:%M')}<br>"
                     f"{'BUY' if is_long else 'SELL'} @ {sig.entry_price:.2f}<br>"
                     f"SL={sig.stop_loss:.2f} TP={sig.take_profit:.2f}<br>"
                     f"R:R={sig.rr_ratio:.1f} | Conf={sig.confidence:.0%}<br>"
                     f"{sub_scores}")

        signal_indices.append(len(fig.data))
        fig.add_trace(go.Scatter(
            x=[sig_idx], y=[sig.entry_price], mode="markers",
            marker=dict(symbol="diamond", size=10, color=color, line=dict(width=1, color="white")),
            text=[sig_hover], hoverinfo="text", showlegend=False,
        ))
        # TP zone
        fig.add_shape(type="rect", x0=sig_idx - box_half, x1=sig_idx + box_half,
                      y0=sig.entry_price, y1=sig.take_profit,
                      fillcolor="rgba(0,230,118,0.1)",
                      line=dict(color="rgba(0,230,118,0.4)", width=1, dash="dot"))
        # SL zone
        fig.add_shape(type="rect", x0=sig_idx - box_half, x1=sig_idx + box_half,
                      y0=sig.entry_price, y1=sig.stop_loss,
                      fillcolor="rgba(255,23,68,0.1)",
                      line=dict(color="rgba(255,23,68,0.4)", width=1, dash="dot"))
    signal_shape_end = len(fig.layout.shapes) if fig.layout.shapes else 0

    # 5. Pin bar markers (hidden by default)
    pinbar_indices = []
    for direction, dir_label in [(Direction.LONG, "Bullish"), (Direction.SHORT, "Bearish")]:
        pbs = [pb for pb in result.pin_bars if pb.direction == direction]
        if not pbs:
            continue
        is_long = direction == Direction.LONG
        color = "#26a69a" if is_long else "#ef5350"
        symbol = "triangle-up" if is_long else "triangle-down"
        x_vals, y_vals, texts = [], [], []
        for pb in pbs:
            y_pos = pb.candle.low if is_long else pb.candle.high
            y_marker = (y_pos - pb.candle.range * 0.3) if is_long else (y_pos + pb.candle.range * 0.3)
            sr_info = ""
            if pb.nearest_sr:
                sr_info = (f"S/R={pb.nearest_sr.price:.2f} "
                           f"({pb.nearest_sr.source_tf.value if pb.nearest_sr.source_tf else 'local'}) "
                           f"dist={pb.sr_distance_pct:.4f}")
            x_vals.append(pb.index)
            y_vals.append(y_marker)
            texts.append(f"{pb.candle.timestamp.strftime('%m/%d %H:%M')}<br>"
                         f"{dir_label} Pin Bar<br>Wick ratio: {pb.wick_ratio:.1f}<br>{sr_info}")

        pinbar_indices.append(len(fig.data))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="markers",
            marker=dict(symbol=symbol, size=8, color=color, line=dict(width=1, color="white")),
            text=texts, hoverinfo="text", showlegend=False,
        ))

    # Layout — tick labels from visible window only
    vis_timestamps = df.iloc[vis_start:]["timestamp"]
    tick_pos, tick_labels = _build_tick_labels(vis_timestamps, result.timeframe)
    # Offset tick positions to match the global x indices
    tick_pos = [p + vis_start for p in tick_pos]
    xaxis_cfg = dict(
        tickmode="array", tickvals=tick_pos, ticktext=tick_labels,
        rangeslider=dict(visible=False),
        range=[vis_start - 5, n + 5],
        tickangle=-45,
    )

    # Auto-fit y-axis to visible bars
    y_lo = float(vis_df["low"].min())
    y_hi = float(vis_df["high"].max())
    y_pad = (y_hi - y_lo) * 0.05

    tf_display = {
        Timeframe.D1: "1D", Timeframe.H4: "4H",
        Timeframe.H1: "1H", Timeframe.M5: "5m",
    }.get(result.timeframe, result.timeframe.value)
    chart_font = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

    # TradingView-style watermark: large faded symbol in chart center
    fig.add_annotation(
        text=f"{result.symbol}, {tf_display}",
        xref="paper", yref="paper", x=0.5, y=0.55,
        showarrow=False, font=dict(family=chart_font, size=56, color="rgba(255,255,255,0.10)"),
    )

    fig.update_layout(
        title=None,
        yaxis_title=None,
        xaxis=xaxis_cfg,
        yaxis=dict(
            range=[y_lo - y_pad, y_hi + y_pad], autorange=False,
            side="right",
            tickfont=dict(family=chart_font, size=11, color="#555"),
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=False,
        ),
        xaxis_tickfont=dict(family=chart_font, size=10, color="#555"),
        xaxis_gridcolor="rgba(255,255,255,0.12)",
        xaxis_zeroline=False,
        font=dict(family=chart_font),
        template="plotly_dark",
        height=700,
        margin=dict(l=30, r=80, t=70, b=50),
        paper_bgcolor="#000000", plot_bgcolor="#000000",
    )

    return fig, pinbar_indices, signal_indices, (signal_shape_start, signal_shape_end)


# ---------------------------------------------------------------------------
# Single chart (standalone HTML)
# ---------------------------------------------------------------------------

def create_chart(
    df: pd.DataFrame,
    result: AnalysisResult,
    signals: list[Signal] | None = None,
    output_path: Path | None = None,
) -> go.Figure:
    fig, _pb_idx, _sig_idx, _sig_shapes = _build_figure(df, result, signals)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
    return fig


# ---------------------------------------------------------------------------
# Tabbed multi-TF chart
# ---------------------------------------------------------------------------

def create_tabbed_chart(
    data: dict[Timeframe, pd.DataFrame],
    results: dict[Timeframe, AnalysisResult],
    trading_tf: Timeframe,
    signals: list[Signal] | None = None,
    output_path: Path | None = None,
) -> None:
    """HTML page with tab buttons (1D / 4H / 1H / 5m) + pin bar toggle."""
    tf_order = [Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M5]
    available_tfs = [tf for tf in tf_order if tf in data]

    trading_result = results.get(trading_tf)
    shared_sr = trading_result.sr_levels if trading_result else []

    # Build each TF chart — store figure JSON for JS-based rendering
    chart_data = {}  # tf -> (fig_json, pinbar_indices)
    for tf in available_tfs:
        tf_result = results.get(tf)
        merged_result = AnalysisResult(
            symbol=trading_result.symbol if trading_result else "",
            timeframe=tf,
            sr_levels=list(shared_sr),
            pin_bars=tf_result.pin_bars if tf_result else [],
            inside_bars=tf_result.inside_bars if tf_result else [],
            pivots=tf_result.pivots if tf_result else [],
            bias=tf_result.bias if tf_result else None,
        )
        tf_signals = signals if tf == trading_tf else None
        fig, pb_idx, sig_idx, sig_shapes = _build_figure(data[tf], merged_result, tf_signals)

        fig_json = fig.to_json()
        chart_data[tf] = (fig_json, pb_idx, sig_idx, sig_shapes)

    tab_labels = {Timeframe.M5: "5m", Timeframe.H1: "1H", Timeframe.H4: "4H", Timeframe.D1: "1D"}
    active_tf = trading_tf.value

    tabs_html = ""
    for tf in available_tfs:
        active = "active" if tf == trading_tf else ""
        tabs_html += f'<button class="tab-btn {active}" data-tf="{tf.value}">{tab_labels.get(tf, tf.value)}</button>\n'

    panels_html = ""
    chart_js_parts = []
    for tf in available_tfs:
        fig_json, pb_idx, sig_idx, sig_shapes = chart_data[tf]
        div_id = f"chart-{tf.value}"
        panel_active = " active" if tf == trading_tf else ""
        panels_html += f'<div class="tab-panel{panel_active}" id="panel-{tf.value}"><div id="{div_id}" style="width:100%;height:700px;"></div></div>\n'
        chart_js_parts.append(
            f'(function() {{ var fig = {fig_json}; '
            f'Plotly.newPlot("{div_id}", fig.data, fig.layout, {{responsive:true}}); '
            f'var el = document.getElementById("{div_id}"); '
            f'el._pinbarIndices = {pb_idx if pb_idx else "[]"}; '
            f'el._signalIndices = {sig_idx if sig_idx else "[]"}; '
            f'el._signalShapeRange = {list(sig_shapes)}; }})()'
        )

    init_js = ";\n".join(chart_js_parts)
    symbol = trading_result.symbol if trading_result else ""

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{symbol} Day Trade</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
<style>
:root {{
  --bg: #000000; --bg2: #0a0a0a; --border: #1a1a1a;
  --text: #d1d4dc; --text-muted: #787b86;
  --accent: #2962ff; --slider-bg: #2a2e39;
  --chart-bg: #131722; --grid: rgba(255,255,255,0.04);
  --axis-text: #555; --watermark: rgba(255,255,255,0.06);
  --sr-color: #00BCD4; --sr-label: #00BCD4;
  --candle-up: #ffffff; --candle-down: #ffffff;
}}
:root.light {{
  --bg: #ffffff; --bg2: #f8f9fa; --border: #e0e3eb;
  --text: #131722; --text-muted: #787b86;
  --accent: #2962ff; --slider-bg: #d1d4dc;
  --chart-bg: #ffffff; --grid: rgba(0,0,0,0.06);
  --axis-text: #999; --watermark: rgba(0,0,0,0.05);
  --sr-color: #0097A7; --sr-label: #0097A7;
  --candle-up: #000000; --candle-down: #000000;
}}
body {{ margin:0; padding:0; background:var(--bg); color:var(--text);
       font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
       transition:background 0.3s, color 0.3s; }}
.controls {{ display:flex; align-items:center; gap:12px; padding:10px 20px;
            background:var(--bg2); border-bottom:1px solid var(--border);
            transition:background 0.3s; }}
.tab-btn {{ padding:6px 16px; border:1px solid var(--border); border-radius:6px;
           background:var(--bg2); color:var(--text-muted); font-size:13px; font-weight:500;
           cursor:pointer; transition:all 0.2s; }}
.tab-btn:hover {{ background:var(--border); color:var(--text); }}
.tab-btn.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.toggle-wrap {{ position:fixed; bottom:16px; right:20px; z-index:1000;
               display:flex; align-items:center; gap:8px;
               background:var(--bg2); border:1px solid var(--border); border-radius:8px;
               padding:6px 12px; transition:background 0.3s; }}
.toggle-label {{ color:var(--text-muted); font-size:12px; font-weight:500; }}
.toggle {{ position:relative; display:inline-block; width:44px; height:26px; cursor:pointer; }}
.toggle input {{ opacity:0; width:0; height:0; }}
.toggle .slider {{ position:absolute; top:0; left:0; right:0; bottom:0;
                  background:var(--slider-bg); border-radius:13px; transition:background 0.3s; }}
.toggle .knob {{ position:absolute; top:2px; left:2px; width:22px; height:22px;
                background:#fff; border-radius:50%; transition:transform 0.3s;
                box-shadow:0 1px 3px rgba(0,0,0,0.3); }}
.toggle input:checked + .slider {{ background:var(--accent); }}
.toggle input:checked + .slider + .knob {{ transform:translateX(18px); }}
.theme-icon {{ font-size:14px; cursor:pointer; color:var(--text-muted); transition:color 0.2s; }}
.theme-icon:hover {{ color:var(--text); }}
.tab-panel {{ display:none; }}
.tab-panel.active {{ display:block; }}
</style>
</head><body>

<div class="controls">
{tabs_html}
</div>

{panels_html}

<div class="toggle-wrap" style="flex-direction:column; gap:10px; align-items:stretch;">
    <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
        <span class="toggle-label">Signals</span>
        <label class="toggle">
            <input type="checkbox" id="signal-toggle" checked>
            <span class="slider"></span>
            <span class="knob"></span>
        </label>
    </div>
    <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
        <span class="toggle-label">Pin Bars</span>
        <label class="toggle">
            <input type="checkbox" id="pinbar-toggle" checked>
            <span class="slider"></span>
            <span class="knob"></span>
        </label>
    </div>
    <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; border-top:1px solid var(--border); padding-top:8px;">
        <span class="toggle-label">Theme</span>
        <span class="theme-icon" id="theme-toggle" title="Toggle light/dark" style="width:44px; text-align:center;">&#9790;</span>
    </div>
</div>

<script>
// Render all charts via Plotly.newPlot (JSON data, no binary encoding issues)
{init_js};

// Auto-fit y-axis to visible x-range
function autoFitYAxis(plotDiv) {{
    if (!plotDiv || !plotDiv._ohlcData) return;
    var xRange = plotDiv.layout.xaxis.range;
    if (!xRange) return;
    var x0 = Math.max(0, Math.floor(xRange[0]));
    var x1 = Math.min(plotDiv._ohlcData.length - 1, Math.ceil(xRange[1]));
    if (x0 >= x1) return;
    var lo = Infinity, hi = -Infinity;
    for (var i = x0; i <= x1; i++) {{
        var d = plotDiv._ohlcData[i];
        if (d) {{ if (d[0] < lo) lo = d[0]; if (d[1] > hi) hi = d[1]; }}
    }}
    if (lo === Infinity) return;
    var pad = (hi - lo) * 0.05;
    Plotly.relayout(plotDiv, {{'yaxis.range': [lo - pad, hi + pad]}});
}}

function attachAutoFit(plotDiv) {{
    if (!plotDiv || plotDiv._autoFitAttached) return;
    plotDiv._autoFitAttached = true;
    plotDiv.on('plotly_relayout', function(ed) {{
        if (ed['xaxis.range[0]'] !== undefined || ed['xaxis.range'] !== undefined) {{
            autoFitYAxis(plotDiv);
        }}
    }});
}}

// After charts render, store OHLC data and set up auto-fit
document.querySelectorAll('[id^="chart-"]').forEach(function(pd) {{
    if (pd.data && pd.data[0] && pd.data[0].low) {{
        var ohlc = [];
        for (var i = 0; i < pd.data[0].low.length; i++) {{
            ohlc.push([pd.data[0].low[i], pd.data[0].high[i]]);
        }}
        pd._ohlcData = ohlc;
    }}
    attachAutoFit(pd);
}});

// Tab switching
document.querySelectorAll('.tab-btn').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
        var tf = this.dataset.tf;
        document.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
        this.classList.add('active');
        document.querySelectorAll('.tab-panel').forEach(function(p) {{ p.classList.remove('active'); }});
        var panel = document.getElementById('panel-' + tf);
        panel.classList.add('active');
        requestAnimationFrame(function() {{
            var plotDiv = document.getElementById('chart-' + tf);
            if (plotDiv) {{
                Plotly.Plots.resize(plotDiv);
                var pbToggle = document.getElementById('pinbar-toggle');
                var idx = plotDiv._pinbarIndices;
                if (idx && idx.length > 0) Plotly.restyle(plotDiv, {{visible: pbToggle.checked}}, idx);
                // Sync signal toggle
                var sigToggle = document.getElementById('signal-toggle');
                var sigIdx = plotDiv._signalIndices;
                if (sigIdx && sigIdx.length > 0) Plotly.restyle(plotDiv, {{visible: sigToggle.checked}}, sigIdx);
                attachAutoFit(plotDiv);
            }}
        }});
    }});
}});

// Pin bar toggle
document.getElementById('pinbar-toggle').addEventListener('change', function() {{
    var vis = this.checked;
    document.querySelectorAll('[id^="chart-"]').forEach(function(pd) {{
        var idx = pd._pinbarIndices;
        if (idx && idx.length > 0) Plotly.restyle(pd, {{visible: vis}}, idx);
    }});
}});

// Signal toggle — show/hide signal markers + TP/SL boxes
function toggleSignals(vis) {{
    document.querySelectorAll('[id^="chart-"]').forEach(function(pd) {{
        var idx = pd._signalIndices;
        if (idx && idx.length > 0) Plotly.restyle(pd, {{visible: vis}}, idx);
        // Toggle TP/SL shape visibility
        var sr = pd._signalShapeRange;
        if (sr && sr[1] > sr[0]) {{
            var updates = {{}};
            for (var i = sr[0]; i < sr[1]; i++) {{
                updates['shapes[' + i + '].visible'] = vis;
            }}
            Plotly.relayout(pd, updates);
        }}
    }});
}}
document.getElementById('signal-toggle').addEventListener('change', function() {{
    toggleSignals(this.checked);
}});

// Theme toggle
var isDark = true;
var themes = {{
    dark: {{
        bg: '#000000', grid: 'rgba(255,255,255,0.12)', axisText: '#666',
        watermark: 'rgba(255,255,255,0.06)', srColor: '#00BCD4',
        candleUp: '#26a69a', candleDown: '#ef5350',
        candleUpFill: '#26a69a', candleDownFill: '#ef5350',
        sigLongFill: '#26a69a', sigLongBorder: '#26a69a',
        sigShortFill: '#ef5350', sigShortBorder: '#ef5350',
        labelBg: 'rgba(0,0,0,0.4)',
        icon: '\u263E'
    }},
    light: {{
        bg: '#ffffff', grid: 'rgba(0,0,0,0.06)', axisText: '#999',
        watermark: 'rgba(0,0,0,0.05)', srColor: '#0097A7',
        candleUp: '#000000', candleDown: '#000000',
        candleUpFill: '#ffffff', candleDownFill: '#000000',
        sigLongFill: '#ffffff', sigLongBorder: '#000000',
        sigShortFill: '#000000', sigShortBorder: '#000000',
        labelBg: 'rgba(0,0,0,0.08)',
        icon: '\u2600'
    }}
}};

document.getElementById('theme-toggle').addEventListener('click', function() {{
    isDark = !isDark;
    var t = isDark ? themes.dark : themes.light;
    this.textContent = t.icon;
    document.documentElement.classList.toggle('light', !isDark);

    document.querySelectorAll('[id^="chart-"]').forEach(function(pd) {{
        // Update chart backgrounds, grid, axis colors
        var layoutUpdate = {{
            'paper_bgcolor': t.bg, 'plot_bgcolor': t.bg,
            'xaxis.gridcolor': t.grid, 'yaxis.gridcolor': t.grid,
            'xaxis.tickfont.color': t.axisText, 'yaxis.tickfont.color': t.axisText,
        }};
        // Update watermark annotation (last annotation is always the watermark)
        var annLen = pd.layout.annotations ? pd.layout.annotations.length : 0;
        for (var a = 0; a < annLen; a++) {{
            var ann = pd.layout.annotations[a];
            if (ann.xref === 'paper' && ann.yref === 'paper' && ann.font && ann.font.size >= 40) {{
                layoutUpdate['annotations[' + a + '].font.color'] = t.watermark;
            }}
        }}
        // Update S/R line colors
        for (var i = 1; i < pd.data.length; i++) {{
            var tr = pd.data[i];
            if (tr.type === 'scatter' && tr.line && tr.line.color === (isDark ? '#0097A7' : '#00BCD4')) {{
                Plotly.restyle(pd, {{'line.color': t.srColor}}, [i]);
            }}
        }}
        // Update candle colors (line + fill + width)
        var lw = isDark ? 1 : 0.5;
        Plotly.restyle(pd, {{
            'increasing.line.color': t.candleUp, 'increasing.fillcolor': t.candleUpFill,
            'increasing.line.width': lw,
            'decreasing.line.color': t.candleDown, 'decreasing.fillcolor': t.candleDownFill,
            'decreasing.line.width': lw
        }}, [0]);
        // Update signal + pin bar marker colors (fill + border)
        function updateMarkers(indices) {{
            if (!indices || indices.length === 0) return;
            for (var s = 0; s < indices.length; s++) {{
                var tr = pd.data[indices[s]];
                if (!tr || !tr.marker) continue;
                // Track original direction via _isLong flag
                if (tr._isLong === undefined) {{
                    tr._isLong = (tr.marker.color === '#00E676' ||
                                  tr.marker.symbol === 'triangle-up');
                }}
                var fill = tr._isLong ? t.sigLongFill : t.sigShortFill;
                var border = tr._isLong ? t.sigLongBorder : t.sigShortBorder;
                Plotly.restyle(pd, {{
                    'marker.color': fill,
                    'marker.line.color': border,
                    'marker.line.width': isDark ? 1 : 1.5
                }}, [indices[s]]);
            }}
        }}
        updateMarkers(pd._signalIndices);
        updateMarkers(pd._pinbarIndices);
        // Update S/R annotation text colors
        for (var a = 0; a < annLen; a++) {{
            var ann = pd.layout.annotations[a];
            if (ann.font && ann.font.size === 9) {{
                layoutUpdate['annotations[' + a + '].font.color'] = t.srColor;
            }}
        }}
        Plotly.relayout(pd, layoutUpdate);
    }});
}});
</script>
</body></html>"""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)
