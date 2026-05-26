"""Self-contained S&P 500 market-breadth dashboard HTML renderer.

Reads `data/cache/sp500_breadth_daily.parquet` (long-format: asof_date,
indicator, value) and emits a single deterministic HTML file with:

  * Header summary — today's snapshot in big text (% above 20d / 200d MA,
    new 52w highs / lows, A/D today) + `Last updated: YYYY-MM-DD HH:MM PT`.
  * Four inline SVG charts:
        1. chart-pct-above-ma         — 20d / 50d / 200d lines (last 2y)
        2. chart-new-highs-lows       — paired bars (last 2y)
        3. chart-ad-cumulative        — A/D cumulative line (default 2y;
                                        CSS-only radio toggle to "all")
        4. chart-mcclellan            — oscillator + summation stacked
                                        (default 2y; CSS-only radio toggle)
  * Footer — survivorship-bias disclaimer (verbatim from DESIGN §5.3).
  * Light + dark mode via `prefers-color-scheme`.
  * Green/red bull/bear palette (breadth.app aesthetic). Bull = #1b9e3a,
    bear = #c92a2a; mirrors the sibling ETF page's `--pos` / `--neg`.

Pure function over the parquet DataFrame → HTML string. Caller supplies
`rendered_at_pt` (Pacific wall-clock HH:MM); the renderer never calls
``datetime.now()`` so re-rendering on identical input is byte-stable.

Design refs:
    docs/DESIGN-market-breadth-webpage.md §5 (layout) + §5.3 (disclaimer)
    docs/TASK-PLAN-market-breadth-render-e818.md (acceptance gates)

Render flow:

    ┌── data/cache/sp500_breadth_daily.parquet (long format) ──┐
    │                                                          │
    │  1. pivot long → wide (asof_date × indicator)            │
    │  2. slice trailing 504 trading days (default 2y)         │
    │  3. build SVG paths for the 4 charts                     │
    │  4. extract latest-day header snapshot                   │
    │  5. render jinja2 template (inline CSS + 4 SVG blocks)   │
    │  6. atomic write (write_breadth_html only)               │
    └──────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined, select_autoescape
from markupsafe import Markup

__all__ = [
    "render_breadth_html",
    "write_breadth_html",
]


# ---------------------------------------------------------------------------
# Constants — palette + chart geometry
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_DAYS = 504  # ~2 trading years
CHART_W = 800
CHART_H = 200
CHART_PAD_X = 40
CHART_PAD_Y = 16

# breadth.app-aligned bull/bear pair. Same shade family as the ETF dashboard's
# `--pos` / `--neg` so the /trading/ umbrella reads consistent at a glance.
BULL_GREEN = "#1b9e3a"
BEAR_RED = "#c92a2a"
NEUTRAL_BLUE = "#2b6cb0"   # for the 50d MA line (between bull / bear)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_breadth_html(
    *,
    breadth: pd.DataFrame,
    asof: date,
    rendered_at_pt: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> str:
    """Render the market-breadth dashboard to an HTML string.

    Pure function — no I/O, no DB, no wall-clock reads.

    Parameters
    ----------
    breadth:
        Long-format breadth DataFrame. Must include columns ``asof_date``
        (date or string), ``indicator`` (string), ``value`` (float).
    asof:
        Display + filter date. The header snapshot uses ``asof``'s row
        (or the latest row ≤ asof if ``asof`` itself is absent).
    rendered_at_pt:
        Wall-clock string from the caller (e.g. ``"12:40"``). Kept out
        of the renderer's internal state so output is byte-deterministic.
    window_days:
        Trailing-window cap for charts that benefit from the recency
        slice. Default 504 (~2y). McClellan-summation and A/D-cumulative
        charts ALSO render in a hidden "all" SVG; the radio toggle flips
        which is visible via pure CSS.
    """
    asof_str = asof.isoformat()
    wide = _to_wide(breadth)
    if wide.empty:
        return _render_template(
            asof_str=asof_str,
            rendered_at_pt=rendered_at_pt,
            header=_empty_header(),
            charts={},
            has_data=False,
        )

    # Slice trailing N for default-view charts.
    short = wide.tail(window_days)

    # Latest-day header snapshot.
    header = _build_header(wide, asof_str)

    # Build chart SVG blocks. Each chart-N has TWO renders for the toggle:
    # the default (last 2y) and the "all" variant. Pure CSS swaps visibility.
    charts = {
        "pct_above_ma": _chart_pct_above_ma(short),
        "new_highs_lows": _chart_new_highs_lows(short),
        "ad_cumulative_2y": _chart_single_line(
            short, "ad_cumulative", "chart-ad-cumulative", stroke=BULL_GREEN
        ),
        "ad_cumulative_all": _chart_single_line(
            wide, "ad_cumulative", "chart-alt-ad-cumulative-all", stroke=BULL_GREEN
        ),
        "mcclellan_2y": _chart_mcclellan(short, slug="chart-mcclellan"),
        "mcclellan_all": _chart_mcclellan(wide, slug="chart-alt-mcclellan-all"),
    }

    return _render_template(
        asof_str=asof_str,
        rendered_at_pt=rendered_at_pt,
        header=header,
        charts=charts,
        has_data=True,
    )


def write_breadth_html(
    *,
    breadth_path: str | Path,
    output_path: str | Path,
    asof: date,
    rendered_at_pt: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> Path:
    """Convenience wrapper: load parquet, render, atomic-write output."""
    breadth = pd.read_parquet(breadth_path)
    html = render_breadth_html(
        breadth=breadth,
        asof=asof,
        rendered_at_pt=rendered_at_pt,
        window_days=window_days,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, out)
    return out


# ---------------------------------------------------------------------------
# Long → wide pivot
# ---------------------------------------------------------------------------


def _to_wide(breadth: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long parquet into (asof_date × indicator → value).

    Index is `asof_date` (as ISO strings for stable byte ordering); columns
    are indicator names. Missing combos remain NaN.
    """
    if breadth.empty:
        return pd.DataFrame()
    df = breadth.copy()
    # Normalize asof_date to ISO strings so the pivot index sorts stably.
    if pd.api.types.is_datetime64_any_dtype(df["asof_date"]):
        df["asof_date"] = df["asof_date"].dt.strftime("%Y-%m-%d")
    else:
        df["asof_date"] = df["asof_date"].astype(str)
    wide = df.pivot_table(
        index="asof_date", columns="indicator", values="value", aggfunc="first"
    )
    return wide.sort_index()


# ---------------------------------------------------------------------------
# Header snapshot
# ---------------------------------------------------------------------------


def _build_header(wide: pd.DataFrame, asof_str: str) -> dict:
    """Pull the latest row for the header.

    Picks ``asof_str`` if present, else the last row ≤ asof_str. Empty
    fallback returns ``_empty_header()``.
    """
    candidate = wide.loc[wide.index <= asof_str]
    if candidate.empty:
        return _empty_header()
    row = candidate.iloc[-1]
    return {
        "pct_20": _safe_int(row.get("pct_above_ma_20")),
        "pct_200": _safe_int(row.get("pct_above_ma_200")),
        "new_high": _safe_int(row.get("new_52w_high")),
        "new_low": _safe_int(row.get("new_52w_low")),
        "ad": _safe_int(row.get("ad_diff")),
        "ad_signed": _signed(row.get("ad_diff")),
        "pct_20_class": _bull_bear_class(row.get("pct_above_ma_20"), 50.0),
        "pct_200_class": _bull_bear_class(row.get("pct_above_ma_200"), 50.0),
        "ad_class": _bull_bear_class(row.get("ad_diff"), 0.0),
    }


def _empty_header() -> dict:
    return {
        "pct_20": 0,
        "pct_200": 0,
        "new_high": 0,
        "new_low": 0,
        "ad": 0,
        "ad_signed": "+0",
        "pct_20_class": "neutral",
        "pct_200_class": "neutral",
        "ad_class": "neutral",
    }


def _safe_int(value) -> int:
    if value is None or _is_nan(value):
        return 0
    return int(round(float(value)))


def _signed(value) -> str:
    if value is None or _is_nan(value):
        return "+0"
    v = int(round(float(value)))
    return f"{v:+d}"


def _bull_bear_class(value, threshold: float) -> str:
    if value is None or _is_nan(value):
        return "neutral"
    v = float(value)
    if v > threshold:
        return "bull"
    if v < threshold:
        return "bear"
    return "neutral"


def _is_nan(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# SVG chart builders
# ---------------------------------------------------------------------------


def _scale_x(i: int, n: int) -> float:
    if n <= 1:
        return CHART_PAD_X
    inner = CHART_W - 2 * CHART_PAD_X
    return CHART_PAD_X + (i / (n - 1)) * inner


def _scale_y(v: float, vmin: float, vmax: float) -> float:
    inner = CHART_H - 2 * CHART_PAD_Y
    if vmax <= vmin:
        return CHART_PAD_Y + inner / 2.0
    return CHART_PAD_Y + (1.0 - (v - vmin) / (vmax - vmin)) * inner


def _line_path(series: pd.Series, vmin: float, vmax: float) -> str:
    """Build a single `M x,y L x,y …` path string from ``series``.

    Skips NaN points: the path stays continuous through the next valid
    point (SVG's path "M…L…" naturally handles this by simply not emitting
    a coordinate for the missing day).
    """
    values = series.dropna().tolist()
    n = len(values)
    if n < 2:
        # Flat midline so the test counter still finds a `<path d=…>`.
        mid_y = CHART_H / 2.0
        return f"M{CHART_PAD_X:.2f},{mid_y:.2f} L{CHART_W - CHART_PAD_X:.2f},{mid_y:.2f}"
    pts: list[str] = []
    for i, v in enumerate(values):
        x = _scale_x(i, n)
        y = _scale_y(float(v), vmin, vmax)
        pts.append(f"{x:.2f},{y:.2f}")
    return "M" + " L".join(pts)


def _axes_g() -> str:
    """A tiny baseline + frame group shared by all 4 charts.

    Drawn first so paths overlay. Geometry only — colors come from the
    template's `--axis` CSS var.
    """
    x0, x1 = CHART_PAD_X, CHART_W - CHART_PAD_X
    y0, y1 = CHART_PAD_Y, CHART_H - CHART_PAD_Y
    return (
        f'<g class="axes">'
        f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" />'
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" />'
        f"</g>"
    )


def _chart_pct_above_ma(wide: pd.DataFrame) -> str:
    """3-line chart: pct_above_ma_20 (bull-green), 50 (neutral-blue), 200 (bear-red).

    Color choice — the three lines map to "fast / medium / slow" lookbacks
    rather than bullish / bearish state (state coloring happens in the
    header snapshot). Using BULL_GREEN / NEUTRAL_BLUE / BEAR_RED makes the
    legend self-explanatory at a glance.
    """
    cols = ["pct_above_ma_20", "pct_above_ma_50", "pct_above_ma_200"]
    series = {c: wide[c] if c in wide.columns else pd.Series(dtype=float) for c in cols}
    finite_values: list[float] = []
    for s in series.values():
        finite_values.extend([float(v) for v in s.dropna()])
    if finite_values:
        vmin = min(finite_values)
        vmax = max(finite_values)
        # Pin to the percentage range so the chart's vertical scale is stable
        # across days; otherwise the auto-fit would zoom in on a flat day.
        vmin = min(vmin, 0.0)
        vmax = max(vmax, 100.0)
    else:
        vmin, vmax = 0.0, 100.0

    paths = []
    for col, stroke in zip(
        cols, [BULL_GREEN, NEUTRAL_BLUE, BEAR_RED]
    ):
        d = _line_path(series[col], vmin, vmax)
        paths.append(
            f'<path class="line line-{col}" stroke="{stroke}" d="{d}" />'
        )
    return (
        f'<svg class="chart-pct-above-ma" viewBox="0 0 {CHART_W} {CHART_H}" '
        f'width="100%" preserveAspectRatio="none" role="img" '
        f'aria-label="% of S&amp;P 500 above moving averages">'
        f"{_axes_g()}"
        f'{"".join(paths)}'
        f"</svg>"
    )


def _chart_new_highs_lows(wide: pd.DataFrame) -> str:
    """Paired lines: new 52w highs (bull-green) vs new 52w lows (bear-red)."""
    highs = wide["new_52w_high"] if "new_52w_high" in wide.columns else pd.Series(dtype=float)
    lows = wide["new_52w_low"] if "new_52w_low" in wide.columns else pd.Series(dtype=float)
    finite = [float(v) for v in highs.dropna()] + [float(v) for v in lows.dropna()]
    if finite:
        vmin = min(0.0, min(finite))
        vmax = max(finite)
        if vmax == vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0
    high_path = _line_path(highs, vmin, vmax)
    low_path = _line_path(lows, vmin, vmax)
    return (
        f'<svg class="chart-new-highs-lows" viewBox="0 0 {CHART_W} {CHART_H}" '
        f'width="100%" preserveAspectRatio="none" role="img" '
        f'aria-label="New 52-week highs vs lows">'
        f"{_axes_g()}"
        f'<path class="line line-new-highs" stroke="{BULL_GREEN}" d="{high_path}" />'
        f'<path class="line line-new-lows" stroke="{BEAR_RED}" d="{low_path}" />'
        f"</svg>"
    )


def _chart_single_line(wide: pd.DataFrame, col: str, slug: str, stroke: str) -> str:
    """Single-line chart (used for both the 2y + the "all" toggle variant)."""
    s = wide[col] if col in wide.columns else pd.Series(dtype=float)
    finite = [float(v) for v in s.dropna()]
    if finite:
        vmin, vmax = min(finite), max(finite)
        if vmax == vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0
    d = _line_path(s, vmin, vmax)
    return (
        f'<svg class="{slug}" viewBox="0 0 {CHART_W} {CHART_H}" '
        f'width="100%" preserveAspectRatio="none" role="img" '
        f'aria-label="{col.replace("_", " ")}">'
        f"{_axes_g()}"
        f'<path class="line line-{col}" stroke="{stroke}" d="{d}" />'
        f"</svg>"
    )


def _chart_mcclellan(wide: pd.DataFrame, slug: str) -> str:
    """McClellan oscillator (foreground) + summation (background, light)."""
    osc = (
        wide["mcclellan_oscillator"]
        if "mcclellan_oscillator" in wide.columns
        else pd.Series(dtype=float)
    )
    summ = (
        wide["mcclellan_summation"]
        if "mcclellan_summation" in wide.columns
        else pd.Series(dtype=float)
    )

    osc_finite = [float(v) for v in osc.dropna()]
    if osc_finite:
        osc_vmin, osc_vmax = min(osc_finite), max(osc_finite)
        if osc_vmax == osc_vmin:
            osc_vmax = osc_vmin + 1.0
    else:
        osc_vmin, osc_vmax = -1.0, 1.0

    summ_finite = [float(v) for v in summ.dropna()]
    if summ_finite:
        summ_vmin, summ_vmax = min(summ_finite), max(summ_finite)
        if summ_vmax == summ_vmin:
            summ_vmax = summ_vmin + 1.0
    else:
        summ_vmin, summ_vmax = -1.0, 1.0

    osc_path = _line_path(osc, osc_vmin, osc_vmax)
    summ_path = _line_path(summ, summ_vmin, summ_vmax)
    # Oscillator stroke flips bull/bear by the LATEST value (a state read,
    # not a per-segment color since a single <path> can't gradient-stripe in
    # plain SVG). The summation line stays a calm neutral so the eye keeps
    # both signals.
    last_osc = osc_finite[-1] if osc_finite else 0.0
    osc_stroke = BULL_GREEN if last_osc > 0 else BEAR_RED
    return (
        f'<svg class="{slug}" viewBox="0 0 {CHART_W} {CHART_H}" '
        f'width="100%" preserveAspectRatio="none" role="img" '
        f'aria-label="McClellan oscillator and summation">'
        f"{_axes_g()}"
        f'<path class="line line-summation" stroke="{NEUTRAL_BLUE}" d="{summ_path}" opacity="0.5" />'
        f'<path class="line line-oscillator" stroke="{osc_stroke}" d="{osc_path}" />'
        f"</svg>"
    )


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>S&amp;P 500 Market Breadth — {{ asof_str }}</title>
<style>
:root {
  --bg-page: #ffffff;
  --bg-elevated: #ffffff;
  --fg: #1f2933;
  --mute: #52606d;
  --line: #e4e7eb;
  --bull: #1b9e3a;
  --bear: #c92a2a;
  --axis: #cbd2d9;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg-page: #0d1018;
    --bg-elevated: #161a26;
    --fg: #d2d6df;
    --mute: #8a93a6;
    --line: #2a3040;
    --bull: #2bd86d;
    --bear: #ff6e6e;
    --axis: #3a4156;
    color-scheme: dark;
  }
}
body {
  font-family: 'Atkinson Hyperlegible', -apple-system, BlinkMacSystemFont,
               'Segoe UI', system-ui, Roboto, 'Helvetica Neue', Arial, sans-serif;
  margin: 24px; color: var(--fg); background: var(--bg-page);
  font-size: 14px; max-width: 920px;
}
h1 { font-size: 1.4rem; margin: 0 0 4px; color: var(--fg); }
.subtitle { color: var(--mute); margin: 0 0 16px; font-size: 0.85rem; }
.snapshot {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px 24px; margin: 12px 0 24px;
  padding: 12px 16px; border: 1px solid var(--line); border-radius: 6px;
  background: var(--bg-elevated);
}
.snapshot .label { color: var(--mute); font-size: 0.8rem; }
.snapshot .value { font-size: 1.5rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.bull { color: var(--bull); }
.bear { color: var(--bear); }
.neutral { color: var(--fg); }
.chart-row { margin: 16px 0 24px; }
.chart-row h2 { font-size: 1rem; margin: 0 0 4px; color: var(--fg); }
.chart-row .legend { color: var(--mute); font-size: 0.8rem; margin: 0 0 6px; }
svg[class^="chart-"], svg[class*=" chart-"] {
  display: block; width: 100%; height: auto;
  fill: none; stroke-width: 1.5;
  background: var(--bg-elevated); border: 1px solid var(--line); border-radius: 4px;
}
svg .axes line { stroke: var(--axis); stroke-width: 1; }
/* Hide the toggle radios visually but keep them focusable. The radios live as
   siblings BEFORE `.toggle-row` so the `~` general-sibling selector flips
   visibility of the two SVG variants. */
input[type=radio].window-toggle {
  position: absolute; opacity: 0; pointer-events: none; width: 0; height: 0;
}
.toggle-row { display: flex; gap: 8px; margin: 8px 0; }
.toggle-row label {
  font-size: 0.8rem; color: var(--mute); cursor: pointer;
  padding: 2px 8px; border: 1px solid var(--line); border-radius: 4px;
  user-select: none;
}
/* A/D toggle — default-visible chart hides when "all" is selected; the
   alt-prefixed SVG only appears under :checked, so the default state holds
   exactly 4 chart-* SVGs in DOM order (matches design §5 + acceptance §2). */
#window-ad-2y:checked  ~ .toggle-row label[for=window-ad-2y]  { color: var(--bull); border-color: var(--bull); }
#window-ad-all:checked ~ .toggle-row label[for=window-ad-all] { color: var(--bull); border-color: var(--bull); }
#window-ad-2y:checked  ~ svg.chart-ad-cumulative          { display: block; }
#window-ad-2y:checked  ~ svg.chart-alt-ad-cumulative-all  { display: none; }
#window-ad-all:checked ~ svg.chart-ad-cumulative          { display: none; }
#window-ad-all:checked ~ svg.chart-alt-ad-cumulative-all  { display: block; }
/* McClellan toggle */
#window-mc-2y:checked  ~ .toggle-row label[for=window-mc-2y]  { color: var(--bull); border-color: var(--bull); }
#window-mc-all:checked ~ .toggle-row label[for=window-mc-all] { color: var(--bull); border-color: var(--bull); }
#window-mc-2y:checked  ~ svg.chart-mcclellan          { display: block; }
#window-mc-2y:checked  ~ svg.chart-alt-mcclellan-all  { display: none; }
#window-mc-all:checked ~ svg.chart-mcclellan          { display: none; }
#window-mc-all:checked ~ svg.chart-alt-mcclellan-all  { display: block; }
.disclaimer {
  margin-top: 32px; padding-top: 12px; border-top: 1px solid var(--line);
  color: var(--mute); font-size: 0.8rem; line-height: 1.5;
}
</style>
</head>
<body>
<h1>S&amp;P 500 Market Breadth</h1>
<p class="subtitle">
  Last updated: {{ asof_str }} {{ rendered_at_pt }} PT
  &middot; Universe: current S&amp;P 500 constituents (survivorship-bias applied retroactively — see footer)
</p>

{% if has_data %}
<section class="snapshot" aria-label="Today's snapshot">
  <div>
    <div class="label">% above 20d MA</div>
    <div class="value {{ header.pct_20_class }}">{{ header.pct_20 }}%</div>
  </div>
  <div>
    <div class="label">% above 200d MA</div>
    <div class="value {{ header.pct_200_class }}">{{ header.pct_200 }}%</div>
  </div>
  <div>
    <div class="label">New 52w highs / lows</div>
    <div class="value"><span class="bull">{{ header.new_high }}</span> / <span class="bear">{{ header.new_low }}</span></div>
  </div>
  <div>
    <div class="label">A/D today</div>
    <div class="value {{ header.ad_class }}">{{ header.ad_signed }}</div>
  </div>
</section>

<section class="chart-row">
  <h2>% above moving average</h2>
  <p class="legend">
    <span style="color: #1b9e3a">20d</span> &middot;
    <span style="color: #2b6cb0">50d</span> &middot;
    <span style="color: #c92a2a">200d</span> &middot; trailing 2y
  </p>
  {{ charts.pct_above_ma }}
</section>

<section class="chart-row">
  <h2>New 52-week highs vs lows</h2>
  <p class="legend">
    <span class="bull">new highs</span> &middot;
    <span class="bear">new lows</span> &middot; trailing 2y
  </p>
  {{ charts.new_highs_lows }}
</section>

<input type="radio" name="window-ad" id="window-ad-2y" class="window-toggle" checked>
<input type="radio" name="window-ad" id="window-ad-all" class="window-toggle">
<section class="chart-row">
  <h2>Advance / Decline (cumulative)</h2>
  <p class="legend">cumulative since 2020-01-01</p>
  <div class="toggle-row">
    <label for="window-ad-2y">2y</label>
    <label for="window-ad-all">all</label>
  </div>
  {{ charts.ad_cumulative_2y }}
  {{ charts.ad_cumulative_all }}
</section>

<input type="radio" name="window-mc" id="window-mc-2y" class="window-toggle" checked>
<input type="radio" name="window-mc" id="window-mc-all" class="window-toggle">
<section class="chart-row">
  <h2>McClellan oscillator + summation</h2>
  <p class="legend">
    oscillator (foreground, green &gt; 0 / red &lt; 0) &middot; summation (background, blue)
  </p>
  <div class="toggle-row">
    <label for="window-mc-2y">2y</label>
    <label for="window-mc-all">all</label>
  </div>
  {{ charts.mcclellan_2y }}
  {{ charts.mcclellan_all }}
</section>
{% else %}
<p class="subtitle">No breadth data available for this asof_date.</p>
{% endif %}

<p class="disclaimer">
  Universe = current S&amp;P 500 constituents applied retroactively. Historical view does not adjust for past index add/drop events; breadth may appear slightly stronger than reality. v1 will add point-in-time membership.
</p>
</body>
</html>
"""


_ENV = Environment(
    autoescape=select_autoescape(default=True, default_for_string=True),
    undefined=StrictUndefined,
    trim_blocks=False,
    lstrip_blocks=False,
    keep_trailing_newline=True,
)


def _render_template(
    *,
    asof_str: str,
    rendered_at_pt: str,
    header: dict,
    charts: dict,
    has_data: bool,
) -> str:
    template = _ENV.from_string(_TEMPLATE)
    # Pre-build SVG strings flow through as Markup so autoescape doesn't
    # double-escape `<svg>` / `<path>`. The charts dict is internal — every
    # value originates from `_chart_*` helpers which assemble HTML from
    # numeric data only.
    safe_charts = {k: Markup(v) if v else Markup("") for k, v in charts.items()}
    return template.render(
        asof_str=asof_str,
        rendered_at_pt=rendered_at_pt,
        header=header,
        charts=safe_charts,
        has_data=has_data,
    )
