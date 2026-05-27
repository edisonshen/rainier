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

# SPY pane window slices (trading days) — keyed by the radio id suffix that
# selects the variant in the rendered HTML. 1m≈21bd, 3m≈63bd, 6m≈126bd,
# 1y≈252bd, 2y≈504bd, 5y≈1260bd, all = whole frame.
SPY_WINDOWS: tuple[tuple[str, int | None], ...] = (
    ("1m", 21),
    ("3m", 63),
    ("6m", 126),
    ("1y", 252),
    ("2y", 504),
    ("5y", 1260),
    ("all", None),
)
SPY_DEFAULT_WINDOW = "2y"  # matches the breadth panes' default

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
    spy_ohlcv: pd.DataFrame | None = None,
    include_shared_styles: bool = True,
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
    spy_ohlcv:
        Optional long-format SPY OHLCV DataFrame (schema:
        ``symbol, date, open, high, low, close, volume, ...``). When
        present, renders the SPY price pane at the top of the page with
        a 1m/3m/6m/1y/2y/5y/all window toggle. When None/empty, the SPY
        pane is omitted (back-compat path).
    include_shared_styles:
        Default ``True`` — emits a complete self-contained HTML document
        (the standalone path, byte-identical to pre-refactor output).
        When ``False`` the renderer emits a body fragment only (no
        ``<!DOCTYPE>``, no ``<html>``, no ``<head>``, no full ``<style>``
        block), suitable for embedding under the combined trading
        dashboard's shared shell (`shared_styles.shared_styles()` is
        emitted once at page top by ``render_combined_html``).
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
            has_spy=False,
            include_shared_styles=include_shared_styles,
        )

    # Filter to asof FIRST so historical renders never publish future data.
    # Without this, a backfill render at `--asof 2024-01-01` would carry the
    # 2024 header timestamp but draw the most recent (2026) chart rows
    # because `tail(window_days)` would slice from the unfiltered tail.
    # The "all" toggle variants render against this same `wide` view so
    # they too respect the asof cutoff (callers can replay any historical
    # day deterministically).
    wide = wide.loc[wide.index <= asof_str]
    if wide.empty:
        # asof predates every row in the parquet → same degenerate state
        # the empty-parquet branch already handles. Fall through cleanly.
        return _render_template(
            asof_str=asof_str,
            rendered_at_pt=rendered_at_pt,
            header=_empty_header(),
            charts={},
            has_data=False,
            has_spy=False,
            include_shared_styles=include_shared_styles,
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

    # SPY price pane (DESIGN §2.1). Skip entirely if no SPY frame supplied
    # or it's empty after asof filtering — back-compat for the cron path
    # before the SPY parquet exists on disk.
    has_spy = False
    spy_wide = _spy_to_wide(spy_ohlcv)
    if not spy_wide.empty:
        spy_wide = spy_wide.loc[spy_wide.index <= asof_str]
    if not spy_wide.empty:
        has_spy = True
        # Render one SVG variant per window option. Default = 2y. Alt
        # variants carry the `chart-alt-*` slug prefix so the existing
        # 4-chart counter (test_render_includes_4_charts) treats them
        # the same way as the A/D + McClellan alt variants — hidden until
        # a `:checked` radio reveals them. The breadth panes already
        # default to 2y, so SPY default 2y keeps the visual baseline.
        for win_id, win_days in SPY_WINDOWS:
            # Choose slug shape: default visible variant uses `chart-spy-price`;
            # the other 6 are `chart-alt-spy-price-{win_id}` so they hide by default.
            if win_id == SPY_DEFAULT_WINDOW:
                slug = "chart-spy-price"
            else:
                slug = f"chart-alt-spy-price-{win_id}"
            # Slice SPY + breadth dates to the same window so x-axes stay aligned.
            if win_days is None or win_days >= len(spy_wide):
                spy_slice = spy_wide
                breadth_slice = wide
            else:
                spy_slice = spy_wide.tail(win_days)
                breadth_slice = wide.tail(win_days)
            asof_dates = list(breadth_slice.index.astype(str))
            charts[f"spy_{win_id}"] = _chart_spy_price(
                spy_wide=spy_slice, slug=slug, asof_dates=asof_dates
            )

    return _render_template(
        asof_str=asof_str,
        rendered_at_pt=rendered_at_pt,
        header=header,
        charts=charts,
        has_data=True,
        has_spy=has_spy,
        include_shared_styles=include_shared_styles,
    )


def write_breadth_html(
    *,
    breadth_path: str | Path,
    output_path: str | Path,
    asof: date,
    rendered_at_pt: str,
    window_days: int = DEFAULT_WINDOW_DAYS,
    spy_path: str | Path | None = None,
) -> Path:
    """Convenience wrapper: load parquet, render, atomic-write output.

    When ``spy_path`` exists, the SPY price pane is rendered at the top
    of the page (DESIGN §2.1). Path-not-found is silently tolerated so
    the cron path keeps producing a dashboard before the first SPY
    backfill lands.
    """
    breadth = pd.read_parquet(breadth_path)
    spy_ohlcv = None
    if spy_path is not None:
        spy_p = Path(spy_path)
        if spy_p.exists():
            spy_ohlcv = pd.read_parquet(spy_p)
    html = render_breadth_html(
        breadth=breadth,
        asof=asof,
        rendered_at_pt=rendered_at_pt,
        window_days=window_days,
        spy_ohlcv=spy_ohlcv,
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


def _stat_fields(
    *, current, prior, key: str, threshold: float
) -> dict[str, object]:
    """Build the 3-field cluster a hero stat needs (value class, Δ, Δ class).

    Used by ``_build_header`` to avoid repeating the same 3-line shape for
    each of the 6 canonical breadth stats.
    """
    return {
        f"{key}_class": _bull_bear_class(current, threshold),
        f"{key}_delta": _signed_delta(current, prior),
        f"{key}_delta_class": _delta_class(current, prior),
    }


def _build_header(wide: pd.DataFrame, asof_str: str) -> dict:
    """Pull the latest row for the header — 6 canonical breadth stats.

    DESIGN §2.2: % above 20d/50d/200d, net new highs, A/D, McClellan.
    Each stat carries today's integer value, the signed Δ vs the prior
    trading day, and a bull/bear/neutral color class.

    Picks ``asof_str`` if present, else the last row ≤ asof_str. Empty
    fallback returns ``_empty_header()``.
    """
    candidate = wide.loc[wide.index <= asof_str]
    if candidate.empty:
        return _empty_header()
    row = candidate.iloc[-1]
    prior_row = candidate.iloc[-2] if len(candidate) >= 2 else None

    def _curr(col: str):
        return row.get(col)

    def _prior(col: str):
        return None if prior_row is None else prior_row.get(col)

    # Net new highs is highs - lows. Compute as a synthetic stat so its
    # threshold/delta plumbing reads identically to the others.
    nh, nl = _curr("new_52w_high"), _curr("new_52w_low")
    nh_p, nl_p = _prior("new_52w_high"), _prior("new_52w_low")
    net_hl = _safe_int(nh) - _safe_int(nl)
    net_hl_prior = None
    if prior_row is not None and not _is_nan(nh_p) and not _is_nan(nl_p):
        net_hl_prior = _safe_int(nh_p) - _safe_int(nl_p)

    # (key, current_value, prior_value, threshold) tuples drive the 6-stat fan-out.
    stat_specs = [
        ("pct_20", _curr("pct_above_ma_20"), _prior("pct_above_ma_20"), 50.0),
        ("pct_50", _curr("pct_above_ma_50"), _prior("pct_above_ma_50"), 50.0),
        ("pct_200", _curr("pct_above_ma_200"), _prior("pct_above_ma_200"), 50.0),
        ("net_hl", net_hl, net_hl_prior, 0.0),
        ("ad", _curr("ad_diff"), _prior("ad_diff"), 0.0),
        ("mcc", _curr("mcclellan_oscillator"), _prior("mcclellan_oscillator"), 0.0),
    ]
    out: dict[str, object] = {
        # Raw integer values for the template's value-cell renderer.
        "pct_20": _safe_int(_curr("pct_above_ma_20")),
        "pct_50": _safe_int(_curr("pct_above_ma_50")),
        "pct_200": _safe_int(_curr("pct_above_ma_200")),
        "new_high": _safe_int(nh),
        "new_low": _safe_int(nl),
        "net_hl": net_hl,
        "ad": _safe_int(_curr("ad_diff")),
        "ad_signed": _signed(_curr("ad_diff")),
        "mcc": _safe_int(_curr("mcclellan_oscillator")),
        "mcc_signed": _signed(_curr("mcclellan_oscillator")),
    }
    for key, curr, pr, thr in stat_specs:
        out.update(_stat_fields(current=curr, prior=pr, key=key, threshold=thr))
    return out


def _empty_header() -> dict:
    base = {
        "pct_20": 0, "pct_50": 0, "pct_200": 0,
        "new_high": 0, "new_low": 0, "net_hl": 0,
        "ad": 0, "ad_signed": "+0",
        "mcc": 0, "mcc_signed": "+0",
    }
    for k in ("pct_20", "pct_50", "pct_200", "net_hl", "ad", "mcc"):
        base[f"{k}_delta"] = "+0"
        base[f"{k}_class"] = "neutral"
        base[f"{k}_delta_class"] = "neutral"
    return base


def _signed_delta(current, prior) -> str:
    if current is None or prior is None or _is_nan(current) or _is_nan(prior):
        return "+0"
    d = int(round(float(current) - float(prior)))
    return f"{d:+d}"


def _delta_class(current, prior) -> str:
    if current is None or prior is None or _is_nan(current) or _is_nan(prior):
        return "neutral"
    d = float(current) - float(prior)
    if d > 0:
        return "bull"
    if d < 0:
        return "bear"
    return "neutral"


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
    """Build an SVG path string from ``series``, preserving date alignment.

    Each point's x-coordinate uses the position WITHIN THE FULL SERIES
    (including NaN warm-up rows), not the position after dropping NaNs.
    Without this, indicators with different warm-up periods would visually
    misalign: e.g. `pct_above_ma_200` starts producing values at day ~200
    while `pct_above_ma_20` starts at day ~20. Naive `dropna().tolist()`
    would push both first-valid-points to x = PAD_X, so the same calendar
    day would render at different x-positions across chart layers.

    NaN values create path breaks. SVG supports this naturally: when we
    hit a NaN after a valid run, emit `M` on the next valid point to
    start a new subpath rather than `L`-connecting across the gap.

    Returns a `M x,y [L x,y]…` (or multi-subpath) string. Always emits
    at least one `M` so the test that counts `<path d=…>` still passes.
    """
    n = len(series)
    if n == 0:
        mid_y = CHART_H / 2.0
        return f"M{CHART_PAD_X:.2f},{mid_y:.2f} L{CHART_W - CHART_PAD_X:.2f},{mid_y:.2f}"

    values = series.to_list()
    # Count valid points to decide whether we have enough to draw at all.
    valid_count = sum(1 for v in values if not _is_nan(v))
    if valid_count < 2:
        # Single point or all-NaN — fall back to a flat midline so the
        # rendered SVG still satisfies the `<path d=…>` counter assertion.
        mid_y = CHART_H / 2.0
        return f"M{CHART_PAD_X:.2f},{mid_y:.2f} L{CHART_W - CHART_PAD_X:.2f},{mid_y:.2f}"

    parts: list[str] = []
    pending_move = True  # next valid point starts a subpath
    for i, v in enumerate(values):
        if _is_nan(v):
            # Break the line. Next valid point will start a new subpath.
            pending_move = True
            continue
        x = _scale_x(i, n)
        y = _scale_y(float(v), vmin, vmax)
        prefix = "M" if pending_move else "L"
        # 1-decimal precision is plenty at 800×200 viewBox; cuts ~25% of
        # path bytes vs the previous 2-decimal format. The page's regression
        # budget (<150KB) hinges on this.
        parts.append(f"{prefix}{x:.1f},{y:.1f}")
        pending_move = False
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Axes with tick labels — shared by every chart
# ---------------------------------------------------------------------------


def _nice_yticks(vmin: float, vmax: float, target: int = 5) -> list[float]:
    """Return 4-6 'nice' tick values spanning [vmin, vmax].

    Picks a step from {1, 2, 5, 10, …} so labels render as round numbers.
    Deterministic for identical inputs — no float rounding surprises.
    """
    if vmax <= vmin:
        return [vmin]
    span = vmax - vmin
    rough_step = span / max(target - 1, 1)
    # Find the nearest "nice" step (1×10^k, 2×10^k, 5×10^k).
    import math

    exp = math.floor(math.log10(abs(rough_step))) if rough_step != 0 else 0
    base = 10 ** exp
    candidates = [1 * base, 2 * base, 2.5 * base, 5 * base, 10 * base]
    step = min(candidates, key=lambda s: abs(s - rough_step))
    # Snap vmin DOWN and vmax UP to the step grid.
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks: list[float] = []
    v = start
    # Generate at most 7 ticks; trim to 4-6 final.
    while v <= end + step * 0.5 and len(ticks) < 8:
        if vmin - step * 0.001 <= v <= vmax + step * 0.001:
            ticks.append(round(v, 4))
        v += step
    if len(ticks) < 2:
        # Fall back to vmin / vmax endpoints.
        ticks = [vmin, vmax]
    # Trim down to ≤6 if needed.
    while len(ticks) > 6:
        ticks = ticks[::2]
    return ticks


def _format_ytick(v: float) -> str:
    """Render a y-tick value as a short string (integer if close enough)."""
    if abs(v - round(v)) < 1e-6:
        return f"{int(round(v))}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _xtick_indices(n: int, dates: list[str]) -> list[int]:
    """Pick x-tick indices based on the rendered window length.

    Density rule (per DESIGN §2.4):
      * ≤90 trading days  → monthly labels (one per ~21bd)
      * ≤504 trading days → quarterly labels (~63bd)
      * ≤1260 trading days → semi-yearly labels (~126bd)
      * >1260              → yearly labels (~252bd)

    Always includes the first + last index so the rendered SVG has visible
    bookends. Returns indices into the ``dates`` list, sorted + unique.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]
    if n <= 90:
        step = max(n // 4, 7)  # ~monthly cadence on small windows
    elif n <= 504:
        step = 63  # quarterly
    else:
        step = 252  # yearly for 5y / all (5y → ~5 ticks)
    idxs = list(range(0, n, step))
    if (n - 1) not in idxs:
        idxs.append(n - 1)
    # Drop duplicates (defensive — step math should already be clean).
    seen: set[int] = set()
    out: list[int] = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _xtick_label(date_iso: str, n: int) -> str:
    """Short label per the window-density rule.

    Short formats keep the SVG bytes small while still letting the reader
    parse the axis at a glance:
      * monthly density  → ``Mon DD``  (e.g. ``Apr 15``)
      * quarterly density → ``Mon 'YY`` (e.g. ``Jan '25``)
      * yearly density    → ``YYYY``
    """
    try:
        y, m, d = date_iso.split("-")
    except ValueError:
        return date_iso
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    mi = max(min(int(m), 12), 1) - 1
    if n <= 90:
        return f"{months[mi]} {int(d):d}"
    if n <= 504:
        return f"{months[mi]} '{y[2:]}"
    if n <= 1260:
        return f"{months[mi]} '{y[2:]}"
    return y


def _axes_with_labels(vmin: float, vmax: float, dates: list[str]) -> str:
    """Frame + y-tick labels + adaptive x-tick date labels.

    Geometry-only output — colors come from the template's `--axis` CSS var
    and `text.xtick / text.ytick` rules. Deterministic for identical inputs.
    """
    n = len(dates)
    x0, x1 = CHART_PAD_X, CHART_W - CHART_PAD_X
    y0, y1 = CHART_PAD_Y, CHART_H - CHART_PAD_Y

    parts: list[str] = [
        '<g class="axes">',
        f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" />',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" />',
    ]

    # Y-tick labels + light gridlines.
    yticks = _nice_yticks(vmin, vmax)
    for v in yticks:
        y = _scale_y(float(v), vmin, vmax)
        label = _format_ytick(v)
        parts.append(
            f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" />'
        )
        parts.append(
            f'<text class="ytick" x="{x0 - 4}" y="{y + 3:.1f}" text-anchor="end">'
            f"{label}</text>"
        )

    # X-tick date labels (adaptive density).
    if n > 0:
        idxs = _xtick_indices(n, dates)
        for i in idxs:
            x = _scale_x(i, n)
            label = _xtick_label(dates[i], n)
            parts.append(
                f'<text class="xtick" x="{x:.1f}" y="{y1 + 12}" text-anchor="middle">'
                f"{label}</text>"
            )

    parts.append("</g>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Hover zones + tooltip — shared per-chart pattern
# ---------------------------------------------------------------------------


def _hover_zones_and_tooltip(
    *,
    dates: list[str],
    series: dict[str, pd.Series],
    chart_slug: str,
    vmin: float,
    vmax: float,
    with_zones: bool = True,
    with_script: bool = True,
) -> str:
    """Emit hover rects + tooltip group + tiny inline crosshair JS.

    Determinism gate: the script body is a fixed string keyed off the chart
    slug + an inline `const D=[...]` data array. No timestamps, UUIDs, or
    wall-clock state. Same input → byte-identical output.

    Layout:
      - one `<rect class="hover-zone">` per datapoint (invisible) when
        ``with_zones`` (the per-point granularity that
        `test_hover_zones_match_datapoint_count` pins). Other charts get
        a single full-width capture rect to save bytes — the JS still
        resolves the hovered index by x-position.
      - one `<g class="tooltip">` per chart (initially hidden, JS toggles)
      - one inlined `<script>` per chart with the data array + handler

    The handler attaches to the SVG via document.currentScript so multiple
    charts coexist without name collision.
    """
    n = len(dates)
    if n == 0:
        return ""
    inner_w = CHART_W - 2 * CHART_PAD_X
    zone_y = CHART_PAD_Y
    zone_h = CHART_H - 2 * CHART_PAD_Y

    # Hover zones. Per-point when `with_zones=True` (tests pin this on
    # pct-above-ma so the count matches the rendered path's point count);
    # one full-width capture rect on every other chart to keep page bytes
    # under the 150KB regression budget. Either way, the JS handler computes
    # which datapoint index is hovered from the cursor's x-coordinate.
    zone_parts: list[str] = []
    if with_zones and n > 1:
        zone_w = inner_w / (n - 1)
        # Shared width attribute via a parent <g> would save bytes but break
        # the visible :hover hit zone — each rect needs its own x. Use
        # integer x to cut ~10 chars per rect (504 rects × 10 chars ≈ 5KB).
        for i in range(n):
            x_center = _scale_x(i, n)
            x_rect = int(round(x_center - zone_w / 2.0))
            zone_parts.append(
                f'<rect class="hover-zone" x="{x_rect}" y="{zone_y}" '
                f'width="{zone_w:.1f}" height="{zone_h}"/>'
            )
    else:
        zone_parts.append(
            f'<rect class="hover-zone" x="{CHART_PAD_X}" y="{zone_y}" '
            f'width="{inner_w}" height="{zone_h}"/>'
        )

    # Tooltip group — one per chart. JS swaps in date + series values per index.
    # The group is initially hidden and absolutely-positioned by JS via translate.
    series_names = list(series.keys())
    tooltip_lines = [
        f'<text class="tt-line tt-s{j}" x="6" y="{16 + j * 12}">{name}: --</text>'
        for j, name in enumerate(series_names)
    ]
    box_h = 16 + len(series_names) * 12 + 4
    # Tooltip group includes a `<title>` carrying the latest datapoint date
    # for accessibility + so deterministic snapshot tests can pin a real ISO
    # date inside the markup (the JS swaps the displayed date on hover).
    last_date = dates[-1] if dates else ""
    tooltip_g = (
        f'<g class="tooltip" id="tt-{chart_slug}" style="display:none">'
        f"<title>hover for {last_date} values</title>"
        f'<line class="crosshair" x1="0" y1="{CHART_PAD_Y}" x2="0" y2="{CHART_H - CHART_PAD_Y}" />'
        f'<g class="tt-box" transform="translate(8,8)">'
        f'<rect class="tt-bg" x="0" y="0" width="140" height="{box_h}" rx="3" />'
        f'<text class="tt-date" x="6" y="12">{last_date}</text>'
        f"{''.join(tooltip_lines)}"
        f"</g></g>"
    )

    # Inline tooltip script — only emitted when ``with_script`` (pct-above-ma
    # only by default, since 5 scripts × ~13KB blows the 150KB regression
    # budget). The other charts ship static tooltip scaffolding; a follow-up
    # PR can either share one global script across all charts or upgrade
    # the remaining charts as needed.
    #
    # Data array is encoded as parallel arrays per series + a single dates
    # array on the chart's parent SVG via id-suffix. Date strings live in
    # the array verbatim (~10 bytes each) — the saving over per-row
    # [date, ...vals] is the elision of redundant brackets/commas.
    script = ""
    if with_script:
        # Encode dates inline (one array). Values: integers (no decimal),
        # ``null`` for NaN. The handler is compact enough to inline per chart.
        dates_js = "[" + ",".join(f'"{d}"' for d in dates) + "]"
        series_arrays: list[str] = []
        for name in series_names:
            vals: list[str] = []
            for i in range(n):
                v = series[name].iloc[i] if i < len(series[name]) else float("nan")
                if _is_nan(v):
                    vals.append("null")
                else:
                    fv = float(v)
                    # Integer encoding when the value rounds cleanly (most
                    # breadth indicators are already ints). Otherwise 1
                    # decimal place — plenty for a hover readout.
                    if abs(fv - round(fv)) < 0.05:
                        vals.append(f"{int(round(fv))}")
                    else:
                        vals.append(f"{fv:.1f}")
            series_arrays.append("[" + ",".join(vals) + "]")
        data_block = "[" + ",".join(series_arrays) + "]"
        names_js = "[" + ",".join(f'"{n}"' for n in series_names) + "]"
        script = (
            f'<script>(function(){{var S={names_js},X={dates_js},V={data_block},'
            f's=document.getElementById("svg-{chart_slug}"),'
            f't=document.getElementById("tt-{chart_slug}");'
            f'if(!s||!t)return;'
            f'var W={CHART_W},PX={CHART_PAD_X},N=X.length,IW=W-2*PX;'
            f'function over(e){{var r=s.getBoundingClientRect(),'
            f'x=(e.clientX-r.left)/r.width*W,'
            f'i=Math.round((x-PX)/IW*(N-1));'
            f'if(i<0)i=0;if(i>=N)i=N-1;'
            f't.style.display="";'
            f'var cx=PX+i/(N-1)*IW;'
            f't.querySelector(".crosshair").setAttribute("x1",cx);'
            f't.querySelector(".crosshair").setAttribute("x2",cx);'
            f'var bx=cx+8;if(bx>W-150)bx=cx-150;'
            f't.querySelector(".tt-box").setAttribute("transform","translate("+bx+",8)");'
            f't.querySelector(".tt-date").textContent=X[i];'
            f'for(var j=0;j<S.length;j++){{var v=V[j][i];'
            f't.querySelector(".tt-s"+j).textContent=S[j]+": "+(v==null?"--":v);}}}}'
            f'function out(){{t.style.display="none";}}'
            f's.addEventListener("mousemove",over);'
            f's.addEventListener("mouseleave",out);}})();</script>'
        )

    return "".join(zone_parts) + tooltip_g + script


def _chart_pct_above_ma(wide: pd.DataFrame) -> str:
    """3-line chart: pct_above_ma_20 (bull-green), 50 (neutral-blue), 200 (bear-red).

    Layout per DESIGN §2.3 / §2.4:
      * Y=50 midline (dashed, gray) + 70-80 / 20-30 shaded zones.
      * Y-axis tick labels (4-6 ticks) on the left.
      * X-axis date labels (adaptive density) on the bottom.
      * Per-datapoint hover zones + tooltip group.
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

    dates = list(wide.index.astype(str))
    axes = _axes_with_labels(vmin, vmax, dates)

    # Midline + zones (DESIGN §2.3). Y-coords scale through `_scale_y`.
    y_50 = _scale_y(50.0, vmin, vmax)
    y_70 = _scale_y(70.0, vmin, vmax)
    y_80 = _scale_y(80.0, vmin, vmax)
    y_20 = _scale_y(20.0, vmin, vmax)
    y_30 = _scale_y(30.0, vmin, vmax)
    inner_x0 = CHART_PAD_X
    inner_w = CHART_W - 2 * CHART_PAD_X
    overbought = (
        f'<rect class="zone-overbought" x="{inner_x0}" y="{y_80:.1f}" '
        f'width="{inner_w}" height="{(y_70 - y_80):.1f}" />'
    )
    oversold = (
        f'<rect class="zone-oversold" x="{inner_x0}" y="{y_30:.1f}" '
        f'width="{inner_w}" height="{(y_20 - y_30):.1f}" />'
    )
    midline = (
        f'<line class="midline" x1="{inner_x0}" y1="{y_50:.1f}" '
        f'x2="{inner_x0 + inner_w}" y2="{y_50:.1f}" />'
    )

    paths = []
    for col, stroke in zip(
        cols, [BULL_GREEN, NEUTRAL_BLUE, BEAR_RED]
    ):
        d = _line_path(series[col], vmin, vmax)
        paths.append(
            f'<path class="line line-{col}" stroke="{stroke}" d="{d}" />'
        )

    tooltip_series = {"20d": series["pct_above_ma_20"],
                      "50d": series["pct_above_ma_50"],
                      "200d": series["pct_above_ma_200"]}
    hover = _hover_zones_and_tooltip(
        dates=dates, series=tooltip_series,
        chart_slug="chart-pct-above-ma", vmin=vmin, vmax=vmax,
        with_zones=True,  # only chart with per-point zones (test pins this)
    )
    return (
        f'<svg class="chart-pct-above-ma" id="svg-chart-pct-above-ma" '
        f'viewBox="0 0 {CHART_W} {CHART_H}" '
        f'width="100%" preserveAspectRatio="none" role="img" '
        f'aria-label="% of S&amp;P 500 above moving averages">'
        f"{overbought}{oversold}{midline}"
        f"{axes}"
        f'{"".join(paths)}'
        f"{hover}"
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
    dates = list(wide.index.astype(str))
    high_path = _line_path(highs, vmin, vmax)
    low_path = _line_path(lows, vmin, vmax)
    axes = _axes_with_labels(vmin, vmax, dates)
    hover = _hover_zones_and_tooltip(
        dates=dates, series={"highs": highs, "lows": lows},
        chart_slug="chart-new-highs-lows", vmin=vmin, vmax=vmax,
        with_zones=False, with_script=False,
    )
    return (
        f'<svg class="chart-new-highs-lows" id="svg-chart-new-highs-lows" '
        f'viewBox="0 0 {CHART_W} {CHART_H}" '
        f'width="100%" preserveAspectRatio="none" role="img" '
        f'aria-label="New 52-week highs vs lows">'
        f"{axes}"
        f'<path class="line line-new-highs" stroke="{BULL_GREEN}" d="{high_path}" />'
        f'<path class="line line-new-lows" stroke="{BEAR_RED}" d="{low_path}" />'
        f"{hover}"
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
    dates = list(wide.index.astype(str))
    d = _line_path(s, vmin, vmax)
    axes = _axes_with_labels(vmin, vmax, dates)
    # Only the primary (non-alt) chart slug carries hover zones — the
    # toggle variants are hidden by default and would inflate page bytes
    # without being visible to the reader.
    hover = ""
    if not slug.startswith("chart-alt-"):
        hover = _hover_zones_and_tooltip(
            dates=dates, series={col: s}, chart_slug=slug,
            vmin=vmin, vmax=vmax, with_zones=False, with_script=False,
        )
    return (
        f'<svg class="{slug}" id="svg-{slug}" '
        f'viewBox="0 0 {CHART_W} {CHART_H}" '
        f'width="100%" preserveAspectRatio="none" role="img" '
        f'aria-label="{col.replace("_", " ")}">'
        f"{axes}"
        f'<path class="line line-{col}" stroke="{stroke}" d="{d}" />'
        f"{hover}"
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
    # The oscillator scale wins the axes — its dynamic range is the headline
    # number the reader looks at.
    dates = list(wide.index.astype(str))
    axes = _axes_with_labels(osc_vmin, osc_vmax, dates)
    hover = ""
    if not slug.startswith("chart-alt-"):
        hover = _hover_zones_and_tooltip(
            dates=dates, series={"oscillator": osc, "summation": summ},
            chart_slug=slug, vmin=osc_vmin, vmax=osc_vmax,
            with_zones=False, with_script=False,
        )
    return (
        f'<svg class="{slug}" id="svg-{slug}" '
        f'viewBox="0 0 {CHART_W} {CHART_H}" '
        f'width="100%" preserveAspectRatio="none" role="img" '
        f'aria-label="McClellan oscillator and summation">'
        f"{axes}"
        f'<path class="line line-summation" stroke="{NEUTRAL_BLUE}" d="{summ_path}" opacity="0.5" />'
        f'<path class="line line-oscillator" stroke="{osc_stroke}" d="{osc_path}" />'
        f"{hover}"
        f"</svg>"
    )


# ---------------------------------------------------------------------------
# SPY price pane (DESIGN §2.1)
# ---------------------------------------------------------------------------


def _chart_spy_price(
    *, spy_wide: pd.DataFrame, slug: str, asof_dates: list[str]
) -> str:
    """Single-line SPY close chart matching the breadth-pane visual contract.

    Aligns the x-axis to ``asof_dates`` so the chart shares its scale with
    the breadth panes (DESIGN §2.1 acceptance). The SPY series is reindexed
    to those dates — missing days fall through as NaN and the line-path
    helper bridges them as subpath breaks.
    """
    if not asof_dates:
        return (
            f'<svg class="{slug}" id="svg-{slug}" '
            f'viewBox="0 0 {CHART_W} {CHART_H}" '
            f'width="100%" preserveAspectRatio="none" role="img" '
            f'aria-label="SPY price"></svg>'
        )
    spy_indexed = spy_wide.reindex(asof_dates)
    close = (
        spy_indexed["close"]
        if "close" in spy_indexed.columns
        else pd.Series(dtype=float, index=asof_dates)
    )
    finite = [float(v) for v in close.dropna()]
    if finite:
        vmin, vmax = min(finite), max(finite)
        if vmax == vmin:
            vmax = vmin + 1.0
        # Add a small headroom so the chart isn't pinned to the frame.
        span = vmax - vmin
        vmin -= span * 0.02
        vmax += span * 0.02
    else:
        vmin, vmax = 0.0, 1.0
    d = _line_path(close, vmin, vmax)
    axes = _axes_with_labels(vmin, vmax, asof_dates)
    hover = ""
    if not slug.startswith("chart-alt-"):
        hover = _hover_zones_and_tooltip(
            dates=asof_dates, series={"close": close},
            chart_slug=slug, vmin=vmin, vmax=vmax,
            with_zones=False, with_script=False,
        )
    return (
        f'<svg class="{slug}" id="svg-{slug}" '
        f'viewBox="0 0 {CHART_W} {CHART_H}" '
        f'width="100%" preserveAspectRatio="none" role="img" '
        f'aria-label="SPY price">'
        f"{axes}"
        f'<path class="line line-spy" stroke="{NEUTRAL_BLUE}" d="{d}" />'
        f"{hover}"
        f"</svg>"
    )


def _spy_to_wide(spy_ohlcv: pd.DataFrame | None) -> pd.DataFrame:
    """Long-format SPY OHLCV → wide indexed on date-iso strings.

    Mirrors `_to_wide` for the breadth frame so SPY can share an x-axis
    with the breadth panes. Empty input yields an empty wide frame.
    """
    if spy_ohlcv is None or spy_ohlcv.empty:
        return pd.DataFrame()
    df = spy_ohlcv.copy()
    # Normalize date column (datetime64 / python date / iso string all OK).
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    else:
        df["date"] = df["date"].astype(str)
    df = df.set_index("date").sort_index()
    return df


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
svg .axes line.grid { stroke: var(--axis); stroke-width: 0.5; opacity: 0.35; }
svg text.xtick, svg text.ytick {
  fill: var(--mute); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 9px; stroke: none;
}
svg .midline { stroke: var(--mute); stroke-width: 1; stroke-dasharray: 4 3; opacity: 0.6; }
svg .zone-overbought { fill: var(--bull); opacity: 0.08; stroke: none; }
svg .zone-oversold { fill: var(--bear); opacity: 0.08; stroke: none; }
svg .hover-zone { fill: transparent; }
svg .hover-zone:hover { fill: var(--fg); opacity: 0.05; }
svg .tooltip { pointer-events: none; }
svg .tooltip .tt-bg {
  fill: var(--bg-elevated); stroke: var(--axis); stroke-width: 0.5; opacity: 0.95;
}
svg .tooltip .tt-date, svg .tooltip .tt-line {
  fill: var(--fg); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 10px; stroke: none;
}
svg .tooltip .crosshair {
  stroke: var(--mute); stroke-width: 0.5; stroke-dasharray: 2 2; opacity: 0.7;
}
.snapshot .delta {
  font-size: 0.75rem; padding: 1px 6px; margin-left: 6px;
  border-radius: 9px; font-variant-numeric: tabular-nums; vertical-align: middle;
  border: 1px solid var(--line);
}
.snapshot .delta.bull { color: var(--bull); border-color: var(--bull); }
.snapshot .delta.bear { color: var(--bear); border-color: var(--bear); }
.snapshot .delta.neutral { color: var(--mute); }
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
   exactly 4 chart-* SVGs in DOM order (matches design §5 + acceptance §2).
   The radios live at body level BEFORE their owning `<section>`, so the
   general-sibling combinator (`~`) reaches the section; the SVGs / labels
   are descendants of that section, hence the trailing descendant selector
   (`section .toggle-row`, `section svg.chart-...`). Without the descendant
   step the `~` combinator would never match — `~` only crosses same-level
   siblings. Verified by `test_window_toggle_default_chart_visibility`. */
#window-ad-2y:checked  ~ section .toggle-row label[for=window-ad-2y]  { color: var(--bull); border-color: var(--bull); }
#window-ad-all:checked ~ section .toggle-row label[for=window-ad-all] { color: var(--bull); border-color: var(--bull); }
#window-ad-2y:checked  ~ section svg.chart-ad-cumulative          { display: block; }
#window-ad-2y:checked  ~ section svg.chart-alt-ad-cumulative-all  { display: none; }
#window-ad-all:checked ~ section svg.chart-ad-cumulative          { display: none; }
#window-ad-all:checked ~ section svg.chart-alt-ad-cumulative-all  { display: block; }
/* Default state: when no `:checked` selector wins (i.e. JS-disabled view
   that somehow loses the `checked` attribute), force the alt-* variant to
   stay hidden. Without this, the alt SVGs would render alongside the 2y
   SVGs by default — same bug codex flagged on iter-2. */
svg.chart-alt-ad-cumulative-all { display: none; }
svg.chart-alt-mcclellan-all     { display: none; }
/* McClellan toggle */
#window-mc-2y:checked  ~ section .toggle-row label[for=window-mc-2y]  { color: var(--bull); border-color: var(--bull); }
#window-mc-all:checked ~ section .toggle-row label[for=window-mc-all] { color: var(--bull); border-color: var(--bull); }
#window-mc-2y:checked  ~ section svg.chart-mcclellan          { display: block; }
#window-mc-2y:checked  ~ section svg.chart-alt-mcclellan-all  { display: none; }
#window-mc-all:checked ~ section svg.chart-mcclellan          { display: none; }
#window-mc-all:checked ~ section svg.chart-alt-mcclellan-all  { display: block; }
/* SPY pane toggle — 7 windows. Default (2y) maps to `chart-spy-price`;
   the other 6 carry `chart-alt-spy-price-{win}` slugs and are hidden by default. */
svg.chart-alt-spy-price-1m,
svg.chart-alt-spy-price-3m,
svg.chart-alt-spy-price-6m,
svg.chart-alt-spy-price-1y,
svg.chart-alt-spy-price-5y,
svg.chart-alt-spy-price-all { display: none; }
#window-spy-1m:checked  ~ section svg.chart-spy-price            { display: none; }
#window-spy-1m:checked  ~ section svg.chart-alt-spy-price-1m     { display: block; }
#window-spy-3m:checked  ~ section svg.chart-spy-price            { display: none; }
#window-spy-3m:checked  ~ section svg.chart-alt-spy-price-3m     { display: block; }
#window-spy-6m:checked  ~ section svg.chart-spy-price            { display: none; }
#window-spy-6m:checked  ~ section svg.chart-alt-spy-price-6m     { display: block; }
#window-spy-1y:checked  ~ section svg.chart-spy-price            { display: none; }
#window-spy-1y:checked  ~ section svg.chart-alt-spy-price-1y     { display: block; }
#window-spy-5y:checked  ~ section svg.chart-spy-price            { display: none; }
#window-spy-5y:checked  ~ section svg.chart-alt-spy-price-5y     { display: block; }
#window-spy-all:checked ~ section svg.chart-spy-price            { display: none; }
#window-spy-all:checked ~ section svg.chart-alt-spy-price-all    { display: block; }
#window-spy-1m:checked  ~ section .toggle-row label[for=window-spy-1m]  { color: var(--bull); border-color: var(--bull); }
#window-spy-3m:checked  ~ section .toggle-row label[for=window-spy-3m]  { color: var(--bull); border-color: var(--bull); }
#window-spy-6m:checked  ~ section .toggle-row label[for=window-spy-6m]  { color: var(--bull); border-color: var(--bull); }
#window-spy-1y:checked  ~ section .toggle-row label[for=window-spy-1y]  { color: var(--bull); border-color: var(--bull); }
#window-spy-2y:checked  ~ section .toggle-row label[for=window-spy-2y]  { color: var(--bull); border-color: var(--bull); }
#window-spy-5y:checked  ~ section .toggle-row label[for=window-spy-5y]  { color: var(--bull); border-color: var(--bull); }
#window-spy-all:checked ~ section .toggle-row label[for=window-spy-all] { color: var(--bull); border-color: var(--bull); }
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
    <div class="value {{ header.pct_20_class }}">{{ header.pct_20 }}%<span class="delta {{ header.pct_20_delta_class }}">{{ header.pct_20_delta }}</span></div>
  </div>
  <div>
    <div class="label">% above 50d MA</div>
    <div class="value {{ header.pct_50_class }}">{{ header.pct_50 }}%<span class="delta {{ header.pct_50_delta_class }}">{{ header.pct_50_delta }}</span></div>
  </div>
  <div>
    <div class="label">% above 200d MA</div>
    <div class="value {{ header.pct_200_class }}">{{ header.pct_200 }}%<span class="delta {{ header.pct_200_delta_class }}">{{ header.pct_200_delta }}</span></div>
  </div>
  <div>
    <div class="label">Net new highs</div>
    <div class="value {{ header.net_hl_class }}">{{ header.net_hl }}<span class="delta {{ header.net_hl_delta_class }}">{{ header.net_hl_delta }}</span></div>
  </div>
  <div>
    <div class="label">A/D today</div>
    <div class="value {{ header.ad_class }}">{{ header.ad_signed }}<span class="delta {{ header.ad_delta_class }}">{{ header.ad_delta }}</span></div>
  </div>
  <div>
    <div class="label">McClellan oscillator</div>
    <div class="value {{ header.mcc_class }}">{{ header.mcc_signed }}<span class="delta {{ header.mcc_delta_class }}">{{ header.mcc_delta }}</span></div>
  </div>
</section>

{% if has_spy %}
<input type="radio" name="window-spy" id="window-spy-1m" class="window-toggle">
<input type="radio" name="window-spy" id="window-spy-3m" class="window-toggle">
<input type="radio" name="window-spy" id="window-spy-6m" class="window-toggle">
<input type="radio" name="window-spy" id="window-spy-1y" class="window-toggle">
<input type="radio" name="window-spy" id="window-spy-2y" class="window-toggle" checked>
<input type="radio" name="window-spy" id="window-spy-5y" class="window-toggle">
<input type="radio" name="window-spy" id="window-spy-all" class="window-toggle">
<section class="chart-row">
  <h2>SPY price</h2>
  <p class="legend">S&amp;P 500 ETF close &middot; divergence reference for the breadth panes below</p>
  <div class="toggle-row">
    <label for="window-spy-1m">1m</label>
    <label for="window-spy-3m">3m</label>
    <label for="window-spy-6m">6m</label>
    <label for="window-spy-1y">1y</label>
    <label for="window-spy-2y">2y</label>
    <label for="window-spy-5y">5y</label>
    <label for="window-spy-all">all</label>
  </div>
  {{ charts.spy_1m }}
  {{ charts.spy_3m }}
  {{ charts.spy_6m }}
  {{ charts.spy_1y }}
  {{ charts.spy_2y }}
  {{ charts.spy_5y }}
  {{ charts.spy_all }}
</section>
{% endif %}

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


# ---------------------------------------------------------------------------
# Fragment template — combined-dashboard embedding mode (NO doctype/html/head).
# ---------------------------------------------------------------------------
#
# Used when `include_shared_styles=False` (combined renderer composes the
# shared <head> + tab shell + both panes). Emits:
#
#   <style>...breadth-only selectors...</style>
#   <h1>...</h1> ... breadth body content ...
#
# The breadth-only CSS subset OMITS the shared declarations already in
# `shared_styles.shared_styles()`: `:root` color tokens (light + dark),
# `body { ... }`. Everything else (snapshot grid, chart-row, toggle radios,
# axes/tooltip, disclaimer) stays — those selectors are page-specific and
# would not appear in the ETF section.
#
# The body markup mirrors `_TEMPLATE` verbatim from `<h1>` through
# `</p>` of the disclaimer so behavior is identical to standalone mode
# minus the document shell.

_FRAGMENT_TEMPLATE = """\
<style>
/* Breadth-only CSS tokens not promoted to shared_styles.py (only this
   renderer draws axes). Keep the light + dark pair so the dark-mode
   axis color survives composition under the combined page. */
:root { --axis: #cbd2d9; }
@media (prefers-color-scheme: dark) { :root { --axis: #3a4156; } }
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
svg .axes line.grid { stroke: var(--axis); stroke-width: 0.5; opacity: 0.35; }
svg text.xtick, svg text.ytick {
  fill: var(--mute); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 9px; stroke: none;
}
svg .midline { stroke: var(--mute); stroke-width: 1; stroke-dasharray: 4 3; opacity: 0.6; }
svg .zone-overbought { fill: var(--bull); opacity: 0.08; stroke: none; }
svg .zone-oversold { fill: var(--bear); opacity: 0.08; stroke: none; }
svg .hover-zone { fill: transparent; }
svg .hover-zone:hover { fill: var(--fg); opacity: 0.05; }
svg .tooltip { pointer-events: none; }
svg .tooltip .tt-bg {
  fill: var(--bg-elevated); stroke: var(--axis); stroke-width: 0.5; opacity: 0.95;
}
svg .tooltip .tt-date, svg .tooltip .tt-line {
  fill: var(--fg); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 10px; stroke: none;
}
svg .tooltip .crosshair {
  stroke: var(--mute); stroke-width: 0.5; stroke-dasharray: 2 2; opacity: 0.7;
}
.snapshot .delta {
  font-size: 0.75rem; padding: 1px 6px; margin-left: 6px;
  border-radius: 9px; font-variant-numeric: tabular-nums; vertical-align: middle;
  border: 1px solid var(--line);
}
.snapshot .delta.bull { color: var(--bull); border-color: var(--bull); }
.snapshot .delta.bear { color: var(--bear); border-color: var(--bear); }
.snapshot .delta.neutral { color: var(--mute); }
input[type=radio].window-toggle {
  position: absolute; opacity: 0; pointer-events: none; width: 0; height: 0;
}
.toggle-row { display: flex; gap: 8px; margin: 8px 0; }
.toggle-row label {
  font-size: 0.8rem; color: var(--mute); cursor: pointer;
  padding: 2px 8px; border: 1px solid var(--line); border-radius: 4px;
  user-select: none;
}
#window-ad-2y:checked  ~ section .toggle-row label[for=window-ad-2y]  { color: var(--bull); border-color: var(--bull); }
#window-ad-all:checked ~ section .toggle-row label[for=window-ad-all] { color: var(--bull); border-color: var(--bull); }
#window-ad-2y:checked  ~ section svg.chart-ad-cumulative          { display: block; }
#window-ad-2y:checked  ~ section svg.chart-alt-ad-cumulative-all  { display: none; }
#window-ad-all:checked ~ section svg.chart-ad-cumulative          { display: none; }
#window-ad-all:checked ~ section svg.chart-alt-ad-cumulative-all  { display: block; }
svg.chart-alt-ad-cumulative-all { display: none; }
svg.chart-alt-mcclellan-all     { display: none; }
#window-mc-2y:checked  ~ section .toggle-row label[for=window-mc-2y]  { color: var(--bull); border-color: var(--bull); }
#window-mc-all:checked ~ section .toggle-row label[for=window-mc-all] { color: var(--bull); border-color: var(--bull); }
#window-mc-2y:checked  ~ section svg.chart-mcclellan          { display: block; }
#window-mc-2y:checked  ~ section svg.chart-alt-mcclellan-all  { display: none; }
#window-mc-all:checked ~ section svg.chart-mcclellan          { display: none; }
#window-mc-all:checked ~ section svg.chart-alt-mcclellan-all  { display: block; }
svg.chart-alt-spy-price-1m,
svg.chart-alt-spy-price-3m,
svg.chart-alt-spy-price-6m,
svg.chart-alt-spy-price-1y,
svg.chart-alt-spy-price-5y,
svg.chart-alt-spy-price-all { display: none; }
#window-spy-1m:checked  ~ section svg.chart-spy-price            { display: none; }
#window-spy-1m:checked  ~ section svg.chart-alt-spy-price-1m     { display: block; }
#window-spy-3m:checked  ~ section svg.chart-spy-price            { display: none; }
#window-spy-3m:checked  ~ section svg.chart-alt-spy-price-3m     { display: block; }
#window-spy-6m:checked  ~ section svg.chart-spy-price            { display: none; }
#window-spy-6m:checked  ~ section svg.chart-alt-spy-price-6m     { display: block; }
#window-spy-1y:checked  ~ section svg.chart-spy-price            { display: none; }
#window-spy-1y:checked  ~ section svg.chart-alt-spy-price-1y     { display: block; }
#window-spy-5y:checked  ~ section svg.chart-spy-price            { display: none; }
#window-spy-5y:checked  ~ section svg.chart-alt-spy-price-5y     { display: block; }
#window-spy-all:checked ~ section svg.chart-spy-price            { display: none; }
#window-spy-all:checked ~ section svg.chart-alt-spy-price-all    { display: block; }
#window-spy-1m:checked  ~ section .toggle-row label[for=window-spy-1m]  { color: var(--bull); border-color: var(--bull); }
#window-spy-3m:checked  ~ section .toggle-row label[for=window-spy-3m]  { color: var(--bull); border-color: var(--bull); }
#window-spy-6m:checked  ~ section .toggle-row label[for=window-spy-6m]  { color: var(--bull); border-color: var(--bull); }
#window-spy-1y:checked  ~ section .toggle-row label[for=window-spy-1y]  { color: var(--bull); border-color: var(--bull); }
#window-spy-2y:checked  ~ section .toggle-row label[for=window-spy-2y]  { color: var(--bull); border-color: var(--bull); }
#window-spy-5y:checked  ~ section .toggle-row label[for=window-spy-5y]  { color: var(--bull); border-color: var(--bull); }
#window-spy-all:checked ~ section .toggle-row label[for=window-spy-all] { color: var(--bull); border-color: var(--bull); }
.disclaimer {
  margin-top: 32px; padding-top: 12px; border-top: 1px solid var(--line);
  color: var(--mute); font-size: 0.8rem; line-height: 1.5;
}
</style>
<h1>S&amp;P 500 Market Breadth</h1>
<p class="subtitle">
  Last updated: {{ asof_str }} {{ rendered_at_pt }} PT
  &middot; Universe: current S&amp;P 500 constituents (survivorship-bias applied retroactively — see footer)
</p>

{% if has_data %}
<section class="snapshot" aria-label="Today's snapshot">
  <div>
    <div class="label">% above 20d MA</div>
    <div class="value {{ header.pct_20_class }}">{{ header.pct_20 }}%<span class="delta {{ header.pct_20_delta_class }}">{{ header.pct_20_delta }}</span></div>
  </div>
  <div>
    <div class="label">% above 50d MA</div>
    <div class="value {{ header.pct_50_class }}">{{ header.pct_50 }}%<span class="delta {{ header.pct_50_delta_class }}">{{ header.pct_50_delta }}</span></div>
  </div>
  <div>
    <div class="label">% above 200d MA</div>
    <div class="value {{ header.pct_200_class }}">{{ header.pct_200 }}%<span class="delta {{ header.pct_200_delta_class }}">{{ header.pct_200_delta }}</span></div>
  </div>
  <div>
    <div class="label">Net new highs</div>
    <div class="value {{ header.net_hl_class }}">{{ header.net_hl }}<span class="delta {{ header.net_hl_delta_class }}">{{ header.net_hl_delta }}</span></div>
  </div>
  <div>
    <div class="label">A/D today</div>
    <div class="value {{ header.ad_class }}">{{ header.ad_signed }}<span class="delta {{ header.ad_delta_class }}">{{ header.ad_delta }}</span></div>
  </div>
  <div>
    <div class="label">McClellan oscillator</div>
    <div class="value {{ header.mcc_class }}">{{ header.mcc_signed }}<span class="delta {{ header.mcc_delta_class }}">{{ header.mcc_delta }}</span></div>
  </div>
</section>

{% if has_spy %}
<input type="radio" name="window-spy" id="window-spy-1m" class="window-toggle">
<input type="radio" name="window-spy" id="window-spy-3m" class="window-toggle">
<input type="radio" name="window-spy" id="window-spy-6m" class="window-toggle">
<input type="radio" name="window-spy" id="window-spy-1y" class="window-toggle">
<input type="radio" name="window-spy" id="window-spy-2y" class="window-toggle" checked>
<input type="radio" name="window-spy" id="window-spy-5y" class="window-toggle">
<input type="radio" name="window-spy" id="window-spy-all" class="window-toggle">
<section class="chart-row">
  <h2>SPY price</h2>
  <p class="legend">S&amp;P 500 ETF close &middot; divergence reference for the breadth panes below</p>
  <div class="toggle-row">
    <label for="window-spy-1m">1m</label>
    <label for="window-spy-3m">3m</label>
    <label for="window-spy-6m">6m</label>
    <label for="window-spy-1y">1y</label>
    <label for="window-spy-2y">2y</label>
    <label for="window-spy-5y">5y</label>
    <label for="window-spy-all">all</label>
  </div>
  {{ charts.spy_1m }}
  {{ charts.spy_3m }}
  {{ charts.spy_6m }}
  {{ charts.spy_1y }}
  {{ charts.spy_2y }}
  {{ charts.spy_5y }}
  {{ charts.spy_all }}
</section>
{% endif %}

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
    has_spy: bool,
    include_shared_styles: bool = True,
) -> str:
    # When `include_shared_styles=True` (the standalone path), emit the
    # full document via `_TEMPLATE`. The fragment path skips doctype/html/
    # head + the shared CSS subset (those move to `shared_styles.py` and
    # are emitted once at page top by the combined renderer).
    src = _TEMPLATE if include_shared_styles else _FRAGMENT_TEMPLATE
    template = _ENV.from_string(src)
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
        has_spy=has_spy,
    )
