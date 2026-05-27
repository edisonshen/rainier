"""Shared CSS + regime helpers for the combined trading dashboard.

Two responsibilities:

  1. `shared_styles()` returns the CSS subset common to both the breadth
     and ETF ranks renderers (typography, color tokens, dark-mode `:root`
     vars, regime chip styles, top-level radio-tab pattern, hero card
     shell). The combined renderer emits it once at page top so the
     two nested section renderers don't duplicate it.

  2. `_regime_from_pct_above_200d(pct)` + `regime_chip_html(pct)` —
     map %-above-200d to a (label, css-class) pair and emit the chip
     fragment for the shared header. Thresholds follow StockCharts
     convention (DESIGN-trading-dashboard-combined-v1.md §3.1):

         > 70   strong-bull   deep green
         50–70  bullish       light green
         30–50  mixed         yellow
         < 30   bearish       red

The standalone breadth and ETF pages still ship their own complete
`<style>` blocks for backward byte-compat (PRs #90-98). This module
exists so the combined page doesn't double-include those declarations
when the two sub-renderers are invoked in fragment mode.

ASCII flow:

    standalone breadth/ETF  ──▶  full <style> block (unchanged)

    combined renderer
        │  emits shared_styles() ONCE at page top
        ├──▶ render_breadth_html(include_shared_styles=False)
        │       └─▶ inner content only (no shared CSS)
        └──▶ render_etf_html(include_shared_styles=False)
                └─▶ inner content only (no shared CSS)
"""

from __future__ import annotations

__all__ = [
    "shared_styles",
    "regime_chip_html",
    "_regime_from_pct_above_200d",
]


# ---------------------------------------------------------------------------
# Regime helper
# ---------------------------------------------------------------------------


def _regime_from_pct_above_200d(pct: float) -> tuple[str, str]:
    """Map %-above-200d to (label, css-class).

    Bands (inclusive on the upper edge of each lower band):
        > 70   ("Strong bull",     "regime-strong-bull")
        50-70  ("Bullish breadth", "regime-bullish")
        30-50  ("Mixed",           "regime-mixed")
        < 30   ("Bearish breadth", "regime-bearish")

    The inclusive boundary picks the *upper* band at the threshold — i.e.
    pct == 70.5 is "Strong bull", pct == 50.5 is "Bullish breadth",
    pct == 30.5 is "Mixed". A clean integer-boundary rendering
    (pct == 70.0 exactly) lands in "Bullish breadth" because the strict
    `> 70` test fails. This matches the StockCharts convention in
    DESIGN §3.1 and is what the threshold-helper tests pin.
    """
    p = float(pct)
    if p > 70.0:
        return ("Strong bull", "regime-strong-bull")
    if p > 50.0:
        return ("Bullish breadth", "regime-bullish")
    if p > 30.0:
        return ("Mixed", "regime-mixed")
    return ("Bearish breadth", "regime-bearish")


def regime_chip_html(pct: float) -> str:
    """Pre-rendered chip with `regime-chip` base + threshold-derived modifier.

    Used in the combined-dashboard's shared header. Numeric-only inputs +
    no template engine — safe to drop into the page verbatim.
    """
    label, css = _regime_from_pct_above_200d(pct)
    return (
        f'<span class="regime-chip {css}">'
        f'{label} &middot; {int(round(pct))}% above 200d'
        "</span>"
    )


# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------
#
# Color tokens, typography base, the top-level tab radio pattern, the
# regime chip, and the mobile breakpoint live here. The chart-specific
# selectors (breadth SVG layers, ETF rank gradient, sparkline `.spark`)
# stay in their respective renderers — they're not shared and moving them
# here would force every consumer to ship CSS for charts it doesn't draw.


_SHARED_STYLES = """\
:root {
  --bg-page: #ffffff;
  --bg-elevated: #ffffff;
  --fg: #1f2933;
  --mute: #52606d;
  --line: #e4e7eb;
  --bull: #1b9e3a;
  --bear: #c92a2a;
  --regime-strong-bull-bg: #1b9e3a;
  --regime-strong-bull-fg: #ffffff;
  --regime-bullish-bg: #b7f0c8;
  --regime-bullish-fg: #0d4a1d;
  --regime-mixed-bg: #ffe39a;
  --regime-mixed-fg: #5a4400;
  --regime-bearish-bg: #c92a2a;
  --regime-bearish-fg: #ffffff;
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
    --regime-strong-bull-bg: #2bd86d;
    --regime-strong-bull-fg: #06140a;
    --regime-bullish-bg: #1f4a2b;
    --regime-bullish-fg: #b7f0c8;
    --regime-mixed-bg: #4a3c12;
    --regime-mixed-fg: #ffe39a;
    --regime-bearish-bg: #ff6e6e;
    --regime-bearish-fg: #1a0808;
    color-scheme: dark;
  }
}
body {
  font-family: 'Atkinson Hyperlegible', -apple-system, BlinkMacSystemFont,
               'Segoe UI', system-ui, Roboto, 'Helvetica Neue', Arial, sans-serif;
  margin: 24px; color: var(--fg); background: var(--bg-page);
  font-size: 14px; max-width: 1100px;
}
h1.dashboard-brand { font-size: 1.4rem; margin: 0 0 4px; color: var(--fg); }
.dashboard-header {
  display: flex; flex-wrap: wrap; align-items: center; gap: 12px;
  padding: 12px 16px; margin: 0 0 16px;
  background: var(--bg-elevated); border: 1px solid var(--line); border-radius: 6px;
}
.dashboard-header .header-meta { color: var(--mute); font-size: 0.85rem; }
.regime-chip {
  display: inline-block; padding: 2px 10px; border-radius: 12px;
  font-size: 0.8rem; font-weight: 600; letter-spacing: 0.02em;
  border: 1px solid transparent; white-space: nowrap;
}
.regime-chip.regime-strong-bull {
  background: var(--regime-strong-bull-bg); color: var(--regime-strong-bull-fg);
}
.regime-chip.regime-bullish {
  background: var(--regime-bullish-bg); color: var(--regime-bullish-fg);
}
.regime-chip.regime-mixed {
  background: var(--regime-mixed-bg); color: var(--regime-mixed-fg);
}
.regime-chip.regime-bearish {
  background: var(--regime-bearish-bg); color: var(--regime-bearish-fg);
}
/* Top-level trading-tab radio pattern — Breadth / ETF Ranks. Mirrors the
   ETF dashboard's per-tab pattern; the radios live as siblings BEFORE the
   `.trading-tabs` label row + the `<section class="tab-pane">` panes so
   the `~` general-sibling combinator can flip both visibility + active
   state. Pure CSS, no JS, byte-deterministic. */
input[type=radio].trading-tab-radio {
  position: absolute; opacity: 0; pointer-events: none; width: 0; height: 0;
}
.trading-tabs {
  display: flex; gap: 4px; margin: 0 0 8px;
  border-bottom: 1px solid var(--line);
}
.trading-tabs label {
  padding: 8px 18px; cursor: pointer; user-select: none;
  color: var(--mute); font-size: 0.95rem; font-weight: 600;
  border-bottom: 2px solid transparent;
}
.trading-tabs label:hover { color: var(--fg); }
#tab-breadth:checked ~ .trading-tabs label[for=tab-breadth] {
  color: var(--bull); border-bottom-color: var(--bull);
}
#tab-etf:checked ~ .trading-tabs label[for=tab-etf] {
  color: var(--bull); border-bottom-color: var(--bull);
}
section.tab-pane { display: none; }
#tab-breadth:checked ~ section.tab-pane.pane-breadth { display: block; }
#tab-etf:checked     ~ section.tab-pane.pane-etf     { display: block; }
/* Mobile breakpoint — under 720px the tabs collapse to stacked sections
   so the operator can scroll through both views without tapping. Hides
   the radios entirely (they're already visually hidden, but on mobile we
   want both panes visible at once regardless of `:checked` state). */
@media (max-width: 720px) {
  input[type=radio].trading-tab-radio { display: none; }
  .trading-tabs { display: none; }
  section.tab-pane { display: block !important; }
}
"""


def shared_styles() -> str:
    """Return the shared CSS string (no surrounding `<style>` tags).

    Caller wraps in `<style>...</style>`. Keeps the function easy to
    splice into a larger `<style>` block if the combined renderer ever
    needs to merge with renderer-specific selectors.
    """
    return _SHARED_STYLES
