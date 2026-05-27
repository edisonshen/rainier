"""Combined trading dashboard renderer — `/trading/`.

One HTML page that hosts both the S&P 500 breadth view AND the ETF ranks
table, with a shared site header (brand + asof + regime chip + rendered-at-pt)
and pure-CSS top-level tabs (Breadth / ETF Ranks).

URL: DESIGN-trading-dashboard-combined-v1.md §4 D1 (operator override
2026-05-27 "the url will be /trading"). The publish script
`scripts/publish-trading-dashboard.sh` invokes the generic publisher with
`--root` so this HTML lands at `public/trading/index.html` rather than the
default `public/trading/<name>/index.html`.

Composition model (DESIGN-trading-dashboard-combined-v1.md §3):

    ┌──────────────────────────────────────────────────────────┐
    │  <head> + shared_styles() — emitted ONCE                  │
    ├──────────────────────────────────────────────────────────┤
    │  <body>                                                   │
    │    <header class="dashboard-header">                      │
    │      brand · asof · regime-chip · rendered-at-pt          │
    │    </header>                                              │
    │                                                           │
    │    <input type=radio name="trading-tab" id="tab-breadth"  │
    │           class="trading-tab-radio" checked>              │
    │    <input type=radio name="trading-tab" id="tab-etf"      │
    │           class="trading-tab-radio">                      │
    │    <nav class="trading-tabs">                             │
    │      <label for=tab-breadth>Breadth</label>               │
    │      <label for=tab-etf>ETF Ranks</label>                 │
    │    </nav>                                                 │
    │                                                           │
    │    <section class="tab-pane pane-breadth">                │
    │      render_breadth_html(include_shared_styles=False)     │
    │    </section>                                             │
    │    <section class="tab-pane pane-etf">                    │
    │      render_etf_html(include_shared_styles=False)         │
    │    </section>                                             │
    │  </body>                                                  │
    └──────────────────────────────────────────────────────────┘

Both nested renderers run in fragment mode (no document shell, no shared
CSS subset). Breadth is the default `checked` top-level tab (D2). Mobile
breakpoint at 720px (D7) collapses tabs to stacked sections — defined in
`shared_styles.shared_styles()` so both standalone callers don't need to
ship the rule.

Pure function — caller supplies `rendered_at_pt` so re-rendering on
identical inputs is byte-stable. No `datetime.now()`, no `Math.random()`,
no uuid generation. The 3x byte-identical regression test enforces this.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
from markupsafe import escape

from rainier.dashboard.render_etf import render_etf_html
from rainier.dashboard.shared_styles import (
    regime_chip_html,
    shared_styles,
)
from rainier.market_breadth.render import render_breadth_html

__all__ = [
    "render_combined_html",
    "write_combined_html",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_combined_html(
    *,
    breadth: pd.DataFrame,
    features: pd.DataFrame,
    registry: pd.DataFrame,
    asof: date,
    rendered_at_pt: str,
    spy_ohlcv: pd.DataFrame | None = None,
    history_days: int = 30,
    window_days: int = 504,
    names_path: str | Path | None = None,
) -> str:
    """Render the combined trading dashboard to a single HTML string.

    Parameters mirror the per-section renderers' kwargs, plus this
    composition step picks the regime-chip pct from the latest
    ``pct_above_ma_200`` row in ``breadth``.

    Pure function — no I/O, no DB, no wall-clock reads inside (the
    optional ``names_path`` read is delegated to ``render_etf_html``,
    which handles a missing / unreadable path by falling back to the
    symbol-only display — same semantics as the standalone path in
    PR #100).

    ``names_path`` threads PR #100's ETF Name-column behavior through
    the combined-dashboard composition. When provided + readable, the
    ETF tab inside ``/trading/`` renders yfinance ``longName`` in the
    Name column; otherwise it falls back to the symbol itself, matching
    the standalone ``/trading/etf-ranks/`` page byte-for-byte on
    identical inputs.
    """
    asof_str = asof.isoformat()
    # Defense-in-depth — `asof.isoformat()` always returns safe YYYY-MM-DD,
    # but `rendered_at_pt` is caller-supplied free-form text that lands in
    # the public `/trading/` page. Both sub-renderers autoescape these
    # values through their jinja templates; the combined-shell path
    # hand-assembles HTML with `str.join`, so escape here too for parity.
    asof_str_safe = str(escape(asof_str))
    rendered_at_pt_safe = str(escape(rendered_at_pt))
    pct_above_200d = _latest_pct_above_200d(breadth, asof_str)
    regime_chip = (
        regime_chip_html(pct_above_200d)
        if pct_above_200d is not None
        else '<span class="regime-chip regime-mixed">No data</span>'
    )

    breadth_fragment = render_breadth_html(
        breadth=breadth,
        asof=asof,
        rendered_at_pt=rendered_at_pt,
        window_days=window_days,
        spy_ohlcv=spy_ohlcv,
        include_shared_styles=False,
    )
    etf_fragment = render_etf_html(
        features=features,
        registry=registry,
        asof=asof,
        rendered_at_pt=rendered_at_pt,
        history_days=history_days,
        include_shared_styles=False,
        names_path=names_path,
    )

    # Plain concatenation — `str.format()` is unsafe here because the
    # ETF fragment's inline JS contains literal `{` / `}` (arrow functions,
    # object literals) that would crash `format()` with KeyError. The shell
    # is fixed-shape so simple list-join is the lowest-risk composition.
    # `asof_str_safe` and `rendered_at_pt_safe` are pre-escaped via
    # markupsafe to match the autoescape behavior of the nested jinja
    # renderers; `regime_chip` is renderer-generated HTML (numeric input
    # only — no caller-controlled strings inside).
    return "".join(
        [
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n',
            '<meta charset="utf-8">\n',
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n',
            f"<title>Trading Dashboard — {asof_str_safe}</title>\n",
            "<style>\n",
            shared_styles(),
            "</style>\n",
            "</head>\n",
            "<body>\n",
            '<header class="dashboard-header">\n',
            '  <h1 class="dashboard-brand">rainier &middot; trading dashboard</h1>\n',
            f'  <span class="header-meta">asof {asof_str_safe}</span>\n',
            "  ",
            regime_chip,
            "\n",
            f'  <span class="header-meta">rendered {rendered_at_pt_safe} PT</span>\n',
            "</header>\n\n",
            '<input type="radio" name="trading-tab" id="tab-breadth" class="trading-tab-radio" checked>\n',
            '<input type="radio" name="trading-tab" id="tab-etf" class="trading-tab-radio">\n',
            '<nav class="trading-tabs" aria-label="Trading dashboard sections">\n',
            '  <label for="tab-breadth">Breadth</label>\n',
            '  <label for="tab-etf">ETF Ranks</label>\n',
            "</nav>\n\n",
            '<section class="tab-pane pane-breadth" aria-label="S&amp;P 500 market breadth">\n',
            breadth_fragment,
            "\n</section>\n\n",
            '<section class="tab-pane pane-etf" aria-label="ETF ranks">\n',
            etf_fragment,
            "\n</section>\n",
            "</body>\n</html>\n",
        ]
    )


def write_combined_html(
    *,
    breadth_path: str | Path,
    features_path: str | Path,
    registry_path: str | Path,
    output_path: str | Path,
    asof: date,
    rendered_at_pt: str,
    spy_path: str | Path | None = None,
    history_days: int = 30,
    window_days: int = 504,
    names_path: str | Path | None = None,
) -> Path:
    """Convenience wrapper: load parquets, render, atomic-write the HTML.

    ``names_path`` is forwarded to ``render_combined_html`` → the ETF
    sub-renderer; see that docstring for fallback semantics.
    """
    breadth = pd.read_parquet(breadth_path)
    features = pd.read_parquet(features_path)
    registry = pd.read_parquet(registry_path)
    spy_ohlcv = None
    if spy_path is not None:
        spy_p = Path(spy_path)
        if spy_p.exists():
            spy_ohlcv = pd.read_parquet(spy_p)
    html = render_combined_html(
        breadth=breadth,
        features=features,
        registry=registry,
        spy_ohlcv=spy_ohlcv,
        asof=asof,
        rendered_at_pt=rendered_at_pt,
        history_days=history_days,
        window_days=window_days,
        names_path=names_path,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, out)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest_pct_above_200d(breadth: pd.DataFrame, asof_str: str) -> float | None:
    """Pick the latest `pct_above_ma_200` value on or before `asof_str`.

    Returns None when the indicator column is missing entirely OR the
    latest matching row's value is NaN (partial-population case — the
    standalone breadth renderer treats this as missing too). The caller
    substitutes a "no data" chip.
    """
    if breadth.empty:
        return None
    df = breadth
    if pd.api.types.is_datetime64_any_dtype(df["asof_date"]):
        asof_col = df["asof_date"].dt.strftime("%Y-%m-%d")
    else:
        asof_col = df["asof_date"].astype(str)
    mask = (df["indicator"] == "pct_above_ma_200") & (asof_col <= asof_str)
    rows = df.loc[mask].copy()
    if rows.empty:
        return None
    rows["asof_date"] = asof_col[mask]
    rows = rows.sort_values("asof_date")
    raw = rows.iloc[-1]["value"]
    # Missing-value guard — partial breadth data can leave the latest row's
    # value as NaN, pd.NA, or None (nullable/object dtypes appear when a
    # row is partially populated). `float(pd.NA)` and `float(None)` both
    # raise TypeError; `float(nan)` succeeds but yields nan which then
    # crashes `regime_chip_html` at `int(round(nan))`. `pd.isna` covers
    # all three so the caller falls back to the "No data" chip uniformly.
    if pd.isna(raw):
        return None
    return float(raw)


# NOTE: The combined-page shell is hand-assembled in `render_combined_html`
# via `"".join([...])` rather than a jinja template or `str.format()`. The
# ETF fragment's inline JS contains literal `{` / `}` (arrow functions,
# object literals); `str.format()` would crash with KeyError on those, and
# jinja autoescape would mangle the already-rendered fragments. The shell
# is fixed-shape, so the list-join keeps the seam simple + obvious.
