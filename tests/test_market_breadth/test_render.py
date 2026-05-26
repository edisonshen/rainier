"""Unit tests for `rainier.market_breadth.render`.

The renderer is a pure function over a long-format breadth parquet → HTML
string. Tests exercise it on a deterministic fixture
(`tests/fixtures/sp500_breadth_small.parquet` — 12 indicators × 750
trading days ending 2026-05-25) and assert on the rendered HTML.

Acceptance gates (docs/TASK-PLAN-market-breadth-render-e818.md §Tests):
    1. self-contained HTML — no external CSS, no remote scripts/images
    2. exactly 4 chart SVG blocks (svg.chart-*)
    3. % above MA chart has 3 `<path>` elements (20d / 50d / 200d lines)
    4. default window slices to ~504 trading days (last ~2y)
    5. CSS-only window toggle: <input type="radio"> + :checked selector
    6. green/red bull/bear palette — bull-green + bear-red hex codes present
    7. header timestamp literal `Last updated: 2026-05-25 12:40 PT`
    8. footer disclaimer present (verbatim from DESIGN §5.3)
    9. byte-deterministic — same input → same output
   10. header snapshot values match the latest row in the fixture parquet
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BREADTH_FIX = FIXTURES / "sp500_breadth_small.parquet"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def breadth_df() -> pd.DataFrame:
    return pd.read_parquet(BREADTH_FIX)


@pytest.fixture
def rendered_html(breadth_df) -> str:
    from rainier.market_breadth.render import render_breadth_html

    return render_breadth_html(
        breadth=breadth_df,
        asof=date(2026, 5, 25),
        rendered_at_pt="12:40",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_render_html_is_self_contained(rendered_html):
    """No external CSS, no remote <script src>, no remote <img src>."""
    html = rendered_html
    assert not re.search(
        r'<link\b[^>]*rel=["\']stylesheet["\'][^>]*href=["\']https?:', html
    ), "external stylesheet link found"
    assert not re.search(r"<script\b[^>]*\bsrc=", html), "external script src found"
    assert not re.search(r'<img\b[^>]*src=["\']https?:', html), "external img src found"
    assert "src=\"//" not in html and "href=\"//" not in html, "protocol-relative URL"


def test_render_includes_4_charts(rendered_html):
    """Output contains exactly 4 primary chart SVG blocks.

    The two toggle-window variants (A/D `all` + McClellan `all`) carry a
    `chart-alt-*` prefix so the default DOM has exactly 4 primary charts
    visible; the `alt-` variants are hidden until their radio is selected.
    """
    html = rendered_html
    svgs = re.findall(
        r'<svg\b[^>]*\bclass="chart-(?!alt-)([a-z0-9_-]+)"', html
    )
    assert len(svgs) == 4, f"expected 4 primary chart SVGs, got {len(svgs)}: {svgs}"
    assert len(set(svgs)) == 4, f"chart slugs not unique: {svgs}"
    # Sanity-check: the 2 toggle-window variants ARE in the rendered HTML
    # (hidden until :checked). The CSS-only window toggle test below verifies
    # the :checked selectors actually exist.
    alt_svgs = re.findall(r'<svg\b[^>]*\bclass="chart-alt-([a-z0-9_-]+)"', html)
    assert len(alt_svgs) == 2, f"expected 2 alt-window SVGs, got {len(alt_svgs)}: {alt_svgs}"


def test_chart_path_count_matches_data(rendered_html):
    """% above MA chart has 3 `<path>` elements (20d / 50d / 200d lines)."""
    html = rendered_html
    # Pull the % above MA chart block (anything between the open svg and its close).
    block = re.search(
        r'(<svg\b[^>]*\bclass="chart-pct-above-ma"[^>]*>.*?</svg>)', html, re.DOTALL
    )
    assert block, "chart-pct-above-ma SVG block not found"
    path_count = len(re.findall(r"<path\b[^>]*\bd=", block.group(1)))
    assert path_count == 3, (
        f"expected 3 line paths (20d / 50d / 200d) in pct-above-ma chart, got {path_count}"
    )


def test_default_window_is_2y(rendered_html, breadth_df):
    """Given 750d of fixture data, default-rendered chart paths cover only ~504d.

    The path's `M x,y L x,y …` coordinate count == number of data points.
    The renderer's default 504d window means the longest path has at most
    504 data points (not the full 750).
    """
    html = rendered_html
    block = re.search(
        r'(<svg\b[^>]*\bclass="chart-pct-above-ma"[^>]*>.*?</svg>)', html, re.DOTALL
    )
    assert block, "chart-pct-above-ma SVG block not found"
    # Pull the FIRST path's d-attribute (the 20d line) and count points.
    path_d = re.search(r'<path\b[^>]*\bd="([^"]+)"', block.group(1))
    assert path_d, "no path in chart-pct-above-ma"
    point_count = len(re.findall(r"[ML]\s*[-\d.]+,[-\d.]+", path_d.group(1)))
    assert 480 <= point_count <= 510, (
        f"expected ~504 points in default 2y window, got {point_count}"
    )
    # And confirm the fixture itself has more than 504 days so the slice is real.
    days = breadth_df["asof_date"].nunique()
    assert days > 510, f"fixture sanity: needs >510 days, got {days}"


def test_window_toggle_present(rendered_html):
    """Output contains an <input type="radio"> AND a :checked CSS selector."""
    html = rendered_html
    assert re.search(r"<input\b[^>]*\btype=[\"']radio[\"']", html), (
        "no <input type=radio> for the window toggle"
    )
    assert ":checked" in html, "no :checked CSS selector for the toggle"


def test_palette_is_green_red(rendered_html):
    """Output contains bull-green / bear-red hex codes (breadth.app aesthetic).

    Asserts on a stable pair the renderer picks. The renderer must NOT carry
    the fengshen-site blue accent (#2337ff) — that's the explicit override
    per the task plan.
    """
    html = rendered_html
    # Bull / bear pair — same shade family as the ETF dashboard's pos/neg
    # CSS vars (#1b9e3a / #c92a2a) so the look stays consistent across the
    # /trading/ umbrella.
    assert "#1b9e3a" in html.lower() or "#1b9e3a" in html, "bull-green hex not in output"
    assert "#c92a2a" in html.lower() or "#c92a2a" in html, "bear-red hex not in output"
    # The fengshen-site blue accent must NOT be applied to charts on this page.
    # (It can appear inside an unrelated comment, so we restrict the check to
    # CSS variable definitions and stroke/fill assignments.)
    assert "--accent: #2337ff" not in html, "fengshen blue accent leaked into breadth page"


def test_header_timestamp_pt_format(rendered_html):
    """Header contains literal `Last updated: 2026-05-25 12:40 PT`."""
    assert "Last updated: 2026-05-25 12:40 PT" in rendered_html


def test_disclaimer_present(rendered_html):
    """Footer contains the survivorship-bias paragraph verbatim from DESIGN §5.3.

    The plaintext design text says ``S&P 500`` — under autoescape this lands
    in the rendered HTML as ``S&amp;P 500``. We match the rendered form so
    the assertion is exact against what ships to the public page.
    """
    html = rendered_html
    expected = (
        "Universe = current S&amp;P 500 constituents applied retroactively. "
        "Historical view does not adjust for past index add/drop events; "
        "breadth may appear slightly stronger than reality. "
        "v1 will add point-in-time membership."
    )
    assert expected in html, "survivorship-bias disclaimer missing or altered"


def test_render_deterministic(breadth_df):
    """Same input → byte-identical output across two consecutive renders."""
    from rainier.market_breadth.render import render_breadth_html

    kwargs = dict(
        breadth=breadth_df,
        asof=date(2026, 5, 25),
        rendered_at_pt="12:40",
    )
    out1 = render_breadth_html(**kwargs)
    out2 = render_breadth_html(**kwargs)
    assert out1 == out2, "renderer is not deterministic across consecutive calls"


def test_today_snapshot_values_match_input(rendered_html, breadth_df):
    """Header headline numbers match the latest row in the fixture parquet."""
    html = rendered_html
    asof_max = breadth_df["asof_date"].max()
    latest = breadth_df[breadth_df["asof_date"] == asof_max].set_index("indicator")["value"]

    pct_20 = int(round(latest["pct_above_ma_20"]))   # 65
    pct_200 = int(round(latest["pct_above_ma_200"])) # 62
    nh = int(latest["new_52w_high"])                  # 24
    nl = int(latest["new_52w_low"])                   # 7
    ad = int(latest["ad_diff"])                       # 217

    # The header MUST render these as integers (no `.0`).
    assert f"{pct_20}%" in html, f"header missing pct_above_ma_20 = {pct_20}%"
    assert f"{pct_200}%" in html, f"header missing pct_above_ma_200 = {pct_200}%"
    assert f">{nh}<" in html or f" {nh} " in html, f"header missing new_52w_high={nh}"
    assert f">{nl}<" in html or f" {nl} " in html, f"header missing new_52w_low={nl}"
    # A/D rendered with explicit sign.
    assert f"+{ad}" in html or f"{ad:+d}" in html, f"header missing ad_diff={ad}"
