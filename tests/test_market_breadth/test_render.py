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
    """Output contains an <input type="radio"> AND a :checked CSS selector.

    Also asserts the **default-on** radios are the 2y variants for both
    A/D-cumulative and McClellan toggles. Without this, a future CSS edit
    could swap the `checked` attribute onto the `-all` variants and the
    default view would render the full-history chart — silently breaking
    the "last 2y default" acceptance gate.
    """
    html = rendered_html
    assert re.search(r"<input\b[^>]*\btype=[\"']radio[\"']", html), (
        "no <input type=radio> for the window toggle"
    )
    assert ":checked" in html, "no :checked CSS selector for the toggle"
    # The two 2y radios MUST carry the `checked` attribute so the CSS-only
    # default state renders the trailing-2y SVGs (not the hidden "all"
    # variants). Match against the input element rather than the CSS
    # selector so a stylesheet rename can't fool the assertion.
    assert re.search(
        r'<input\b[^>]*\bid=["\']window-ad-2y["\'][^>]*\bchecked\b', html
    ), "default A/D radio is not the 2y variant"
    assert re.search(
        r'<input\b[^>]*\bid=["\']window-mc-2y["\'][^>]*\bchecked\b', html
    ), "default McClellan radio is not the 2y variant"
    # And the `-all` radios must NOT be checked by default.
    assert not re.search(
        r'<input\b[^>]*\bid=["\']window-ad-all["\'][^>]*\bchecked\b', html
    ), "A/D 'all' radio should not be default-checked"
    assert not re.search(
        r'<input\b[^>]*\bid=["\']window-mc-all["\'][^>]*\bchecked\b', html
    ), "McClellan 'all' radio should not be default-checked"


def test_window_toggle_css_targets_section_descendants(rendered_html):
    """CSS toggle selectors must traverse INTO the section, not stop at siblings.

    Codex review iter-2 found the original selectors used
    `:checked ~ svg.chart-...` — but the radios were body-level siblings of
    `<section>`, while the SVGs were descendants of the section. The general
    sibling combinator (`~`) only matches same-level siblings, so the CSS
    never matched in practice. Both 2y AND `-all` variants would render by
    default, AND clicking labels couldn't switch views.

    The fix targets the section first, then descends to the SVG. This test
    pins that selector shape so a future style edit can't silently regress.
    """
    html = rendered_html
    # Each :checked rule must include the `section` descendant step.
    must_have_patterns = [
        r"#window-ad-2y:checked\s*~\s*section\s+svg\.chart-ad-cumulative",
        r"#window-ad-all:checked\s*~\s*section\s+svg\.chart-alt-ad-cumulative-all",
        r"#window-mc-2y:checked\s*~\s*section\s+svg\.chart-mcclellan",
        r"#window-mc-all:checked\s*~\s*section\s+svg\.chart-alt-mcclellan-all",
    ]
    for pattern in must_have_patterns:
        assert re.search(pattern, html), (
            f"toggle CSS missing section-descendant selector: {pattern}"
        )
    # The non-section-traversing form must NOT remain. Catches a partial revert.
    forbidden_patterns = [
        # The classic bug: `~ svg.chart-...` without the `section` step.
        r"#window-ad-2y:checked\s*~\s*svg\.chart-ad-cumulative\b",
        r"#window-ad-all:checked\s*~\s*svg\.chart-alt-ad-cumulative-all\b",
        r"#window-mc-2y:checked\s*~\s*svg\.chart-mcclellan\b",
        r"#window-mc-all:checked\s*~\s*svg\.chart-alt-mcclellan-all\b",
    ]
    for pattern in forbidden_patterns:
        assert not re.search(pattern, html), (
            f"toggle CSS still uses broken sibling-only selector: {pattern}"
        )


def test_alt_variants_hidden_by_default(rendered_html):
    """Alt-variant SVGs (`-all` toggle target) must be `display: none` by default.

    Belt-and-suspenders for the toggle CSS: if no `:checked` rule wins
    (e.g. a future stylesheet edit removes the `checked` attribute, or a
    UA strips it), the alt-variant SVGs MUST still be hidden by default.
    Without this, the page would render with BOTH the 2y chart and the
    all-history chart visible — confusing the operator and doubling the
    rendered chart count.
    """
    html = rendered_html
    # Match the explicit default-hide rule for each alt variant.
    assert re.search(
        r"svg\.chart-alt-ad-cumulative-all\s*{\s*display:\s*none",
        html,
    ), "alt A/D chart not default-hidden"
    assert re.search(
        r"svg\.chart-alt-mcclellan-all\s*{\s*display:\s*none",
        html,
    ), "alt McClellan chart not default-hidden"


def test_historical_asof_filters_chart_data(breadth_df):
    """`--asof 2025-01-15` must not let 2026 chart data leak into the render.

    Codex review iter-2 (P2): the chart-slice path used
    `wide.tail(window_days)` over the UNFILTERED wide frame, so historical
    `--asof` renders carried the requested header date but the most recent
    chart rows (2026 data on a 2025 page). Filter `wide <= asof` BEFORE
    slicing so historical replays are byte-deterministic per-day.
    """
    from rainier.market_breadth.render import render_breadth_html

    historical_asof = date(2025, 1, 15)
    html = render_breadth_html(
        breadth=breadth_df,
        asof=historical_asof,
        rendered_at_pt="12:40",
    )
    # Header must reflect the historical asof.
    assert "Last updated: 2025-01-15 12:40 PT" in html, (
        "historical asof not honored in header"
    )
    # Pull the pct-above-ma chart block and count points in the first <path>.
    block = re.search(
        r'(<svg\b[^>]*\bclass="chart-pct-above-ma"[^>]*>.*?</svg>)',
        html,
        re.DOTALL,
    )
    assert block, "chart-pct-above-ma block not found in historical render"
    path_d = re.search(r'<path\b[^>]*\bd="([^"]+)"', block.group(1))
    assert path_d, "no <path d=> in historical chart"
    point_count = len(re.findall(r"[ML]\s*[-\d.]+,[-\d.]+", path_d.group(1)))

    # The fixture ends 2026-05-25 with 750 trading days back, so
    # 2025-01-15 sits in the middle. After filtering to <= 2025-01-15
    # the post-filter dataframe has ~370 rows (from ~mid-2023 to 2025-01-15).
    # The 504d window then keeps all of them (since 370 < 504). The point
    # count MUST be MUCH less than the full 750-day count and the no-filter
    # 504-day count we already test elsewhere.
    asof_iso = historical_asof.isoformat()
    expected_max = breadth_df[
        breadth_df["asof_date"].astype(str) <= asof_iso
    ]["asof_date"].nunique()
    assert point_count <= expected_max, (
        f"chart leaked future data: {point_count} points > {expected_max} "
        f"available days up to {asof_iso}"
    )
    # Sanity: a healthy fixture has hundreds of days up to 2025-01-15.
    assert point_count > 100, (
        f"historical chart suspiciously sparse: {point_count} points "
        f"(fixture has {expected_max} days up to {asof_iso})"
    )


def test_line_path_preserves_date_alignment_with_warmup_nans():
    """Codex iter-2 [P2]: `dropna()` reindexed warm-up gaps to the left edge.

    With different warm-up periods per indicator (`pct_above_ma_20` starts
    producing values at day 20, `pct_above_ma_200` at day 200), naively
    dropping NaN before scaling x positions would push both first-valid
    points to `x = PAD_X`. The same calendar day would then render at
    different x-positions across chart layers — silently misleading.

    Fix: x-position is the original index in the full series; NaNs become
    SVG subpath breaks. This test pins that.
    """
    from rainier.market_breadth.render import (
        CHART_PAD_X,
        CHART_W,
        _line_path,
    )

    # Two series with the SAME length and the SAME last valid value but
    # different warm-up periods. After fix: the last L-point in both paths
    # must land at the same x-coordinate (the right edge).
    n = 100
    series_short_warmup = pd.Series(
        [float("nan")] * 5 + [10.0 + i * 0.1 for i in range(n - 5)]
    )
    series_long_warmup = pd.Series(
        [float("nan")] * 50 + [10.0 + i * 0.1 for i in range(n - 50)]
    )
    path_short = _line_path(series_short_warmup, 0.0, 100.0)
    path_long = _line_path(series_long_warmup, 0.0, 100.0)

    # Extract the last x-coordinate from each path.
    last_x_short = float(
        re.findall(r"[ML]([-\d.]+),", path_short)[-1]
    )
    last_x_long = float(
        re.findall(r"[ML]([-\d.]+),", path_long)[-1]
    )
    # Both must hit the right edge (i.e., be very close to each other).
    assert abs(last_x_short - last_x_long) < 0.01, (
        f"warm-up indicators misalign on right edge: "
        f"short_warmup_last_x={last_x_short}, long_warmup_last_x={last_x_long}"
    )
    # And the first valid x-coordinate of the long-warmup series must NOT
    # be at the left edge (PAD_X) — proves we're not reindexing.
    first_x_long = float(
        re.findall(r"[ML]([-\d.]+),", path_long)[0]
    )
    assert first_x_long > CHART_PAD_X + 1.0, (
        f"long-warmup series starts at pad x={first_x_long:.2f}; "
        f"NaN warm-up was incorrectly reindexed to the left edge"
    )
    # Sanity: short warm-up's first valid x is near (but past) the left pad.
    first_x_short = float(
        re.findall(r"[ML]([-\d.]+),", path_short)[0]
    )
    assert first_x_short < first_x_long, (
        f"short_warmup first valid x ({first_x_short:.2f}) must be left of "
        f"long_warmup first valid x ({first_x_long:.2f})"
    )
    # And the right edge for both is approximately CHART_W - CHART_PAD_X.
    assert abs(last_x_short - (CHART_W - CHART_PAD_X)) < 0.5


def test_line_path_breaks_subpath_on_interior_nan():
    """Interior NaN gaps must break the line — not connect across the gap."""
    from rainier.market_breadth.render import _line_path

    # 10 days with a NaN gap in the middle: values exist at indices 0-3 and 6-9.
    series = pd.Series([1.0, 2.0, 3.0, 4.0, float("nan"), float("nan"),
                        7.0, 8.0, 9.0, 10.0])
    path = _line_path(series, 0.0, 10.0)
    # Must contain at least 2 `M` markers (one per subpath).
    m_count = len(re.findall(r"\bM", path))
    assert m_count >= 2, (
        f"interior NaN should split into ≥2 subpaths, got {m_count} 'M' "
        f"in path: {path!r}"
    )


def test_rendered_html_under_size_budget(rendered_html):
    """Render output stays under the 150KB byte budget.

    Operator's acceptance gate is "sub-100KB" (observed at 97KB during the
    worker turn). 150KB is the **regression bar**: above that, the page
    becomes too heavy to ship through fengshen-site's edge cache cleanly,
    AND the gap usually means a chart blew up its point count or someone
    embedded a binary blob. Catch it here rather than at deploy time.
    """
    byte_len = len(rendered_html.encode("utf-8"))
    assert byte_len < 150_000, (
        f"rendered HTML exceeds 150KB regression budget: {byte_len:,} bytes "
        f"(operator target: <100KB, ratchet at >150KB)"
    )


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


# ---------------------------------------------------------------------------
# CLI fail-loud regression — `rainier market-breadth render-html` must NOT
# silently publish a degenerate page when the input parquet is empty or has
# only NaT asof_dates. Without this, `breadth["asof_date"].max()` returns
# `pd.NaT`, `NaT.date()` returns NaT (not a real date), and the renderer's
# `asof.isoformat()` writes the literal string "NaT" into the published HTML.
# ---------------------------------------------------------------------------


def test_cli_render_html_fails_loud_on_empty_parquet(tmp_path):
    """Empty input parquet → non-zero exit + stderr error, never a stale render."""
    from click.testing import CliRunner

    from rainier.cli import cli

    empty_path = tmp_path / "empty.parquet"
    # Zero-row parquet — schema matches what compute.py emits but no rows.
    pd.DataFrame(
        {
            "asof_date": pd.Series(dtype="datetime64[ns]"),
            "indicator": pd.Series(dtype=str),
            "value": pd.Series(dtype=float),
        }
    ).to_parquet(empty_path, index=False)
    out_path = tmp_path / "market-breadth.html"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "market-breadth",
            "render-html",
            "--input",
            str(empty_path),
            "--rendered-at-pt",
            "12:40",
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code != 0, (
        f"expected non-zero exit on empty parquet, got {result.exit_code}; "
        f"output={result.output!r}"
    )
    assert "no rows" in result.output.lower(), (
        f"expected fail-loud output mentioning 'no rows', got {result.output!r}"
    )
    # And critically: no HTML must have been written to the public path.
    assert not out_path.exists(), (
        f"output HTML must NOT be written on empty-parquet failure: {out_path}"
    )


def test_cli_render_html_fails_loud_on_all_nat_asof(tmp_path):
    """Non-empty parquet whose every asof_date is NaT → fail-loud, no stale render."""
    from click.testing import CliRunner

    from rainier.cli import cli

    nat_path = tmp_path / "nat.parquet"
    pd.DataFrame(
        {
            "asof_date": [pd.NaT, pd.NaT, pd.NaT],
            "indicator": ["pct_above_ma_20", "pct_above_ma_50", "pct_above_ma_200"],
            "value": [50.0, 55.0, 60.0],
        }
    ).to_parquet(nat_path, index=False)
    out_path = tmp_path / "market-breadth.html"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "market-breadth",
            "render-html",
            "--input",
            str(nat_path),
            "--rendered-at-pt",
            "12:40",
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code != 0, (
        f"expected non-zero exit on all-NaT asof_date, got {result.exit_code}; "
        f"output={result.output!r}"
    )
    assert "asof_date" in result.output.lower(), (
        f"expected fail-loud output mentioning 'asof_date', got {result.output!r}"
    )
    assert not out_path.exists(), (
        f"output HTML must NOT be written on all-NaT failure: {out_path}"
    )


def test_cli_render_html_auto_asof_uses_latest_row(tmp_path, breadth_df):
    """Healthy parquet + no `--asof` → CLI resolves the latest row and renders."""
    from click.testing import CliRunner

    from rainier.cli import cli

    in_path = tmp_path / "breadth.parquet"
    breadth_df.to_parquet(in_path, index=False)
    out_path = tmp_path / "market-breadth.html"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "market-breadth",
            "render-html",
            "--input",
            str(in_path),
            "--rendered-at-pt",
            "12:40",
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, (
        f"expected success on healthy parquet, got {result.exit_code}; "
        f"output={result.output!r}"
    )
    assert out_path.exists(), "output HTML must be written on success"
    # The latest row in the fixture is 2026-05-25 — verify the auto-asof picked it.
    html = out_path.read_text(encoding="utf-8")
    assert "Last updated: 2026-05-25 12:40 PT" in html, (
        "auto-asof did not resolve to the fixture's latest row"
    )
