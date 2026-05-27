"""Unit tests for `rainier.dashboard.render_combined`.

Combined page = shared header (brand + asof + regime chip + rendered-at-pt)
+ top-level radio tabs (Breadth / ETF Ranks) + both section renderers
nested with shared styles emitted exactly once.

Acceptance gates (docs/TASK-PLAN-trading-dashboard-combined-v1.md §Tests):
    1. renders both sections (breadth + ETF) in one HTML.
    2. shared header includes brand line + asof + regime chip + rendered-at-pt.
    3. regime chip carries the threshold-derived css class.
    4. top-level tabs use the CSS radio pattern (no JS routing).
    5. Breadth is the default-active top-level tab (`checked`).
    6. Shared styles emitted exactly once (no duplicate `:root { --bg-page` etc).
    7. Standalone breadth/ETF HTML byte-identical to pre-refactor golden file.
    8. Combined HTML byte-deterministic on identical inputs (3x render).
    9. Mobile breakpoint @media (max-width: 720px) present.
   10. CLI smoke + missing-input cleanly errors.

Determinism is the load-bearing invariant — operator's "any byte change in
standalone outputs = P0 violation, roll back" applies here.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BREADTH_FIX = FIXTURES / "sp500_breadth_small.parquet"
FEATURES_FIX = FIXTURES / "etf_features_small.parquet"
REGISTRY_FIX = FIXTURES / "etf_sector_registry_small.parquet"


# ---------------------------------------------------------------------------
# Fixtures — mirror the existing breadth + ETF test fixtures so the combined
# renderer sees the same inputs the standalone renderers do.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def breadth_df() -> pd.DataFrame:
    return pd.read_parquet(BREADTH_FIX)


@pytest.fixture(scope="module")
def spy_df(breadth_df) -> pd.DataFrame:
    """Synthetic SPY OHLCV frame aligned to the breadth fixture dates.

    Mirrors `tests/test_market_breadth/test_render.py:spy_df` exactly so the
    combined output uses the same SPY pane bytes as the standalone breadth.
    """
    dates = sorted(pd.to_datetime(breadth_df["asof_date"].unique()).date.tolist())
    rows = []
    for i, d in enumerate(dates):
        close = 400.0 + i * 0.1
        rows.append(
            {
                "symbol": "SPY",
                "date": d,
                "open": close - 0.5,
                "high": close + 0.7,
                "low": close - 0.9,
                "close": close,
                "volume": 50_000_000 + i * 1000,
                "fetched_at": datetime(2026, 5, 25, tzinfo=timezone.utc),
                "yfinance_version": "stub",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_FIX)


@pytest.fixture(scope="module")
def registry_df() -> pd.DataFrame:
    return pd.read_parquet(REGISTRY_FIX)


@pytest.fixture
def rendered_combined(breadth_df, spy_df, features_df, registry_df) -> str:
    from rainier.dashboard.render_combined import render_combined_html

    return render_combined_html(
        breadth=breadth_df,
        spy_ohlcv=spy_df,
        features=features_df,
        registry=registry_df,
        asof=date(2026, 5, 25),
        rendered_at_pt="14:32",
    )


# ---------------------------------------------------------------------------
# Combined page structure (5)
# ---------------------------------------------------------------------------


def test_combined_renders_both_sections(rendered_combined):
    """The combined HTML embeds both the breadth charts AND the ETF table.

    Heuristic: breadth signature = the % above MA chart class; ETF signature
    = the ETF rank table id (`etf-all` for the All tab).
    """
    html = rendered_combined
    assert 'class="chart-pct-above-ma"' in html or "chart-pct-above-ma" in html, (
        "breadth section missing — expected `chart-pct-above-ma` SVG"
    )
    assert 'id="etf-all"' in html, "ETF section missing — expected the etf-all table id"


def test_combined_has_shared_header(rendered_combined):
    """Shared header is brand line + asof + regime chip + rendered-at-pt."""
    html = rendered_combined
    assert "rainier" in html and "trading dashboard" in html, "missing brand line"
    assert "2026-05-25" in html, "missing asof date in header"
    assert "14:32" in html, "missing rendered_at_pt in header"
    assert "PT" in html, "missing PT marker in header"
    # Regime chip is part of the header — verified at class level by the next test.


def test_combined_regime_chip_present_and_correct_class(rendered_combined, breadth_df):
    """The header regime chip's css class matches the threshold rule.

    Picks `pct_above_ma_200` for the asof_date and asserts the chip carries
    the class that matches the 70/50/30 threshold bands.
    """
    from rainier.dashboard.shared_styles import _regime_from_pct_above_200d

    # Pull the actual pct_above_ma_200 from the fixture for asof=2026-05-25.
    df = breadth_df.copy()
    if pd.api.types.is_datetime64_any_dtype(df["asof_date"]):
        df["asof_date"] = df["asof_date"].dt.strftime("%Y-%m-%d")
    else:
        df["asof_date"] = df["asof_date"].astype(str)
    df = df[df["asof_date"] <= "2026-05-25"]
    latest = df[df["indicator"] == "pct_above_ma_200"].sort_values("asof_date").iloc[-1]
    pct = float(latest["value"])
    label, css_class = _regime_from_pct_above_200d(pct)

    html = rendered_combined
    # Chip must include the css class verbatim and the label text.
    chip_re = re.compile(r'<[^>]*class="[^"]*regime-chip[^"]*' + re.escape(css_class), re.IGNORECASE)
    assert chip_re.search(html), (
        f"regime chip with css class {css_class!r} not found (pct={pct})"
    )
    assert label in html, f"regime label {label!r} not found in rendered HTML"


def test_combined_top_level_tabs_use_radio_pattern(rendered_combined):
    """Top-level tabs are pure-CSS radio inputs with name='trading-tab'."""
    html = rendered_combined
    # Two radios under the trading-tab namespace — one Breadth, one ETF Ranks.
    radios = re.findall(
        r'<input\b[^>]*\bname="trading-tab"[^>]*>', html
    )
    assert len(radios) == 2, (
        f"expected 2 trading-tab radio inputs, got {len(radios)}\n"
        + "\n".join(radios)
    )
    # No JS-driven tab routing.
    assert 'onclick=' not in html.lower() or 'sorttable' in html.lower() or 'sortetftable' in html.lower(), (
        "top-level tab switching must use CSS radios, not onclick handlers "
        "(only ETF table column-sort JS is permitted)"
    )


def test_combined_breadth_is_default_active_tab(rendered_combined):
    """The Breadth radio is `checked` by default."""
    html = rendered_combined
    breadth_radio_re = re.compile(
        r'<input\b[^>]*\bname="trading-tab"[^>]*\bid="tab-breadth"[^>]*\bchecked',
        re.IGNORECASE,
    )
    assert breadth_radio_re.search(html), (
        "breadth radio (id=tab-breadth, name=trading-tab) must be `checked`"
    )


# ---------------------------------------------------------------------------
# Shared styles refactor (3)
# ---------------------------------------------------------------------------


def test_shared_styles_emits_once_in_combined(rendered_combined):
    """Shared style block is emitted exactly once.

    Heuristic: the shared CSS module declares `--bg-page` in two `:root`
    blocks (one light, one dark-mode `@media (prefers-color-scheme: dark)`).
    The combined output must hold exactly those 2 declarations — neither
    sub-renderer is allowed to duplicate them when invoked with
    `include_shared_styles=False`.

    Equivalently: the count is identical to the count in `shared_styles()`
    alone; any extra is duplication.
    """
    from rainier.dashboard.shared_styles import shared_styles

    expected = shared_styles().count("--bg-page:")
    actual = rendered_combined.count("--bg-page:")
    assert actual == expected, (
        f"--bg-page declared {actual}x in combined HTML; expected exactly "
        f"{expected} (shared styles must emit once at the top, not "
        "duplicate per section)"
    )


def test_standalone_breadth_html_unchanged(breadth_df, spy_df):
    """Standalone breadth output must be byte-identical to the pre-refactor golden.

    This is the load-bearing invariant — the operator's rule:
        "Any byte change in the standalone outputs = P0 violation, roll back."

    The golden file `tests/fixtures/golden/standalone_breadth.html` was
    captured BEFORE refactoring `render_breadth_html` to accept
    `include_shared_styles`. With default `True`, the standalone path
    must produce the same bytes forever (PRs #90-98 protection).
    """
    from rainier.market_breadth.render import render_breadth_html

    golden = (FIXTURES / "golden" / "standalone_breadth.html").read_text(encoding="utf-8")
    actual = render_breadth_html(
        breadth=breadth_df,
        spy_ohlcv=spy_df,
        asof=date(2026, 5, 25),
        rendered_at_pt="12:40",
    )
    assert actual == golden, (
        "standalone breadth HTML drifted from golden file "
        "(see tests/fixtures/golden/standalone_breadth.html). "
        "P0 invariant — roll back any refactor that breaks standalone bytes."
    )


def test_standalone_etf_html_unchanged(features_df, registry_df):
    """Standalone ETF output must be byte-identical to the pre-refactor golden."""
    from rainier.dashboard.render_etf import render_etf_html

    golden = (FIXTURES / "golden" / "standalone_etf.html").read_text(encoding="utf-8")
    actual = render_etf_html(
        features=features_df,
        registry=registry_df,
        asof=date(2026, 5, 25),
        rendered_at_pt="12:40",
        history_days=30,
    )
    assert actual == golden, (
        "standalone ETF HTML drifted from golden file "
        "(see tests/fixtures/golden/standalone_etf.html). "
        "P0 invariant — roll back any refactor that breaks standalone bytes."
    )


# ---------------------------------------------------------------------------
# Regime helper (3)
# ---------------------------------------------------------------------------


def test_regime_from_pct_above_200d_thresholds():
    """Thresholds: >70 strong-bull, 50-70 bullish, 30-50 mixed, <30 bearish.

    Boundary convention: inclusive on the upper side of each band (>=70 = strong-bull,
    >=50 = bullish, >=30 = mixed, <30 = bearish). This matches StockCharts
    convention noted in DESIGN §3.1.
    """
    from rainier.dashboard.shared_styles import _regime_from_pct_above_200d

    # Strong bull — pct > 70.
    assert _regime_from_pct_above_200d(85.0)[0].lower().startswith("strong")
    assert _regime_from_pct_above_200d(70.5)[0].lower().startswith("strong")

    # Bullish — 50 < pct <= 70.
    assert "bull" in _regime_from_pct_above_200d(65.0)[0].lower()
    assert "bull" in _regime_from_pct_above_200d(50.5)[0].lower()

    # Mixed — 30 < pct <= 50.
    assert "mixed" in _regime_from_pct_above_200d(40.0)[0].lower()
    assert "mixed" in _regime_from_pct_above_200d(30.5)[0].lower()

    # Bearish — pct <= 30.
    assert "bear" in _regime_from_pct_above_200d(25.0)[0].lower()
    assert "bear" in _regime_from_pct_above_200d(10.0)[0].lower()


def test_regime_color_class_mapping():
    """Color classes are deterministic per band: strong-bull / bullish / mixed / bearish."""
    from rainier.dashboard.shared_styles import _regime_from_pct_above_200d

    _, c_strong = _regime_from_pct_above_200d(85.0)
    _, c_bull = _regime_from_pct_above_200d(60.0)
    _, c_mixed = _regime_from_pct_above_200d(40.0)
    _, c_bear = _regime_from_pct_above_200d(20.0)
    # Distinct classes per band so CSS can color each chip independently.
    assert len({c_strong, c_bull, c_mixed, c_bear}) == 4, (
        f"regime css classes collapsed to fewer bands: "
        f"{[c_strong, c_bull, c_mixed, c_bear]}"
    )
    # Sanity: each class string is non-empty and css-safe.
    for cls in (c_strong, c_bull, c_mixed, c_bear):
        assert cls and re.match(r"[a-z0-9-]+$", cls), (
            f"css class {cls!r} not lowercase-css-safe"
        )


def test_regime_chip_renders_with_correct_label():
    """`regime_chip_html(pct)` emits a span with both the css class and label text."""
    from rainier.dashboard.shared_styles import (
        _regime_from_pct_above_200d,
        regime_chip_html,
    )

    pct = 65.0  # bullish band
    label, css = _regime_from_pct_above_200d(pct)
    chip = regime_chip_html(pct)
    assert css in chip, f"regime_chip_html({pct}) missing css class {css!r}"
    assert label in chip, f"regime_chip_html({pct}) missing label {label!r}"
    # Must include the regime-chip base class so the shared CSS targets it.
    assert "regime-chip" in chip, "regime_chip_html missing `regime-chip` base class"


# ---------------------------------------------------------------------------
# Determinism + mobile (2)
# ---------------------------------------------------------------------------


def test_combined_byte_identical_on_same_inputs(
    breadth_df, spy_df, features_df, registry_df
):
    """3x render on identical inputs produces identical bytes.

    Catches accidental `datetime.now()` / uuid / dict-order leaks.
    """
    from rainier.dashboard.render_combined import render_combined_html

    kwargs = dict(
        breadth=breadth_df,
        spy_ohlcv=spy_df,
        features=features_df,
        registry=registry_df,
        asof=date(2026, 5, 25),
        rendered_at_pt="14:32",
    )
    a = render_combined_html(**kwargs)
    b = render_combined_html(**kwargs)
    c = render_combined_html(**kwargs)
    assert a == b == c, "combined render is non-deterministic across runs"


def test_combined_responsive_breakpoint_present(rendered_combined):
    """Mobile breakpoint: @media (max-width: 720px) rule present.

    Effect at <720px: hide the radio inputs + force-show both tab panes so
    the page reads as stacked sections (per DESIGN D7).
    """
    html = rendered_combined
    media_re = re.compile(
        r"@media\s*\(\s*max-width\s*:\s*720px\s*\)", re.IGNORECASE
    )
    assert media_re.search(html), (
        "missing @media (max-width: 720px) rule for mobile tab stacking"
    )


def test_combined_escapes_rendered_at_pt(breadth_df, spy_df, features_df, registry_df):
    """`rendered_at_pt` is caller-supplied free-form text that lands in the
    public `/trading/` HTML. The standalone breadth/ETF jinja templates
    autoescape it; the combined renderer hand-assembles HTML via str.join,
    so it must escape too. Otherwise an operator passing a malformed value
    would inject executable markup into the published page (stored XSS).

    Codex iter-2 surfaced this gap.
    """
    from rainier.dashboard.render_combined import render_combined_html

    malicious = '<img src=x onerror="alert(1)">'
    html = render_combined_html(
        breadth=breadth_df,
        spy_ohlcv=spy_df,
        features=features_df,
        registry=registry_df,
        asof=date(2026, 5, 25),
        rendered_at_pt=malicious,
    )
    # The raw `<img ...>` payload must NOT appear unescaped anywhere.
    assert malicious not in html, (
        "rendered_at_pt was NOT escaped — XSS payload landed verbatim in the "
        "public HTML. Header writes must go through markupsafe.escape() so "
        "they match the autoescape behavior of the nested jinja templates."
    )
    # The escaped form must appear (proving the value was actually emitted,
    # just safely). Match either &lt; or &amp;lt; (paranoid double-escape).
    assert "&lt;img" in html or "&amp;lt;img" in html, (
        "expected escaped form of the rendered_at_pt payload in the HTML; "
        "renderer may be dropping the value entirely instead of escaping it"
    )


# ---------------------------------------------------------------------------
# CLI (2)
# ---------------------------------------------------------------------------


def test_render_combined_cli_smoke(tmp_path):
    """`rainier dashboard render-combined` writes a valid combined HTML.

    Spawns a subprocess so we exercise click wiring end-to-end. Uses the
    same parquet fixtures the unit tests rely on; SPY parquet is generated
    inline so we don't need a fixture parquet on disk.
    """
    import subprocess
    from datetime import datetime, timezone

    # Generate SPY parquet on disk (CLI reads parquet, not in-memory frames).
    breadth = pd.read_parquet(BREADTH_FIX)
    dates = sorted(pd.to_datetime(breadth["asof_date"].unique()).date.tolist())
    spy_rows = []
    for i, d in enumerate(dates):
        close = 400.0 + i * 0.1
        spy_rows.append(
            {
                "symbol": "SPY",
                "date": d,
                "open": close - 0.5,
                "high": close + 0.7,
                "low": close - 0.9,
                "close": close,
                "volume": 50_000_000 + i * 1000,
                "fetched_at": datetime(2026, 5, 25, tzinfo=timezone.utc),
                "yfinance_version": "stub",
            }
        )
    spy_path = tmp_path / "spy.parquet"
    pd.DataFrame(spy_rows).to_parquet(spy_path)

    out_path = tmp_path / "trading.html"
    cmd = [
        "uv", "run", "rainier", "dashboard", "render-combined",
        "--breadth-input", str(BREADTH_FIX),
        "--etf-features", str(FEATURES_FIX),
        "--etf-registry", str(REGISTRY_FIX),
        "--spy-path", str(spy_path),
        "--asof", "2026-05-25",
        "--rendered-at-pt", "14:32",
        "--output", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        f"CLI failed (exit={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert out_path.exists(), f"CLI did not write output at {out_path}"
    html = out_path.read_text(encoding="utf-8")
    # Spot-check: both sections present.
    assert "chart-pct-above-ma" in html, "CLI output missing breadth chart"
    assert 'id="etf-all"' in html, "CLI output missing ETF table"


def test_render_combined_missing_input_errors_cleanly(tmp_path):
    """Missing input parquet → non-zero exit + a useful error message."""
    import subprocess

    out_path = tmp_path / "trading.html"
    cmd = [
        "uv", "run", "rainier", "dashboard", "render-combined",
        "--breadth-input", str(tmp_path / "missing.parquet"),
        "--etf-features", str(FEATURES_FIX),
        "--etf-registry", str(REGISTRY_FIX),
        "--asof", "2026-05-25",
        "--rendered-at-pt", "14:32",
        "--output", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0, (
        "CLI should exit non-zero when --breadth-input is missing\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    # Click's Path(exists=True) check or our own error should mention the file.
    combined = (proc.stdout + proc.stderr).lower()
    assert "missing" in combined or "does not exist" in combined or "no such" in combined, (
        f"CLI error message should mention missing input; got:\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
