"""Unit tests for `rainier.dashboard.render_etf`.

The renderer is a pure function over parquet inputs → HTML output. Tests
exercise it on a small deterministic fixture (3 sectors × 4 tickers × 35
days) and assert on the rendered HTML.

Acceptance gates (from docs/TASK-PLAN-etf-dashboard-renderer-efed.md):
    - self-contained HTML (no external <link>, <script src>, <img src=http>)
    - default sort = sector ASC → rank DESC within sector
    - one <path> per ticker in the All-ETFs tab sparkline column
    - Top-15 tab filters rank ≥ 85
    - Movers tab filters |Δ1d|≥10 OR |Δ5d|≥15
    - header timestamp: literal "Last updated: YYYY-MM-DD HH:MM PT"
    - @media (prefers-color-scheme: dark) block present
    - byte-equal output on re-render
    - never opens a DB session
    - graceful render when a ticker has <30 days of history
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Fixture path is committed alongside the test, generated once via
# scripts/_make_etf_fixture.py. Sibling registry parquet at same path.
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FEATURES_FIX = FIXTURES / "etf_features_small.parquet"
REGISTRY_FIX = FIXTURES / "etf_sector_registry_small.parquet"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_FIX)
    return df


@pytest.fixture(scope="module")
def registry_df() -> pd.DataFrame:
    return pd.read_parquet(REGISTRY_FIX)


@pytest.fixture
def rendered_html(features_df, registry_df) -> str:
    from rainier.dashboard.render_etf import render_etf_html

    return render_etf_html(
        features=features_df,
        registry=registry_df,
        asof=date(2026, 5, 25),
        rendered_at_pt="12:40",
        history_days=30,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_render_html_is_self_contained(rendered_html):
    """No external CSS, no remote <script src>, no remote <img src>."""
    html = rendered_html
    # <link rel="stylesheet" href="http..."> not allowed.
    assert not re.search(
        r'<link\b[^>]*rel=["\']stylesheet["\'][^>]*href=["\']https?:', html
    ), "external stylesheet link found in rendered HTML"
    # <script src="..."> not allowed (inline <script>...</script> is fine).
    assert not re.search(r"<script\b[^>]*\bsrc=", html), "external script src found"
    # <img src="http..."> not allowed.
    assert not re.search(r'<img\b[^>]*src=["\']https?:', html), "external img src found"
    # No CDN-style protocol-relative either.
    assert "src=\"//" not in html and "href=\"//" not in html, "protocol-relative URL"


def test_columns_symbol_first_sector_second(rendered_html):
    """Header row of the All-ETFs table is Symbol, Sector, Rank, Δ1d, Δ5d, R5, R10, R20, YTD, 30d."""
    html = rendered_html
    all_block = _extract_tab_block(html, "all")
    # Pull the <thead> row's <th> texts.
    thead = re.search(r"<thead>(.*?)</thead>", all_block, re.DOTALL)
    assert thead, "no <thead> in All-ETFs table"
    th_texts = [_TAGS_RE.sub("", c).strip() for c in re.findall(r"<th\b[^>]*>(.*?)</th>", thead.group(1), re.DOTALL)]
    # The Δ characters come through as &Delta;; normalise for compare.
    th_texts = [t.replace("&Delta;", "Δ") for t in th_texts]
    assert th_texts[0] == "Symbol", f"first column should be Symbol, got: {th_texts}"
    assert th_texts[1] == "Sector", f"second column should be Sector, got: {th_texts}"
    assert th_texts[2] == "Rank", f"third column should be Rank, got: {th_texts}"
    # Tail columns unchanged.
    assert th_texts[3] == "Δ1d"
    assert th_texts[4] == "Δ5d"
    assert th_texts[5] == "R5"
    assert th_texts[6] == "R10"
    assert th_texts[7] == "R20"
    assert th_texts[8] == "YTD"
    assert th_texts[9] == "30d"


def test_tier_default_is_3():
    """`IMPORTANCE_TIER.get("UNKNOWN_SYMBOL", TIER_DEFAULT)` returns 3."""
    from rainier.dashboard.render_etf import IMPORTANCE_TIER, TIER_DEFAULT

    assert TIER_DEFAULT == 3
    assert IMPORTANCE_TIER.get("UNKNOWN_SYMBOL", TIER_DEFAULT) == 3
    # Spot-check a few known mappings.
    assert IMPORTANCE_TIER["SMH"] == 0
    assert IMPORTANCE_TIER["XLE"] == 1
    assert IMPORTANCE_TIER["XBI"] == 2


def _build_tier_features(rows: list[dict]) -> pd.DataFrame:
    """Build a features DataFrame for tier-sort tests. Each row needs the
    required numeric columns; tests vary symbol/sector_id/rank."""
    base_cols = {
        "rank_delta_1d": 0,
        "rank_delta_5d": 0,
        "r_5": 50,
        "r_10": 50,
        "r_20": 50,
        "ret_ytd": 0.05,
        "top15_streak": 0,
    }
    full_rows = []
    for r in rows:
        full_rows.append({**base_cols, "asof_date": "2026-05-25", **r})
    df = pd.DataFrame(full_rows)
    df["asof_date"] = df["asof_date"].astype("string")
    return df


def test_default_sort_is_importance_then_rank_desc():
    """Tier dominates rank: tier 0 first, then 1, then 2, then 3."""
    from rainier.dashboard.render_etf import render_etf_html

    features = _build_tier_features([
        {"symbol": "XLE",  "sector_id": 1, "rank": 80},   # tier 1
        {"symbol": "XBI",  "sector_id": 2, "rank": 95},   # tier 2
        {"symbol": "MEME", "sector_id": 3, "rank": 99},   # tier 3
        {"symbol": "SMH",  "sector_id": 4, "rank": 50},   # tier 0
    ])
    registry = pd.DataFrame([
        {"sector_id": 1, "sector_name": "energy"},
        {"sector_id": 2, "sector_name": "biotech"},
        {"sector_id": 3, "sector_name": "meme"},
        {"sector_id": 4, "sector_name": "semis"},
    ])
    html = render_etf_html(
        features=features, registry=registry,
        asof=date(2026, 5, 25), rendered_at_pt="12:40",
    )
    all_block = _extract_tab_block(html, "all")
    symbols = _extract_symbols(all_block)
    assert symbols == ["SMH", "XLE", "XBI", "MEME"], (
        f"tier sort wrong; got {symbols}, expected [SMH, XLE, XBI, MEME]"
    )


def test_default_sort_within_tier_uses_rank_desc():
    """Within the same tier, higher rank comes first."""
    from rainier.dashboard.render_etf import render_etf_html

    # Two tier-1 ETFs with different ranks.
    features = _build_tier_features([
        {"symbol": "XLE", "sector_id": 1, "rank": 80},
        {"symbol": "XLF", "sector_id": 1, "rank": 90},
    ])
    registry = pd.DataFrame([{"sector_id": 1, "sector_name": "energy"}])
    html = render_etf_html(
        features=features, registry=registry,
        asof=date(2026, 5, 25), rendered_at_pt="12:40",
    )
    all_block = _extract_tab_block(html, "all")
    symbols = _extract_symbols(all_block)
    assert symbols == ["XLF", "XLE"], (
        f"within-tier rank-desc wrong; got {symbols}, expected [XLF, XLE]"
    )


def test_within_tier_ties_sort_alphabetically():
    """Within the same (tier, rank), tiebreak by symbol ascending."""
    from rainier.dashboard.render_etf import render_etf_html

    # Two tier-1 ETFs with identical ranks.
    features = _build_tier_features([
        {"symbol": "XLC", "sector_id": 1, "rank": 85},
        {"symbol": "XLB", "sector_id": 1, "rank": 85},
    ])
    registry = pd.DataFrame([{"sector_id": 1, "sector_name": "materials"}])
    html = render_etf_html(
        features=features, registry=registry,
        asof=date(2026, 5, 25), rendered_at_pt="12:40",
    )
    all_block = _extract_tab_block(html, "all")
    symbols = _extract_symbols(all_block)
    assert symbols == ["XLB", "XLC"], (
        f"alphabetical tiebreak wrong; got {symbols}, expected [XLB, XLC]"
    )


def test_sparkline_path_count(rendered_html, features_df):
    """One <path> per ticker in the All-ETFs tab (12 tickers in fixture)."""
    html = rendered_html
    # Parse just the All-ETFs tab so the Top-15 / Movers subset paths don't
    # inflate the count.
    all_block = _extract_tab_block(html, "all")
    # Count <path d="..."> elements inside the table block.
    path_count = len(re.findall(r"<path\b[^>]*\bd=", all_block))
    latest_asof = features_df["asof_date"].max()
    expected_tickers = features_df.loc[features_df["asof_date"] == latest_asof, "symbol"].nunique()
    assert path_count == expected_tickers, (
        f"expected {expected_tickers} sparkline paths in All-ETFs tab, got {path_count}"
    )


def test_top15_tab_filters_rank_ge_85(rendered_html, features_df):
    """Top-15 tab section renders only rows where rank >= 85 in fixture."""
    html = rendered_html
    block = _extract_tab_block(html, "top15")
    rows = _parse_table_rows(block)
    assert rows, "no rows in Top-15 tab"
    for _sec, _sym, rk in rows:
        assert rk >= 85, f"Top-15 row has rank<85: {_sym}={rk}"
    # Cross-check against the fixture: count of rank≥85 on latest day.
    latest_asof = features_df["asof_date"].max()
    latest = features_df.loc[features_df["asof_date"] == latest_asof]
    expected = int((latest["rank"] >= 85).sum())
    assert len(rows) == expected, f"expected {expected} Top-15 rows, got {len(rows)}"


def test_movers_tab_filters_correctly(rendered_html, features_df):
    """Movers tab renders only rows where |Δ1d|≥10 OR |Δ5d|≥15."""
    html = rendered_html
    block = _extract_tab_block(html, "movers")
    symbols = _extract_symbols(block)
    assert symbols, "no rows in Movers tab"
    latest_asof = features_df["asof_date"].max()
    latest = features_df.loc[features_df["asof_date"] == latest_asof]
    expected = set(
        latest.loc[
            (latest["rank_delta_1d"].abs() >= 10) | (latest["rank_delta_5d"].abs() >= 15),
            "symbol",
        ]
    )
    assert set(symbols) == expected, (
        f"Movers tab mismatch — got {sorted(symbols)} expected {sorted(expected)}"
    )


def test_header_timestamp_is_pt(features_df, registry_df):
    """Header includes literal `Last updated: 2026-05-25 12:40 PT`."""
    from rainier.dashboard.render_etf import render_etf_html

    html = render_etf_html(
        features=features_df,
        registry=registry_df,
        asof=date(2026, 5, 25),
        rendered_at_pt="12:40",
        history_days=30,
    )
    assert "Last updated: 2026-05-25 12:40 PT" in html


def test_dark_mode_block_present(rendered_html):
    """Rendered CSS contains a `@media (prefers-color-scheme: dark)` rule."""
    assert "@media (prefers-color-scheme: dark)" in rendered_html


def test_render_is_deterministic(features_df, registry_df):
    """Same inputs → byte-identical output across two consecutive renders."""
    from rainier.dashboard.render_etf import render_etf_html

    kwargs = dict(
        features=features_df,
        registry=registry_df,
        asof=date(2026, 5, 25),
        rendered_at_pt="12:40",
        history_days=30,
    )
    out1 = render_etf_html(**kwargs)
    out2 = render_etf_html(**kwargs)
    assert out1 == out2, "renderer is not deterministic across consecutive calls"


def test_render_does_not_touch_db(features_df, registry_df):
    """`render_etf_html` must not open a DB session — pure parquet read."""
    from rainier.dashboard import render_etf as mod

    # Patch the core database accessors to explode if called. The renderer
    # lives under src/rainier/dashboard/, sibling to data.py which DOES
    # import get_session — render_etf must NOT.
    sentinel = RuntimeError("renderer touched the database")

    def _explode(*_a, **_kw):
        raise sentinel

    with patch("rainier.core.database.get_session", side_effect=_explode), patch(
        "rainier.core.database.get_engine", side_effect=_explode
    ):
        out = mod.render_etf_html(
            features=features_df,
            registry=registry_df,
            asof=date(2026, 5, 25),
            rendered_at_pt="12:40",
            history_days=30,
        )
    assert "<html" in out.lower()


def test_missing_cells_get_data_missing_marker():
    """Regression: cells with missing data must carry `data-missing="1"`.

    The client-side `sortEtfTable` reads `data-missing` to push missing
    rows to the bottom regardless of sort direction. Without the marker
    the very-negative `ytd_sort_key` would cluster missing-YTD rows at the
    top of ascending order — wrong: missing-data is not the same as
    "lowest value", it's "no value". The marker fixes this contract.
    """
    from rainier.dashboard.render_etf import render_etf_html

    nan = float("nan")
    features = pd.DataFrame(
        [
            {
                "asof_date": "2026-05-25",
                "symbol": "MISS",
                "sector_id": 1,
                "rank": -1,
                "rank_delta_1d": 0,
                "rank_delta_5d": 0,
                "r_5": -1,
                "r_10": -1,
                "r_20": -1,
                "ret_ytd": nan,
                "top15_streak": 0,
            },
            {
                "asof_date": "2026-05-25",
                "symbol": "FULL",
                "sector_id": 1,
                "rank": 50,
                "rank_delta_1d": 0,
                "rank_delta_5d": 0,
                "r_5": 50,
                "r_10": 50,
                "r_20": 50,
                "ret_ytd": 0.1,
                "top15_streak": 0,
            },
        ]
    )
    features["asof_date"] = features["asof_date"].astype("string")
    registry = pd.DataFrame([{"sector_id": 1, "sector_name": "energy"}])
    html = render_etf_html(
        features=features, registry=registry,
        asof=date(2026, 5, 25), rendered_at_pt="12:40",
    )

    miss_block = re.search(r"<tr>\s*<td>MISS</td>\s*<td>energy</td>(.*?)</tr>", html, re.DOTALL)
    full_block = re.search(r"<tr>\s*<td>FULL</td>\s*<td>energy</td>(.*?)</tr>", html, re.DOTALL)
    assert miss_block and full_block, "rows not found"

    # MISS row: every numeric cell with a missing value must carry data-missing="1"
    miss_cells = miss_block.group(1)
    assert miss_cells.count('data-missing="1"') == 5, (
        f"expected 5 data-missing markers (rank,r5,r10,r20,ytd) on MISS row, "
        f"got {miss_cells.count('data-missing=')!r}: {miss_cells!r}"
    )

    # FULL row: no data-missing markers — all values are real.
    full_cells = full_block.group(1)
    assert 'data-missing=' not in full_cells, (
        f"FULL row should not have data-missing markers: {full_cells!r}"
    )

    # Sort JS knows to push them last.
    assert 'data-missing' in html and "ma && !mb" in html, (
        "sortEtfTable JS must consult data-missing"
    )


def test_negative_rank_sentinels_render_as_em_dash():
    """Regression: `rank == -1` and `r_5/r_10/r_20 == -1` must display as '—'.

    The upstream features parquet uses -1 as the "no data yet" sentinel
    for rank-like columns during a ticker's first ~20 trading days.
    Without explicit handling these surface as a literal `-1` in the
    Rank / R5 / R10 / R20 cells of the published dashboard. Replace
    them with an em-dash; sort keys keep the raw int so column sorting
    is deterministic (missing rows cluster at the bottom of ascending).
    """
    from rainier.dashboard.render_etf import render_etf_html

    features = pd.DataFrame(
        [
            {
                "asof_date": "2026-05-25",
                "symbol": "NEW",
                "sector_id": 1,
                "rank": -1,
                "rank_delta_1d": 0,
                "rank_delta_5d": 0,
                "r_5": -1,
                "r_10": -1,
                "r_20": -1,
                "ret_ytd": 0.0,
                "top15_streak": 0,
            },
            {
                "asof_date": "2026-05-25",
                "symbol": "OLD",
                "sector_id": 1,
                "rank": 90,
                "rank_delta_1d": 0,
                "rank_delta_5d": 0,
                "r_5": 50,
                "r_10": 60,
                "r_20": 70,
                "ret_ytd": 0.10,
                "top15_streak": 5,
            },
        ]
    )
    features["asof_date"] = features["asof_date"].astype("string")
    registry = pd.DataFrame([{"sector_id": 1, "sector_name": "energy"}])
    html = render_etf_html(
        features=features, registry=registry,
        asof=date(2026, 5, 25), rendered_at_pt="12:40",
    )

    # NEW row's <td> cells should NOT contain a literal `>-1<` for any
    # rank-like column.
    new_block = re.search(r"<tr>\s*<td>NEW</td>\s*<td>energy</td>(.*?)</tr>", html, re.DOTALL)
    assert new_block, "NEW row not found in rendered HTML"
    new_cells = new_block.group(1)
    assert ">-1<" not in new_cells, (
        f"raw -1 sentinel leaked into displayed cell: {new_cells!r}"
    )
    # The em-dash is present in those cells.
    assert "—" in new_cells, "missing-data em-dash absent from NEW row"
    # Sort keys must still be the raw -1 so column-click sorting works.
    assert 'data-sort="-1"' in new_cells, "rank sort key not preserved"

    # OLD row still displays its real values.
    old_block = re.search(r"<tr>\s*<td>OLD</td>\s*<td>energy</td>(.*?)</tr>", html, re.DOTALL)
    assert old_block, "OLD row not found"
    old_cells = old_block.group(1)
    assert ">90<" in old_cells, "OLD rank display value missing"
    assert ">50<" in old_cells, "OLD r5 display value missing"


def test_render_handles_nan_numeric_fields():
    """Regression: NaN in any numeric field must not crash or render '+nan%'.

    The prod parquet emits NaN for fields that aren't populated yet (a
    newly added ETF before its first YTD base date, missing intermediate
    momentum returns, etc.). Naive `x or default` does NOT substitute the
    default because NaN is truthy; `int(NaN)` raises ValueError. Every
    numeric pull goes through _safe_int / _is_missing so a missing value
    renders as '—' (for percentages) or a sensible sentinel (for ints).
    """
    from rainier.dashboard.render_etf import render_etf_html

    nan = float("nan")
    features = pd.DataFrame(
        [
            {
                "asof_date": "2026-05-25",
                "symbol": "NEW",
                "sector_id": 1,
                "rank": 50,
                "rank_delta_1d": nan,
                "rank_delta_5d": nan,
                "r_5": nan,
                "r_10": nan,
                "r_20": nan,
                "ret_ytd": nan,
                "top15_streak": nan,
            },
            {
                "asof_date": "2026-05-25",
                "symbol": "OLD",
                "sector_id": 1,
                "rank": 90,
                "rank_delta_1d": 0,
                "rank_delta_5d": 0,
                "r_5": 50,
                "r_10": 50,
                "r_20": 50,
                "ret_ytd": 0.15,
                "top15_streak": 5,
            },
        ]
    )
    features["asof_date"] = features["asof_date"].astype("string")
    registry = pd.DataFrame([{"sector_id": 1, "sector_name": "energy"}])

    # This must not raise (was crashing with int(NaN) before the fix).
    html = render_etf_html(
        features=features, registry=registry,
        asof=date(2026, 5, 25), rendered_at_pt="12:40",
    )
    # No '+nan%' or 'nan%' or 'NaN' anywhere in rendered output.
    assert "nan" not in html.lower() or "atkinson" in html.lower(), (
        "NaN leaked into rendered HTML"
    )
    # Spot-check: the case-insensitive 'nan' substring lives inside
    # 'Atkinson Hyperlegible' (the font name has 'nan' inside 'Atkinson').
    # Strip the font reference before checking.
    no_font = html.replace("Atkinson Hyperlegible", "")
    assert "nan" not in no_font.lower(), f"NaN leak: {[m for m in no_font.lower().split() if 'nan' in m][:5]}"
    # OLD's YTD still renders normally.
    assert "+15.0%" in html, "OLD ticker's YTD missing or malformed"
    # NEW's YTD missing token present.
    assert "—" in html, "missing-YTD token '—' not rendered"


def test_sparkline_skips_missing_rank_sentinel():
    """Regression: `rank == -1` (missing-data sentinel) must not paint as 0.

    The upstream pipeline writes `rank = -1` for a ticker's first ~20
    trading days. Naive `_sparkline_svg` clamps negative ranks to 0, which
    renders as a real bottom-of-range point and creates a fake collapse
    curve. The history lookup must filter -1 BEFORE the SVG is built.
    """
    from rainier.dashboard.render_etf import render_etf_html

    # Ticker NEW has 5 sentinel days then 5 valid days. Sparkline should
    # only reflect the valid 5.
    rows = []
    for i, d in enumerate(["2026-05-16","2026-05-17","2026-05-18","2026-05-19","2026-05-20"]):
        rows.append({"asof_date": d, "symbol": "NEW", "sector_id": 1, "rank": -1,
                     "rank_delta_1d": 0, "rank_delta_5d": 0,
                     "r_5": 50, "r_10": 50, "r_20": 50, "ret_ytd": 0.1, "top15_streak": 0})
    for i, d in enumerate(["2026-05-21","2026-05-22","2026-05-23","2026-05-24","2026-05-25"]):
        rows.append({"asof_date": d, "symbol": "NEW", "sector_id": 1, "rank": 90 - i,
                     "rank_delta_1d": 0, "rank_delta_5d": 0,
                     "r_5": 50, "r_10": 50, "r_20": 50, "ret_ytd": 0.1, "top15_streak": 0})
    features = pd.DataFrame(rows)
    features["asof_date"] = features["asof_date"].astype("string")
    registry = pd.DataFrame([{"sector_id": 1, "sector_name": "energy"}])
    html = render_etf_html(
        features=features, registry=registry,
        asof=date(2026, 5, 25), rendered_at_pt="12:40",
    )

    # Pull the NEW row's sparkline path d-attribute. Each path looks like:
    # `<path d="M0.00,3.30 L20.00,3.50 L40.00,3.70 ...">`. We extract the
    # numeric y-values and assert NONE are clamped to the bottom-of-range
    # (which would be _SPARK_H - _SPARK_PAD ≈ 18.5 — what y looks like
    # when rank=0 after the invert). A flat midline rendered for missing
    # data sits at y=10 (the _SPARK_H/2 fallback).
    new_row = re.search(r"<td>NEW</td>(.*?)</tr>", html, re.DOTALL)
    assert new_row, "NEW row not in rendered HTML"
    spark_path = re.search(r'<path d="([^"]+)"', new_row.group(1))
    assert spark_path, "NEW row missing sparkline path"
    d_attr = spark_path.group(1)
    # Pull y values from `Mx,y` and `Lx,y` tokens.
    ys = [float(m.group(1)) for m in re.finditer(r"[ML][^,]+,([0-9.]+)", d_attr)]
    # Allow a 0.5 tolerance below 18.5 — the bottom-clamp boundary.
    bottom_clamped = [y for y in ys if y >= 18.0]
    assert not bottom_clamped, (
        f"sparkline has bottom-clamped points {bottom_clamped} from -1 sentinels; "
        f"path={d_attr}"
    )


def test_tab_radios_are_hidden_by_a_matching_selector(rendered_html):
    """Regression: the radio inputs powering the CSS tabs must be hidden.

    The radios live as siblings BEFORE `<div class="tabs">` (so the `~`
    general-sibling selector that toggles `:checked ~ section` works).
    A selector like `.tabs input[type=radio]` will NOT match them and the
    raw OS radio buttons leak above the tab labels in the published page.
    """
    html = rendered_html
    # Find a CSS rule that targets the radios specifically and hides them.
    # We accept either `display: none` or the visually-hidden absolute pattern,
    # as long as the rule actually mentions one of the three radio ids.
    style_block = re.search(r"<style>(.*?)</style>", html, re.DOTALL)
    assert style_block, "no <style> block found"
    css = style_block.group(1)
    targets_radio = re.search(
        r"(input\[type=radio\]#tab-(?:all|top15|movers)|#tab-(?:all|top15|movers)\b[^,{]*\{[^}]*\b(?:display\s*:\s*none|opacity\s*:\s*0)[^}]*\})",
        css,
    )
    assert targets_radio, (
        "no CSS rule hides the tab radio inputs by id — they will render "
        "as visible OS radio buttons above the tab labels"
    )


def test_html_escapes_untrusted_registry_strings():
    """Regression: sector_name from sector_registry.parquet must be HTML-escaped.

    The rendered dashboard is published to a public path. Even though the
    registry is operator-edited, treating it as trusted means a typo with
    a stray `<` breaks the page and a copy/paste with `<script>` becomes
    an XSS in any browser that opens the file. Autoescape MUST be on.
    """
    from rainier.dashboard.render_etf import render_etf_html

    features = pd.DataFrame(
        [
            {
                "asof_date": "2026-05-25",
                "symbol": "XLE",
                "sector_id": 1,
                "rank": 50,
                "rank_delta_1d": 0,
                "rank_delta_5d": 0,
                "r_5": 50,
                "r_10": 50,
                "r_20": 50,
                "ret_ytd": 0.1,
                "top15_streak": 0,
            }
        ]
    )
    features["asof_date"] = features["asof_date"].astype("string")
    payload = "<script>document.title='PWNED'</script>"
    registry = pd.DataFrame([{"sector_id": 1, "sector_name": payload}])

    html = render_etf_html(
        features=features,
        registry=registry,
        asof=date(2026, 5, 25),
        rendered_at_pt="12:40",
    )
    assert payload not in html, "untrusted sector_name leaked into HTML unescaped"
    assert "&lt;script&gt;" in html, "sector_name was not HTML-escaped"
    # Sparkline SVG (pre-rendered, marked Markup) must still pass through
    # raw — verify the autoescape didn't double-escape pre-built HTML.
    assert "<svg" in html and "<path" in html, "sparkline SVG was double-escaped"


def test_render_works_with_missing_history(features_df, registry_df):
    """Ticker with <30 days of history renders without crashing (flat sparkline ok)."""
    from rainier.dashboard.render_etf import render_etf_html

    # The fixture deliberately has XES with only ~20 days of history.
    latest_asof = features_df["asof_date"].max()
    xes_history = features_df.loc[features_df["symbol"] == "XES"]
    assert len(xes_history) < 30, "fixture sanity: XES must have <30 days of history"
    assert (xes_history["asof_date"] == latest_asof).any(), (
        "fixture sanity: XES must still appear on the latest asof_date"
    )

    html = render_etf_html(
        features=features_df,
        registry=registry_df,
        asof=date(2026, 5, 25),
        rendered_at_pt="12:40",
        history_days=30,
    )
    # XES must appear in the rendered All-ETFs tab body.
    all_block = _extract_tab_block(html, "all")
    assert ">XES<" in all_block, "XES (missing-history ticker) absent from rendered HTML"


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------
#
# The HTML structure is a contract between the renderer and the test:
#   - Each tab is a <section data-tab="all|top15|movers"> ... </section>
#   - Inside each section there's a <table class="etf-table"> ... </table>
#   - Each data <tr> has cells in order: sector, symbol, rank, ...
# Tests parse via cheap regex; if the renderer changes the surface the
# tests update alongside.


_TAB_BLOCK_RE = re.compile(
    r"<section[^>]*\bdata-tab=\"(?P<tab>[a-z0-9]+)\"[^>]*>(?P<body>.*?)</section>",
    re.DOTALL,
)


def _extract_tab_block(html: str, tab: str) -> str:
    for match in _TAB_BLOCK_RE.finditer(html):
        if match.group("tab") == tab:
            return match.group("body")
    raise AssertionError(f"tab section '{tab}' not found in rendered HTML")


_TR_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.DOTALL)
_TD_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.DOTALL)
_TAGS_RE = re.compile(r"<[^>]+>")


def _parse_table_rows(html_block: str) -> list[tuple[str, str, int]]:
    """Return list of (sector, symbol, rank) tuples from data rows.

    Column order in the rendered table is Symbol | Sector | Rank | ...,
    but the returned tuple keeps the legacy (sector, symbol, rank) shape
    so test assertions don't need to flip.
    """
    rows: list[tuple[str, str, int]] = []
    for tr in _TR_RE.finditer(html_block):
        cells = _TD_RE.findall(tr.group(1))
        if len(cells) < 3:
            continue
        # Strip inner tags + whitespace. Column order: symbol, sector, rank.
        symbol = _TAGS_RE.sub("", cells[0]).strip()
        sector = _TAGS_RE.sub("", cells[1]).strip()
        rank_text = _TAGS_RE.sub("", cells[2]).strip()
        try:
            rank = int(rank_text)
        except ValueError:
            # header row or non-numeric — skip.
            continue
        if not sector or not symbol:
            continue
        rows.append((sector, symbol, rank))
    return rows


def _extract_symbols(html_block: str) -> list[str]:
    return [sym for _sec, sym, _rk in _parse_table_rows(html_block)]
