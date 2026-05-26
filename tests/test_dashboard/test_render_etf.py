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


def test_sort_is_sector_then_rank_desc(rendered_html):
    """Default All-ETFs table is sorted by sector ASC, then rank DESC within sector."""
    html = rendered_html
    # Pull the All-ETFs table block.
    all_block = _extract_tab_block(html, "all")
    rows = _parse_table_rows(all_block)
    # Rows are (sector, symbol, rank); pair-wise check.
    assert rows, "no rows parsed from All ETFs tab"
    # Sectors must appear in alphabetical order overall.
    sector_order = []
    seen = set()
    for sector, _sym, _rank in rows:
        if sector not in seen:
            sector_order.append(sector)
            seen.add(sector)
    assert sector_order == sorted(sector_order), f"sectors not alphabetical: {sector_order}"
    # Within each sector, rank must be monotone non-increasing.
    by_sector: dict[str, list[int]] = {}
    for sec, _sym, rk in rows:
        by_sector.setdefault(sec, []).append(rk)
    for sec, ranks in by_sector.items():
        assert ranks == sorted(ranks, reverse=True), f"sector={sec} ranks not desc: {ranks}"


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
    """Return list of (sector, symbol, rank) tuples from data rows."""
    rows: list[tuple[str, str, int]] = []
    for tr in _TR_RE.finditer(html_block):
        cells = _TD_RE.findall(tr.group(1))
        if len(cells) < 3:
            continue
        # Strip inner tags + whitespace.
        sector = _TAGS_RE.sub("", cells[0]).strip()
        symbol = _TAGS_RE.sub("", cells[1]).strip()
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
