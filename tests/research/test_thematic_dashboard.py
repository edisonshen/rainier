"""Dashboard render tests — byte-identical on byte-identical input.

Design ref: docs/DESIGN-thematic-ranks-dashboard.md §6 + [D-009].
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rainier.viz.thematic_dashboard import render_dashboard


def _build_features_df() -> pd.DataFrame:
    """Build a deterministic features DataFrame for one asof_date."""
    sectors = ["tech", "tech", "healthcare", "healthcare", "energy"]
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    sector_ids = [1, 1, 2, 2, 3]
    ranks = [90, 70, 50, 30, 10]
    rows = []
    for i, sym in enumerate(symbols):
        rows.append(
            {
                "asof_date": date(2024, 11, 8),
                "trading_day_ordinal": np.int32(100),
                "symbol": sym,
                "ticker_id": np.int16(i + 1),
                "sector_id": np.int8(sector_ids[i]),
                "sector_name": sectors[i],
                "close": np.float64(100.0 + i),
                "ret_1d": np.float32(0.01),
                "rel_5": np.float32(0.02),
                "rel_10": np.float32(0.03),
                "rel_20": np.float32(0.04),
                "ret_ytd": np.float32(0.10),
                "relvol_5": np.float32(0.5),
                "relvol_10": np.float32(0.5),
                "relvol_20": np.float32(0.5),
                "r_5": np.int8(ranks[i]),
                "r_10": np.int8(ranks[i]),
                "r_20": np.int8(ranks[i]),
                "rank": np.int8(ranks[i]),
                "rank_delta_1d": np.int8(0),
                "rank_delta_5d": np.int8(0),
                "top15_streak": np.int16(0),
                "rank_minus_sector_mean": np.float32(0.0),
                "universe_size": np.int16(5),
                "universe_yaml_sha": "abc123",
                "computed_at": pd.Timestamp("2024-11-08T16:30:00Z"),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def features_df() -> pd.DataFrame:
    return _build_features_df()


# ---------------------------------------------------------------------------
# Render produces an HTML file
# ---------------------------------------------------------------------------


def test_render_writes_html_file(features_df, tmp_path: Path):
    out = tmp_path / "thematic-ranks-2024-11-08.html"
    render_dashboard(features_df, out_path=out, asof=date(2024, 11, 8))
    assert out.exists()
    content = out.read_text()
    assert content.startswith("<!DOCTYPE html") or content.lstrip().startswith("<!DOCTYPE")
    assert "</html>" in content


def test_render_contains_all_tickers(features_df, tmp_path: Path):
    out = tmp_path / "dashboard.html"
    render_dashboard(features_df, out_path=out, asof=date(2024, 11, 8))
    html = out.read_text()
    for sym in ("AAA", "BBB", "CCC", "DDD", "EEE"):
        assert sym in html, f"ticker {sym} missing from HTML"


def test_render_contains_sector_labels(features_df, tmp_path: Path):
    out = tmp_path / "dashboard.html"
    render_dashboard(features_df, out_path=out, asof=date(2024, 11, 8))
    html = out.read_text()
    for sector in ("tech", "healthcare", "energy"):
        assert sector in html, f"sector {sector} missing from HTML"


def test_render_contains_rank_values(features_df, tmp_path: Path):
    out = tmp_path / "dashboard.html"
    render_dashboard(features_df, out_path=out, asof=date(2024, 11, 8))
    html = out.read_text()
    # Rank values should appear (90, 70, 50, 30, 10).
    for r in (90, 70, 50, 30, 10):
        assert str(r) in html, f"rank value {r} missing from HTML"


# ---------------------------------------------------------------------------
# Byte-identical determinism
# ---------------------------------------------------------------------------


def test_render_byte_identical_on_identical_input(features_df, tmp_path: Path):
    """Same DataFrame input → byte-identical HTML output."""
    out_a = tmp_path / "a.html"
    out_b = tmp_path / "b.html"
    render_dashboard(features_df.copy(), out_path=out_a, asof=date(2024, 11, 8))
    render_dashboard(features_df.copy(), out_path=out_b, asof=date(2024, 11, 8))

    bytes_a = out_a.read_bytes()
    bytes_b = out_b.read_bytes()
    assert hashlib.sha256(bytes_a).hexdigest() == hashlib.sha256(bytes_b).hexdigest(), (
        "dashboard HTML not byte-identical on identical input"
    )


def test_render_sort_js_groups_by_sector(features_df, tmp_path: Path):
    """Regression — codex iter-2 [P2]: the per-column sort JS must preserve
    sector grouping. The earlier implementation filtered headers out of the
    row list and appended only data rows, leaving headers detached at the
    top of the tbody after any sort click. The new sort walks tbody once,
    partitions into per-sector groups, re-appends header then sorted data
    inside each group.
    """
    out = tmp_path / "dashboard.html"
    render_dashboard(features_df, out_path=out, asof=date(2024, 11, 8))
    html = out.read_text()
    # Group-aware sort: must build a `groups` array AND re-append header
    # then data rows inside the forEach. Stale impl had no `groups` var.
    assert "const groups = []" in html, "sort JS missing per-sector grouping"
    assert "g.header" in html, "sort JS missing header re-append"
    # Belt-and-braces: the old filter-then-append shape must not be present.
    assert "rows.forEach(r => tbody.appendChild(r));" not in html, (
        "old filter-then-append sort shape detected — sector headers will detach"
    )


def test_render_self_contained_no_external_assets(features_df, tmp_path: Path):
    """No <script src=...>, no <link href=...> — fully self-contained.

    Operators preview locally with `open <path>` — the HTML must work without
    network and without companion files (per DESIGN-thematic dashboard §6 spec).
    """
    out = tmp_path / "dashboard.html"
    render_dashboard(features_df, out_path=out, asof=date(2024, 11, 8))
    html = out.read_text()
    # Allow embedded <style> and inline CSS; reject external <link href=
    # pointing at remote resources or local stylesheet files.
    import re

    external_links = re.findall(
        r'<link[^>]*rel=["\']stylesheet["\'][^>]*href=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    for href in external_links:
        # Allow data: URIs.
        assert href.startswith("data:"), (
            f"external stylesheet found: {href!r}"
        )
    # External script src=... is also forbidden.
    external_scripts = re.findall(
        r'<script[^>]*src=["\']([^"\']+)["\']', html, re.IGNORECASE
    )
    for src in external_scripts:
        assert src.startswith("data:"), f"external script found: {src!r}"
