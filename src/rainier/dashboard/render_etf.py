"""Self-contained ETF-ranks dashboard HTML renderer.

Reads `thematic_features_daily.parquet` + `sector_registry.parquet` and
emits a single deterministic HTML file with three CSS-tab views (All
ETFs / Top 15 / Movers), inline SVG sparklines, a green→yellow→red
rank-color gradient, and `prefers-color-scheme` dark-mode styling that
mirrors fengshen-site CSS variables.

Pure function over parquet inputs → string. No DB session, no wall-clock
reads inside the renderer (caller passes `rendered_at_pt`), so the
output is byte-identical on identical inputs.

Design refs:
    docs/DESIGN-etf-dashboard-publish.md §3 (layout) + §4 (build pipeline)
    docs/TASK-PLAN-etf-dashboard-renderer-efed.md (acceptance gates)

Render flow:

    ┌── thematic_features_daily.parquet ──┐
    │                                     │
    │  1. project asof slice              │
    │  2. join sector_name via registry   │
    │  3. compute per-symbol 30d history  │  ──▶ sparkline SVG paths
    │  4. sort: sector ASC, rank DESC     │
    │  5. render jinja2 template:         │
    │       • <style> with CSS vars + dark
    │       • 3 radio-tab table blocks    │
    │       • inline SVG sparklines       │
    │       • inline sort.js              │
    │  6. write atomic                    │
    └─────────────────────────────────────┘
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined, select_autoescape
from markupsafe import Markup

__all__ = [
    "render_etf_html",
    "write_etf_html",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_etf_html(
    *,
    features: pd.DataFrame,
    registry: pd.DataFrame,
    asof: date,
    rendered_at_pt: str,
    history_days: int = 30,
) -> str:
    """Render the ETF-ranks dashboard to an HTML string.

    Pure function — no I/O, no DB, no wall-clock reads.

    Parameters
    ----------
    features:
        Long-format ``thematic_features_daily`` DataFrame. Must include
        columns ``asof_date``, ``symbol``, ``sector_id``, ``rank``,
        ``rank_delta_1d``, ``rank_delta_5d``, ``r_5``, ``r_10``, ``r_20``,
        ``ret_ytd``, ``top15_streak``.
    registry:
        Sector registry parquet — ``sector_id``, ``sector_name``.
    asof:
        Display + filter date. The renderer slices the features DataFrame
        to rows where ``asof_date == asof``.
    rendered_at_pt:
        Wall-clock string from the operator (e.g. ``"12:40"``). The renderer
        does NOT call ``datetime.now()`` — kills determinism. The wrapper
        script (or CLI) supplies this.
    history_days:
        Sparkline history window. Tickers with shorter history degrade to a
        flat midpoint line.
    """
    asof_str = asof.isoformat()
    features = _normalise_asof_column(features)
    latest = features.loc[features["asof_date"] == asof_str].copy()
    if latest.empty:
        # Best-effort: render empty page so the cron output is still a valid
        # file (the publish step can detect emptiness via byte size).
        return _render_template(
            sections=[],
            asof_str=asof_str,
            rendered_at_pt=rendered_at_pt,
            universe_size=0,
            sparklines_present=False,
        )

    # Join sector_name onto the latest slice.
    sector_map = dict(zip(registry["sector_id"].astype(int), registry["sector_name"].astype(str)))
    latest["sector_name"] = latest["sector_id"].astype(int).map(sector_map).fillna("unknown")

    # Sparklines: take up to `history_days` of prior `rank` values per symbol,
    # ending on `asof`. Order chronologically. Tickers with <2 points degrade
    # to a single mid-line.
    history_lookup = _build_history_lookup(features, asof_str, history_days)

    # Default sort: sector ASC → rank DESC (within sector) → symbol ASC tiebreaker.
    latest = latest.sort_values(
        by=["sector_name", "rank", "symbol"],
        ascending=[True, False, True],
        kind="mergesort",  # stable
    ).reset_index(drop=True)

    rows_all = [_row_to_dict(r, history_lookup) for r in latest.to_dict(orient="records")]

    # Tab filters — operate on the already-rendered rows so the sparkline
    # SVG string is reused (cheap; no re-build).
    rows_top15 = [r for r in rows_all if r["rank_int"] >= 85]
    rows_movers = [
        r
        for r in rows_all
        if abs(int(r["rank_delta_1d"])) >= 10 or abs(int(r["rank_delta_5d"])) >= 15
    ]

    sections = [
        {"tab": "all", "label": "All ETFs", "rows": rows_all},
        {"tab": "top15", "label": "Top 15", "rows": rows_top15},
        {"tab": "movers", "label": "Movers", "rows": rows_movers},
    ]

    return _render_template(
        sections=sections,
        asof_str=asof_str,
        rendered_at_pt=rendered_at_pt,
        universe_size=len(rows_all),
        sparklines_present=True,
    )


def write_etf_html(
    *,
    features_path: str | Path,
    registry_path: str | Path,
    output_path: str | Path,
    asof: date,
    rendered_at_pt: str,
    history_days: int = 30,
) -> Path:
    """Convenience wrapper: load parquets, render, atomic-write output."""
    features = pd.read_parquet(features_path)
    registry = pd.read_parquet(registry_path)
    html = render_etf_html(
        features=features,
        registry=registry,
        asof=asof,
        rendered_at_pt=rendered_at_pt,
        history_days=history_days,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, out)
    return out


# ---------------------------------------------------------------------------
# Internals — data shaping
# ---------------------------------------------------------------------------


def _normalise_asof_column(features: pd.DataFrame) -> pd.DataFrame:
    """Coerce `asof_date` to an isoformat string for stable comparison.

    The prod parquet ships `asof_date` as object/string already, but some
    callers may pass through `pd.to_datetime`. Normalising once at the top
    of the renderer keeps every downstream comparison/sort byte-stable.
    """
    if "asof_date" not in features.columns:
        return features
    col = features["asof_date"]
    if pd.api.types.is_datetime64_any_dtype(col):
        return features.assign(asof_date=col.dt.strftime("%Y-%m-%d"))
    return features.assign(asof_date=col.astype(str))


def _build_history_lookup(
    features: pd.DataFrame, asof_str: str, history_days: int
) -> dict[str, list[int]]:
    """For each symbol, return up to `history_days` of valid `rank` values, oldest first.

    Only includes rows on or before `asof_str`. Truncates to the most recent
    `history_days` after sorting.

    Filters out the `rank == -1` missing-data sentinel that the upstream
    pipeline writes during a ticker's first ~20 trading days. Letting -1
    leak into the sparkline gets clamped to 0 by `_sparkline_svg` and
    paints a misleading "collapse → rebound" curve instead of "no data".
    A ticker with no valid history degrades to the flat-midline branch
    in `_sparkline_svg` (same as `<2 points`).
    """
    valid = features.loc[
        (features["asof_date"] <= asof_str) & (features["rank"] >= 0),
        ["symbol", "asof_date", "rank"],
    ]
    valid = valid.sort_values(by=["symbol", "asof_date"], kind="mergesort")
    out: dict[str, list[int]] = {}
    for sym, grp in valid.groupby("symbol", sort=False):
        ranks = grp["rank"].tolist()[-history_days:]
        out[str(sym)] = [int(v) for v in ranks]
    return out


def _row_to_dict(row: dict, history_lookup: dict[str, list[int]]) -> dict:
    """Pre-format every cell for jinja consumption.

    The template is dumb on purpose — strings only, no conditionals beyond
    the trivial truthy checks. Pre-formatting here keeps the renderer
    deterministic and the template tiny.

    NaN handling: the prod parquet emits `NaN` for fields that aren't
    populated yet (e.g. `ret_ytd` for a newly-added ETF before its first
    YTD base date). Naive ``int(x or 0)`` and ``float(x or 0)`` both fail:
    NaN is truthy AND ``int(NaN)`` raises. Every numeric pull goes through
    the _safe_* helpers below so a missing/NaN value renders as a stable
    "—" display token + a deterministic sort key.
    """
    rank = _safe_int(row.get("rank"), default=-1)
    sym = str(row["symbol"])
    ret_ytd_raw = row.get("ret_ytd")
    ret_ytd_missing = _is_missing(ret_ytd_raw)
    ret_ytd = 0.0 if ret_ytd_missing else float(ret_ytd_raw)
    history = history_lookup.get(sym, [])
    sparkline_svg = _sparkline_svg(history)
    # Negative rank-like values are the upstream "missing data" sentinel
    # — bake them into the displayed cell as an em-dash so the published
    # dashboard never surfaces a literal `-1`. Sort keys keep the raw int
    # so column-click sorting clusters missing rows at the bottom of asc.
    r5_raw = _safe_int(row.get("r_5"), default=-1)
    r10_raw = _safe_int(row.get("r_10"), default=-1)
    r20_raw = _safe_int(row.get("r_20"), default=-1)
    return {
        "sector_name": str(row["sector_name"]),
        "symbol": sym,
        "rank_int": rank,
        "rank_text": "—" if rank < 0 else str(rank),
        # rank_color comes from _rank_color() — a hex literal we just built,
        # never touched by an external string. Safe to mark explicitly.
        "rank_color": Markup(_rank_color(rank)),
        "rank_delta_1d": _safe_int(row.get("rank_delta_1d"), default=0),
        "rank_delta_5d": _safe_int(row.get("rank_delta_5d"), default=0),
        "rank_delta_1d_text": _signed_int(row.get("rank_delta_1d")),
        "rank_delta_5d_text": _signed_int(row.get("rank_delta_5d")),
        "r5": r5_raw,
        "r10": r10_raw,
        "r20": r20_raw,
        "r5_text": "—" if r5_raw < 0 else str(r5_raw),
        "r10_text": "—" if r10_raw < 0 else str(r10_raw),
        "r20_text": "—" if r20_raw < 0 else str(r20_raw),
        # Missing YTD → display "—" but sort to the bottom of either direction
        # (use a stable very-negative key so NaN rows cluster predictably).
        "ytd_pct": "—" if ret_ytd_missing else f"{ret_ytd * 100:+.1f}%",
        "ytd_sort_key": "-9.999999" if ret_ytd_missing else f"{ret_ytd:.6f}",
        "top15_streak": _safe_int(row.get("top15_streak"), default=0),
        # Pre-rendered SVG built from numeric data ONLY (_sparkline_svg()
        # accepts a list[int] and emits a fixed-shape <svg><path/></svg>).
        # Marked Markup so the autoescaping template emits it as raw HTML.
        "sparkline_svg": Markup(sparkline_svg),
    }


def _is_missing(value) -> bool:
    """True iff value is None, pd.NA, or NaN.

    Centralizes the "missing-data sentinel" check. ``bool(NaN)`` is True,
    so naive ``x or default`` does NOT substitute the default — we need
    an explicit isna() probe.
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _safe_int(value, *, default: int) -> int:
    """Coerce to int, falling back to ``default`` on NaN/None/pd.NA.

    ``int(NaN)`` raises ValueError, and ``NaN or 0`` returns NaN because
    NaN is truthy. Neither idiom is safe for fields that arrive as NaN
    in the prod parquet.
    """
    if _is_missing(value):
        return default
    return int(value)


def _signed_int(value) -> str:
    if _is_missing(value):
        return "0"
    v = int(value)
    if v == 0:
        return "0"
    return f"{v:+d}"


# ---------------------------------------------------------------------------
# Color — rank gradient
# ---------------------------------------------------------------------------


def _rank_color(rank: int) -> str:
    """Map ``rank`` in [0..99] to a hex color: red (low) → yellow → green (high).

    Negative ranks (sentinel for missing data) → neutral gray. Deterministic
    on input (pure integer math).
    """
    if rank < 0:
        return "#cccccc"
    rank = max(0, min(99, int(rank)))
    if rank <= 50:
        # red (201, 42, 42) → yellow (220, 205, 44)
        t = rank / 50.0
        r = int(201 + (220 - 201) * t)
        g = int(42 + (205 - 42) * t)
        b = int(42 + (44 - 42) * t)
    else:
        # yellow (220, 205, 44) → green (27, 158, 58)
        t = (rank - 50) / 49.0
        r = int(220 + (27 - 220) * t)
        g = int(205 + (158 - 205) * t)
        b = int(44 + (58 - 44) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Sparkline — single SVG <path>
# ---------------------------------------------------------------------------

_SPARK_W = 80
_SPARK_H = 20
_SPARK_PAD = 1.5


def _sparkline_svg(ranks: list[int]) -> str:
    """Render a single-<path> SVG of the rank history.

    Always emits an `<svg>` containing exactly one `<path>`. Tickers with
    fewer than 2 points get a flat line at the vertical midpoint (the row
    still has a `<path>` so the per-row counter test holds).
    """
    if not ranks or len(ranks) < 2:
        # Flat midline at rank=50 equivalent.
        mid_y = _SPARK_H / 2.0
        d = f"M0,{mid_y:.2f} L{_SPARK_W},{mid_y:.2f}"
        return (
            f'<svg class="spark" viewBox="0 0 {_SPARK_W} {_SPARK_H}" '
            f'width="{_SPARK_W}" height="{_SPARK_H}" preserveAspectRatio="none">'
            f'<path d="{d}" /></svg>'
        )
    n = len(ranks)
    x_step = _SPARK_W / max(1, n - 1)
    inner_h = _SPARK_H - 2 * _SPARK_PAD
    pts = []
    for i, rk in enumerate(ranks):
        v = max(0, min(99, int(rk)))
        x = i * x_step
        # Invert Y so higher rank = higher on chart.
        y = _SPARK_PAD + (1.0 - v / 99.0) * inner_h
        pts.append(f"{x:.2f},{y:.2f}")
    d = "M" + " L".join(pts)
    return (
        f'<svg class="spark" viewBox="0 0 {_SPARK_W} {_SPARK_H}" '
        f'width="{_SPARK_W}" height="{_SPARK_H}" preserveAspectRatio="none">'
        f'<path d="{d}" /></svg>'
    )


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ETF Ranks — {{ asof_str }}</title>
<style>
:root {
  --accent: #2337ff;
  --bg-page: #ffffff;
  --bg-elevated: #ffffff;
  --gray-fg: #1f2933;
  --gray-mute: #52606d;
  --gray-line: #e4e7eb;
  --gray-stripe: #f7fafc;
  --pos: #1b9e3a;
  --neg: #c92a2a;
  --spark: #2337ff;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root {
    --accent: #8b9dff;
    --bg-page: #0d1018;
    --bg-elevated: #161a26;
    --gray-fg: #d2d6df;
    --gray-mute: #8a93a6;
    --gray-line: #2a3040;
    --gray-stripe: #11141d;
    --pos: #2bd86d;
    --neg: #ff6e6e;
    --spark: #8b9dff;
    color-scheme: dark;
  }
}
body {
  font-family: 'Atkinson Hyperlegible', -apple-system, BlinkMacSystemFont,
               'Segoe UI', system-ui, Roboto, 'Helvetica Neue', Arial, sans-serif;
  margin: 24px; color: var(--gray-fg); background: var(--bg-page);
  font-size: 14px;
}
h1 { font-size: 1.4rem; margin: 0 0 4px; color: var(--gray-fg); }
.subtitle {
  color: var(--gray-mute);
  margin: 0 0 16px;
  font-size: 0.85rem;
}
.tabs { display: flex; gap: 4px; margin: 16px 0 0; border-bottom: 1px solid var(--gray-line); }
/* The tab radios live as siblings BEFORE `.tabs` (the `~` general-sibling
   selector needs them to precede the section/label nodes), so hide them by
   id rather than by descendant-of-.tabs which doesn't match. */
input[type=radio]#tab-all,
input[type=radio]#tab-top15,
input[type=radio]#tab-movers { position: absolute; opacity: 0; pointer-events: none; width: 0; height: 0; }
.tabs label {
  padding: 6px 16px;
  cursor: pointer;
  color: var(--gray-mute);
  font-size: 0.9rem;
  border-bottom: 2px solid transparent;
  user-select: none;
}
.tabs label:hover { color: var(--accent); }
section[data-tab] { display: none; margin-top: 16px; }
#tab-all:checked    ~ .tabs label[for=tab-all]    { color: var(--accent); border-bottom-color: var(--accent); }
#tab-top15:checked  ~ .tabs label[for=tab-top15]  { color: var(--accent); border-bottom-color: var(--accent); }
#tab-movers:checked ~ .tabs label[for=tab-movers] { color: var(--accent); border-bottom-color: var(--accent); }
#tab-all:checked    ~ section[data-tab="all"]    { display: block; }
#tab-top15:checked  ~ section[data-tab="top15"]  { display: block; }
#tab-movers:checked ~ section[data-tab="movers"] { display: block; }
table.etf-table {
  border-collapse: collapse;
  background: var(--bg-elevated);
  width: 100%;
  font-size: 0.85rem;
}
table.etf-table th, table.etf-table td {
  padding: 6px 12px;
  text-align: right;
  border-bottom: 1px solid var(--gray-line);
  font-variant-numeric: tabular-nums;
}
table.etf-table th {
  background: var(--gray-stripe);
  cursor: pointer;
  user-select: none;
  font-weight: 600;
  color: var(--gray-mute);
}
table.etf-table th:first-child, table.etf-table td:first-child { text-align: left; }
table.etf-table th:nth-child(2), table.etf-table td:nth-child(2) {
  text-align: left; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-weight: 600;
}
.rank-cell {
  font-weight: 700;
  color: #1a1a1a;
  text-align: center;
  min-width: 30px;
}
.pos { color: var(--pos); }
.neg { color: var(--neg); }
svg.spark { vertical-align: middle; fill: none; stroke: var(--spark); stroke-width: 1.25; }
</style>
</head>
<body>
<h1>ETF Ranks — as of {{ asof_str }}</h1>
<p class="subtitle">
  Last updated: {{ asof_str }} {{ rendered_at_pt }} PT
  &middot; Universe: {{ universe_size }} sector/theme ETFs
</p>

<input type="radio" name="tab" id="tab-all" checked>
<input type="radio" name="tab" id="tab-top15">
<input type="radio" name="tab" id="tab-movers">
<div class="tabs">
  <label for="tab-all">All ETFs</label>
  <label for="tab-top15">Top 15</label>
  <label for="tab-movers">Movers</label>
</div>

{% for sec in sections %}
<section data-tab="{{ sec.tab }}">
  <table class="etf-table" id="etf-{{ sec.tab }}">
    <thead><tr>
      <th onclick="sortEtfTable('etf-{{ sec.tab }}', 0, 'text')">Sector</th>
      <th onclick="sortEtfTable('etf-{{ sec.tab }}', 1, 'text')">Symbol</th>
      <th onclick="sortEtfTable('etf-{{ sec.tab }}', 2, 'num')">Rank</th>
      <th onclick="sortEtfTable('etf-{{ sec.tab }}', 3, 'num')">&Delta;1d</th>
      <th onclick="sortEtfTable('etf-{{ sec.tab }}', 4, 'num')">&Delta;5d</th>
      <th onclick="sortEtfTable('etf-{{ sec.tab }}', 5, 'num')">R5</th>
      <th onclick="sortEtfTable('etf-{{ sec.tab }}', 6, 'num')">R10</th>
      <th onclick="sortEtfTable('etf-{{ sec.tab }}', 7, 'num')">R20</th>
      <th onclick="sortEtfTable('etf-{{ sec.tab }}', 8, 'num')">YTD</th>
      <th>30d</th>
    </tr></thead>
    <tbody>
    {% for r in sec.rows %}
      <tr>
        <td>{{ r.sector_name }}</td>
        <td>{{ r.symbol }}</td>
        <td class="rank-cell" data-sort="{{ r.rank_int }}" style="background: {{ r.rank_color }}">{{ r.rank_text }}</td>
        <td data-sort="{{ r.rank_delta_1d }}" class="{% if r.rank_delta_1d > 0 %}pos{% elif r.rank_delta_1d < 0 %}neg{% endif %}">{{ r.rank_delta_1d_text }}</td>
        <td data-sort="{{ r.rank_delta_5d }}" class="{% if r.rank_delta_5d > 0 %}pos{% elif r.rank_delta_5d < 0 %}neg{% endif %}">{{ r.rank_delta_5d_text }}</td>
        <td data-sort="{{ r.r5 }}">{{ r.r5_text }}</td>
        <td data-sort="{{ r.r10 }}">{{ r.r10_text }}</td>
        <td data-sort="{{ r.r20 }}">{{ r.r20_text }}</td>
        <td data-sort="{{ r.ytd_sort_key }}">{{ r.ytd_pct }}</td>
        <td>{{ r.sparkline_svg }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</section>
{% endfor %}

<script>
function sortEtfTable(tableId, idx, kind) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const tbody = table.tBodies[0];
  const key = 'sortDir' + idx;
  const dir = table.dataset[key] === 'asc' ? 'desc' : 'asc';
  table.dataset[key] = dir;
  const rows = Array.from(tbody.rows);
  rows.sort((a, b) => {
    const ca = a.cells[idx], cb = b.cells[idx];
    const va = ca.dataset.sort !== undefined ? ca.dataset.sort : ca.textContent;
    const vb = cb.dataset.sort !== undefined ? cb.dataset.sort : cb.textContent;
    if (kind === 'num') {
      const na = parseFloat(va), nb = parseFloat(vb);
      return dir === 'asc' ? na - nb : nb - na;
    }
    return dir === 'asc'
      ? String(va).localeCompare(String(vb))
      : String(vb).localeCompare(String(va));
  });
  rows.forEach(r => tbody.appendChild(r));
}
</script>
</body>
</html>
"""


# Module-level jinja environment — autoescape ON, StrictUndefined for safety.
#
# Why autoescape ON: the rendered HTML is published to a public path (per
# DESIGN-etf-dashboard-publish.md), so every string-valued field that flows
# through the template must be entity-encoded by default. `sector_name`
# comes from `sector_registry.parquet` (human-edited config), and even the
# ticker `symbol` is not load-bearing enough to trust unescaped. Pre-built
# HTML fragments we DO want raw (sparkline SVG, rank-color hex) are wrapped
# in `Markup(...)` at the `_row_to_dict` site so they pass through verbatim.
_ENV = Environment(
    autoescape=select_autoescape(default=True, default_for_string=True),
    undefined=StrictUndefined,
    trim_blocks=False,
    lstrip_blocks=False,
    keep_trailing_newline=True,
)


def _render_template(
    *,
    sections: list[dict],
    asof_str: str,
    rendered_at_pt: str,
    universe_size: int,
    sparklines_present: bool,
) -> str:
    template = _ENV.from_string(_TEMPLATE)
    return template.render(
        sections=sections,
        asof_str=asof_str,
        rendered_at_pt=rendered_at_pt,
        universe_size=universe_size,
        sparklines_present=sparklines_present,
    )
