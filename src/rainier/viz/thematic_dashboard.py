"""Self-contained HTML dashboard for thematic ranks.

Renders a sortable table grouped by sector with color-scaled R cells.
Byte-identical render on byte-identical input (no timestamps, no UUIDs,
no library-version strings embedded in the output).

Design ref:
    docs/DESIGN-thematic-ranks-dashboard.md §6 (architecture) + [D-009] (no
    publish hook in v0 — render-only).
"""

from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Color scale — sequential green-to-red over [0, 99]
# ---------------------------------------------------------------------------


def _rank_color(rank: int) -> str:
    """Map rank centile (0..99) to a hex color via a fixed gradient.

    99 (top) -> green (#1b9e3a); 50 -> yellow (#dccd2c); 0 (bottom) -> red (#c92a2a).
    Deterministic on identical input (no random / library calls).
    Negative ranks (sentinel for missing data) → neutral gray.
    """
    if rank < 0:
        return "#cccccc"
    rank = max(0, min(99, int(rank)))
    # Two-stop linear interp: 0 -> red; 50 -> yellow; 99 -> green.
    if rank <= 50:
        # red (201, 42, 42) -> yellow (220, 205, 44)
        t = rank / 50.0
        r = int(201 + (220 - 201) * t)
        g = int(42 + (205 - 42) * t)
        b = int(42 + (44 - 42) * t)
    else:
        # yellow (220, 205, 44) -> green (27, 158, 58)
        t = (rank - 50) / 49.0
        r = int(220 + (27 - 220) * t)
        g = int(205 + (158 - 205) * t)
        b = int(44 + (58 - 44) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
       Roboto, "Helvetica Neue", Arial, sans-serif;
       margin: 24px; color: #1f2933; background: #f7fafc; }
h1 { font-size: 1.5rem; margin: 0 0 8px; }
.subtitle { color: #52606d; margin: 0 0 24px; font-size: 0.9rem; }
table { border-collapse: collapse; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border-radius: 6px; overflow: hidden; }
th, td { padding: 6px 12px; text-align: right; font-size: 0.85rem;
         border-bottom: 1px solid #e4e7eb; }
th { background: #f0f4f8; cursor: pointer; user-select: none;
     text-align: right; font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
th:nth-child(2), td:nth-child(2) { text-align: left; font-weight: 500; }
.rank-cell { font-weight: 600; color: #fff; min-width: 28px;
             text-align: center; padding: 6px 10px; }
.sector-header td { background: #e4e7eb; font-weight: 700;
                    padding: 8px 12px; text-align: left;
                    text-transform: capitalize; font-size: 0.95rem; }
.numeric { font-variant-numeric: tabular-nums; }
"""


_SORT_JS = """
function sortTable(idx) {
  const table = document.getElementById('ranks');
  const tbody = table.tBodies[0];
  const dir = table.dataset.sortDir === 'asc' ? 'desc' : 'asc';
  table.dataset.sortDir = dir;

  // Walk tbody rows once, partitioning into sector groups. Each group is
  // {header: <tr.sector-header or null>, data: [<tr>, ...]}. We preserve
  // group order from the rendered DOM and only re-sort the data rows
  // within each group — keeps headers stuck to their rows on every click.
  const groups = [];
  let current = null;
  Array.from(tbody.rows).forEach(r => {
    if (r.classList.contains('sector-header')) {
      current = { header: r, data: [] };
      groups.push(current);
    } else {
      if (current === null) {
        current = { header: null, data: [] };
        groups.push(current);
      }
      current.data.push(r);
    }
  });

  const cmp = (a, b) => {
    const va = a.cells[idx].dataset.sort || a.cells[idx].textContent;
    const vb = b.cells[idx].dataset.sort || b.cells[idx].textContent;
    const na = parseFloat(va);
    const nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) {
      return dir === 'asc' ? na - nb : nb - na;
    }
    return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
  };

  // Re-append in group order: header then sorted data rows.
  groups.forEach(g => {
    if (g.header) tbody.appendChild(g.header);
    g.data.sort(cmp).forEach(r => tbody.appendChild(r));
  });
}
"""


_COLUMN_SPECS = [
    ("symbol", "Ticker", "text"),
    ("sector_name", "Sector", "text"),
    ("close", "Close", "float"),
    ("ret_1d", "1d %", "pct"),
    ("rel_5", "5d %", "pct"),
    ("rel_10", "10d %", "pct"),
    ("rel_20", "20d %", "pct"),
    ("ret_ytd", "YTD %", "pct"),
    ("r_5", "R5", "rank"),
    ("r_10", "R10", "rank"),
    ("r_20", "R20", "rank"),
    ("rank", "Rank", "rank"),
    ("top15_streak", "Streak", "int"),
]


def _format_cell(value, kind: str) -> tuple[str, str]:
    """Return ``(display_text, sort_key)`` for the given value + type."""
    if pd.isna(value):
        return ("—", "")
    if kind == "text":
        return (str(value), str(value))
    if kind == "float":
        return (f"{value:.2f}", f"{value:.6f}")
    if kind == "pct":
        return (f"{value * 100:.2f}%", f"{value:.6f}")
    if kind == "rank":
        v = int(value)
        if v < 0:
            return ("—", "-1")
        return (str(v), str(v))
    if kind == "int":
        return (str(int(value)), str(int(value)))
    return (str(value), str(value))


def render_dashboard(
    features: pd.DataFrame,
    out_path: str | Path,
    asof: date,
) -> Path:
    """Render the thematic-ranks dashboard for a single ``asof_date``.

    Parameters
    ----------
    features:
        Long-format Layer A DataFrame. MUST include ``sector_name`` (the
        renderer groups rows by sector for display). If absent, callers
        should join sector_name onto the features before calling.
    out_path:
        Destination HTML file. Parent directory created if missing.
    asof:
        Display-only date for the header. Does not filter the input — the
        caller projects ``features`` to the desired snapshot.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if features.empty:
        out.write_text(_render_empty(asof))
        return out

    if "sector_name" not in features.columns:
        # Defensive default: every ticker in 'unknown' sector.
        features = features.copy()
        features["sector_name"] = "unknown"

    # Deterministic ordering: by (sector_name, -rank, symbol) for the initial
    # table layout. Sort key is fully deterministic on the input DataFrame.
    df = features.copy()
    df = df.sort_values(
        by=["sector_name", "rank", "symbol"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    # Build HTML.
    parts: list[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append(f"<title>Thematic Ranks — {asof.isoformat()}</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append("</head><body>")
    parts.append(f"<h1>Thematic Ranks — {asof.isoformat()}</h1>")
    parts.append(
        f'<p class="subtitle">Universe size: {len(df)} tickers; ranks computed '
        f"vs cross-sectional universe. Click column headers to sort.</p>"
    )
    parts.append('<table id="ranks" data-sort-dir="asc">')

    # Header row
    parts.append("<thead><tr>")
    for i, (_col, label, _kind) in enumerate(_COLUMN_SPECS):
        parts.append(f'<th onclick="sortTable({i})">{escape(label)}</th>')
    parts.append("</tr></thead><tbody>")

    # Body: grouped by sector with a separator row per group.
    for sector_name, sec_group in df.groupby("sector_name", sort=True):
        parts.append(
            f'<tr class="sector-header"><td colspan="{len(_COLUMN_SPECS)}">'
            f"{escape(str(sector_name))}</td></tr>"
        )
        for _, row in sec_group.iterrows():
            parts.append("<tr>")
            for col, _label, kind in _COLUMN_SPECS:
                if col not in row.index:
                    parts.append("<td>—</td>")
                    continue
                text, sort_key = _format_cell(row[col], kind)
                if kind == "rank":
                    rank_val = int(row[col]) if pd.notna(row[col]) else -1
                    color = _rank_color(rank_val)
                    parts.append(
                        f'<td class="rank-cell" data-sort="{escape(sort_key)}" '
                        f'style="background:{color}">{escape(text)}</td>'
                    )
                elif kind in ("float", "pct", "int"):
                    parts.append(
                        f'<td class="numeric" data-sort="{escape(sort_key)}">'
                        f"{escape(text)}</td>"
                    )
                else:
                    parts.append(
                        f'<td data-sort="{escape(sort_key)}">{escape(text)}</td>'
                    )
            parts.append("</tr>")

    parts.append("</tbody></table>")
    parts.append(f"<script>{_SORT_JS}</script>")
    parts.append("</body></html>")

    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def _render_empty(asof: date) -> str:
    return (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'>"
        f"<title>Thematic Ranks — {asof.isoformat()}</title>"
        f"<style>{_CSS}</style>"
        "</head><body>"
        f"<h1>Thematic Ranks — {asof.isoformat()}</h1>"
        "<p>No data for this asof_date.</p>"
        "</body></html>"
    )
