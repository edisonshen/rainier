"""HTML report for the TQQQ/SQQQ SMA sweep.

Produces a single self-contained HTML file in the design-doc "hub" style:
sticky two-col layout, numbered sections, embedded CSS, no JS, no external
asset fetches at view time. Plotly figures are emitted as inline ``<div>``
blocks via ``to_html(include_plotlyjs="inline")`` on the first chart and
``"cdn"`` is intentionally not used so the file remains viewable offline.

Sections (all 9 from the task spec):

    1. Header + framing
    2. Top-50 winners table (with in-sample vs out-of-sample delta highlight)
    3. Buy-and-hold baselines (numbers + equity overlay)
    4. Heatmaps (two slices through 4D space)
    5. Distribution of outcomes histogram
    6. Equity curves (top-5 + baselines)
    7. Walk-forward delta scatter
    8. Honest discussion
    9. Reproducibility footer
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rainier.backtest.tqqq_sma_sweep import (
    LONG_TQQQ,
    SHORT_SQQQ,
    dedup_by_strategy_id,
    precompute_sma_signals,
)


def _strategy_type(time_in_long: float, time_in_short: float) -> str:
    """Classify a strategy by which legs actually fired.

    A leg that never fires (time fraction == 0 exactly) is dormant. With
    phase-1 constraints + validity gating this happens for entire sub-grids:
    e.g. any combo with buy_S=1 has a structurally-false short-entry signal,
    so the short leg never trades regardless of sell_S.

    Returns one of: ``LONG-only`` | ``SHORT-only`` | ``BOTH-legs``.
    A strategy that is 100% cash is reported as ``BOTH-legs`` since neither
    leg's dormancy is structural — the call is "neither signal fired in this
    universe", not "this leg can never fire".
    """
    if time_in_short == 0.0 and time_in_long > 0.0:
        return "LONG-only"
    if time_in_long == 0.0 and time_in_short > 0.0:
        return "SHORT-only"
    return "BOTH-legs"


def _dormant_cell(value: int, dormant: bool) -> str:
    """Render an integer param cell, or a literal ``-`` when the leg is dormant.

    Used by the top-50 table to make it visually obvious that e.g. sell_S
    has no effect on a LONG-only strategy. Pre-dedup the leaderboard could
    show 50 rows whose differences were entirely cosmetic; post-dedup the
    surviving row from each group must still telegraph that fact at a glance.
    """
    return "-" if dormant else f"{int(value)}"

# Teal accent matches the design-doc template palette
_TEAL = "#0f766e"
_TEAL_LIGHT = "#7bd9c7"
_MUTED = "#626b73"
_AMBER = "#b87333"


def _git_sha(path: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _equity_curve_for(prices: pd.DataFrame, combo: tuple[int, int, int, int],
                      slippage_bp: float, max_window: int) -> pd.Series:
    """Replay one combo in-Python (slow but only called for top-5) and return an equity series.

    Must mirror :func:`rainier.backtest.tqqq_sma_sweep.run_backtest` exactly,
    including the validity gating on entry/exit signals — otherwise the
    equity curve drawn in the report diverges from the sweep's final_value.
    """
    qqq = prices["qqq"].to_numpy(dtype=np.float64)
    tqqq = prices["tqqq"].to_numpy(dtype=np.float64)
    sqqq = prices["sqqq"].to_numpy(dtype=np.float64)
    above, valid = precompute_sma_signals(qqq, max_window=max_window)
    tqqq_ret = (tqqq[1:] / tqqq[:-1]) - 1.0
    sqqq_ret = (sqqq[1:] / sqqq[:-1]) - 1.0
    bT, sT, bS, sS = combo
    n = qqq.shape[0]
    slip = slippage_bp * 1e-4

    eq = np.empty(n)
    state = 0  # CASH
    equity = 1.0

    col_bT = above[:, bT - 1]
    col_sT = above[:, sT - 1]
    col_bS = above[:, bS - 1]
    col_sS = above[:, sS - 1]
    v_bT = valid[:, bT - 1]
    v_sT = valid[:, sT - 1]
    v_bS = valid[:, bS - 1]
    v_sS = valid[:, sS - 1]

    if v_bT[0] and col_bT[0]:
        state = LONG_TQQQ
        equity *= 1.0 - slip
    elif v_bS[0] and not col_bS[0]:
        state = SHORT_SQQQ
        equity *= 1.0 - slip
    eq[0] = equity

    for d in range(1, n):
        if state == LONG_TQQQ:
            equity *= 1.0 + tqqq_ret[d - 1]
        elif state == SHORT_SQQQ:
            equity *= 1.0 + sqqq_ret[d - 1]
        if state == LONG_TQQQ and v_sT[d] and not col_sT[d]:
            state = 0
            equity *= 1.0 - slip
        elif state == SHORT_SQQQ and v_sS[d] and col_sS[d]:
            state = 0
            equity *= 1.0 - slip
        if state == 0:
            if v_bT[d] and col_bT[d]:
                state = LONG_TQQQ
                equity *= 1.0 - slip
            elif v_bS[d] and not col_bS[d]:
                state = SHORT_SQQQ
                equity *= 1.0 - slip
        eq[d] = equity

    return pd.Series(eq, index=prices.index, name=f"({bT},{sT},{bS},{sS})")


def _fig_to_div(fig: go.Figure, include_js: bool) -> str:
    return fig.to_html(
        include_plotlyjs="inline" if include_js else False,
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
    )


def _style_figure(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        template="simple_white",
        height=height,
        margin=dict(l=48, r=24, t=32, b=48),
        font=dict(family="ui-sans-serif, system-ui, -apple-system", size=12, color="#1d2024"),
        colorway=[_TEAL, _AMBER, "#1d4ed8", "#9f1239", "#4d7c0f", _TEAL_LIGHT],
        plot_bgcolor="#fffefb",
        paper_bgcolor="#fffefb",
    )
    return fig


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------


def _equity_overlay(prices: pd.DataFrame, top_combos: list[tuple[int, int, int, int]],
                    slippage_bp: float, max_window: int) -> go.Figure:
    fig = go.Figure()
    base_qqq = prices["qqq"] / prices["qqq"].iloc[0]
    base_tqqq = prices["tqqq"] / prices["tqqq"].iloc[0]
    base_sqqq = prices["sqqq"] / prices["sqqq"].iloc[0]
    fig.add_trace(go.Scatter(x=prices.index, y=base_qqq, name="QQQ B&H",
                             line=dict(color=_MUTED, dash="dot")))
    fig.add_trace(go.Scatter(x=prices.index, y=base_tqqq, name="TQQQ B&H",
                             line=dict(color="#9f1239", dash="dash")))
    fig.add_trace(go.Scatter(x=prices.index, y=base_sqqq, name="SQQQ B&H",
                             line=dict(color="#1d4ed8", dash="dash")))
    for combo in top_combos[:5]:
        eq = _equity_curve_for(prices, combo, slippage_bp=slippage_bp, max_window=max_window)
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=f"strat {combo}"))
    fig.update_yaxes(type="log", title="equity (log, base = 1.0)")
    fig.update_xaxes(title="date")
    return _style_figure(fig, height=460)


def _heatmap_buyT_sellT(df: pd.DataFrame, fixed_buyS: int, fixed_sellS: int) -> go.Figure:
    sub = df[(df.buy_S == fixed_buyS) & (df.sell_S == fixed_sellS)]
    pivot = sub.pivot_table(index="buy_T", columns="sell_T", values="final_value")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns, y=pivot.index,
        colorscale=[[0, "#fff5e8"], [0.3, "#fde6c5"], [0.7, _TEAL_LIGHT], [1, _TEAL]],
        colorbar=dict(title="final_value"),
    ))
    fig.update_xaxes(title="sell_T (long-leg exit SMA)")
    fig.update_yaxes(title="buy_T (long-leg entry SMA)")
    return _style_figure(fig, height=460)


def _heatmap_buyS_sellS(df: pd.DataFrame, fixed_buyT: int, fixed_sellT: int) -> go.Figure:
    sub = df[(df.buy_T == fixed_buyT) & (df.sell_T == fixed_sellT)]
    pivot = sub.pivot_table(index="buy_S", columns="sell_S", values="final_value")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns, y=pivot.index,
        colorscale=[[0, "#fff5e8"], [0.3, "#fde6c5"], [0.7, _TEAL_LIGHT], [1, _TEAL]],
        colorbar=dict(title="final_value"),
    ))
    fig.update_xaxes(title="sell_S (short-leg exit SMA)")
    fig.update_yaxes(title="buy_S (short-leg entry SMA)")
    return _style_figure(fig, height=460)


def _distribution_hist(df: pd.DataFrame, baselines: dict[str, float]) -> go.Figure:
    # final_value can span 0 → many thousands; log10 axis for readability.
    # Pre-bin in numpy to keep the embedded plotly payload to ~120 floats
    # (instead of 3.35M / 12.96M points). Without this the HTML file is
    # ~50 MB (Phase 1) or ~200 MB (Phase 2).
    vals = df["final_value"].to_numpy()
    log_vals = np.log10(np.clip(vals, 1e-3, None))
    counts, edges = np.histogram(log_vals, bins=120)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0])
    n_combos = len(df)
    # Pretty-print combo count: "3.35M combos" or "12.96M combos"
    if n_combos >= 1_000_000:
        combo_label = f"{n_combos / 1_000_000:.2f}M combos"
    else:
        combo_label = f"{n_combos:,} combos"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centers, y=counts, width=width,
        marker_color=_TEAL, opacity=0.85, name=combo_label,
        hovertemplate="log10(final)=%{x:.2f}<br>count=%{y:,}<extra></extra>",
    ))
    for name, v in baselines.items():
        if v <= 0:
            continue
        fig.add_vline(x=np.log10(v), line=dict(color=_AMBER, width=2, dash="dash"),
                      annotation_text=f"{name} = {v:.2f}×",
                      annotation_position="top")
    fig.update_xaxes(title="log10(final_value)")
    fig.update_yaxes(title="combo count")
    return _style_figure(fig, height=420)


def _walkforward_scatter(wf: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wf["final_value_train"], y=wf["final_value_test"],
        mode="markers",
        marker=dict(size=8, color=_TEAL, opacity=0.78,
                    line=dict(color="white", width=1)),
        text=[f"({r.buy_T},{r.sell_T},{r.buy_S},{r.sell_S})"
              for r in wf.itertuples()],
        hovertemplate="combo=%{text}<br>train=%{x:.2f}×<br>test=%{y:.2f}×<extra></extra>",
        name="top-N",
    ))
    # y = x reference: a perfectly-generalizing strategy lands here
    if len(wf):
        lo = min(wf["final_value_train"].min(), wf["final_value_test"].min())
        hi = max(wf["final_value_train"].max(), wf["final_value_test"].max())
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
            line=dict(color=_MUTED, dash="dot"),
        ))
    fig.update_xaxes(title="in-sample final_value (train: 2010–2018)")
    fig.update_yaxes(title="out-of-sample final_value (test: 2019–today)")
    return _style_figure(fig, height=460)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------


_HTML_HEAD = """<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>TQQQ/SQQQ SMA-Grid Backtest</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #fbfaf7; --paper: #fffefb; --surface: #f4f1ea; --surface-2: #ede8dc;
  --fg: #1d2024; --muted: #626b73; --subtle: #8a9299;
  --border: #ded7ca; --border-strong: #c9bfad;
  --accent: #0f766e; --accent-light: #7bd9c7; --code-bg: #efebe2; --row-alt: #f6f3ec;
  --warn: #b87333;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #101312; --paper: #171b19; --surface: #1d2320; --surface-2: #242b27;
    --fg: #e9e5dc; --muted: #a2aaa4; --subtle: #727c75;
    --border: #303832; --border-strong: #465047;
    --accent: #7bd9c7; --accent-light: #0f766e; --code-bg: #222821; --row-alt: #1b211e;
    --warn: #d9a36e;
  }
}
html { background: var(--bg); color: var(--fg); scroll-behavior: smooth; }
body {
  font: 16px/1.68 ui-sans-serif, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
  display: grid; grid-template-columns: 250px minmax(0, 880px);
  column-gap: 56px; max-width: 1240px; margin: 0 auto;
  padding: 56px 32px 112px; background: var(--bg); color: var(--fg);
}
body > header, body > section, body > footer { grid-column: 2; }
header { border-bottom: 1px solid var(--border); padding: 0 0 28px; margin-bottom: 40px; }
h1 { font-size: clamp(2.2rem, 4vw, 3.5rem); line-height: 1.04; font-weight: 720; margin: 0 0 12px; max-width: 14ch; }
h2 { font-size: 1.42rem; line-height: 1.22; margin: 64px 0 18px; font-weight: 680; }
h3 { font-size: 1.08rem; line-height: 1.34; margin: 32px 0 10px; font-weight: 650; }
section:first-of-type h2 { margin-top: 0; }
p, li { max-width: 72ch; }
.meta { color: var(--muted); font-size: 0.92rem; }
.subtitle { color: var(--muted); font-size: 1rem; margin: 0; }
nav.toc {
  grid-column: 1; grid-row: 1 / span 100;
  position: sticky; top: 28px; align-self: start;
  font-size: 0.88rem; color: var(--muted);
  border-left: 1px solid var(--border); padding: 8px 0 8px 16px;
}
nav.toc ol { list-style: none; padding: 0; margin: 0; }
nav.toc li { margin: 6px 0; line-height: 1.4; }
nav.toc a { color: var(--muted); text-decoration: none; }
nav.toc a:hover { color: var(--accent); }
@media (max-width: 900px) {
  body { grid-template-columns: 1fr; }
  body > header, body > section, body > footer { grid-column: 1; }
  nav.toc { position: static; border-left: none; border-top: 1px solid var(--border); padding: 16px 0; }
}
table.kv, table.data {
  width: 100%; border-collapse: collapse; font-size: 0.92rem; margin: 14px 0 22px;
  background: var(--paper); border: 1px solid var(--border);
}
table.kv th, table.kv td, table.data th, table.data td {
  text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border);
}
table.kv th, table.data th {
  background: var(--surface); color: var(--fg); font-weight: 620;
  border-bottom: 1px solid var(--border-strong);
}
table.data tr:nth-child(odd) td { background: var(--row-alt); }
td.num { font-variant-numeric: tabular-nums; text-align: right; }
td.delta-pos { color: var(--accent); font-weight: 620; font-variant-numeric: tabular-nums; text-align: right; }
td.delta-neg { color: #9f1239; font-weight: 620; font-variant-numeric: tabular-nums; text-align: right; }
code, kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.88em; background: var(--code-bg); padding: 1px 5px; border-radius: 4px; }
blockquote {
  border-left: 3px solid var(--accent); padding: 4px 0 4px 16px;
  color: var(--fg); background: var(--surface); margin: 18px 0;
}
.qbox {
  background: var(--paper); border: 1px solid var(--border-strong);
  border-left: 3px solid var(--warn);
  padding: 16px 18px; margin: 16px 0; border-radius: 6px;
}
.figure { margin: 18px 0 28px; }
.figure-caption { color: var(--muted); font-size: 0.86rem; margin-top: 6px; }
footer { margin-top: 80px; padding-top: 24px; border-top: 1px solid var(--border);
  color: var(--muted); font-size: 0.86rem; }
</style>
</head>
<body>
"""


def _section(num: int, title: str, anchor: str, body: str) -> str:
    return (f'<section id="{anchor}">\n'
            f'<h2>{num}. {title}</h2>\n{body}\n</section>\n')


def _top50_table(top: pd.DataFrame, wf: pd.DataFrame) -> str:
    """Top-50 table; highlight in-sample vs out-of-sample delta from walkforward.

    Each row is one DEDUPED strategy (unique equity curve). Dormant-leg cells
    render as ``-`` instead of an arbitrary parameter value to make it obvious
    that the SMA window is structurally irrelevant for that row (e.g. sell_S
    on a LONG-only strategy). The ``type`` column classifies the row, and
    ``n_equivalent`` shows how many grid combos collapsed into it.
    """
    # Top-50 keyed by combo. Pull WF columns when present.
    # wf may be the empty `pd.DataFrame()` fallback when walkforward parquet is
    # missing — guard the indexing so the table still renders (just without
    # train/test/delta columns populated).
    required = {"buy_T", "sell_T", "buy_S", "sell_S", "final_value_train", "final_value_test"}
    if not wf.empty and required.issubset(wf.columns):
        wf_indexed = wf.set_index(["buy_T", "sell_T", "buy_S", "sell_S"])[
            ["final_value_train", "final_value_test"]
        ].to_dict("index")
    else:
        wf_indexed = {}

    has_neq = "n_equivalent" in top.columns

    rows = ["<table class=\"data\"><thead><tr>"
            "<th>#</th><th>type</th><th>buy_T</th><th>sell_T</th><th>buy_S</th><th>sell_S</th>"
            "<th>n_eq</th>"
            "<th>final×</th><th>Sharpe</th><th>max_dd</th><th>Calmar</th>"
            "<th>trades</th><th>%long</th><th>%short</th>"
            "<th>train×</th><th>test×</th><th>delta</th>"
            "</tr></thead><tbody>"]
    for i, row in enumerate(top.itertuples(index=False), start=1):
        key = (int(row.buy_T), int(row.sell_T), int(row.buy_S), int(row.sell_S))
        wfr = wf_indexed.get(key)
        train = wfr["final_value_train"] if wfr else float("nan")
        test = wfr["final_value_test"] if wfr else float("nan")
        if not np.isnan(train) and not np.isnan(test) and train > 0:
            delta = (test - train) / train
            delta_cls = "delta-pos" if delta >= 0 else "delta-neg"
            delta_s = f"{delta*100:+.1f}%"
        else:
            delta_cls = "num"
            delta_s = "—"

        long_dormant = float(row.time_in_long) == 0.0
        short_dormant = float(row.time_in_short) == 0.0
        type_str = _strategy_type(float(row.time_in_long), float(row.time_in_short))
        bT_s = _dormant_cell(row.buy_T, long_dormant)
        sT_s = _dormant_cell(row.sell_T, long_dormant)
        bS_s = _dormant_cell(row.buy_S, short_dormant)
        sS_s = _dormant_cell(row.sell_S, short_dormant)
        n_eq = int(getattr(row, "n_equivalent")) if has_neq else 1

        rows.append(
            f"<tr>"
            f"<td class=\"num\">{i}</td>"
            f"<td>{type_str}</td>"
            f"<td class=\"num\">{bT_s}</td><td class=\"num\">{sT_s}</td>"
            f"<td class=\"num\">{bS_s}</td><td class=\"num\">{sS_s}</td>"
            f"<td class=\"num\">{n_eq:,}</td>"
            f"<td class=\"num\">{row.final_value:.2f}×</td>"
            f"<td class=\"num\">{row.sharpe:.2f}</td>"
            f"<td class=\"num\">{row.max_dd*100:.1f}%</td>"
            f"<td class=\"num\">{row.calmar if not np.isnan(row.calmar) else 0:.2f}</td>"
            f"<td class=\"num\">{int(row.n_trades)}</td>"
            f"<td class=\"num\">{row.time_in_long*100:.1f}%</td>"
            f"<td class=\"num\">{row.time_in_short*100:.1f}%</td>"
            f"<td class=\"num\">{train:.2f}×</td>"
            f"<td class=\"num\">{test:.2f}×</td>"
            f"<td class=\"{delta_cls}\">{delta_s}</td>"
            f"</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _baseline_kv_table(prices: pd.DataFrame, baselines: dict[str, float]) -> str:
    rows = ["<table class=\"kv\"><thead><tr>"
            "<th>baseline</th><th class=\"num\">final×</th><th class=\"num\">CAGR</th>"
            "<th class=\"num\">max_dd</th></tr></thead><tbody>"]
    n_days = len(prices)
    years = n_days / 252.0
    # CAGR / dd for the baselines
    for name in ["QQQ B&H", "TQQQ B&H", "SQQQ B&H", "1/N(QQQ, TQQQ, SQQQ)"]:
        v = baselines[name]
        cagr = v ** (1.0 / years) - 1.0 if v > 0 and years > 0 else float("nan")
        eq = baselines[f"_curve::{name}"]
        peak = eq.cummax()
        dd = (peak - eq) / peak
        max_dd = dd.max()
        rows.append(
            f"<tr><td>{name}</td>"
            f"<td class=\"num\">{v:.2f}×</td>"
            f"<td class=\"num\">{cagr*100:.1f}%</td>"
            f"<td class=\"num\">{max_dd*100:.1f}%</td></tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _phase1_vs_phase2_section(df_raw: pd.DataFrame) -> str:
    """Build the headline comparison: does any anti-trend combo beat the
    trend-following winner?

    Splits the full Phase-2 grid into the trend-following subset
    (``sell_T >= buy_T`` AND ``sell_S >= buy_S``) and its complement (the
    anti-trend additions only Phase 2 explores), then reports the best
    final_value in each. The headline answer is whether the anti-trend best
    exceeds the trend-following best.

    Degenerate-winner disclosure (review iter-1): a combo with ``n_trades == 1``
    entered the market once and never exited — it is effectively leveraged
    buy-and-hold with a one-time slippage hit, not a "strategy". The
    unconstrained Phase 2 grid surfaces these (e.g. ``sell_T=1`` with the
    SMA(1) validity mask makes the long-exit signal structurally never fire,
    so the inner backtest never exits LONG_TQQQ). When the headline winner is
    of this shape, the section calls it out explicitly instead of presenting
    it as a discovered strategy that beats trend-following — the user needs
    to know they are looking at "buy and hold TQQQ" wearing an SMA mask, not
    a real mean-reversion strategy.
    """
    p1_mask = (df_raw["sell_T"] >= df_raw["buy_T"]) & (df_raw["sell_S"] >= df_raw["buy_S"])
    p1 = df_raw[p1_mask]
    anti = df_raw[~p1_mask]

    n_p1 = len(p1)
    n_anti = len(anti)
    n_total = len(df_raw)

    p1_best = p1.nlargest(1, "final_value").iloc[0]
    anti_best = anti.nlargest(1, "final_value").iloc[0] if n_anti else None
    overall_best = df_raw.nlargest(1, "final_value").iloc[0]

    p1_combo = (int(p1_best.buy_T), int(p1_best.sell_T),
                int(p1_best.buy_S), int(p1_best.sell_S))
    p1_final = float(p1_best.final_value)
    p1_sharpe = float(p1_best.sharpe)
    p1_trades = int(p1_best.n_trades)

    if anti_best is not None:
        anti_combo = (int(anti_best.buy_T), int(anti_best.sell_T),
                      int(anti_best.buy_S), int(anti_best.sell_S))
        anti_final = float(anti_best.final_value)
        anti_sharpe = float(anti_best.sharpe)
        anti_trades = int(anti_best.n_trades)
        anti_beats_p1 = anti_final > p1_final
        delta_pct = (anti_final - p1_final) / p1_final * 100.0
    else:
        anti_combo = None
        anti_final = float("nan")
        anti_sharpe = float("nan")
        anti_trades = 0
        anti_beats_p1 = False
        delta_pct = float("nan")

    overall_combo = (int(overall_best.buy_T), int(overall_best.sell_T),
                     int(overall_best.buy_S), int(overall_best.sell_S))
    overall_final = float(overall_best.final_value)

    # Quantile placement: how does anti-trend best rank among trend-following?
    anti_pctile_in_p1 = (
        100.0 * (p1["final_value"] < anti_final).mean()
        if anti_best is not None else float("nan")
    )

    # Degenerate detection: n_trades == 1 means the strategy entered once and
    # never exited — effective buy-and-hold (TQQQ-only if it entered LONG, or
    # SQQQ-short if it entered the short leg). Report this honestly rather
    # than dressing it up as a discovered anti-trend strategy.
    anti_is_degenerate = anti_best is not None and anti_trades == 1
    p1_is_degenerate = p1_trades == 1

    # DEGENERATE banner fires whenever the anti-trend representative has
    # n_trades=1 — entered once and never exited. This is the buy-and-hold
    # equivalent we want to surface honestly, regardless of whether it
    # nominally "beats" the Phase-1 winner. (On real data anti beats Phase 1
    # by the slippage-on-entry gap; on monotonic synthetic data they tie
    # because P1's winner also enters once and never exits. Either way the
    # reader needs to know the headline is buy-and-hold-equivalent.)
    if anti_is_degenerate:
        headline_class = "delta-neg"
        comparator = (
            f"Trend-following winner {p1_combo} ({p1_final:.2f}×, "
            f"n_trades={p1_trades}) is the meaningful comparator."
            if not p1_is_degenerate else
            f"Phase-1 winner {p1_combo} is ALSO n_trades=1 — both regimes' "
            f"top rows are buy-and-hold-equivalents."
        )
        if anti_beats_p1:
            beat_clause = (
                f"nominally beats trend-following winner {p1_combo} "
                f"({p1_final:.2f}×) by {delta_pct:+.1f}%, but"
            )
        else:
            beat_clause = (
                f"matches/trails trend-following winner {p1_combo} "
                f"({p1_final:.2f}×) by {delta_pct:+.1f}% and"
            )
        headline_text = (
            f"DEGENERATE — best anti-trend combo {anti_combo} returns "
            f"{anti_final:.2f}×; it {beat_clause} has n_trades=1 (entered "
            f"once and never exited). This is effectively buy-and-hold TQQQ "
            f"with a single slippage hit, not a real anti-trend strategy. "
            f"The exit signal is structurally dormant (e.g. sell_T=1 → "
            f"SMA(1) validity mask prevents the exit firing). {comparator} "
            f"Phase 2's unconstrained grid surfaces these "
            f"buy-and-hold-equivalents — interpret the headline accordingly."
        )
    elif anti_beats_p1:
        headline_class = "delta-pos"
        headline_text = (
            f"YES — best anti-trend combo {anti_combo} returns "
            f"{anti_final:.2f}× (n_trades={anti_trades}), beating "
            f"trend-following winner {p1_combo} ({p1_final:.2f}×, "
            f"n_trades={p1_trades}) by {delta_pct:+.1f}%."
        )
    else:
        headline_class = "delta-neg"
        headline_text = (
            f"NO — trend-following winner {p1_combo} ({p1_final:.2f}×) "
            f"remains the overall best. Best anti-trend combo {anti_combo} "
            f"lands at {anti_final:.2f}× ({delta_pct:+.1f}% vs trend-following "
            f"winner), in the {anti_pctile_in_p1:.1f}th percentile of "
            f"trend-following combos."
        )

    def _trade_cell(n: int) -> str:
        if n == 1:
            return f'{n} <span style="color: var(--warn);">(buy-and-hold)</span>'
        return f'{n}'

    table = (
        '<table class="data"><thead><tr>'
        '<th>regime</th><th class="num">combos</th>'
        '<th>best combo</th><th class="num">final×</th>'
        '<th class="num">Sharpe</th><th class="num">trades</th>'
        '</tr></thead><tbody>'
        f'<tr><td>Phase 1 (trend-following: sell ≥ buy)</td>'
        f'<td class="num">{n_p1:,}</td>'
        f'<td><code>{p1_combo}</code></td>'
        f'<td class="num">{p1_final:.2f}×</td>'
        f'<td class="num">{p1_sharpe:.2f}</td>'
        f'<td class="num">{_trade_cell(p1_trades)}</td></tr>'
        f'<tr><td>Phase 2 only (anti-trend: sell &lt; buy on at least one leg)</td>'
        f'<td class="num">{n_anti:,}</td>'
        f'<td><code>{anti_combo}</code></td>'
        f'<td class="num">{anti_final:.2f}×</td>'
        f'<td class="num">{anti_sharpe:.2f}</td>'
        f'<td class="num">{_trade_cell(anti_trades)}</td></tr>'
        f'<tr><td><strong>Phase 2 (full grid)</strong></td>'
        f'<td class="num"><strong>{n_total:,}</strong></td>'
        f'<td><code>{overall_combo}</code></td>'
        f'<td class="num"><strong>{overall_final:.2f}×</strong></td>'
        f'<td class="num">{float(overall_best.sharpe):.2f}</td>'
        f'<td class="num">{_trade_cell(int(overall_best.n_trades))}</td></tr>'
        '</tbody></table>'
    )

    # Border color: green only when anti-trend strictly beats AND isn't
    # degenerate; otherwise warn — either anti-trend lost OR the "winner" is
    # a buy-and-hold-equivalent that shouldn't be celebrated. We also warn
    # when the Phase-1 winner itself is degenerate (the leaderboard is being
    # led by a buy-and-hold-equivalent), so the reader doesn't take the
    # comparison at face value.
    is_meaningful_win = (
        anti_beats_p1 and not anti_is_degenerate and not p1_is_degenerate
    )
    qbox_color = "var(--accent)" if is_meaningful_win else "var(--warn)"

    # Optional supporting paragraph called out when the winners are degenerate
    # — gives the reader the cause (SMA(1) validity gating) without burying it
    # in the discussion section.
    degenerate_note = ""
    if anti_is_degenerate or p1_is_degenerate:
        legs = []
        if p1_is_degenerate:
            legs.append(f"the Phase-1 winner {p1_combo}")
        if anti_is_degenerate:
            legs.append(f"the anti-trend winner {anti_combo}")
        legs_phrase = " and ".join(legs)
        degenerate_note = (
            f'<p class="meta"><strong>note.</strong> {legs_phrase} '
            f'shows n_trades=1 — entered once, never exited. The exit leg is '
            f'structurally dormant (typically <code>sell_T=1</code> or '
            f'<code>sell_S=1</code>, where the SMA(1) validity mask in '
            f'<code>precompute_sma_signals</code> prevents the exit signal '
            f'from firing). The result is equivalent to leveraged buy-and-hold '
            f'starting from the first entry signal, with one slippage hit. '
            f'These rows are visible on the leaderboard because the dedup pass '
            f'keeps one representative per equity curve; the <code>n_eq</code> '
            f'column in §3 shows how many grid combos collapse to each curve.</p>'
        )

    return (
        '<p>Phase 2 drops the <code>sell ≥ buy</code> trend-following '
        'constraint and explores the full 4-D <code>{1..60}^4</code> grid. '
        'Of the 12,960,000 combos, 3,348,900 are the original trend-following '
        'set; the remaining 9,611,100 are anti-trend regions where one or '
        'both legs use a shorter exit-SMA than entry-SMA — i.e. mean-reversion '
        'rather than momentum.</p>'
        f'<div class="qbox" style="border-left-color: {qbox_color};">'
        f'<strong>headline.</strong> <span class="{headline_class}">'
        f'{headline_text}</span></div>'
        + table +
        degenerate_note +
        '<p class="meta">Walk-forward generalization for these comparisons '
        'lives in §8 (the in-sample / out-of-sample scatter); the headline '
        'above is the FULL-sample top combo from each regime, not the '
        'cross-validated one.</p>'
    )


def render_report(
    prices: pd.DataFrame,
    results_path: Path,
    walkforward_path: Path,
    output_path: Path,
    sweep_wall_seconds: float,
    slippage_bp: float = 5.0,
    max_window: int = 60,
    phase: int = 1,
) -> Path:
    """Write the design-doc style HTML report to ``output_path``.

    Parameters
    ----------
    prices:
        Frame with columns qqq/tqqq/sqqq used for the sweep (and equity curves).
    results_path:
        Parquet of all combo results.
    walkforward_path:
        Parquet with the top-N walk-forward results (the file produced by
        :func:`walk_forward_top_n` and written by the CLI).
    output_path:
        Where to write the HTML.
    sweep_wall_seconds:
        Wall-clock time of the producing sweep, embedded in the framing section.
    slippage_bp, max_window:
        Reproducibility metadata, also reused for the equity-curve replay.
    """
    results_path = Path(results_path)
    walkforward_path = Path(walkforward_path)
    output_path = Path(output_path)

    df_raw = pd.read_parquet(results_path)
    wf = pd.read_parquet(walkforward_path) if walkforward_path.exists() else pd.DataFrame()

    n_combos = len(df_raw)
    # Dedup by equity-curve fingerprint before any leaderboard slicing so the
    # top-50 contains 50 truly-distinct strategies, not 50 copies of one
    # underlying strategy with dormant parameters varying.
    df = dedup_by_strategy_id(df_raw)
    n_distinct = len(df)
    # Sanity invariant: dedup preserves the total combo count via n_equivalent
    # (one row per group, summed weights == input row count).
    assert int(df["n_equivalent"].sum()) == n_combos, (
        f"dedup lost rows: n_equivalent sum = {int(df['n_equivalent'].sum())}, "
        f"expected {n_combos}"
    )
    top50 = df.nlargest(50, "final_value").reset_index(drop=True)
    top_combos = [(int(r.buy_T), int(r.sell_T), int(r.buy_S), int(r.sell_S))
                  for r in top50.itertuples()]
    winner = top_combos[0]

    # Baselines
    qqq_curve = prices["qqq"] / prices["qqq"].iloc[0]
    tqqq_curve = prices["tqqq"] / prices["tqqq"].iloc[0]
    sqqq_curve = prices["sqqq"] / prices["sqqq"].iloc[0]
    one_n_curve = (qqq_curve + tqqq_curve + sqqq_curve) / 3.0
    baselines = {
        "QQQ B&H": float(qqq_curve.iloc[-1]),
        "TQQQ B&H": float(tqqq_curve.iloc[-1]),
        "SQQQ B&H": float(sqqq_curve.iloc[-1]),
        "1/N(QQQ, TQQQ, SQQQ)": float(one_n_curve.iloc[-1]),
        "_curve::QQQ B&H": qqq_curve,
        "_curve::TQQQ B&H": tqqq_curve,
        "_curve::SQQQ B&H": sqqq_curve,
        "_curve::1/N(QQQ, TQQQ, SQQQ)": one_n_curve,
    }

    # Figures. Heatmaps + distribution operate on the FULL undeduped grid
    # (df_raw): slicing the 4-D grid by fixed buy_S/sell_S requires every
    # combo to be present, and the distribution histogram is "what does the
    # full search surface look like" — not "what does it look like after we
    # collapse equivalents". Only the leaderboard tables and equity-curve
    # replay use the deduped df.
    fig_equity = _equity_overlay(prices, top_combos, slippage_bp=slippage_bp, max_window=max_window)
    fig_heatT = _heatmap_buyT_sellT(df_raw, fixed_buyS=winner[2], fixed_sellS=winner[3])
    fig_heatS = _heatmap_buyS_sellS(df_raw, fixed_buyT=winner[0], fixed_sellT=winner[1])
    fig_dist = _distribution_hist(
        df_raw,
        baselines={k: v for k, v in baselines.items() if not k.startswith("_curve::")},
    )
    fig_wf = _walkforward_scatter(wf) if len(wf) else None

    # Compose
    parts = [_HTML_HEAD]

    # TOC — Phase 2 inserts an extra "Phase 1 vs Phase 2 comparison" section
    # between framing and top-50, renumbering subsequent sections.
    if phase == 1:
        parts.append(
            '<nav class="toc"><strong>contents</strong><ol>'
            '<li><a href="#framing">1. framing</a></li>'
            '<li><a href="#top50">2. top-50 winners</a></li>'
            '<li><a href="#baselines">3. buy-and-hold baselines</a></li>'
            '<li><a href="#heatmaps">4. heatmaps</a></li>'
            '<li><a href="#distribution">5. distribution of outcomes</a></li>'
            '<li><a href="#equity">6. equity curves</a></li>'
            '<li><a href="#walkforward">7. walk-forward delta</a></li>'
            '<li><a href="#discussion">8. honest discussion</a></li>'
            '<li><a href="#repro">9. reproducibility</a></li>'
            '</ol></nav>'
        )
    else:
        parts.append(
            '<nav class="toc"><strong>contents</strong><ol>'
            '<li><a href="#framing">1. framing</a></li>'
            '<li><a href="#compare">2. Phase 1 vs Phase 2</a></li>'
            '<li><a href="#top50">3. top-50 winners</a></li>'
            '<li><a href="#baselines">4. buy-and-hold baselines</a></li>'
            '<li><a href="#heatmaps">5. heatmaps</a></li>'
            '<li><a href="#distribution">6. distribution of outcomes</a></li>'
            '<li><a href="#equity">7. equity curves</a></li>'
            '<li><a href="#walkforward">8. walk-forward delta</a></li>'
            '<li><a href="#discussion">9. honest discussion</a></li>'
            '<li><a href="#repro">10. reproducibility</a></li>'
            '</ol></nav>'
        )

    # Header
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if phase == 1:
        subtitle = (
            f'Phase 1 trend-following sweep · {n_combos:,} combos '
            f'({n_distinct:,} distinct equity curves) · '
            f'data {prices.index[0].date()} → {prices.index[-1].date()} · {today}'
        )
    else:
        subtitle = (
            f'Phase 2 unconstrained sweep · {n_combos:,} combos '
            f'({n_distinct:,} distinct equity curves) · '
            f'data {prices.index[0].date()} → {prices.index[-1].date()} · {today}'
        )
    parts.append(
        f'<header><h1>TQQQ/SQQQ SMA-grid backtest</h1>'
        f'<p class="subtitle">{subtitle}</p></header>'
    )

    # 1. Framing — phase-aware constraint description
    if phase == 1:
        constraint_para = (
            '<p>Phase 1 of this sweep enforces <code>sell_T ≥ buy_T</code> and '
            '<code>sell_S ≥ buy_S</code> — trend-following only. Each SMA window '
            'is drawn from <code>{1..60}</code>, so the constrained grid has '
            '<code>1830 × 1830 = 3,348,900</code> combos.</p>'
        )
    else:
        constraint_para = (
            '<p>Phase 2 of this sweep drops the trend-following constraint and '
            'explores the full <code>{1..60}^4 = 12,960,000</code> combo grid. '
            'This adds 9,611,100 anti-trend combos (<code>sell &lt; buy</code> on '
            'one or both legs) on top of the 3,348,900 trend-following combos '
            'from Phase 1. Anti-trend regions correspond to mean-reversion '
            'strategies: a tighter exit SMA than entry SMA implies the strategy '
            'exits the position as soon as the trend confirms — only useful in '
            'a regime where short-horizon momentum is reverting, not '
            'continuing.</p>'
        )
    framing_body = (
        '<p>The strategy is a three-state rotation driven by a single signal '
        'instrument (QQQ) against four simple-moving-average windows. On every '
        'daily bar the state machine is</p>'
        '<pre style="background:var(--code-bg); padding:14px; border-radius:6px; '
        'font-size:0.86rem; overflow-x:auto;">CASH        → LONG_TQQQ    when QQQ &gt; SMA(buy_T)\n'
        'LONG_TQQQ   → CASH         when QQQ &lt; SMA(sell_T)\n'
        'CASH        → SHORT_SQQQ   when QQQ &lt; SMA(buy_S)\n'
        'SHORT_SQQQ  → CASH         when QQQ &gt; SMA(sell_S)</pre>'
        + constraint_para +
        f'<p class="meta">Sweep wall-clock: {sweep_wall_seconds/60:.1f} min · '
        f'~{n_combos / max(sweep_wall_seconds, 1e-3):,.0f} backtests/sec. '
        f'Slippage: {slippage_bp:.1f} bp per state transition; $0 commission; '
        f'adjusted-close prices.</p>'
        '<blockquote>This document tells you what the search surface actually '
        'looks like — not just the single best combo. The histogram further '
        'down is the antidote to the leaderboard above it.</blockquote>'
    )
    parts.append(_section(1, "framing", "framing", framing_body))

    # Numbering offset: Phase 2 inserts a new section at index 2.
    sec_off = 1 if phase == 2 else 0

    # 2. Phase 1 vs Phase 2 (only in Phase 2 reports)
    if phase == 2:
        compare_body = _phase1_vs_phase2_section(df_raw)
        parts.append(_section(2, "Phase 1 vs Phase 2 comparison", "compare", compare_body))

    # 2/3. Top-50
    top50_body = (
        '<p>Sorted by final equity multiple. Each row is one DISTINCT equity '
        'curve — combos sharing a curve (e.g. dormant short-leg parameters '
        'varying freely when <code>buy_S=1</code> structurally never fires) '
        'are collapsed by a deterministic equity-curve fingerprint. '
        '<code>n_eq</code> is the size of the equivalence class; '
        '<code>type</code> classifies which legs actually traded; dormant '
        f'parameters render as <code>-</code>. Of the {n_combos:,} grid '
        f'combos, {n_distinct:,} produce distinct equity curves.</p>'
        '<p><code>delta</code> is the '
        '<code>(test − train) / train</code> ratio from the walk-forward — '
        'green = generalizes up, red = generalizes down.</p>'
        + _top50_table(top50, wf)
    )
    parts.append(_section(2 + sec_off, "top-50 winners", "top50", top50_body))

    # 3/4. Baselines
    baseline_body = (
        '<p>The four passive baselines a strategy must beat to be interesting:</p>'
        + _baseline_kv_table(prices, baselines)
    )
    parts.append(_section(3 + sec_off, "buy-and-hold baselines", "baselines", baseline_body))

    # 4/5. Heatmaps
    heatmap_body = (
        f'<p>Two 2-D slices through the 4-D grid. The winner combo is '
        f'<code>{winner}</code>.</p>'
        f'<h3>{4 + sec_off}a. (buy_T, sell_T) with short-leg fixed at the winner</h3>'
        '<div class="figure">' + _fig_to_div(fig_heatT, include_js=True) + '</div>'
        f'<h3>{4 + sec_off}b. (buy_S, sell_S) with long-leg fixed at the winner</h3>'
        '<div class="figure">' + _fig_to_div(fig_heatS, include_js=False) + '</div>'
    )
    parts.append(_section(4 + sec_off, "heatmaps", "heatmaps", heatmap_body))

    # 5/6. Distribution
    dist_label = "3.35M-combo" if phase == 1 else "12.96M-combo"
    dist_body = (
        f'<p>Where do the buy-and-hold baselines sit in the {dist_label} '
        'final-value distribution? Most of the mass is mediocre; the long '
        'right tail is where the headline numbers live.</p>'
        '<div class="figure">' + _fig_to_div(fig_dist, include_js=False) + '</div>'
    )
    parts.append(_section(5 + sec_off, "distribution of outcomes", "distribution", dist_body))

    # 6/7. Equity
    equity_body = (
        '<p>Top-5 strategies vs the three buy-and-hold baselines. Log scale '
        'on the y-axis so the leveraged baselines don\'t flatten the rest.</p>'
        '<div class="figure">' + _fig_to_div(fig_equity, include_js=False) + '</div>'
    )
    parts.append(_section(6 + sec_off, "equity curves", "equity", equity_body))

    # 7/8. Walk-forward
    if fig_wf is not None:
        wf_body = (
            '<p>For each of the top-' + str(len(wf)) + ' DEDUPED combos by '
            'full-sample final value (one representative per equity-curve '
            'equivalence class), we re-ran the backtest on <em>train</em> '
            '(2010-02 → 2018-12) and <em>test</em> (2019-01 → today). Points '
            'hugging the diagonal generalize; points dropping below it overfit.</p>'
            '<div class="figure">' + _fig_to_div(fig_wf, include_js=False) + '</div>'
        )
        # Quick generalization stat
        if "final_value_train" in wf.columns and "final_value_test" in wf.columns:
            mask = (wf["final_value_train"] > 1.0)
            gen = ((wf["final_value_test"] > 1.0) & mask).sum()
            total = int(mask.sum())
            if total > 0:
                wf_body += (
                    f'<p class="meta">Of the {total} combos that beat cash '
                    f'in-sample, {gen} also beat cash out-of-sample '
                    f'({gen/total*100:.1f}%).</p>'
                )
    else:
        wf_body = '<p class="meta">Walk-forward parquet not found — re-run with --top-n-walkforward&nbsp;&gt;&nbsp;0.</p>'
    parts.append(_section(7 + sec_off, "walk-forward delta", "walkforward", wf_body))

    # 8/9. Honest discussion
    discussion_combo_label = "3.35M combos" if phase == 1 else "12.96M combos"
    extra_qbox = ""
    if phase == 2:
        extra_qbox = (
            '<div class="qbox"><strong>anti-trend regions are noisy.</strong> '
            'Dropping <code>sell ≥ buy</code> adds 9.61M combos where the exit '
            'SMA is tighter than the entry SMA. Most of these correspond to '
            'high-churn mean-reversion strategies that fall apart out-of-sample. '
            'The walk-forward delta is the honest test — full-sample winners '
            'in anti-trend territory are especially suspect because the '
            'optimizer has 3× more grid to lottery-pick from.</div>'
        )
    discussion_body = (
        '<div class="qbox"><strong>survivorship bias.</strong> QQQ, TQQQ, and '
        'SQQQ all survived the sweep period. A real strategy needs to handle '
        'tickers that fail or get delisted; we did not.</div>'
        '<div class="qbox"><strong>leveraged-ETF decay.</strong> TQQQ and SQQQ '
        'are 3x daily-rebalanced — they bleed in choppy markets and explode '
        'in trending ones. The backtest uses adjusted close which handles '
        'splits, but daily decay is structural, not a fee.</div>'
        '<div class="qbox"><strong>slippage assumption.</strong> 5 bp per '
        'transition is generous for retail liquidity in these tickers, but '
        'optimistic during market stress. Combos with thousands of trades '
        'feel the slippage more than the headline number suggests.</div>'
        f'<div class="qbox"><strong>{discussion_combo_label} guarantee lucky picks.</strong> '
        'Even pure noise generates extreme outliers at this sample size. The '
        'walk-forward is the honest filter; the headline table is the '
        'seduction.</div>'
        + extra_qbox +
        '<div class="qbox"><strong>signal source ≠ traded instrument.</strong> '
        'We trigger on QQQ but trade TQQQ/SQQQ — there is an implicit '
        'assumption that the leveraged-fund tracking error stays small. '
        'Historically true on a daily basis; not guaranteed in a stressed '
        'market.</div>'
    )
    parts.append(_section(8 + sec_off, "honest discussion", "discussion", discussion_body))

    # 9/10. Reproducibility
    git_sha = _git_sha(Path(__file__).resolve().parent)
    parquet_hash = _file_sha256(results_path)
    wf_hash = _file_sha256(walkforward_path) if walkforward_path.exists() else "missing"
    # Dataset hash derived from the qqq column to stay independent of pyarrow buffer layout
    ds_hash = hashlib.sha256(prices.to_csv().encode()).hexdigest()
    repro_cmd = f"uv run rainier sma-sweep --phase {phase}"
    repro_body = (
        '<table class="kv"><tbody>'
        f'<tr><th>git SHA</th><td><code>{git_sha}</code></td></tr>'
        f'<tr><th>results.parquet SHA-256</th><td><code>{parquet_hash}</code></td></tr>'
        f'<tr><th>walkforward.parquet SHA-256</th><td><code>{wf_hash}</code></td></tr>'
        f'<tr><th>dataset SHA-256</th><td><code>{ds_hash}</code></td></tr>'
        f'<tr><th>run date (UTC)</th><td>{today}</td></tr>'
        f'<tr><th>sweep wall-clock</th><td>{sweep_wall_seconds/60:.2f} min</td></tr>'
        f'<tr><th>max_window</th><td>{max_window}</td></tr>'
        f'<tr><th>slippage</th><td>{slippage_bp:.2f} bp/transition</td></tr>'
        f'<tr><th>phase</th><td>{phase}</td></tr>'
        f'<tr><th>combos</th><td>{n_combos:,}</td></tr>'
        f'<tr><th>distinct equity curves</th><td>{n_distinct:,}</td></tr>'
        '</tbody></table>'
        f'<p class="meta">To reproduce: <code>{repro_cmd}</code></p>'
    )
    parts.append(_section(9 + sec_off, "reproducibility", "repro", repro_body))

    parts.append(
        f'<footer>Generated {today} UTC · git <code>{git_sha[:12]}</code> · '
        f'{n_combos:,} backtests in {sweep_wall_seconds/60:.1f} min.</footer>'
    )
    parts.append("</body></html>\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(parts), encoding="utf-8")
    return output_path
