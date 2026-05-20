"""HTML report for the SQQQ-only SMA sweep.

Mirrors the design-doc "hub" style used by ``tqqq_sma_report.py`` (sticky
TOC, numbered sections, embedded CSS, no external assets) but for the
simpler 2-state SQQQ-only strategy. The report MUST surface the SMA(1)
degeneracy honestly via a DEGENERATE banner when the headline rows are
all-cash (``n_trades == 0``) or one-shot (``n_trades == 1``).

Sections (all 9 from the task spec):
    1. Header + framing — what was tested, data range, total combos
    2. Top-50 winners table — strategy_id dedup, dormant ``-`` rendering,
       DEGENERATE banner if applicable
    3. Buy-and-hold baselines — QQQ, SQQQ B&H, Phase-1 SMA winner for context
    4. Heatmap — (buy_S, sell_S) → final_value across the full grid
    5. Distribution of outcomes
    6. Equity curves of top-5 vs SQQQ B&H, QQQ B&H, cash
    7. Walk-forward delta for top-50
    8. Honest discussion — can timing SQQQ entry/exit beat cash?
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

from rainier.backtest.sqqq_sma_sweep import (
    LONG_SQQQ,
    dedup_by_strategy_id,
)
from rainier.backtest.tqqq_sma_sweep import precompute_sma_signals

# Palette matches the TQQQ report so the two documents read as a series
_TEAL = "#0f766e"
_TEAL_LIGHT = "#7bd9c7"
_MUTED = "#626b73"
_AMBER = "#b87333"
_RED = "#9f1239"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _strategy_type(n_trades: int, time_in_short: float) -> str:
    """Classify each row's structural status.

    Outcomes:
        - ``ALL-CASH``: n_trades == 0 → strategy never entered. Trivially
          equal to cash (final = 1.0). Buy_S=1 is the canonical example.
        - ``ONE-SHOT``: n_trades == 1 → entered once, never exited. Equity
          curve is SQQQ B&H from entry forward. sell_S=1 forces this when
          buy_S has fired.
        - ``ACTIVE``: more than one trade → real strategy with at least one
          round-trip.

    The DEGENERATE banner fires when the leaderboard #1 is ALL-CASH or
    ONE-SHOT — neither is a real "discovered" SQQQ-timing strategy.
    """
    if n_trades == 0:
        return "ALL-CASH"
    if n_trades == 1:
        return "ONE-SHOT"
    return "ACTIVE"


def _dormant_cell(value: int, dormant: bool) -> str:
    """Render an integer param as ``-`` when the short leg never fired.

    For an ALL-CASH row, BOTH buy_S and sell_S are structurally irrelevant
    (no trade was made). For a ONE-SHOT row, ONLY sell_S is dormant (it
    governs exit, which never fired). The caller decides which leg is
    dormant; this helper just renders.
    """
    return "-" if dormant else f"{int(value)}"


def _equity_curve_for(prices: pd.DataFrame, combo: tuple[int, int],
                      slippage_bp: float, max_window: int) -> pd.Series:
    """Replay one combo in-Python (slow but only for top-5 visualization).

    Must mirror :func:`rainier.backtest.sqqq_sma_sweep.run_backtest` exactly,
    including validity gating, so the equity curve shown in the report
    matches the sweep's ``final_value`` for that combo.
    """
    qqq = prices["qqq"].to_numpy(dtype=np.float64)
    sqqq = prices["sqqq"].to_numpy(dtype=np.float64)
    above, valid = precompute_sma_signals(qqq, max_window=max_window)
    sqqq_ret = (sqqq[1:] / sqqq[:-1]) - 1.0
    bS, sS = combo
    n = qqq.shape[0]
    slip = slippage_bp * 1e-4

    eq = np.empty(n)
    state = 0  # CASH
    equity = 1.0

    col_bS = above[:, bS - 1]
    col_sS = above[:, sS - 1]
    v_bS = valid[:, bS - 1]
    v_sS = valid[:, sS - 1]

    if v_bS[0] and not col_bS[0]:
        state = LONG_SQQQ
        equity *= 1.0 - slip
    eq[0] = equity

    for d in range(1, n):
        if state == LONG_SQQQ:
            equity *= 1.0 + sqqq_ret[d - 1]
        if state == LONG_SQQQ and v_sS[d] and col_sS[d]:
            state = 0
            equity *= 1.0 - slip
        if state == 0 and v_bS[d] and not col_bS[d]:
            state = LONG_SQQQ
            equity *= 1.0 - slip
        eq[d] = equity

    return pd.Series(eq, index=prices.index, name=f"({bS},{sS})")


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
        colorway=[_TEAL, _AMBER, "#1d4ed8", _RED, "#4d7c0f", _TEAL_LIGHT],
        plot_bgcolor="#fffefb",
        paper_bgcolor="#fffefb",
    )
    return fig


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------


def _equity_overlay(prices: pd.DataFrame, top_combos: list[tuple[int, int]],
                    slippage_bp: float, max_window: int) -> go.Figure:
    fig = go.Figure()
    base_qqq = prices["qqq"] / prices["qqq"].iloc[0]
    base_sqqq = prices["sqqq"] / prices["sqqq"].iloc[0]
    fig.add_trace(go.Scatter(x=prices.index, y=base_qqq, name="QQQ B&H",
                             line=dict(color=_MUTED, dash="dot")))
    fig.add_trace(go.Scatter(x=prices.index, y=base_sqqq, name="SQQQ B&H",
                             line=dict(color=_RED, dash="dash")))
    # Cash baseline (constant 1.0)
    fig.add_trace(go.Scatter(
        x=prices.index, y=np.ones(len(prices)), name="cash (1.00×)",
        line=dict(color="#94a3b8", dash="dot"),
    ))
    for combo in top_combos[:5]:
        eq = _equity_curve_for(prices, combo, slippage_bp=slippage_bp, max_window=max_window)
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name=f"strat {combo}"))
    fig.update_yaxes(type="log", title="equity (log, base = 1.0)")
    fig.update_xaxes(title="date")
    return _style_figure(fig, height=460)


def _heatmap_full_grid(df: pd.DataFrame) -> go.Figure:
    """Full 2-D heatmap of (buy_S, sell_S) → final_value.

    Unlike the TQQQ sweep (which needed 2-D slices through a 4-D grid),
    the SQQQ-only sweep is intrinsically 2-D — the entire search surface
    fits in a single heatmap.
    """
    pivot = df.pivot_table(index="buy_S", columns="sell_S", values="final_value")
    # Cap z at a reasonable percentile so the colorscale isn't dominated by
    # one or two extreme outliers (which would flatten the rest of the grid
    # to a single color).
    z = pivot.values
    z_cap = float(np.nanpercentile(z[~np.isnan(z)], 99)) if np.any(~np.isnan(z)) else 1.0
    fig = go.Figure(go.Heatmap(
        z=z,
        x=pivot.columns, y=pivot.index,
        zmin=0.0, zmax=max(z_cap, 1.0),
        colorscale=[[0, "#fff5e8"], [0.1, "#fde6c5"], [0.5, _TEAL_LIGHT], [1, _TEAL]],
        colorbar=dict(title="final_value (clipped at p99)"),
        hovertemplate="buy_S=%{y}<br>sell_S=%{x}<br>final=%{z:.3f}×<extra></extra>",
    ))
    fig.update_xaxes(title="sell_S (exit SMA)")
    fig.update_yaxes(title="buy_S (entry SMA)")
    return _style_figure(fig, height=520)


def _distribution_hist(df: pd.DataFrame, baselines: dict[str, float]) -> go.Figure:
    vals = df["final_value"].to_numpy()
    log_vals = np.log10(np.clip(vals, 1e-3, None))
    counts, edges = np.histogram(log_vals, bins=80)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0])
    n_combos = len(df)
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
                      annotation_text=f"{name} = {v:.3f}×",
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
        text=[f"({r.buy_S},{r.sell_S})" for r in wf.itertuples()],
        hovertemplate="combo=%{text}<br>train=%{x:.3f}×<br>test=%{y:.3f}×<extra></extra>",
        name="top-N",
    ))
    if len(wf):
        finite = wf[["final_value_train", "final_value_test"]].dropna()
        if len(finite):
            lo = float(min(finite["final_value_train"].min(), finite["final_value_test"].min()))
            hi = float(max(finite["final_value_train"].max(), finite["final_value_test"].max()))
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
                line=dict(color=_MUTED, dash="dot"),
            ))
    fig.update_xaxes(title="in-sample final_value (train: 2010–2018)")
    fig.update_yaxes(title="out-of-sample final_value (test: 2019–today)")
    return _style_figure(fig, height=460)


# ---------------------------------------------------------------------------
# HTML head + scaffolding (shares look-and-feel with tqqq_sma_report.py)
# ---------------------------------------------------------------------------


_HTML_HEAD = """<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>SQQQ-only SMA-Grid Backtest</title>
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


# ---------------------------------------------------------------------------
# Body builders
# ---------------------------------------------------------------------------


def _degenerate_banner(top: pd.DataFrame) -> str:
    """Return a DEGENERATE qbox if the leaderboard's #1 row is all-cash or
    one-shot, otherwise empty string.

    On a SQQQ-only sweep this is the load-bearing honesty mechanism — many
    realistic runs (especially on data dominated by QQQ uptrends) produce
    leaderboards led by ALL-CASH (n_trades=0) winners. The reader must know
    that's not a "discovered strategy" but the absence of one.
    """
    if not len(top):
        return ""
    row = top.iloc[0]
    n_trades = int(row.n_trades)
    if n_trades >= 2:
        return ""
    bS = int(row.buy_S)
    sS = int(row.sell_S)
    final = float(row.final_value)
    if n_trades == 0:
        kind = "ALL-CASH"
        explanation = (
            "the strategy never entered SQQQ. This happens when "
            "<code>buy_S=1</code> makes the entry signal "
            "<code>QQQ &lt; SMA(1) = QQQ &lt; QQQ</code> structurally False "
            "(the SMA(1) validity mask in <code>precompute_sma_signals</code> "
            "forces it off), or when no day in the sample crossed below the "
            "longer-window SMA. The headline final value of "
            f"<strong>{final:.3f}×</strong> is just cash."
        )
    else:  # n_trades == 1
        kind = "ONE-SHOT"
        explanation = (
            "the strategy entered SQQQ exactly once and never exited. This "
            "happens when <code>sell_S=1</code> makes the exit signal "
            "<code>QQQ &gt; SMA(1) = QQQ &gt; QQQ</code> structurally False, "
            "so any combo with <code>sell_S=1</code> that fires an entry is "
            "stuck. The equity curve is SQQQ buy-and-hold from the entry "
            f"date — final <strong>{final:.3f}×</strong> reflects whatever "
            "SQQQ did from then on, not a real timing strategy."
        )
    return (
        f'<div class="qbox" style="border-left-color: var(--warn);">'
        f'<strong>DEGENERATE — headline winner is {kind}.</strong> Best combo '
        f'is <code>(buy_S={bS}, sell_S={sS})</code> with <code>n_trades='
        f'{n_trades}</code> — {explanation} '
        'Look further down the leaderboard for the best <em>actually-trading</em> '
        'combo (n_trades &ge; 2) before drawing any conclusion.</div>'
    )


def _top50_table(top: pd.DataFrame, wf: pd.DataFrame) -> str:
    """Top-50 table with dormant-cell rendering and walk-forward delta.

    Dormancy logic:
        - ``ALL-CASH`` (n_trades=0): both buy_S and sell_S render as ``-``
          (the entry leg never fired regardless of which SMA was chosen).
        - ``ONE-SHOT`` (n_trades=1): sell_S renders as ``-`` (the exit leg
          never fired); buy_S keeps its real value because it controlled
          the single entry.
        - ``ACTIVE`` (n_trades>=2): both cells render their real values.
    """
    required = {"buy_S", "sell_S", "final_value_train", "final_value_test"}
    if not wf.empty and required.issubset(wf.columns):
        wf_indexed = wf.set_index(["buy_S", "sell_S"])[
            ["final_value_train", "final_value_test"]
        ].to_dict("index")
    else:
        wf_indexed = {}

    has_neq = "n_equivalent" in top.columns

    rows = ["<table class=\"data\"><thead><tr>"
            "<th>#</th><th>type</th><th>buy_S</th><th>sell_S</th>"
            "<th>n_eq</th>"
            "<th>final×</th><th>Sharpe</th><th>max_dd</th><th>Calmar</th>"
            "<th>trades</th><th>%short</th>"
            "<th>train×</th><th>test×</th><th>delta</th>"
            "</tr></thead><tbody>"]
    for i, row in enumerate(top.itertuples(index=False), start=1):
        key = (int(row.buy_S), int(row.sell_S))
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

        n_tr = int(row.n_trades)
        type_str = _strategy_type(n_tr, float(row.time_in_short))
        # ALL-CASH → both cells dormant; ONE-SHOT → sell dormant; ACTIVE → neither
        buy_dormant = (type_str == "ALL-CASH")
        sell_dormant = (type_str in ("ALL-CASH", "ONE-SHOT"))
        bS_s = _dormant_cell(row.buy_S, buy_dormant)
        sS_s = _dormant_cell(row.sell_S, sell_dormant)
        n_eq = int(getattr(row, "n_equivalent")) if has_neq else 1

        rows.append(
            f"<tr>"
            f"<td class=\"num\">{i}</td>"
            f"<td>{type_str}</td>"
            f"<td class=\"num\">{bS_s}</td><td class=\"num\">{sS_s}</td>"
            f"<td class=\"num\">{n_eq:,}</td>"
            f"<td class=\"num\">{row.final_value:.3f}×</td>"
            f"<td class=\"num\">{row.sharpe:.2f}</td>"
            f"<td class=\"num\">{row.max_dd*100:.1f}%</td>"
            f"<td class=\"num\">{row.calmar if not np.isnan(row.calmar) else 0:.2f}</td>"
            f"<td class=\"num\">{n_tr}</td>"
            f"<td class=\"num\">{row.time_in_short*100:.1f}%</td>"
            f"<td class=\"num\">{train:.3f}×</td>"
            f"<td class=\"num\">{test:.3f}×</td>"
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
    for name in ["QQQ B&H", "SQQQ B&H", "cash"]:
        v = baselines[name]
        cagr = v ** (1.0 / years) - 1.0 if v > 0 and years > 0 else float("nan")
        eq = baselines[f"_curve::{name}"]
        peak = eq.cummax()
        # Guard div-by-zero on SQQQ curves that touch zero (the realistic case
        # over 2010-2026): mask peak<=0 days before computing drawdown.
        peak_safe = peak.where(peak > 0, np.nan)
        dd = (peak_safe - eq) / peak_safe
        max_dd = float(dd.max()) if dd.notna().any() else float("nan")
        rows.append(
            f"<tr><td>{name}</td>"
            f"<td class=\"num\">{v:.3f}×</td>"
            f"<td class=\"num\">{cagr*100:.1f}%</td>"
            f"<td class=\"num\">{max_dd*100:.1f}%</td></tr>"
        )
    # Optional: Phase-1 TQQQ/SQQQ rotation winner for context
    if "tqqq_sma_phase1_winner_final" in baselines:
        v = float(baselines["tqqq_sma_phase1_winner_final"])
        rows.append(
            f"<tr><td>TQQQ/SQQQ Phase-1 SMA winner (context)</td>"
            f"<td class=\"num\">{v:.2f}×</td>"
            f"<td class=\"num\">—</td>"
            f"<td class=\"num\">—</td></tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _honest_discussion(prices: pd.DataFrame, df: pd.DataFrame,
                       top: pd.DataFrame) -> str:
    """Build the honest-discussion section — primary answer to the headline
    question: is there a profitable SQQQ-timing strategy over 2010-2026?

    The function makes the actual call based on the deduped top rows. If the
    best ACTIVE strategy (n_trades >= 2) beats cash, we say YES with the
    caveats; otherwise we say NO and explain why timing SQQQ is structurally
    hard in a market that spends most of its time going up.
    """
    sqqq_bh = float(prices["sqqq"].iloc[-1] / prices["sqqq"].iloc[0])
    qqq_bh = float(prices["qqq"].iloc[-1] / prices["qqq"].iloc[0])

    active = top[top["n_trades"] >= 2].copy()
    if len(active):
        best_active = active.nlargest(1, "final_value").iloc[0]
        best_combo = (int(best_active.buy_S), int(best_active.sell_S))
        best_final = float(best_active.final_value)
        best_trades = int(best_active.n_trades)
        active_beats_cash = best_final > 1.0
        if active_beats_cash:
            headline = (
                f'<div class="qbox" style="border-left-color: var(--accent);">'
                f'<strong>Headline answer: YES, just barely.</strong> Best '
                f'<em>actually-trading</em> combo is <code>{best_combo}</code> '
                f'with <code>n_trades={best_trades}</code> and final '
                f'<strong>{best_final:.3f}×</strong> — beating cash, but the '
                f'walk-forward in §7 is the honest test of whether that holds '
                f'out-of-sample. SQQQ B&amp;H over the same window went to '
                f'<code>{sqqq_bh:.4f}×</code> (~zero); QQQ B&amp;H went to '
                f'<code>{qqq_bh:.2f}×</code>. The win is &quot;avoided '
                f'catastrophic SQQQ decay while catching the 2022 drawdown&quot;.</div>'
            )
        else:
            headline = (
                f'<div class="qbox" style="border-left-color: var(--warn);">'
                f'<strong>Headline answer: NO.</strong> No actively-trading '
                f'combo (<code>n_trades &ge; 2</code>) beats cash. Best is '
                f'<code>{best_combo}</code> at <code>{best_final:.3f}×</code> '
                f'with <code>n_trades={best_trades}</code> — worse than just '
                'holding cash. SQQQ-timing over 2010–2026 fails because the '
                'index spent &gt;75% of trading days above its 200-day SMA: '
                "you'd be short during long uptrends and the slippage drag "
                'plus SQQQ decay overwhelms the 2022 win.</div>'
            )
    else:
        headline = (
            '<div class="qbox" style="border-left-color: var(--warn);">'
            '<strong>Headline answer: NO actively-trading combo found.</strong> '
            f'Every top-50 row is either ALL-CASH (n_trades=0) or ONE-SHOT '
            f'(n_trades=1). SQQQ B&amp;H went to <code>{sqqq_bh:.4f}×</code> '
            f'over the sample — sitting in cash beat every "real" SQQQ-timing '
            "strategy the grid surfaced.</div>"
        )

    return (
        headline +
        '<div class="qbox"><strong>SQQQ buy-and-hold goes to ~zero.</strong> '
        f'Over 2010-2026 SQQQ B&amp;H returned <code>{sqqq_bh:.4f}×</code> — '
        'effectively a complete loss. The 3x daily-rebalance leveraged-ETF '
        'decay structure means that even if QQQ ends roughly flat over a '
        'long window, SQQQ bleeds to zero through volatility drag. Any '
        'strategy that holds SQQQ for stretches longer than weeks pays this '
        'cost.</div>'
        '<div class="qbox"><strong>2022 is the only realistic profit '
        'window.</strong> QQQ fell ~35% peak-to-trough in 2022; SQQQ '
        'compounded positively for ~9 months. Any strategy that catches '
        '2022 and stays out the rest of the time will look great — but the '
        'leaderboard is full of combos that catch 2022 by happening to be '
        'long at the right moment, not because they "knew" 2022 was coming. '
        'The walk-forward §7 is the only filter that distinguishes these.</div>'
        '<div class="qbox"><strong>SMA(1) degeneracy.</strong> Combos with '
        '<code>buy_S=1</code> never enter (signal is structurally False); '
        'combos with <code>sell_S=1</code> never exit once entered. The '
        'leaderboard collapses these into a single equity-curve equivalence '
        'class under dedup (see <code>n_eq</code> in §2), but the &quot;winner&quot; '
        'in the all-cash class is just <code>1.00×</code> — not a discovery. '
        'The DEGENERATE banner in §2 flags this honestly when applicable.</div>'
        '<div class="qbox"><strong>slippage assumption.</strong> 5 bp per '
        'transition is generous for retail liquidity in SQQQ, but optimistic '
        'in stress periods (which is exactly when this strategy needs to '
        'trade). Combos with thousands of trades feel the slippage more than '
        'the headline number suggests.</div>'
        '<div class="qbox"><strong>signal source ≠ traded instrument.</strong> '
        'We trigger on QQQ but trade SQQQ — the leveraged-fund tracking '
        'assumption is the same as the TQQQ sweep, and same caveats apply.</div>'
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def render_report(
    prices: pd.DataFrame,
    results_path: Path,
    walkforward_path: Path,
    output_path: Path,
    sweep_wall_seconds: float,
    slippage_bp: float = 5.0,
    max_window: int = 60,
    tqqq_phase1_winner_final: float | None = None,
) -> Path:
    """Write the design-doc style HTML report to ``output_path``.

    Parameters
    ----------
    prices:
        Frame with columns ``qqq``, ``sqqq`` used for the sweep.
    results_path:
        Parquet of all combo results.
    walkforward_path:
        Parquet with the top-N walk-forward results.
    output_path:
        Where to write the HTML.
    sweep_wall_seconds:
        Wall-clock of the producing sweep, embedded in framing.
    slippage_bp, max_window:
        Reproducibility metadata, also reused for the top-5 equity-curve
        replay.
    tqqq_phase1_winner_final:
        Optional context — the final equity multiple of the Phase-1
        TQQQ/SQQQ SMA rotation winner. Surfaced in §3 baselines so the
        reader sees what a more complex strategy on the same signals
        achieved, for comparison.
    """
    results_path = Path(results_path)
    walkforward_path = Path(walkforward_path)
    output_path = Path(output_path)

    df_raw = pd.read_parquet(results_path)
    wf = pd.read_parquet(walkforward_path) if walkforward_path.exists() else pd.DataFrame()

    n_combos = len(df_raw)
    df = dedup_by_strategy_id(df_raw)
    n_distinct = len(df)
    assert int(df["n_equivalent"].sum()) == n_combos, (
        f"dedup lost rows: n_equivalent sum = {int(df['n_equivalent'].sum())}, "
        f"expected {n_combos}"
    )
    top50 = df.nlargest(50, "final_value").reset_index(drop=True)
    top_combos = [(int(r.buy_S), int(r.sell_S)) for r in top50.itertuples()]

    # Baselines: QQQ, SQQQ, cash (constant). Phase-1 TQQQ winner is optional context.
    qqq_curve = prices["qqq"] / prices["qqq"].iloc[0]
    sqqq_curve = prices["sqqq"] / prices["sqqq"].iloc[0]
    cash_curve = pd.Series(1.0, index=prices.index, name="cash")
    baselines: dict[str, float | pd.Series] = {
        "QQQ B&H": float(qqq_curve.iloc[-1]),
        "SQQQ B&H": float(sqqq_curve.iloc[-1]),
        "cash": 1.0,
        "_curve::QQQ B&H": qqq_curve,
        "_curve::SQQQ B&H": sqqq_curve,
        "_curve::cash": cash_curve,
    }
    if tqqq_phase1_winner_final is not None:
        baselines["tqqq_sma_phase1_winner_final"] = float(tqqq_phase1_winner_final)

    # Figures: heatmap + distribution on the FULL grid; leaderboard + equity on dedup.
    fig_equity = _equity_overlay(prices, top_combos, slippage_bp=slippage_bp, max_window=max_window)
    fig_heat = _heatmap_full_grid(df_raw)
    # Filter out NaNs and non-positives for the histogram baselines
    fig_dist = _distribution_hist(
        df_raw,
        baselines={k: v for k, v in baselines.items()
                   if not str(k).startswith("_curve::")
                   and isinstance(v, (int, float))
                   and v > 0},
    )
    fig_wf = _walkforward_scatter(wf) if len(wf) else None

    parts = [_HTML_HEAD]

    parts.append(
        '<nav class="toc"><strong>contents</strong><ol>'
        '<li><a href="#framing">1. framing</a></li>'
        '<li><a href="#top50">2. top-50 winners</a></li>'
        '<li><a href="#baselines">3. baselines</a></li>'
        '<li><a href="#heatmap">4. heatmap</a></li>'
        '<li><a href="#distribution">5. distribution</a></li>'
        '<li><a href="#equity">6. equity curves</a></li>'
        '<li><a href="#walkforward">7. walk-forward</a></li>'
        '<li><a href="#discussion">8. honest discussion</a></li>'
        '<li><a href="#repro">9. reproducibility</a></li>'
        '</ol></nav>'
    )

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    subtitle = (
        f'SQQQ-only SMA sweep · {n_combos:,} combos '
        f'({n_distinct:,} distinct equity curves) · '
        f'data {prices.index[0].date()} → {prices.index[-1].date()} · {today}'
    )
    parts.append(
        f'<header><h1>SQQQ-only SMA-grid backtest</h1>'
        f'<p class="subtitle">{subtitle}</p></header>'
    )

    # 1. Framing
    _grid_size_analytic = max_window ** 2
    framing_body = (
        '<p>The strategy is a 2-state machine driven by QQQ closes against '
        'two simple-moving-average windows. There is no TQQQ involvement: '
        'the strategy is either in cash or long SQQQ.</p>'
        '<pre style="background:var(--code-bg); padding:14px; border-radius:6px; '
        'font-size:0.86rem; overflow-x:auto;">CASH       → LONG_SQQQ    when QQQ &lt; SMA(buy_S)   [bet QQQ goes down]\n'
        'LONG_SQQQ  → CASH         when QQQ &gt; SMA(sell_S)  [QQQ recovering, exit]</pre>'
        f'<p>The parameter grid is unconstrained — every '
        f'<code>(buy_S, sell_S) in {{1..{max_window}}}^2</code> is swept, '
        f'totalling <code>{_grid_size_analytic:,}</code> combos. At the default '
        '<code>max_window=60</code> that is 3,600 — single-process compute, '
        'no parallelism needed.</p>'
        f'<p class="meta">Sweep wall-clock: {sweep_wall_seconds:.2f} sec · '
        f'~{n_combos / max(sweep_wall_seconds, 1e-3):,.0f} backtests/sec. '
        f'Slippage: {slippage_bp:.1f} bp per state transition; $0 commission; '
        f'adjusted-close prices.</p>'
        '<blockquote>SQQQ B&amp;H over 2010-2026 lost essentially all its '
        'value through leveraged-ETF decay. The question this report '
        'answers: can timing SQQQ entry/exit recover any of that — or '
        'better, beat cash? The DEGENERATE banner in §2 fires when the '
        'leaderboard winner is an artifact of the SMA(1) signal '
        'degeneracy, not a real strategy.</blockquote>'
    )
    parts.append(_section(1, "framing", "framing", framing_body))

    # 2. Top-50 winners (with DEGENERATE banner if applicable)
    degenerate = _degenerate_banner(top50)
    top50_body = (
        '<p>Sorted by final equity multiple. Each row is one DISTINCT equity '
        'curve — combos sharing a curve (e.g. all 60 combos with '
        '<code>buy_S=1</code> produce the identical all-cash trajectory) '
        'collapse by a deterministic equity-curve fingerprint. <code>n_eq</code> '
        'is the equivalence-class size; <code>type</code> classifies the row '
        '(<code>ALL-CASH</code> / <code>ONE-SHOT</code> / <code>ACTIVE</code>); '
        'dormant parameters render as <code>-</code>.</p>'
        f'<p>Of the {n_combos:,} grid combos, {n_distinct:,} produce distinct '
        'equity curves.</p>'
        + degenerate +
        '<p><code>delta</code> is <code>(test − train) / train</code> from '
        'the walk-forward — green = generalizes up, red = generalizes down.</p>'
        + _top50_table(top50, wf)
    )
    parts.append(_section(2, "top-50 winners", "top50", top50_body))

    # 3. Baselines
    baseline_body = (
        '<p>What a strategy must beat to be interesting. Cash (constant '
        '1.00×) is the load-bearing one here — SQQQ B&amp;H goes to '
        'effectively zero over the sample, so beating SQQQ B&amp;H is '
        'trivial; beating cash is the real bar.</p>'
        + _baseline_kv_table(prices, baselines)
    )
    parts.append(_section(3, "buy-and-hold baselines", "baselines", baseline_body))

    # 4. Heatmap (full grid — 2D fits in one figure)
    heatmap_body = (
        '<p>Full <code>(buy_S, sell_S)</code> grid. Color = final equity '
        'multiple, clipped at the 99th percentile so a single extreme '
        'outlier doesn&apos;t flatten the rest of the surface.</p>'
        '<div class="figure">' + _fig_to_div(fig_heat, include_js=True) + '</div>'
        '<p class="meta">The bottom row (<code>buy_S=1</code>) is uniformly '
        '1.00× — those combos never enter (SMA(1) degeneracy). The first '
        'column (<code>sell_S=1</code>) is the ONE-SHOT band — strategies '
        'that enter once and ride SQQQ buy-and-hold from there.</p>'
    )
    parts.append(_section(4, "heatmap", "heatmap", heatmap_body))

    # 5. Distribution
    dist_body = (
        f'<p>Where do the buy-and-hold baselines sit in the {n_combos:,}-combo '
        'final-value distribution? Most of the mass is at-or-below cash; the '
        'right tail is the hopeful region.</p>'
        '<div class="figure">' + _fig_to_div(fig_dist, include_js=False) + '</div>'
    )
    parts.append(_section(5, "distribution of outcomes", "distribution", dist_body))

    # 6. Equity curves
    equity_body = (
        '<p>Top-5 strategies vs QQQ B&amp;H, SQQQ B&amp;H, and cash. Log '
        'scale on the y-axis so the SQQQ collapse doesn&apos;t flatten the '
        'rest.</p>'
        '<div class="figure">' + _fig_to_div(fig_equity, include_js=False) + '</div>'
    )
    parts.append(_section(6, "equity curves", "equity", equity_body))

    # 7. Walk-forward
    if fig_wf is not None:
        wf_body = (
            f'<p>For each of the top-{len(wf)} DEDUPED combos by full-sample '
            'final value, we re-ran the backtest on <em>train</em> '
            '(2010-02 → 2018-12) and <em>test</em> (2019-01 → today). Points '
            'hugging the diagonal generalize; points dropping below it '
            'overfit. For SQQQ timing strategies the train period is mostly '
            'QQQ uptrend (favorable for sitting in cash); the test period '
            'contains the 2022 bear and the 2020 COVID drawdown (where '
            'actually-trading combos can profit).</p>'
            '<div class="figure">' + _fig_to_div(fig_wf, include_js=False) + '</div>'
        )
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
        wf_body = ('<p class="meta">Walk-forward parquet not found — re-run '
                   'with <code>--top-n-walkforward &gt; 0</code>.</p>')
    parts.append(_section(7, "walk-forward delta", "walkforward", wf_body))

    # 8. Honest discussion
    discussion_body = _honest_discussion(prices, df, top50)
    parts.append(_section(8, "honest discussion", "discussion", discussion_body))

    # 9. Reproducibility footer
    git_sha = _git_sha(Path(__file__).resolve().parent)
    parquet_hash = _file_sha256(results_path)
    wf_hash = _file_sha256(walkforward_path) if walkforward_path.exists() else "missing"
    ds_hash = hashlib.sha256(prices.to_csv().encode()).hexdigest()
    repro_cmd = "uv run rainier sqqq-sma-sweep"
    repro_body = (
        '<table class="kv"><tbody>'
        f'<tr><th>git SHA</th><td><code>{git_sha}</code></td></tr>'
        f'<tr><th>results.parquet SHA-256</th><td><code>{parquet_hash}</code></td></tr>'
        f'<tr><th>walkforward.parquet SHA-256</th><td><code>{wf_hash}</code></td></tr>'
        f'<tr><th>dataset SHA-256</th><td><code>{ds_hash}</code></td></tr>'
        f'<tr><th>run date (UTC)</th><td>{today}</td></tr>'
        f'<tr><th>sweep wall-clock</th><td>{sweep_wall_seconds:.2f} sec</td></tr>'
        f'<tr><th>max_window</th><td>{max_window}</td></tr>'
        f'<tr><th>slippage</th><td>{slippage_bp:.2f} bp/transition</td></tr>'
        f'<tr><th>combos</th><td>{n_combos:,}</td></tr>'
        f'<tr><th>distinct equity curves</th><td>{n_distinct:,}</td></tr>'
        '</tbody></table>'
        f'<p class="meta">To reproduce: <code>{repro_cmd}</code></p>'
    )
    parts.append(_section(9, "reproducibility", "repro", repro_body))

    parts.append(
        f'<footer>Generated {today} UTC · git <code>{git_sha[:12]}</code> · '
        f'{n_combos:,} backtests in {sweep_wall_seconds:.2f} sec.</footer>'
    )
    parts.append("</body></html>\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(parts), encoding="utf-8")
    return output_path
