"""Self-contained HTML report for the F&G + QQQ-MA backtest.

One file, embedded CSS, no JS, no external asset fetches (viewable offline).
Renders: framing + research-grade banner, the auto-verdict, the per-logic
train-selected combo scored against its paired MA-only arm on the OOS holdout,
the full combo ranking (train vs OOS Calmar/Sharpe/MaxDD), and the benchmarks.

The report is a regenerable render (``docs/*-backtest-*.html`` is gitignored);
the committed source-of-truth is this code + ``rainier fng-backtest``.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

from rainier.backtest.fear_greed_signal import (
    ArmMetrics,
    BacktestResult,
    Logic,
)

_LOGIC_RULE = {
    Logic.CONTRARIAN: "QQQ &gt; MA and F&amp;G &le; low",
    Logic.MOMENTUM: "QQQ &gt; MA and F&amp;G &ge; high",
    Logic.GREED_AVOID: "QQQ &gt; MA and F&amp;G &lt; extreme",
}


def _fmt(x: float, pct: bool = False, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and x != x):  # NaN
        return "&mdash;"
    if pct:
        return f"{x * 100:.{nd}f}%"
    return f"{x:.{nd}f}"


def _metric_cells(m: ArmMetrics) -> str:
    return (
        f"<td>{_fmt(m.cagr, pct=True)}</td>"
        f"<td>{_fmt(m.max_dd, pct=True)}</td>"
        f"<td>{_fmt(m.calmar)}</td>"
        f"<td>{_fmt(m.sharpe)}</td>"
        f"<td>{_fmt(m.exposure, pct=True, nd=1)}</td>"
        f"<td>{m.switches}</td>"
        f"<td>{_fmt(m.hit_rate, pct=True, nd=1)}</td>"
    )


def _verdict_block(result: BacktestResult) -> str:
    eligible = result.eligible_logics
    if eligible:
        names = ", ".join(l.value for l in eligible)
        cls, headline = "verdict-pass", f"ELIGIBLE for a Phase-3 gate: {names}"
        sub = (
            "At least one logic cleared all three OOS inequalities vs its paired "
            "MA-only arm. Phase 3 remains a separate gated decision (shadow first)."
        )
    else:
        cls, headline = "verdict-fail", "F&amp;G stays context-only (dashboard 4c)"
        sub = (
            "No logic cleared the pre-registered bar out-of-sample. Do NOT gate "
            "entries on F&amp;G; keep it as a read-only dashboard/context signal."
        )
    rows = []
    for logic in Logic:
        sel = result.selections.get(logic)
        if sel is None:
            continue
        paired = result.ma_only.get(sel.combo.ma)
        ok = result.verdicts.get(logic, False)
        c = sel.oos
        b = paired.oos if paired else None
        calmar_ratio = (
            _fmt(c.calmar / b.calmar) if (b and b.calmar and b.calmar == b.calmar and b.calmar != 0) else "&mdash;"
        )
        rows.append(
            f"<tr><td>{logic.value}</td>"
            f"<td>MA={sel.combo.ma}, thr={sel.combo.threshold:g}</td>"
            f"<td>{_fmt(c.calmar)}</td><td>{_fmt(b.calmar) if b else '&mdash;'}</td>"
            f"<td>{calmar_ratio}&times;</td>"
            f"<td>{_fmt(c.sharpe)} / {_fmt(b.sharpe) if b else '&mdash;'}</td>"
            f"<td>{_fmt(c.max_dd, pct=True)} / {_fmt(b.max_dd, pct=True) if b else '&mdash;'}</td>"
            f"<td>{_fmt(c.exposure, pct=True, nd=1)}</td>"
            f"<td class='{'ok' if ok else 'no'}'>{'PASS' if ok else 'fail'}</td></tr>"
        )
    table = (
        "<table><thead><tr><th>Logic</th><th>Train-selected combo</th>"
        "<th>OOS Calmar</th><th>MA-only Calmar</th><th>ratio (bar &ge;1.15&times;)</th>"
        "<th>Sharpe (logic/MA)</th><th>MaxDD (logic/MA)</th>"
        "<th>OOS Expo</th><th>Verdict</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return (
        f"<div class='verdict {cls}'><div class='v-head'>{headline}</div>"
        f"<div class='v-sub'>{sub}</div></div>"
        "<p>Bar (all three, OOS, vs the MA-only arm sharing the selected MA): "
        "Calmar &ge; 1.15&times;, Sharpe &ge;, MaxDD &le; (annualized 252, rf=0). "
        "<b>Read Calmar with the exposure column:</b> a very-low-exposure combo "
        "sits in cash most of the OOS window, so its tiny drawdown can inflate "
        "Calmar off a near-zero denominator &mdash; a PASS at &lt;~15% exposure is "
        "fragile, not a robust edge (the design's Calmar-instability caveat).</p>"
        f"{table}"
    )


def _combo_rows(result: BacktestResult) -> str:
    # Rank by OOS Calmar desc (NaN last) for the headline ranking.
    def key(cr):
        c = cr.oos.calmar
        return c if c == c else float("-inf")

    rows = []
    for cr in sorted(result.combo_results, key=key, reverse=True):
        selected = result.selections.get(cr.combo.logic)
        star = " &#9733;" if selected is not None and selected.combo == cr.combo else ""
        rows.append(
            f"<tr><td>{cr.combo.logic.value}{star}</td>"
            f"<td>{cr.combo.ma}</td><td>{cr.combo.threshold:g}</td>"
            f"<td>{_fmt(cr.train.calmar)}</td><td>{_fmt(cr.oos.calmar)}</td>"
            f"<td>{_fmt(cr.train.sharpe)}</td><td>{_fmt(cr.oos.sharpe)}</td>"
            f"<td>{_fmt(cr.oos.max_dd, pct=True)}</td>"
            f"<td>{_fmt(cr.oos.cagr, pct=True)}</td>"
            f"<td>{_fmt(cr.oos.exposure, pct=True, nd=1)}</td>"
            f"<td>{cr.oos.switches}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Logic</th><th>MA</th><th>Thr</th>"
        "<th>Train Calmar</th><th>OOS Calmar</th><th>Train Sharpe</th>"
        "<th>OOS Sharpe</th><th>OOS MaxDD</th><th>OOS CAGR</th>"
        "<th>OOS Expo</th><th>OOS Sw</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _bench_rows(result: BacktestResult) -> str:
    arms = [result.buy_and_hold] + [result.ma_only[w] for w in sorted(result.ma_only)]
    body = []
    for arm in arms:
        body.append(
            f"<tr><td>{arm.label}</td>{_metric_cells(arm.full)}{_metric_cells(arm.oos)}</tr>"
        )
    head = (
        "<tr><th rowspan=2>Arm</th>"
        "<th colspan=7>Full span</th><th colspan=7>OOS holdout</th></tr>"
        "<tr>"
        + "".join(f"<th>{h}</th>" for h in ("CAGR", "MaxDD", "Calmar", "Sharpe", "Expo", "Sw", "Hit")) * 2
        + "</tr>"
    )
    return f"<table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


_CSS = """
:root{color-scheme:light dark}
body{font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
 margin:0;padding:2rem;max-width:1100px;margin-inline:auto;color:#1a1a1a;background:#fafafa}
h1{font-size:1.6rem;margin:0 0 .2rem}h2{font-size:1.15rem;margin:2rem 0 .6rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}
.meta{color:#666;font-size:.85rem}
.banner{background:#fff8e1;border:1px solid #f0d98a;border-radius:6px;padding:.7rem 1rem;margin:1rem 0;font-size:.9rem}
.verdict{border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}
.verdict-pass{background:#e6f6e9;border:1px solid #8ccf9a}
.verdict-fail{background:#fdecec;border:1px solid #e0a3a3}
.v-head{font-size:1.15rem;font-weight:700}.v-sub{color:#444;margin-top:.3rem;font-size:.9rem}
table{border-collapse:collapse;width:100%;font-size:.82rem;margin:.6rem 0;overflow-x:auto;display:block}
th,td{border:1px solid #ddd;padding:.3rem .5rem;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{background:#f0f0f0}
td.ok{color:#1b7a2e;font-weight:700}td.no{color:#a12}
tbody tr:nth-child(even){background:#f6f6f6}
.foot{color:#888;font-size:.78rem;margin-top:2rem}
@media(prefers-color-scheme:dark){body{background:#16181c;color:#e6e6e6}
 h2{border-color:#333}.meta,.foot{color:#999}thead th{background:#22252b}
 th,td{border-color:#333}tbody tr:nth-child(even){background:#1c1f24}
 .banner{background:#2b2617;border-color:#5a4d1f}
 .verdict-pass{background:#16301c;border-color:#2e5d38}.verdict-fail{background:#341a1a;border-color:#5d2e2e}
 .v-sub{color:#bbb}}
"""


def render_report(result: BacktestResult, output_path: Path) -> Path:
    """Write the self-contained HTML report; return the path."""
    output_path = Path(output_path)
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if result.research_grade:
        banner = (
            "<div class='banner'><b>Research-grade data.</b> Every F&amp;G row is "
            "<code>source_version=backfill</code> (CNN's revised series) &mdash; no live "
            "point-in-time capture exists yet (derived boundary "
            "<code>MIN(observed_at) WHERE source_version='daily'</code> is empty). "
            "These are the endpoint's restated values, not what was knowable in real time. "
            "Treat OOS numbers as indicative, not a live track record.</div>"
        )
    else:
        banner = (
            f"<div class='banner'>Live-capture boundary: "
            f"<code>{html.escape(str(result.daily_boundary))}</code>. Rows before it are "
            "research-grade backfill (revised); rows on/after are true point-in-time.</div>"
        )

    rules = "".join(
        f"<li><b>{l.value}</b>: {_LOGIC_RULE[l]}</li>" for l in Logic
    )

    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>F&amp;G + QQQ-MA backtest &mdash; {html.escape(result.symbol)}</title>
<style>{_CSS}</style></head><body>
<h1>Fear &amp; Greed + QQQ-MA market-entry backtest</h1>
<div class=meta>Traded symbol <b>{html.escape(result.symbol)}</b> &middot;
span {result.start} &rarr; {result.end} &middot;
train/OOS split {result.split_date} &middot;
slippage {result.slippage_bp:g} bp/transition &middot; rendered {now}</div>
{banner}
<p>Trend + sentiment are market-level (always QQQ vs its own SMA + the composite
F&amp;G). The traded instrument is <b>{html.escape(result.symbol)}</b>. Execution is
one-day-lagged: the mask from <code>F&amp;G[D]</code>+<code>close[D]</code> earns
<code>return[D+1]</code>. Selection is train-only; the gate is the untouched OOS
holdout. Logics:</p>
<ul>{rules}</ul>

<h2>1. Verdict (auto-computed, OOS holdout)</h2>
{_verdict_block(result)}

<h2>2. All combos ranked (by OOS Calmar)</h2>
<p>&#9733; marks each logic's train-selected combo. Compare train vs OOS columns
to see how much of the edge survives out-of-sample.</p>
{_combo_rows(result)}

<h2>3. Benchmarks</h2>
<p>Buy-and-hold and the MA-only arms (QQQ&gt;MA, no F&amp;G), on the traded symbol.
The verdict pairs each logic against the MA-only arm sharing its MA window.</p>
{_bench_rows(result)}

<div class=foot>Regenerable render &mdash; reproduce with
<code>rainier fng-backtest --symbol {html.escape(result.symbol)}</code>.
Prices: yfinance adjusted-close parquet cache. F&amp;G: legacy
<code>fear_greed_index</code> table (earliest observation per date).</div>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc)
    return output_path
