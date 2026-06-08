"""Render the resonance-gate §6 study to a self-contained HTML report.

Aesthetic mirrors the repo's other exploratory reports (dark, sticky header,
decision boxes). The verdict is rendered honestly — if the gate loses, the
banner says "ship the SMA gate".
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rainier.backtest.resonance_study import (
    BUY_GRID,
    SELL_GRID,
    TRAIN_END,
    WEIGHT_MODES,
    ABRow,
    StudyResult,
    run_study,
)

_CSS = """
:root{--bg:#0e1117;--panel:#161b22;--ink:#e6edf3;--mut:#8b93a7;--line:#222a35;
--pos:#3ddc97;--neg:#ff6b6b;--key:#5aa9ff;--warn:#e3b341}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 22px 80px}
h1{font-size:26px;margin:0 0 4px}h2.sec{font-size:19px;margin:34px 0 12px;
border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--mut);margin:0 0 18px}
table{width:100%;border-collapse:collapse;margin:10px 0;font-size:13.5px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:600}
td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:var(--pos)}.neg{color:var(--neg)}
code{background:#0b0f14;padding:1px 5px;border-radius:4px;font-size:12.5px}
.box{border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:14px 0;background:var(--panel)}
.box.key{border-color:var(--key)}.box.warn{border-color:var(--warn)}
.box.ship{border-color:var(--pos);border-width:2px}
.box.reject{border-color:var(--neg);border-width:2px}
.box h4{margin:0 0 6px}.muted{color:var(--mut);font-size:13px}
.foot{color:var(--mut);font-size:12px;margin-top:40px;border-top:1px solid var(--line);padding-top:12px}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11.5px;font-weight:700}
.badge.win{background:rgba(61,220,151,.15);color:var(--pos)}
.badge.lose{background:rgba(255,107,107,.15);color:var(--neg)}
"""


def _pct(x: float) -> str:
    if x == float("inf"):
        return "∞"
    return f"{x * 100:+.1f}%"


def _num(x: float) -> str:
    if x == float("inf"):
        return "∞"
    return f"{x:.2f}"


def _ab_rows(rows: list[ABRow]) -> str:
    out = []
    for r in rows:
        badge = ("<span class='badge win'>beats SMA+B&amp;H</span>" if r.beats_sma_and_bh
                 else "<span class='badge lose'>—</span>")
        out.append(
            f"<tr><td>{r.name}</td>"
            f"<td class='r {'pos' if r.ret > 0 else 'neg'}'>{_pct(r.ret)}</td>"
            f"<td class='r neg'>{_pct(-r.dd)}</td>"
            f"<td class='r'><b>{_num(r.calmar)}</b></td>"
            f"<td class='r'>{r.switches}</td><td>{badge}</td></tr>")
    return "\n".join(out)


def _ab_table(rows: list[ABRow], caption: str) -> str:
    return f"""<p class="muted">{caption}</p>
<table>
<tr><th>Strategy</th><th class="r">Return</th><th class="r">MaxDD</th>
<th class="r">Calmar</th><th class="r">Switches</th><th>Anti-gaming</th></tr>
{_ab_rows(rows)}
</table>"""


def render(study: StudyResult) -> str:
    t = study.thesis
    bucket_rows = "\n".join(
        f"<tr><td>{lbl}</td><td class='r'>{wr*100:.0f}%</td><td class='r'>{eff}</td></tr>"
        for lbl, wr, eff in t.buckets)
    ship = study.verdict.startswith("SHIP THE RESONANCE")
    verdict_cls = "ship" if ship else "reject"

    oos_section = (
        _ab_table(study.oos_ab, "Frozen config run on synthetic 3×QQQ pre-2010 "
                  "(dot-com + GFC); synthetic financing/decay netted into the "
                  "return series; historical T-bill cash. Breadth excluded "
                  "(frozen ex-ante universe, §6.2).")
        if study.oos_ab else
        "<p class='muted'>Pre-2010 OOS slice unavailable in this run.</p>")

    oos_anti = study.oos_anti or {}
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Resonance Gate v1 — §6 A/B Evaluation</title><style>{_CSS}</style></head>
<body><main>
<h1>Multi-Signal Resonance Gate v1 — §6 falsifiable evaluation</h1>
<p class="sub">Real adjusted TQQQ · {study.window} · daily MTM, no lookahead,
turnover + T-bill cost · generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC ·
source <code>rainier.backtest.resonance_study</code></p>

<div class="box {verdict_cls}"><h4>Verdict</h4>
<p style="margin:0;font-size:16px"><b>{study.verdict}</b></p></div>

<div class="box warn"><h4>How to read this (the test is built to reject)</h4>
<p style="margin:0">The window holds one bear (2022). The plan tries to kill the
idea, not flatter it. The resonance gate earns its place ONLY if it beats the
<code>SMA22/44</code> gate <b>and</b> buy-hold on the re-derived 2023→now split
<b>and</b> survives the pre-2010 synthetic OOS — with the anti-gaming rule that a
combo winner must also beat its own SMA component and resonance-only must beat
buy-hold. If §6.1's win-rate slope CI includes zero, the premise fails outright.</p></div>

<h2 class="sec">§6.1 · Thesis CI — does more agreement raise the win-rate?</h2>
<p>Forward-20-day TQQQ win-rate by resonance-score bucket. <b>Effective N</b> ≈
bucket-days ÷ 20 (the forward windows overlap, so raw counts overstate
independence). The slope of bucket → win-rate is block-bootstrapped (20-day
blocks); its 95% CI must exclude 0.</p>
<table><tr><th>Score bucket</th><th class="r">Win-rate (fwd 20d)</th>
<th class="r">Effective N</th></tr>{bucket_rows}</table>
<p class="muted">Slope point estimate <b>{t.slope_point:+.4f}</b> · 95% CI
[{t.slope_ci[0]:+.4f}, {t.slope_ci[1]:+.4f}] ·
<b>{'excludes 0 — significant' if t.excludes_null else 'includes 0 — NOT significant'}</b>.</p>

<h2 class="sec">§6.2(a) · Re-derived split — select on ≤2022, report once on 2023→now</h2>
<p>The entire resonance config (weights mode, BUY, SELL) was re-selected on the
<code>≤{TRAIN_END}</code> slice ONLY (best train Calmar),
discarding any prior picks, then reported a single time on the held-out
2023→now slice. <b>{study.n_configs}</b> configs were tried on train.</p>
<p class="muted">Selected config <code>{study.train_cfg.label()}</code> ·
train Calmar {_num(study.train_calmar)} ·
<b>deflated</b> Calmar (÷(1+ln·#configs)) {_num(study.deflated_train_calmar)}
— the multiple-testing haircut over {study.n_configs} configs.</p>
{_ab_table(study.test_ab, "A/B on the held-out 2023→now slice. Real TQQQ; "
           "turnover + T-bill cost only (no synthetic 3× financing — the "
           "adjusted price already embeds it, §5.5).")}
<p class="muted">Anti-gaming on this slice: resonance-only beats buy-hold =
<b>{study.test_anti.get('res_beats_bh')}</b>; resonance-only beats SMA+B&amp;H =
<b>{study.test_anti.get('resonance_beats_sma_and_bh')}</b>.</p>

<h2 class="sec">§6.2(b) · True-OOS — frozen config on synthetic 3×QQQ pre-2010</h2>
{oos_section}
<p class="muted">Anti-gaming on OOS: resonance-only beats buy-hold =
<b>{oos_anti.get('res_beats_bh')}</b>.</p>

<h2 class="sec">§6.4 · Overfit guard</h2>
<ul>
<li>Free parameters are capped + pre-registered: BUY∈{list(BUY_GRID)},
SELL∈{list(SELL_GRID)}, weight-mode∈{list(WEIGHT_MODES)} →
{study.n_configs} configs.</li>
<li>Reported metric is <b>deflated</b> over #configs (above), not the raw best cell.</li>
<li>Net edge is after realistic cost; switch counts are shown per strategy (the
BUY−SELL gap is the turnover lever).</li>
<li>In-window sub-20% drawdown is optimistic — the bare gate's full-cycle
(1999–2026 synthetic) drawdown was −76%; in-window numbers are in-window.</li>
</ul>

<div class="foot">Multi-Signal Resonance Gate v1 · design
docs/DESIGN-multi-signal-resonance.md · exploratory evaluation, falsifiable by
construction. A losing verdict ("ship the SMA gate") is a valid, expected
outcome — simpler wins.</div>
</main></body></html>"""


def _default_report_path() -> Path:
    """Default report path resolved from the INVOCATION context (not the package
    install tree). Prefer ``<repo-root>/docs`` discovered from cwd (so a
    wheel-installed console script writes into the user's workspace as the CLI
    help promises); fall back to ``<cwd>/docs`` and create it if needed."""
    cwd = Path.cwd()
    for base in (cwd, *cwd.parents):
        if (base / "docs").is_dir():
            return base / "docs" / "REPORT-resonance-gate-v1.html"
    target = cwd / "docs"
    target.mkdir(parents=True, exist_ok=True)
    return target / "REPORT-resonance-gate-v1.html"


def write_report(out_path: Path | None = None, n_boot: int = 2000,
                 csv_dir: Path | None = None) -> Path:
    out_path = out_path or _default_report_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    study = run_study(csv_dir=csv_dir, n_boot=n_boot)
    out_path.write_text(render(study))
    return out_path
