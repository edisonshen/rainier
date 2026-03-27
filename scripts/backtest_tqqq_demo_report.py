"""
TQQQ Strategy — Demo HTML Report (Top 5 combos only)

Lightweight version of the full report for demos.
Uses Plotly CDN instead of embedded JS (~4MB savings).
Shows only the top 5 combos by Calmar ratio.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

INITIAL_CAPITAL = 100_000
CASH_YIELD = 0.045
ENTRY_PERIODS = sorted(set([5, 8, 10, 13, 15, 20, 25, 30] + [3, 7, 11, 17, 19, 23, 29]))
EXIT_PERIODS = sorted(set([5, 10, 15, 20, 25, 30, 40, 50] + [3, 7, 11, 17, 19, 23, 29]))
REPORT_PATH = Path(__file__).parent.parent / "reports" / "tqqq_strategy_demo.html"
PAGES_PATH = Path(__file__).parent.parent / "docs" / "tqqq_strategy_demo.html"
TOP_N = 5

PLOTLY_VERSION = plotly.__version__
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{PLOTLY_VERSION}.min.js"


def fetch_data(years: int = 10) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    qqq = yf.download("QQQ", start=start, end=end, auto_adjust=True)
    tqqq = yf.download("TQQQ", start=start, end=end, auto_adjust=True)

    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = qqq.columns.get_level_values(0)
    if isinstance(tqqq.columns, pd.MultiIndex):
        tqqq.columns = tqqq.columns.get_level_values(0)

    df = pd.DataFrame(index=qqq.index)
    df["qqq_close"] = qqq["Close"]
    df["tqqq_close"] = tqqq["Close"]
    df = df.dropna()
    df["tqqq_return"] = df["tqqq_close"].pct_change().fillna(0)
    df["qqq_return"] = df["qqq_close"].pct_change().fillna(0)
    return df


def run_backtest(
    df: pd.DataFrame,
    entry_period: int,
    exit_period: int,
    entry_type: str = "ewm",
    exit_type: str = "rolling",
    strategy_name: str = "EMA/SMA",
) -> dict:
    data = df.copy()

    if entry_type == "ewm":
        data["entry_ma"] = data["qqq_close"].ewm(span=entry_period, adjust=False).mean()
        entry_prefix = "EMA"
    else:
        data["entry_ma"] = data["qqq_close"].rolling(window=entry_period).mean()
        entry_prefix = "SMA"

    if exit_type == "ewm":
        data["exit_ma"] = data["qqq_close"].ewm(span=exit_period, adjust=False).mean()
        exit_prefix = "EMA"
    else:
        data["exit_ma"] = data["qqq_close"].rolling(window=exit_period).mean()
        exit_prefix = "SMA"

    warmup = max(entry_period, exit_period) + 5
    data = data.dropna()
    data = data.iloc[warmup:].copy()

    position = 0
    positions = []
    for i in range(len(data)):
        close = data["qqq_close"].iloc[i]
        entry_ma = data["entry_ma"].iloc[i]
        exit_ma = data["exit_ma"].iloc[i]

        if position == 0 and close >= entry_ma:
            position = 1
        elif position == 1 and close < exit_ma:
            position = 0
        positions.append(position)

    data["position"] = positions
    data["pos_shifted"] = data["position"].shift(1).fillna(0).astype(int)

    daily_cash = (1 + CASH_YIELD) ** (1 / 252) - 1
    data["strat_return"] = np.where(data["pos_shifted"] == 1, data["tqqq_return"], daily_cash)
    data["equity"] = INITIAL_CAPITAL * (1 + data["strat_return"]).cumprod()

    final = data["equity"].iloc[-1]
    years = len(data) / 252
    cagr = (final / INITIAL_CAPITAL) ** (1 / years) - 1

    peak = data["equity"].cummax()
    dd = (data["equity"] - peak) / peak
    max_dd = dd.min()

    std = data["strat_return"].std()
    sharpe = data["strat_return"].mean() / std * np.sqrt(252) if std > 0 else 0

    downside = data["strat_return"][data["strat_return"] < 0].std()
    sortino = data["strat_return"].mean() / downside * np.sqrt(252) if downside > 0 else 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    time_in_market = data["pos_shifted"].mean()

    trade_changes = data["pos_shifted"].diff().fillna(0)
    entries = data[trade_changes == 1]
    exits = data[trade_changes == -1]
    num_trades = len(entries)

    trade_returns = []
    entry_dates = entries.index.tolist()
    exit_dates = exits.index.tolist()
    for i in range(min(len(entry_dates), len(exit_dates))):
        ei = data.index.get_loc(entry_dates[i])
        xi = data.index.get_loc(exit_dates[i])
        if xi > ei:
            trade_returns.append(data["equity"].iloc[xi] / data["equity"].iloc[ei] - 1)

    trade_wr = np.mean([r > 0 for r in trade_returns]) if trade_returns else 0

    data["year"] = data.index.year
    yearly = data.groupby("year")["strat_return"].apply(lambda x: (1 + x).prod() - 1)

    label = f"{entry_prefix}{entry_period}/{exit_prefix}{exit_period}"

    return {
        "label": label,
        "strategy": strategy_name,
        "cagr": cagr,
        "total_return": final / INITIAL_CAPITAL - 1,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "num_trades": num_trades,
        "trade_wr": trade_wr,
        "time_in_market": time_in_market,
        "final_equity": final,
        "equity_series": data[["equity"]].copy(),
        "drawdown_series": dd.copy(),
        "yearly": yearly.to_dict(),
    }


def compute_benchmark(df: pd.DataFrame, col: str, label: str) -> dict:
    data = df.copy()
    data["bh_return"] = data[col].pct_change().fillna(0)
    data["equity"] = INITIAL_CAPITAL * (1 + data["bh_return"]).cumprod()

    final = data["equity"].iloc[-1]
    years = len(data) / 252
    cagr = (final / INITIAL_CAPITAL) ** (1 / years) - 1

    peak = data["equity"].cummax()
    dd = (data["equity"] - peak) / peak
    max_dd = dd.min()

    std = data["bh_return"].std()
    sharpe = data["bh_return"].mean() / std * np.sqrt(252) if std > 0 else 0

    downside = data["bh_return"][data["bh_return"] < 0].std()
    sortino = data["bh_return"].mean() / downside * np.sqrt(252) if downside > 0 else 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    data["year"] = data.index.year
    yearly = data.groupby("year")["bh_return"].apply(lambda x: (1 + x).prod() - 1)

    return {
        "label": label,
        "cagr": cagr,
        "total_return": final / INITIAL_CAPITAL - 1,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "num_trades": 1,
        "trade_wr": 1.0 if final > INITIAL_CAPITAL else 0.0,
        "time_in_market": 1.0,
        "final_equity": final,
        "equity_series": data[["equity"]].copy(),
        "drawdown_series": dd.copy(),
        "yearly": yearly.to_dict(),
    }


def _weekly(series: pd.Series) -> pd.Series:
    """Downsample a daily series to weekly (last value per week) for smaller HTML."""
    return series.resample("W").last().dropna()


def build_html(top5: list, benchmarks: list) -> str:
    """Build a compact demo HTML with top 5 combos."""

    def fmt_pct(v):
        color = "#4caf50" if v >= 0 else "#ef5350"
        return f'<span style="color:{color}">{v:.1%}</span>'

    def fmt_num(v):
        color = "#4caf50" if v >= 0 else "#ef5350"
        return f'<span style="color:{color}">{v:.2f}</span>'

    def fmt_dollar(v):
        color = "#4caf50" if v >= INITIAL_CAPITAL else "#ef5350"
        return f'<span style="color:{color}">${v:,.0f}</span>'

    # --- Section 1: Summary table (benchmarks + top 5) ---
    rows = ""
    for b in benchmarks:
        rows += f"""<tr style="background:#2a2a3e; font-weight:bold;">
            <td>{b['label']}</td><td>{fmt_pct(b['cagr'])}</td>
            <td>{fmt_pct(b['total_return'])}</td><td>{fmt_pct(b['max_dd'])}</td>
            <td>{fmt_num(b['sharpe'])}</td><td>{fmt_num(b['sortino'])}</td>
            <td>{fmt_num(b['calmar'])}</td><td>{b['num_trades']}</td>
            <td>{fmt_pct(b['trade_wr'])}</td><td>{fmt_pct(b['time_in_market'])}</td>
            <td>{fmt_dollar(b['final_equity'])}</td></tr>"""

    rows += '<tr style="height:4px;background:#555;"><td colspan="11"></td></tr>'

    strat_colors = {"EMA/SMA": "#ff9800", "EMA/EMA": "#2196f3", "SMA/SMA": "#4caf50"}
    for r in top5:
        strat = r.get("strategy", "EMA/SMA")
        sc = strat_colors.get(strat, "#888")
        rows += f"""<tr>
            <td><span style="color:{sc}">{strat}</span> {r['label']}</td>
            <td>{fmt_pct(r['cagr'])}</td><td>{fmt_pct(r['total_return'])}</td>
            <td>{fmt_pct(r['max_dd'])}</td><td>{fmt_num(r['sharpe'])}</td>
            <td>{fmt_num(r['sortino'])}</td><td>{fmt_num(r['calmar'])}</td>
            <td>{r['num_trades']}</td><td>{fmt_pct(r['trade_wr'])}</td>
            <td>{fmt_pct(r['time_in_market'])}</td><td>{fmt_dollar(r['final_equity'])}</td></tr>"""

    summary_html = f"""
    <h2 id="summary">Top {TOP_N} Strategies by Calmar Ratio</h2>
    <p>Sorted by Calmar ratio. Benchmarks pinned at top.</p>
    <table>
        <thead><tr>
            <th>Combo</th><th>CAGR</th><th>Total Return</th><th>Max DD</th>
            <th>Sharpe</th><th>Sortino</th><th>Calmar</th>
            <th># Trades</th><th>Win Rate</th><th>In Market</th><th>Final Equity</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""

    # --- Section 2: Equity curves (top 5 + benchmarks) ---
    fig_eq = go.Figure()
    colors = ["#ff9800", "#2196f3", "#4caf50", "#e91e63", "#9c27b0"]
    for i, r in enumerate(top5):
        eq = _weekly(r["equity_series"]["equity"])
        fig_eq.add_trace(go.Scatter(
            x=eq.index, y=eq.values, name=r["label"],
            line=dict(color=colors[i], width=2),
        ))
    for b in benchmarks:
        eq = _weekly(b["equity_series"]["equity"])
        dash = "dash" if "TQQQ" in b["label"] else "dot"
        fig_eq.add_trace(go.Scatter(
            x=eq.index, y=eq.values, name=b["label"],
            line=dict(color="#888", width=1.5, dash=dash),
        ))
    fig_eq.update_layout(
        title="Equity Curves: Top 5 by Calmar + Benchmarks",
        yaxis_title="Equity ($)", yaxis_type="log", height=500,
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        legend=dict(x=0.01, y=0.99), hovermode="x unified",
    )
    equity_html = f"""
    <h2 id="equity">Equity Curves</h2>
    <div>{fig_eq.to_html(full_html=False, include_plotlyjs=False)}</div>"""

    # --- Section 3: Drawdown chart ---
    fig_dd = go.Figure()
    for i, r in enumerate(top5):
        dd = _weekly(r["drawdown_series"]) * 100
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=r["label"],
            line=dict(color=colors[i], width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba({int(colors[i][1:3],16)},{int(colors[i][3:5],16)},{int(colors[i][5:7],16)},0.1)",
        ))
    tqqq_bh = [b for b in benchmarks if "TQQQ" in b["label"]][0]
    dd = _weekly(tqqq_bh["drawdown_series"]) * 100
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd.values, name="TQQQ B&H",
        line=dict(color="#ef5350", width=1.5, dash="dash"),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.05)",
    ))
    fig_dd.update_layout(
        title="Drawdowns: Top 5 vs TQQQ Buy & Hold",
        yaxis_title="Drawdown (%)", height=400,
        template="plotly_dark", paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        legend=dict(x=0.01, y=-0.15, orientation="h"), hovermode="x unified",
    )
    drawdown_html = f"""
    <h2 id="drawdown">Drawdowns</h2>
    <div>{fig_dd.to_html(full_html=False, include_plotlyjs=False)}</div>"""

    # --- Section 4: Key combo comparison table ---
    all_years = sorted(set(y for c in top5 for y in c["yearly"].keys()))
    header = "<tr><th>Metric</th>" + "".join(f"<th>{c['label']}</th>" for c in top5) + "</tr>"

    metrics = [
        ("CAGR", "cagr", True), ("Total Return", "total_return", True),
        ("Max Drawdown", "max_dd", True), ("Sharpe", "sharpe", False),
        ("Sortino", "sortino", False), ("Calmar", "calmar", False),
        ("# Trades", "num_trades", False), ("Win Rate", "trade_wr", True),
        ("Time in Market", "time_in_market", True), ("Final Equity", "final_equity", False),
    ]
    comp_rows = ""
    for name, key, is_pct in metrics:
        comp_rows += f"<tr><td><strong>{name}</strong></td>"
        for c in top5:
            val = c[key]
            if key == "final_equity":
                comp_rows += f"<td>${val:,.0f}</td>"
            elif key == "num_trades":
                comp_rows += f"<td>{val}</td>"
            elif is_pct:
                color = "#4caf50" if val >= 0 else "#ef5350"
                comp_rows += f'<td style="color:{color}">{val:.1%}</td>'
            else:
                color = "#4caf50" if val >= 0 else "#ef5350"
                comp_rows += f'<td style="color:{color}">{val:.2f}</td>'
        comp_rows += "</tr>"

    for year in all_years:
        comp_rows += f"<tr><td><em>{year}</em></td>"
        for c in top5:
            val = c["yearly"].get(year, 0)
            color = "#4caf50" if val >= 0 else "#ef5350"
            comp_rows += f'<td style="color:{color}">{val:.1%}</td>'
        comp_rows += "</tr>"

    comparison_html = f"""
    <h2 id="comparison">Year-by-Year Comparison</h2>
    <table>
        <thead>{header}</thead>
        <tbody>{comp_rows}</tbody>
    </table>"""

    # --- Assemble full HTML ---
    nav = ' | '.join(
        f'<a href="#{id}">{label}</a>' for id, label in [
            ("summary", "Summary"), ("equity", "Equity Curves"),
            ("drawdown", "Drawdowns"), ("comparison", "Comparison"),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TQQQ Strategy — Top {TOP_N} Demo</title>
    <script src="{PLOTLY_CDN}"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            background: #0d0d1a; color: #e0e0e0;
            padding: 20px; max-width: 1400px; margin: 0 auto;
        }}
        h1 {{ text-align: center; color: #ff9800; margin: 20px 0; font-size: 2em; }}
        h2 {{ color: #64b5f6; border-bottom: 2px solid #333; padding-bottom: 8px; margin: 40px 0 16px 0; }}
        nav {{
            text-align: center; padding: 12px; background: #1a1a2e;
            border-radius: 8px; margin-bottom: 20px; position: sticky; top: 0; z-index: 100;
        }}
        nav a {{ color: #64b5f6; text-decoration: none; padding: 4px 10px; }}
        nav a:hover {{ color: #ff9800; }}
        table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }}
        th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #333; }}
        th {{ background: #1a1a2e; color: #ff9800; position: sticky; top: 52px; }}
        td:first-child, th:first-child {{ text-align: left; }}
        tr:hover {{ background: #1a1a2e; }}
        p {{ color: #999; margin: 8px 0; font-size: 14px; }}
        .meta {{ text-align: center; color: #666; font-size: 12px; margin-top: 40px; }}
    </style>
</head>
<body>
    <h1>TQQQ Strategy — Top {TOP_N} by Calmar</h1>
    <p style="text-align:center; color:#aaa;">
        Entry: Buy TQQQ when QQQ Close &ge; MA(entry) | Exit: Sell when QQQ Close &lt; MA(exit)<br>
        Cash yield: {CASH_YIELD:.1%} | Starting capital: ${INITIAL_CAPITAL:,} | 10-year backtest
    </p>
    <nav>{nav}</nav>

    {summary_html}
    {equity_html}
    {drawdown_html}
    {comparison_html}

    <p class="meta">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} PT |
    Plotly {PLOTLY_VERSION} (CDN) | Data: yfinance</p>
</body>
</html>"""


def main():
    print("Fetching QQQ and TQQQ data (10 years)...")
    df = fetch_data(years=10)
    print(f"Got {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")

    print("Computing benchmarks...")
    tqqq_bh = compute_benchmark(df, "tqqq_close", "TQQQ Buy & Hold")
    qqq_bh = compute_benchmark(df, "qqq_close", "QQQ Buy & Hold")
    benchmarks = [tqqq_bh, qqq_bh]

    PERIODS = sorted(set(ENTRY_PERIODS + EXIT_PERIODS))
    strategy_types = [
        ("EMA/SMA", "ewm", "rolling"),
        ("EMA/EMA", "ewm", "ewm"),
        ("SMA/SMA", "rolling", "rolling"),
    ]

    combos = list(product(PERIODS, PERIODS))
    total = len(combos) * len(strategy_types)
    print(f"Running {total} backtests (3 strategies × {len(combos)} combos)...")

    results = []
    for strat_name, entry_type, exit_type in strategy_types:
        for entry_p, exit_p in combos:
            r = run_backtest(df, entry_p, exit_p, entry_type, exit_type, strat_name)
            results.append(r)

    # Pick top 5 by Calmar
    top5 = sorted(results, key=lambda x: x["calmar"], reverse=True)[:TOP_N]
    print(f"\nTop {TOP_N} by Calmar:")
    for i, r in enumerate(top5, 1):
        print(f"  {i}. {r['strategy']} {r['label']}  Calmar={r['calmar']:.2f}  CAGR={r['cagr']:.1%}  MaxDD={r['max_dd']:.1%}")

    print("\nBuilding demo report (CDN Plotly, no embedded JS)...")
    html = build_html(top5, benchmarks)

    for path in [REPORT_PATH, PAGES_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
    size_kb = REPORT_PATH.stat().st_size / 1024
    print(f"Saved to: {REPORT_PATH.resolve()} ({size_kb:.0f} KB)")
    print(f"GitHub Pages copy: {PAGES_PATH.resolve()}")


if __name__ == "__main__":
    main()
