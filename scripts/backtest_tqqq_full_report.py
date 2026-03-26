"""
TQQQ EMA/SMA Strategy — Full HTML Report with Interactive Plotly Charts

Generates a self-contained HTML report with:
  1. Summary table (all 64 combos + benchmarks)
  2. Yearly returns heatmap
  3. Equity curves (top 5 by Calmar + benchmarks)
  4. Drawdown chart
  5. Risk-return scatter
  6. Yearly bar chart (key combos)
  7. Consistency analysis
  8. Key combo comparison
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

INITIAL_CAPITAL = 100_000
CASH_YIELD = 0.045
ENTRY_EMAS = [5, 8, 10, 13, 15, 20, 25, 30]
EXIT_SMAS = [5, 10, 15, 20, 25, 30, 40, 50]
REPORT_PATH = Path(__file__).parent.parent / "reports" / "tqqq_strategy_report.html"

# Plotly CDN version must match installed version
PLOTLY_VERSION = plotly.__version__
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{PLOTLY_VERSION}.min.js"


def fetch_data(years: int = 10) -> pd.DataFrame:
    """Fetch QQQ and TQQQ data."""
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
    ema_period: int,
    sma_period: int,
    cash_yield: float = CASH_YIELD,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """Run a single backtest. Returns metrics + equity series + yearly returns."""
    data = df.copy()

    data["ema"] = data["qqq_close"].ewm(span=ema_period, adjust=False).mean()
    data["sma"] = data["qqq_close"].rolling(window=sma_period).mean()
    warmup = max(ema_period, sma_period) + 5
    data = data.iloc[warmup:].copy()

    # Generate positions: signal on close, execute next day
    position = 0
    positions = []
    for i in range(len(data)):
        close = data["qqq_close"].iloc[i]
        ema = data["ema"].iloc[i]
        sma = data["sma"].iloc[i]

        if position == 0 and close >= ema:
            position = 1
        elif position == 1 and close < sma:
            position = 0
        positions.append(position)

    data["position"] = positions
    data["pos_shifted"] = data["position"].shift(1).fillna(0).astype(int)

    daily_cash = (1 + cash_yield) ** (1 / 252) - 1

    data["strat_return"] = np.where(
        data["pos_shifted"] == 1,
        data["tqqq_return"],
        daily_cash,
    )

    data["equity"] = initial_capital * (1 + data["strat_return"]).cumprod()

    # Core metrics
    final = data["equity"].iloc[-1]
    years = len(data) / 252
    cagr = (final / initial_capital) ** (1 / years) - 1

    peak = data["equity"].cummax()
    dd = (data["equity"] - peak) / peak
    max_dd = dd.min()

    std = data["strat_return"].std()
    sharpe = data["strat_return"].mean() / std * np.sqrt(252) if std > 0 else 0

    downside = data["strat_return"][data["strat_return"] < 0].std()
    sortino = data["strat_return"].mean() / downside * np.sqrt(252) if downside > 0 else 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    time_in_market = data["pos_shifted"].mean()

    # Trade count and win rate
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

    # Yearly returns
    data["year"] = data.index.year
    yearly = data.groupby("year")["strat_return"].apply(lambda x: (1 + x).prod() - 1)

    return {
        "label": f"EMA{ema_period}/SMA{sma_period}",
        "ema": ema_period,
        "sma": sma_period,
        "cagr": cagr,
        "total_return": final / initial_capital - 1,
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
    """Compute buy-and-hold benchmark metrics."""
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
        "ema": "-",
        "sma": "-",
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


def build_summary_table_html(results: list, benchmarks: list) -> str:
    """Section 1: Sortable summary table."""
    def fmt_pct(v):
        color = "#4caf50" if v >= 0 else "#ef5350"
        return f'<span style="color:{color}">{v:.1%}</span>'

    def fmt_num(v, decimals=2):
        color = "#4caf50" if v >= 0 else "#ef5350"
        return f'<span style="color:{color}">{v:.{decimals}f}</span>'

    def fmt_dollar(v):
        color = "#4caf50" if v >= INITIAL_CAPITAL else "#ef5350"
        return f'<span style="color:{color}">${v:,.0f}</span>'

    rows_html = ""

    # Benchmark rows
    for b in benchmarks:
        rows_html += f"""<tr style="background:#2a2a3e; font-weight:bold;">
            <td>{b['label']}</td>
            <td>{fmt_pct(b['cagr'])}</td>
            <td>{fmt_pct(b['total_return'])}</td>
            <td>{fmt_pct(b['max_dd'])}</td>
            <td>{fmt_num(b['sharpe'])}</td>
            <td>{fmt_num(b['sortino'])}</td>
            <td>{fmt_num(b['calmar'])}</td>
            <td>{b['num_trades']}</td>
            <td>{fmt_pct(b['trade_wr'])}</td>
            <td>{fmt_pct(b['time_in_market'])}</td>
            <td>{fmt_dollar(b['final_equity'])}</td>
        </tr>"""

    # Separator
    rows_html += '<tr style="height:4px;background:#555;"><td colspan="11"></td></tr>'

    # Strategy rows sorted by Calmar
    sorted_results = sorted(results, key=lambda x: x["calmar"], reverse=True)
    for r in sorted_results:
        rows_html += f"""<tr>
            <td>{r['label']}</td>
            <td data-sort="{r['cagr']:.6f}">{fmt_pct(r['cagr'])}</td>
            <td data-sort="{r['total_return']:.6f}">{fmt_pct(r['total_return'])}</td>
            <td data-sort="{r['max_dd']:.6f}">{fmt_pct(r['max_dd'])}</td>
            <td data-sort="{r['sharpe']:.6f}">{fmt_num(r['sharpe'])}</td>
            <td data-sort="{r['sortino']:.6f}">{fmt_num(r['sortino'])}</td>
            <td data-sort="{r['calmar']:.6f}">{fmt_num(r['calmar'])}</td>
            <td data-sort="{r['num_trades']}">{r['num_trades']}</td>
            <td data-sort="{r['trade_wr']:.6f}">{fmt_pct(r['trade_wr'])}</td>
            <td data-sort="{r['time_in_market']:.6f}">{fmt_pct(r['time_in_market'])}</td>
            <td data-sort="{r['final_equity']:.2f}">{fmt_dollar(r['final_equity'])}</td>
        </tr>"""

    return f"""
    <h2 id="summary">Section 1: Summary Table (All Combos)</h2>
    <p>Sorted by Calmar ratio. Click column headers to re-sort. Benchmarks pinned at top.</p>
    <table id="summaryTable" class="sortable">
        <thead>
            <tr>
                <th>Combo</th><th>CAGR</th><th>Total Return</th><th>Max DD</th>
                <th>Sharpe</th><th>Sortino</th><th>Calmar</th>
                <th># Trades</th><th>Win Rate</th><th>In Market</th><th>Final Equity</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """


def build_yearly_heatmap(results: list, benchmarks: list) -> str:
    """Section 2: Yearly returns heatmap."""
    all_items = benchmarks + sorted(results, key=lambda x: x["calmar"], reverse=True)
    all_years = sorted(set(y for item in all_items for y in item["yearly"].keys()))

    labels = [item["label"] for item in all_items]
    z_data = []
    text_data = []
    for item in all_items:
        row = []
        text_row = []
        for year in all_years:
            val = item["yearly"].get(year, None)
            if val is not None:
                row.append(round(val * 100, 1))
                text_row.append(f"{val:.1%}")
            else:
                row.append(None)
                text_row.append("")
        z_data.append(row)
        text_data.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[str(y) for y in all_years],
        y=labels,
        text=text_data,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale=[
            [0, "#d32f2f"],
            [0.35, "#ef5350"],
            [0.5, "#ffffff"],
            [0.65, "#66bb6a"],
            [1.0, "#1b5e20"],
        ],
        zmid=0,
        colorbar=dict(title="Return %", ticksuffix="%"),
    ))

    fig.update_layout(
        title="Yearly Returns by Strategy Combo",
        height=max(600, len(labels) * 22 + 100),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=120),
    )

    return f"""
    <h2 id="heatmap">Section 2: Yearly Returns Heatmap</h2>
    <div>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>
    """


def build_equity_curves(results: list, benchmarks: list) -> str:
    """Section 3: Equity curves for top 5 by Calmar + benchmarks."""
    top5 = sorted(results, key=lambda x: x["calmar"], reverse=True)[:5]

    fig = go.Figure()

    colors = ["#ff9800", "#2196f3", "#4caf50", "#e91e63", "#9c27b0"]
    for i, r in enumerate(top5):
        eq = r["equity_series"]
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq["equity"],
            name=r["label"],
            line=dict(color=colors[i], width=2),
        ))

    # Benchmarks
    for b in benchmarks:
        eq = b["equity_series"]
        dash = "dash" if "TQQQ" in b["label"] else "dot"
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq["equity"],
            name=b["label"],
            line=dict(color="#888", width=1.5, dash=dash),
        ))

    fig.update_layout(
        title="Equity Curves: Top 5 by Calmar + Benchmarks",
        yaxis_title="Equity ($)",
        yaxis_type="log",
        height=600,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
    )

    return f"""
    <h2 id="equity">Section 3: Equity Curves (Top 5 by Calmar)</h2>
    <div>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>
    """


def build_drawdown_chart(results: list, benchmarks: list) -> str:
    """Section 4: Drawdown chart for top 5 + TQQQ B&H."""
    top5 = sorted(results, key=lambda x: x["calmar"], reverse=True)[:5]

    fig = go.Figure()

    colors = ["#ff9800", "#2196f3", "#4caf50", "#e91e63", "#9c27b0"]
    for i, r in enumerate(top5):
        dd = r["drawdown_series"]
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values * 100,
            name=r["label"],
            line=dict(color=colors[i], width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba({int(colors[i][1:3],16)},{int(colors[i][3:5],16)},{int(colors[i][5:7],16)},0.1)",
        ))

    # TQQQ B&H drawdown
    tqqq_bh = [b for b in benchmarks if "TQQQ" in b["label"]][0]
    dd = tqqq_bh["drawdown_series"]
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values * 100,
        name="TQQQ B&H",
        line=dict(color="#ef5350", width=1.5, dash="dash"),
        fill="tozeroy",
        fillcolor="rgba(239,83,80,0.05)",
    ))

    fig.update_layout(
        title="Drawdowns: Top 5 Combos vs TQQQ Buy & Hold",
        yaxis_title="Drawdown (%)",
        height=500,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        legend=dict(x=0.01, y=-0.15, orientation="h"),
        hovermode="x unified",
    )

    return f"""
    <h2 id="drawdown">Section 4: Drawdown Chart</h2>
    <div>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>
    """


def build_risk_return_scatter(results: list, benchmarks: list) -> str:
    """Section 5: Risk-Return scatter."""
    fig = go.Figure()

    # Strategy combos
    x = [abs(r["max_dd"]) * 100 for r in results]
    y = [r["cagr"] * 100 for r in results]
    sizes = [max(r["sharpe"] * 10, 5) for r in results]
    labels = [r["label"] for r in results]

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=8, color="#aaa"),
        marker=dict(
            size=sizes,
            color=[r["calmar"] for r in results],
            colorscale="Viridis",
            colorbar=dict(title="Calmar"),
            line=dict(width=1, color="#fff"),
        ),
        name="Strategies",
    ))

    # Benchmarks
    for b in benchmarks:
        fig.add_trace(go.Scatter(
            x=[abs(b["max_dd"]) * 100],
            y=[b["cagr"] * 100],
            mode="markers+text",
            text=[b["label"]],
            textposition="bottom center",
            textfont=dict(size=10, color="#ef5350"),
            marker=dict(size=16, color="#ef5350", symbol="diamond",
                        line=dict(width=2, color="#fff")),
            name=b["label"],
        ))

    fig.update_layout(
        title="Risk-Return: CAGR vs Max Drawdown",
        xaxis_title="Max Drawdown (%)",
        yaxis_title="CAGR (%)",
        height=600,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        showlegend=False,
    )

    return f"""
    <h2 id="scatter">Section 5: Risk-Return Scatter</h2>
    <p>Point size = Sharpe ratio. Color = Calmar ratio. Diamonds = benchmarks.</p>
    <div>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>
    """


def build_yearly_bar_chart(results: list, benchmarks: list) -> str:
    """Section 6: Yearly bar chart for key combos."""
    key_labels = ["EMA13/SMA20", "EMA30/SMA50", "EMA30/SMA5"]
    key_combos = []
    for label in key_labels:
        match = [r for r in results if r["label"] == label]
        if match:
            key_combos.append(match[0])

    # Add TQQQ B&H
    tqqq_bh = [b for b in benchmarks if "TQQQ" in b["label"]][0]
    key_combos.append(tqqq_bh)

    if not key_combos:
        return "<h2>Section 6: No key combos found</h2>"

    all_years = sorted(set(y for c in key_combos for y in c["yearly"].keys()))

    fig = go.Figure()
    colors = ["#ff9800", "#2196f3", "#4caf50", "#ef5350"]
    for i, combo in enumerate(key_combos):
        vals = [combo["yearly"].get(y, 0) * 100 for y in all_years]
        fig.add_trace(go.Bar(
            x=[str(y) for y in all_years],
            y=vals,
            name=combo["label"],
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        title="Year-by-Year Returns: Key Combos",
        yaxis_title="Return (%)",
        barmode="group",
        height=500,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        legend=dict(x=0.01, y=0.99),
    )

    return f"""
    <h2 id="yearly-bars">Section 6: Yearly Returns (Key Combos)</h2>
    <div>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>
    """


def build_consistency_analysis(results: list) -> str:
    """Section 7: Consistency analysis — how often each combo is top 10 / bottom 5."""
    all_years = sorted(set(y for r in results for y in r["yearly"].keys()))

    top10_count = {r["label"]: 0 for r in results}
    bottom5_count = {r["label"]: 0 for r in results}

    for year in all_years:
        year_returns = []
        for r in results:
            ret = r["yearly"].get(year, None)
            if ret is not None:
                year_returns.append((r["label"], ret))

        year_returns.sort(key=lambda x: x[1], reverse=True)

        for label, _ in year_returns[:10]:
            top10_count[label] += 1
        for label, _ in year_returns[-5:]:
            bottom5_count[label] += 1

    # Build table
    rows = []
    for r in results:
        rows.append({
            "label": r["label"],
            "top10": top10_count[r["label"]],
            "bottom5": bottom5_count[r["label"]],
            "calmar": r["calmar"],
        })
    rows.sort(key=lambda x: (-x["top10"], x["bottom5"]))

    rows_html = ""
    for row in rows:
        highlight = ""
        if row["top10"] >= len(all_years) * 0.6 and row["bottom5"] == 0:
            highlight = 'style="background:#1b5e20;"'
        elif row["bottom5"] >= 3:
            highlight = 'style="background:#4a1010;"'
        rows_html += f"""<tr {highlight}>
            <td>{row['label']}</td>
            <td>{row['top10']}</td>
            <td>{row['bottom5']}</td>
            <td>{row['calmar']:.2f}</td>
        </tr>"""

    return f"""
    <h2 id="consistency">Section 7: Consistency Analysis</h2>
    <p>How many years each combo ranked in the Top 10 or Bottom 5.
    <span style="color:#4caf50;">Green</span> = Top 10 in 60%+ years and never Bottom 5.
    <span style="color:#ef5350;">Red</span> = Bottom 5 in 3+ years.</p>
    <table>
        <thead>
            <tr><th>Combo</th><th>Years in Top 10</th><th>Years in Bottom 5</th><th>Calmar</th></tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """


def build_key_combo_comparison(results: list) -> str:
    """Section 8: Detailed comparison of key combos."""
    key_labels = [
        "EMA13/SMA20", "EMA30/SMA50", "EMA30/SMA5",
        "EMA15/SMA20", "EMA20/SMA40", "EMA20/SMA50",
    ]

    key_combos = []
    for label in key_labels:
        match = [r for r in results if r["label"] == label]
        if match:
            key_combos.append(match[0])

    if not key_combos:
        return "<h2>Section 8: No key combos found</h2>"

    # Metrics table
    metrics = [
        ("CAGR", "cagr", True),
        ("Total Return", "total_return", True),
        ("Max Drawdown", "max_dd", True),
        ("Sharpe", "sharpe", False),
        ("Sortino", "sortino", False),
        ("Calmar", "calmar", False),
        ("# Trades", "num_trades", False),
        ("Win Rate", "trade_wr", True),
        ("Time in Market", "time_in_market", True),
        ("Final Equity", "final_equity", False),
    ]

    header = "<tr><th>Metric</th>" + "".join(f"<th>{c['label']}</th>" for c in key_combos) + "</tr>"
    rows_html = ""
    for name, key, is_pct in metrics:
        rows_html += f"<tr><td><strong>{name}</strong></td>"
        for c in key_combos:
            val = c[key]
            if key == "final_equity":
                rows_html += f'<td>${val:,.0f}</td>'
            elif key == "num_trades":
                rows_html += f'<td>{val}</td>'
            elif is_pct:
                color = "#4caf50" if val >= 0 else "#ef5350"
                rows_html += f'<td style="color:{color}">{val:.1%}</td>'
            else:
                color = "#4caf50" if val >= 0 else "#ef5350"
                rows_html += f'<td style="color:{color}">{val:.2f}</td>'
        rows_html += "</tr>"

    # Yearly breakdown
    all_years = sorted(set(y for c in key_combos for y in c["yearly"].keys()))
    for year in all_years:
        rows_html += f"<tr><td><em>{year}</em></td>"
        for c in key_combos:
            val = c["yearly"].get(year, 0)
            color = "#4caf50" if val >= 0 else "#ef5350"
            rows_html += f'<td style="color:{color}">{val:.1%}</td>'
        rows_html += "</tr>"

    return f"""
    <h2 id="key-combos">Section 8: Key Combo Comparison</h2>
    <table>
        <thead>{header}</thead>
        <tbody>{rows_html}</tbody>
    </table>
    """


def build_html_report(sections: list[str], plotly_js: str) -> str:
    """Wrap sections into full HTML document."""
    nav_items = [
        ("summary", "Summary Table"),
        ("heatmap", "Yearly Heatmap"),
        ("equity", "Equity Curves"),
        ("drawdown", "Drawdowns"),
        ("scatter", "Risk-Return"),
        ("yearly-bars", "Yearly Bars"),
        ("consistency", "Consistency"),
        ("key-combos", "Key Combos"),
    ]
    nav_html = " | ".join(f'<a href="#{id}">{label}</a>' for id, label in nav_items)

    body = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TQQQ EMA/SMA Strategy Backtest Report</title>
    <script>{plotly_js}</script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
            background: #0d0d1a;
            color: #e0e0e0;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #ff9800;
            margin: 20px 0;
            font-size: 2em;
        }}
        h2 {{
            color: #64b5f6;
            border-bottom: 2px solid #333;
            padding-bottom: 8px;
            margin: 40px 0 16px 0;
        }}
        nav {{
            text-align: center;
            padding: 12px;
            background: #1a1a2e;
            border-radius: 8px;
            margin-bottom: 20px;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        nav a {{
            color: #64b5f6;
            text-decoration: none;
            padding: 4px 10px;
        }}
        nav a:hover {{ color: #ff9800; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            font-size: 13px;
        }}
        th, td {{
            padding: 6px 10px;
            text-align: right;
            border-bottom: 1px solid #333;
        }}
        th {{
            background: #1a1a2e;
            color: #ff9800;
            cursor: pointer;
            position: sticky;
            top: 52px;
        }}
        th:hover {{ color: #fff; }}
        td:first-child, th:first-child {{ text-align: left; }}
        tr:hover {{ background: #1a1a2e; }}
        p {{
            color: #999;
            margin: 8px 0;
            font-size: 14px;
        }}
        .meta {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <h1>TQQQ EMA/SMA Strategy Backtest Report</h1>
    <p style="text-align:center; color:#aaa;">
        Entry: Buy TQQQ when QQQ Close &ge; EMA(entry) | Exit: Sell when QQQ Close &lt; SMA(exit)<br>
        Cash yield: {CASH_YIELD:.1%} | Starting capital: ${INITIAL_CAPITAL:,} | 10-year backtest
    </p>
    <nav>{nav_html}</nav>

    {body}

    <p class="meta">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} PT |
    Plotly {PLOTLY_VERSION} | Data: yfinance</p>

    <script>
    // Simple table sorting
    document.querySelectorAll('th').forEach(th => {{
        th.addEventListener('click', () => {{
            const table = th.closest('table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const idx = Array.from(th.parentNode.children).indexOf(th);
            const asc = th.dataset.asc === 'true';

            rows.sort((a, b) => {{
                let aVal = a.children[idx]?.dataset?.sort || a.children[idx]?.textContent || '';
                let bVal = b.children[idx]?.dataset?.sort || b.children[idx]?.textContent || '';
                aVal = aVal.replace(/[$,%]/g, '');
                bVal = bVal.replace(/[$,%]/g, '');
                const aNum = parseFloat(aVal);
                const bNum = parseFloat(bVal);
                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return asc ? aNum - bNum : bNum - aNum;
                }}
                return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }});

            rows.forEach(row => tbody.appendChild(row));
            th.dataset.asc = !asc;
        }});
    }});
    </script>
</body>
</html>"""


def main():
    print("Fetching QQQ and TQQQ data (10 years)...")
    df = fetch_data(years=10)
    print(f"Got {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")

    # Benchmarks
    print("Computing benchmarks...")
    tqqq_bh = compute_benchmark(df, "tqqq_close", "TQQQ Buy & Hold")
    qqq_bh = compute_benchmark(df, "qqq_close", "QQQ Buy & Hold")
    benchmarks = [tqqq_bh, qqq_bh]

    # Run all 64 combos
    combos = list(product(ENTRY_EMAS, EXIT_SMAS))
    print(f"Running {len(combos)} parameter combinations...")
    results = []
    for ema, sma in combos:
        r = run_backtest(df, ema, sma)
        results.append(r)
        print(f"  {r['label']}: CAGR={r['cagr']:.1%} MaxDD={r['max_dd']:.1%} Calmar={r['calmar']:.2f}")

    print(f"\nCompleted {len(results)} backtests. Building report...")

    # Read bundled plotly.js for self-contained HTML
    import plotly as _plotly
    plotly_js_path = Path(_plotly.__file__).parent / "package_data" / "plotly.min.js"
    plotly_js = plotly_js_path.read_text()
    print(f"Embedded plotly.js ({len(plotly_js)//1024}KB)")

    # Build all sections
    sections = [
        build_summary_table_html(results, benchmarks),
        build_yearly_heatmap(results, benchmarks),
        build_equity_curves(results, benchmarks),
        build_drawdown_chart(results, benchmarks),
        build_risk_return_scatter(results, benchmarks),
        build_yearly_bar_chart(results, benchmarks),
        build_consistency_analysis(results),
        build_key_combo_comparison(results),
    ]

    html = build_html_report(sections, plotly_js)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html)
    print(f"\nReport saved to: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
