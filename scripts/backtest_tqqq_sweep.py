"""
TQQQ EMA/SMA Strategy Parameter Sweep

Sweeps:
  - Entry EMA: 5, 8, 10, 13, 15, 20, 25, 30
  - Exit SMA: 10, 15, 20, 25, 30, 40, 50
  - Cash yield: 0%, 4.5% (T-bill rate while in cash)
  - Starting capital: $100,000, 10 years
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product


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
    ema_period: int,
    sma_period: int,
    cash_yield: float = 0.0,
    initial_capital: float = 100_000,
) -> dict:
    """Run a single backtest with given parameters."""
    data = df.copy()

    data["ema"] = data["qqq_close"].ewm(span=ema_period, adjust=False).mean()
    data["sma"] = data["qqq_close"].rolling(window=sma_period).mean()
    warmup = max(ema_period, sma_period) + 5
    data = data.iloc[warmup:].copy()

    # Generate positions
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

    # Daily cash yield
    daily_cash = (1 + cash_yield) ** (1 / 252) - 1

    # Strategy returns
    data["strat_return"] = np.where(
        data["pos_shifted"] == 1,
        data["tqqq_return"],
        daily_cash,
    )

    data["equity"] = initial_capital * (1 + data["strat_return"]).cumprod()

    # Metrics
    final = data["equity"].iloc[-1]
    years = len(data) / 252
    cagr = (final / initial_capital) ** (1 / years) - 1

    peak = data["equity"].cummax()
    dd = (data["equity"] - peak) / peak
    max_dd = dd.min()

    sharpe = data["strat_return"].mean() / data["strat_return"].std() * np.sqrt(252) if data["strat_return"].std() > 0 else 0

    downside = data["strat_return"][data["strat_return"] < 0].std()
    sortino = data["strat_return"].mean() / downside * np.sqrt(252) if downside > 0 else 0

    time_in_market = data["pos_shifted"].mean()

    # Trade count & win rate
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

    trade_returns = pd.Series(trade_returns) if trade_returns else pd.Series(dtype=float)
    trade_wr = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Yearly returns
    data["year"] = data.index.year
    yearly = data.groupby("year")["strat_return"].apply(lambda x: (1 + x).prod() - 1)
    worst_year = yearly.min()
    best_year = yearly.max()

    return {
        "ema": ema_period,
        "sma": sma_period,
        "cash_yield": cash_yield,
        "final_equity": final,
        "total_return": final / initial_capital - 1,
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "num_trades": num_trades,
        "trade_wr": trade_wr,
        "time_in_market": time_in_market,
        "best_year": best_year,
        "worst_year": worst_year,
        "yearly": yearly,
    }


def main():
    print("Fetching QQQ and TQQQ data (10 years)...")
    df = fetch_data(years=10)
    print(f"Got {len(df)} trading days\n")

    # Benchmark: TQQQ buy & hold
    tqqq_total = df["tqqq_close"].iloc[-1] / df["tqqq_close"].iloc[0] - 1
    tqqq_cagr = (1 + tqqq_total) ** (1 / (len(df) / 252)) - 1
    tqqq_peak = df["tqqq_close"].cummax()
    tqqq_dd = ((df["tqqq_close"] - tqqq_peak) / tqqq_peak).min()

    qqq_total = df["qqq_close"].iloc[-1] / df["qqq_close"].iloc[0] - 1
    qqq_cagr = (1 + qqq_total) ** (1 / (len(df) / 252)) - 1

    # Sweep parameters
    ema_periods = [5, 8, 10, 13, 15, 20, 25, 30]
    sma_periods = [10, 15, 20, 25, 30, 40, 50]
    cash_yields = [0.0, 0.045]

    combos = list(product(ema_periods, sma_periods, cash_yields))
    print(f"Running {len(combos)} parameter combinations...\n")

    results = []
    for ema, sma, cy in combos:
        if ema >= sma:  # Entry EMA should be faster than exit SMA
            continue
        r = run_backtest(df, ema, sma, cy)
        results.append(r)

    results_df = pd.DataFrame(results)

    # ========== TOP RESULTS BY DIFFERENT METRICS ==========

    print("=" * 100)
    print("  BENCHMARKS")
    print("=" * 100)
    print(f"  TQQQ Buy & Hold:  CAGR {tqqq_cagr:.1%}  |  Max DD {tqqq_dd:.1%}  |  Total {tqqq_total:.1%}")
    print(f"  QQQ Buy & Hold:   CAGR {qqq_cagr:.1%}  |  Total {qqq_total:.1%}")
    print()

    # --- Without cash yield ---
    no_cash = results_df[results_df["cash_yield"] == 0.0].copy()
    with_cash = results_df[results_df["cash_yield"] == 0.045].copy()

    def print_table(subset: pd.DataFrame, title: str, sort_by: str, ascending: bool = False, top_n: int = 15):
        print("=" * 100)
        print(f"  {title} (sorted by {sort_by})")
        print("=" * 100)
        sorted_df = subset.sort_values(sort_by, ascending=ascending).head(top_n)

        header = (
            f"  {'EMA':>4} {'SMA':>4} {'Cash%':>5} | "
            f"{'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} | "
            f"{'Trades':>6} {'WinR%':>6} {'InMkt%':>6} | "
            f"{'Final$':>12} {'BestYr':>7} {'WorstYr':>8}"
        )
        print(header)
        print("  " + "-" * 96)

        for _, row in sorted_df.iterrows():
            line = (
                f"  {row['ema']:>4} {row['sma']:>4} {row['cash_yield']:>5.1%} | "
                f"{row['cagr']:>7.1%} {row['max_dd']:>8.1%} {row['sharpe']:>7.2f} {row['sortino']:>8.2f} {row['calmar']:>7.2f} | "
                f"{row['num_trades']:>6} {row['trade_wr']:>5.1%} {row['time_in_market']:>5.1%} | "
                f"${row['final_equity']:>11,.0f} {row['best_year']:>7.1%} {row['worst_year']:>8.1%}"
            )
            print(line)
        print()

    # Best by risk-adjusted (Calmar = CAGR / MaxDD)
    print_table(no_cash, "TOP 15 BY CALMAR RATIO (no cash yield)", "calmar")
    print_table(no_cash, "TOP 15 BY CAGR (no cash yield)", "cagr")
    print_table(no_cash, "TOP 15 BY SHARPE (no cash yield)", "sharpe")
    print_table(no_cash, "TOP 15 BY LEAST MAX DRAWDOWN (no cash yield)", "max_dd", ascending=False)

    # With cash yield
    print_table(with_cash, "TOP 15 BY CALMAR RATIO (4.5% cash yield)", "calmar")
    print_table(with_cash, "TOP 15 BY CAGR (4.5% cash yield)", "cagr")

    # ========== DETAILED YEARLY for top 3 Calmar combos ==========
    print("=" * 100)
    print("  YEARLY BREAKDOWN — TOP 3 CALMAR (with 4.5% cash yield)")
    print("=" * 100)

    top3 = with_cash.sort_values("calmar", ascending=False).head(3)
    for _, row in top3.iterrows():
        print(f"\n  EMA{int(row['ema'])}/SMA{int(row['sma'])} | CAGR {row['cagr']:.1%} | MaxDD {row['max_dd']:.1%} | Calmar {row['calmar']:.2f}")
        print(f"  {'Year':<8} {'Return':>8}")
        print(f"  {'-'*20}")
        for year, ret in row["yearly"].items():
            bar_len = int(abs(ret) * 40)
            bar = ("+" * bar_len) if ret >= 0 else ("-" * bar_len)
            print(f"  {year:<8} {ret:>8.1%}  {bar}")

    # ========== Original EMA13/SMA20 comparison ==========
    print("\n")
    print("=" * 100)
    print("  ORIGINAL STRATEGY (EMA13/SMA20) vs BEST CALMAR")
    print("=" * 100)
    original = with_cash[(with_cash["ema"] == 13) & (with_cash["sma"] == 20)]
    best = with_cash.sort_values("calmar", ascending=False).iloc[0]

    if len(original) > 0:
        orig = original.iloc[0]
        print(f"  {'Metric':<25} {'EMA13/SMA20':>15} {'Best (EMA' + str(int(best['ema'])) + '/SMA' + str(int(best['sma'])) + ')':>15}")
        print(f"  {'-'*55}")
        print(f"  {'CAGR':<25} {orig['cagr']:>15.1%} {best['cagr']:>15.1%}")
        print(f"  {'Max Drawdown':<25} {orig['max_dd']:>15.1%} {best['max_dd']:>15.1%}")
        print(f"  {'Calmar':<25} {orig['calmar']:>15.2f} {best['calmar']:>15.2f}")
        print(f"  {'Sharpe':<25} {orig['sharpe']:>15.2f} {best['sharpe']:>15.2f}")
        print(f"  {'Sortino':<25} {orig['sortino']:>15.2f} {best['sortino']:>15.2f}")
        print(f"  {'Trades':<25} {orig['num_trades']:>15.0f} {best['num_trades']:>15.0f}")
        print(f"  {'Trade Win Rate':<25} {orig['trade_wr']:>15.1%} {best['trade_wr']:>15.1%}")
        print(f"  {'Time in Market':<25} {orig['time_in_market']:>15.1%} {best['time_in_market']:>15.1%}")
        print(f"  {'Final Equity':<25} ${orig['final_equity']:>14,.0f} ${best['final_equity']:>14,.0f}")
    print()


if __name__ == "__main__":
    main()
