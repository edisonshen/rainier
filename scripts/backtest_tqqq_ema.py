"""
TQQQ EMA13/SMA20 Crossover Strategy Backtest

Rules:
  - BUY 100% TQQQ when QQQ close >= EMA13
  - SELL all TQQQ when QQQ close < SMA20
  - Signal based on QQQ, execution on TQQQ
  - Starting capital: $100,000
  - Period: last 10 years
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def fetch_data(years: int = 10) -> pd.DataFrame:
    """Fetch QQQ and TQQQ daily data."""
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    qqq = yf.download("QQQ", start=start, end=end, auto_adjust=True)
    tqqq = yf.download("TQQQ", start=start, end=end, auto_adjust=True)

    # Flatten multi-level columns if present
    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = qqq.columns.get_level_values(0)
    if isinstance(tqqq.columns, pd.MultiIndex):
        tqqq.columns = tqqq.columns.get_level_values(0)

    df = pd.DataFrame(index=qqq.index)
    df["qqq_close"] = qqq["Close"]
    df["tqqq_close"] = tqqq["Close"]
    df = df.dropna()

    return df


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA13, SMA20 on QQQ and generate buy/sell signals."""
    df = df.copy()

    df["ema13"] = df["qqq_close"].ewm(span=13, adjust=False).mean()
    df["sma20"] = df["qqq_close"].rolling(window=20).mean()
    df = df.dropna()

    # Position: 1 = in TQQQ, 0 = cash
    position = 0
    positions = []

    for i in range(len(df)):
        close = df["qqq_close"].iloc[i]
        ema13 = df["ema13"].iloc[i]
        sma20 = df["sma20"].iloc[i]

        if position == 0 and close >= ema13:
            position = 1  # BUY
        elif position == 1 and close < sma20:
            position = 0  # SELL

        positions.append(position)

    df["position"] = positions
    # Shift position by 1 day (signal on close, execute next open approximated as next close)
    df["position_shifted"] = df["position"].shift(1).fillna(0).astype(int)

    return df


def backtest(df: pd.DataFrame, initial_capital: float = 100_000) -> dict:
    """Run backtest and compute metrics."""
    df = df.copy()

    # Daily returns
    df["tqqq_return"] = df["tqqq_close"].pct_change().fillna(0)
    df["qqq_return"] = df["qqq_close"].pct_change().fillna(0)

    # Strategy return: TQQQ return when in position, 0 when in cash
    df["strategy_return"] = df["position_shifted"] * df["tqqq_return"]

    # Equity curves
    df["strategy_equity"] = initial_capital * (1 + df["strategy_return"]).cumprod()
    df["buyhold_tqqq_equity"] = initial_capital * (1 + df["tqqq_return"]).cumprod()
    df["buyhold_qqq_equity"] = initial_capital * (1 + df["qqq_return"]).cumprod()

    # --- Metrics ---
    total_days = len(df)
    years = total_days / 252

    # Final values
    strat_final = df["strategy_equity"].iloc[-1]
    tqqq_final = df["buyhold_tqqq_equity"].iloc[-1]
    qqq_final = df["buyhold_qqq_equity"].iloc[-1]

    # CAGR
    strat_cagr = (strat_final / initial_capital) ** (1 / years) - 1
    tqqq_cagr = (tqqq_final / initial_capital) ** (1 / years) - 1
    qqq_cagr = (qqq_final / initial_capital) ** (1 / years) - 1

    # Max drawdown
    def max_drawdown(equity: pd.Series) -> tuple[float, str, str]:
        peak = equity.cummax()
        dd = (equity - peak) / peak
        max_dd = dd.min()
        max_dd_end = dd.idxmin()
        max_dd_start = equity[:max_dd_end].idxmax()
        return max_dd, str(max_dd_start.date()), str(max_dd_end.date())

    strat_dd, strat_dd_start, strat_dd_end = max_drawdown(df["strategy_equity"])
    tqqq_dd, _, _ = max_drawdown(df["buyhold_tqqq_equity"])
    qqq_dd, _, _ = max_drawdown(df["buyhold_qqq_equity"])

    # Sharpe ratio (annualized, assuming 0% risk-free)
    strat_sharpe = df["strategy_return"].mean() / df["strategy_return"].std() * np.sqrt(252)
    tqqq_sharpe = df["tqqq_return"].mean() / df["tqqq_return"].std() * np.sqrt(252)

    # Sortino ratio
    strat_downside = df["strategy_return"][df["strategy_return"] < 0].std()
    strat_sortino = df["strategy_return"].mean() / strat_downside * np.sqrt(252) if strat_downside > 0 else 0

    # Win rate (daily)
    trading_days = df[df["position_shifted"] == 1]
    win_days = (trading_days["strategy_return"] > 0).sum()
    total_trading_days = len(trading_days)
    daily_win_rate = win_days / total_trading_days if total_trading_days > 0 else 0

    # Trade analysis
    df["trade_change"] = df["position_shifted"].diff().fillna(0)
    entries = df[df["trade_change"] == 1]
    exits = df[df["trade_change"] == -1]
    num_trades = len(entries)

    # Time in market
    time_in_market = df["position_shifted"].mean()

    # Trade-level P&L
    trade_returns = []
    entry_dates = entries.index.tolist()
    exit_dates = exits.index.tolist()

    for i in range(min(len(entry_dates), len(exit_dates))):
        entry_idx = df.index.get_loc(entry_dates[i])
        exit_idx = df.index.get_loc(exit_dates[i])
        if exit_idx > entry_idx:
            trade_ret = (df["strategy_equity"].iloc[exit_idx] /
                        df["strategy_equity"].iloc[entry_idx]) - 1
            trade_returns.append(trade_ret)

    trade_returns = pd.Series(trade_returns) if trade_returns else pd.Series(dtype=float)
    winning_trades = (trade_returns > 0).sum() if len(trade_returns) > 0 else 0
    trade_win_rate = winning_trades / len(trade_returns) if len(trade_returns) > 0 else 0

    # Avg win / avg loss
    avg_win = trade_returns[trade_returns > 0].mean() if (trade_returns > 0).any() else 0
    avg_loss = trade_returns[trade_returns < 0].mean() if (trade_returns < 0).any() else 0

    # Best/worst year
    df["year"] = df.index.year
    yearly_returns = df.groupby("year")["strategy_return"].apply(
        lambda x: (1 + x).prod() - 1
    )

    return {
        "df": df,
        "initial_capital": initial_capital,
        "final_equity": strat_final,
        "total_return": strat_final / initial_capital - 1,
        "cagr": strat_cagr,
        "max_drawdown": strat_dd,
        "max_dd_period": f"{strat_dd_start} to {strat_dd_end}",
        "sharpe": strat_sharpe,
        "sortino": strat_sortino,
        "daily_win_rate": daily_win_rate,
        "num_trades": num_trades,
        "trade_win_rate": trade_win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "time_in_market": time_in_market,
        "total_trading_days": total_trading_days,
        "yearly_returns": yearly_returns,
        "tqqq_buyhold_final": tqqq_final,
        "tqqq_cagr": tqqq_cagr,
        "tqqq_max_dd": tqqq_dd,
        "tqqq_sharpe": tqqq_sharpe,
        "qqq_buyhold_final": qqq_final,
        "qqq_cagr": qqq_cagr,
        "qqq_max_dd": qqq_dd,
    }


def print_report(results: dict) -> None:
    """Print backtest results."""
    print("=" * 70)
    print("  TQQQ EMA13/SMA20 STRATEGY BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Period: {results['df'].index[0].date()} to {results['df'].index[-1].date()}")
    print(f"  Initial Capital: ${results['initial_capital']:,.0f}")
    print()

    print("-" * 70)
    print(f"  {'Metric':<30} {'Strategy':>12} {'TQQQ B&H':>12} {'QQQ B&H':>12}")
    print("-" * 70)
    print(f"  {'Final Equity':<30} ${results['final_equity']:>11,.0f} ${results['tqqq_buyhold_final']:>11,.0f} ${results['qqq_buyhold_final']:>11,.0f}")
    print(f"  {'Total Return':<30} {results['total_return']:>11.1%} {results['tqqq_buyhold_final']/results['initial_capital']-1:>11.1%} {results['qqq_buyhold_final']/results['initial_capital']-1:>11.1%}")
    print(f"  {'CAGR':<30} {results['cagr']:>11.1%} {results['tqqq_cagr']:>11.1%} {results['qqq_cagr']:>11.1%}")
    print(f"  {'Max Drawdown':<30} {results['max_drawdown']:>11.1%} {results['tqqq_max_dd']:>11.1%} {results['qqq_max_dd']:>11.1%}")
    print(f"  {'Sharpe Ratio':<30} {results['sharpe']:>11.2f} {results['tqqq_sharpe']:>11.2f} {'':>12}")
    print(f"  {'Sortino Ratio':<30} {results['sortino']:>11.2f} {'':>12} {'':>12}")
    print()

    print("-" * 70)
    print("  TRADE STATISTICS")
    print("-" * 70)
    print(f"  Total Trades:          {results['num_trades']}")
    print(f"  Trade Win Rate:        {results['trade_win_rate']:.1%}")
    print(f"  Avg Winning Trade:     {results['avg_win']:.1%}")
    print(f"  Avg Losing Trade:      {results['avg_loss']:.1%}")
    profit_factor = abs(results['avg_win'] / results['avg_loss']) if results['avg_loss'] != 0 else float('inf')
    print(f"  Profit Factor (avg):   {profit_factor:.2f}")
    print(f"  Time in Market:        {results['time_in_market']:.1%}")
    print(f"  Trading Days:          {results['total_trading_days']}")
    print(f"  Daily Win Rate:        {results['daily_win_rate']:.1%}")
    print(f"  Max DD Period:         {results['max_dd_period']}")
    print()

    print("-" * 70)
    print("  YEARLY RETURNS")
    print("-" * 70)
    for year, ret in results["yearly_returns"].items():
        bar = "+" * int(max(0, ret) * 50) or "-" * int(max(0, -ret) * 50)
        print(f"  {year}:  {ret:>8.1%}  {bar}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    print("Fetching QQQ and TQQQ data (10 years)...")
    df = fetch_data(years=10)
    print(f"Got {len(df)} trading days\n")

    print("Computing signals and running backtest...")
    df = compute_signals(df)
    results = backtest(df)

    print_report(results)
