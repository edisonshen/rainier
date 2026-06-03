"""QU100 money flow ranking backtest — tests if top-ranked stocks outperform.

Strategy:
  Each trading day, select top N stocks from QU100 with 'Long in' money flow.
  Buy at next day's open, hold for `holding_days`.
  Track returns vs SPY benchmark.

Pattern-filtered variant:
  Combine money flow ranking with Caisen pattern detection.
  Only trade stocks matching specified patterns (e.g. false_breakdown_w_bottom,
  false_breakdown, bull_flag). Pick top N by pattern confidence.

Dependency rule: imports only from core/ (pattern detection passed in from CLI).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import structlog
import yfinance as yf
from sqlalchemy import select

from rainier.core.database import get_session
from rainier.core.models import MoneyFlowSnapshot

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QU100Trade:
    """Single trade from QU100 ranking strategy."""

    symbol: str
    rank: int
    long_short: str
    signal_date: date  # day we observed the ranking
    entry_date: date  # next trading day (buy at open)
    entry_price: float
    exit_date: date
    exit_price: float
    return_pct: float
    holding_days: int
    pattern_type: str | None = None  # pattern that triggered the trade


@dataclass
class QU100BacktestResult:
    """Aggregate results from QU100 ranking backtest."""

    trades: list[QU100Trade] = field(default_factory=list)
    total_trades: int = 0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    median_return_pct: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    alpha_pct: float = 0.0
    # Per holding period stats
    holding_days: int = 5
    top_n: int = 20
    # Period
    start_date: date | None = None
    end_date: date | None = None
    # Sector breakdown
    sector_returns: dict[str, float] = field(default_factory=dict)
    # Long/short breakdown
    long_in_avg_return: float = 0.0
    short_in_avg_return: float = 0.0
    # Equity curve (daily)
    equity_curve: list[float] = field(default_factory=list)
    dates: list[date] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rankings_from_db() -> pd.DataFrame:
    """Load all QU100 rankings from the database.

    Returns DataFrame with columns:
        data_date, symbol, rank, ranking_type, long_short, sector, industry
    """
    with get_session() as db:
        rows = db.execute(
            select(
                MoneyFlowSnapshot.data_date,
                MoneyFlowSnapshot.symbol,
                MoneyFlowSnapshot.rank,
                MoneyFlowSnapshot.ranking_type,
                MoneyFlowSnapshot.long_short,
                MoneyFlowSnapshot.sector,
                MoneyFlowSnapshot.industry,
            )
            .order_by(MoneyFlowSnapshot.data_date, MoneyFlowSnapshot.rank)
        ).all()

    if not rows:
        raise ValueError("No QU100 data found in database")

    df = pd.DataFrame(rows, columns=[
        "data_date", "symbol", "rank", "ranking_type",
        "long_short", "sector", "industry",
    ])
    df["data_date"] = pd.to_datetime(df["data_date"]).dt.date
    log.info(
        "rankings_loaded",
        rows=len(df),
        dates=df["data_date"].nunique(),
        symbols=df["symbol"].nunique(),
    )
    return df


def fetch_prices(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance for given symbols.

    Returns DataFrame with MultiIndex columns: (field, symbol).
    """
    # Add buffer for holding period exits
    end_buffered = end + timedelta(days=30)
    tickers = " ".join(symbols)
    log.info("fetching_prices", symbols=len(symbols), start=str(start), end=str(end_buffered))

    df = yf.download(
        tickers,
        start=start.isoformat(),
        end=end_buffered.isoformat(),
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError("No price data returned from yfinance")

    log.info("prices_fetched", rows=len(df), columns=list(df.columns.get_level_values(0).unique()))
    return df


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


def run_qu100_backtest(
    top_n: int = 20,
    holding_days: int = 5,
    long_only: bool = True,
    min_rank: int = 1,
    max_rank: int = 50,
    entry_delay: int = 0,
) -> QU100BacktestResult:
    """Run QU100 money flow ranking backtest.

    Args:
        top_n: Number of top-ranked stocks to buy each day
        holding_days: Days to hold each position
        long_only: Only trade 'Long in' stocks (vs also shorting 'Short in')
        min_rank: Minimum rank to consider (1 = best)
        max_rank: Maximum rank to include

    Returns:
        QU100BacktestResult with all trades and aggregate metrics
    """
    # Load rankings
    rankings = load_rankings_from_db()
    top100 = rankings[rankings["ranking_type"] == "top100"].copy()

    if long_only:
        top100 = top100[top100["long_short"] == "Long in"]

    # Filter by rank range
    top100 = top100[(top100["rank"] >= min_rank) & (top100["rank"] <= max_rank)]

    # Get unique dates and symbols
    all_dates = sorted(top100["data_date"].unique())
    if len(all_dates) < 2:
        raise ValueError(f"Need at least 2 dates, got {len(all_dates)}")

    start_date = all_dates[0]
    end_date = all_dates[-1]

    # Get all symbols that appear in top rankings
    all_symbols = sorted(top100["symbol"].unique())
    log.info(
        "backtest_setup",
        dates=len(all_dates),
        symbols=len(all_symbols),
        start=str(start_date),
        end=str(end_date),
    )

    # Fetch prices for all relevant symbols + SPY benchmark
    symbols_with_bench = list(set(all_symbols + ["SPY"]))
    prices = fetch_prices(symbols_with_bench, start_date, end_date)

    # Extract open prices for entries and close prices for exits
    if isinstance(prices.columns, pd.MultiIndex):
        open_prices = prices["Open"]
        close_prices = prices["Close"]
    else:
        # Single symbol case
        open_prices = prices[["Open"]].rename(columns={"Open": all_symbols[0]})
        close_prices = prices[["Close"]].rename(columns={"Close": all_symbols[0]})

    price_dates = [d.date() for d in open_prices.index]

    # Run the backtest
    trades: list[QU100Trade] = []

    for signal_date in all_dates[:-holding_days]:
        # Get top N for this date
        day_stocks = (
            top100[top100["data_date"] == signal_date]
            .nsmallest(top_n, "rank")
        )

        # Find next trading day for entry (+ optional delay)
        entry_idx = None
        days_after = 0
        for i, d in enumerate(price_dates):
            if d > signal_date:
                days_after += 1
                if days_after > entry_delay:
                    entry_idx = i
                    break
        if entry_idx is None:
            continue

        # Find exit day
        exit_idx = min(entry_idx + holding_days, len(price_dates) - 1)
        entry_date_actual = price_dates[entry_idx]
        exit_date_actual = price_dates[exit_idx]

        for _, row in day_stocks.iterrows():
            sym = row["symbol"]
            if sym not in open_prices.columns:
                continue

            entry_price = open_prices.iloc[entry_idx].get(sym)
            exit_price = close_prices.iloc[exit_idx].get(sym)

            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue

            ret = (exit_price - entry_price) / entry_price

            trades.append(QU100Trade(
                symbol=sym,
                rank=int(row["rank"]),
                long_short=row["long_short"],
                signal_date=signal_date,
                entry_date=entry_date_actual,
                entry_price=float(entry_price),
                exit_date=exit_date_actual,
                exit_price=float(exit_price),
                return_pct=float(ret),
                holding_days=holding_days,
            ))

    if not trades:
        raise ValueError("No trades generated")

    # Compute metrics
    result = _compute_metrics(trades, top_n, holding_days, start_date, end_date, rankings)

    # Compute benchmark (SPY buy & hold)
    if "SPY" in open_prices.columns:
        spy_start_idx = 0
        for i, d in enumerate(price_dates):
            if d >= start_date:
                spy_start_idx = i
                break
        spy_start = open_prices["SPY"].iloc[spy_start_idx]
        spy_end = close_prices["SPY"].iloc[-1]
        if not pd.isna(spy_start) and spy_start > 0:
            result.benchmark_return_pct = float(
                (spy_end - spy_start) / spy_start * 100
            )
            result.alpha_pct = result.total_return_pct - result.benchmark_return_pct

    # Build equity curve (daily P&L from overlapping trades)
    result.equity_curve, result.dates, result.max_drawdown_pct = _build_equity_curve(
        trades, price_dates, open_prices, close_prices,
    )

    return result


def _compute_metrics(
    trades: list[QU100Trade],
    top_n: int,
    holding_days: int,
    start_date: date,
    end_date: date,
    rankings: pd.DataFrame,
) -> QU100BacktestResult:
    """Compute aggregate metrics from trade list."""
    returns = [t.return_pct for t in trades]
    winners = [r for r in returns if r > 0]

    result = QU100BacktestResult(
        trades=trades,
        total_trades=len(trades),
        win_rate=len(winners) / len(returns) if returns else 0,
        avg_return_pct=float(np.mean(returns) * 100),
        median_return_pct=float(np.median(returns) * 100),
        holding_days=holding_days,
        top_n=top_n,
        start_date=start_date,
        end_date=end_date,
    )

    # Total compounded return (average daily portfolio return)
    daily_returns = {}
    for t in trades:
        daily_returns.setdefault(t.entry_date, []).append(t.return_pct)

    portfolio_returns = []
    for d in sorted(daily_returns.keys()):
        avg_ret = np.mean(daily_returns[d])
        portfolio_returns.append(avg_ret)

    if portfolio_returns:
        cumulative = np.prod([1 + r for r in portfolio_returns])
        result.total_return_pct = float((cumulative - 1) * 100)

        # Sharpe (annualized)
        if np.std(portfolio_returns) > 0:
            result.sharpe_ratio = float(
                np.mean(portfolio_returns) / np.std(portfolio_returns)
                * math.sqrt(252 / holding_days)
            )

    # Sector breakdown
    sector_returns: dict[str, list[float]] = {}
    for t in trades:
        # Look up sector from rankings
        match = rankings[
            (rankings["symbol"] == t.symbol)
            & (rankings["data_date"] == t.signal_date)
        ]
        sector = match["sector"].iloc[0] if len(match) > 0 else "Unknown"
        sector_returns.setdefault(sector, []).append(t.return_pct)

    result.sector_returns = {
        s: float(np.mean(rets) * 100)
        for s, rets in sector_returns.items()
    }

    # Long vs Short breakdown
    long_trades = [t.return_pct for t in trades if t.long_short == "Long in"]
    short_trades = [t.return_pct for t in trades if t.long_short == "Short in"]
    if long_trades:
        result.long_in_avg_return = float(np.mean(long_trades) * 100)
    if short_trades:
        result.short_in_avg_return = float(np.mean(short_trades) * 100)

    return result


def _build_equity_curve(
    trades: list[QU100Trade],
    price_dates: list[date],
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
) -> tuple[list[float], list[date], float]:
    """Build daily equity curve from overlapping trades.

    Returns (equity_values, dates, max_drawdown_pct).

    Uses compounded equal-weight daily returns: each day, compute the average
    return across all active positions, then compound onto the prior day's equity.
    """
    if not trades:
        return [], [], 0.0

    # Pre-compute a date→index lookup for price_dates
    date_to_idx: dict[date, int] = {}
    for i, d in enumerate(price_dates):
        date_to_idx[d] = i

    capital = 100_000.0
    equity = [capital]
    curve_dates: list[date] = []
    prev_prices: dict[str, float] = {}  # track previous close for daily returns

    for d in price_dates:
        if d < trades[0].entry_date:
            continue

        # Find active trades for this day
        active = [
            t for t in trades
            if t.entry_date <= d <= t.exit_date
        ]
        if not active:
            equity.append(equity[-1])
            curve_dates.append(d)
            continue

        # Compute equal-weight daily return
        daily_returns: list[float] = []
        for t in active:
            sym = t.symbol
            if sym not in close_prices.columns:
                continue
            idx = date_to_idx.get(d)
            if idx is None:
                continue
            today_price = close_prices.iloc[idx].get(sym)
            if pd.isna(today_price):
                continue

            # Reference price: yesterday's close, or entry price on first day
            key = f"{sym}_{t.signal_date}"
            ref_price = prev_prices.get(key, t.entry_price)
            if ref_price <= 0:
                continue
            daily_returns.append((today_price - ref_price) / ref_price)
            prev_prices[key] = float(today_price)

        if daily_returns:
            avg_daily_ret = np.mean(daily_returns)
            capital = equity[-1] * (1 + avg_daily_ret)
        else:
            capital = equity[-1]

        equity.append(capital)
        curve_dates.append(d)

    # Max drawdown
    peak = equity[0]
    max_dd = 0.0
    for eq in equity:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

    return equity, curve_dates, max_dd


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_qu100_report(result: QU100BacktestResult) -> str:
    """Format QU100 backtest results as readable text."""
    lines = [
        "=" * 70,
        "QU100 MONEY FLOW RANKING BACKTEST",
        "=" * 70,
        "",
        f"  Period: {result.start_date} to {result.end_date}",
        f"  Strategy: Buy top {result.top_n} 'Long in' stocks,",
        f"            hold {result.holding_days} trading days",
        "",
        "PERFORMANCE",
        "-" * 70,
        f"  Total trades:       {result.total_trades:,}",
        f"  Win rate:           {result.win_rate:.1%}",
        f"  Avg return/trade:   {result.avg_return_pct:+.2f}%",
        f"  Median return:      {result.median_return_pct:+.2f}%",
        f"  Total return:       {result.total_return_pct:+.2f}%",
        f"  Sharpe ratio:       {result.sharpe_ratio:.2f}",
        f"  Max drawdown:       {result.max_drawdown_pct:.2%}",
        "",
        "BENCHMARK (SPY buy & hold)",
        "-" * 70,
        f"  SPY return:         {result.benchmark_return_pct:+.2f}%",
        f"  Alpha:              {result.alpha_pct:+.2f}%",
        "",
    ]

    # Top sectors
    if result.sector_returns:
        sorted_sectors = sorted(
            result.sector_returns.items(), key=lambda x: x[1], reverse=True,
        )
        lines.append("TOP SECTORS (avg return)")
        lines.append("-" * 70)
        for sector, ret in sorted_sectors[:10]:
            lines.append(f"  {sector:<35} {ret:+.2f}%")
        lines.append("")

    # Top symbols by frequency
    symbol_counts: dict[str, list[float]] = {}
    for t in result.trades:
        symbol_counts.setdefault(t.symbol, []).append(t.return_pct)

    top_symbols = sorted(
        symbol_counts.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )[:15]

    lines.append("TOP SYMBOLS (by frequency)")
    lines.append("-" * 70)
    lines.append(
        f"  {'Symbol':<8} {'Trades':>7} {'WinRate':>8} "
        f"{'AvgRet':>8} {'TotalRet':>10}"
    )
    lines.append("-" * 70)
    for sym, rets in top_symbols:
        wins = sum(1 for r in rets if r > 0)
        lines.append(
            f"  {sym:<8} {len(rets):>7} {wins/len(rets):>7.1%} "
            f"{np.mean(rets)*100:>+7.2f}% "
            f"{sum(rets)*100:>+9.2f}%"
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def format_discord_report(result: QU100BacktestResult) -> list[dict]:
    """Format QU100 backtest as Discord embeds with tables."""
    n_days = 0
    if result.start_date and result.end_date:
        n_days = (result.end_date - result.start_date).days

    # --- Embed 1: Main performance table ---
    table = (
        "```"
        "\n\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u252c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510"
    )
    rows = [
        ("Period", f"{result.start_date} to {result.end_date}"),
        ("Total trades", f"{result.total_trades:,}"),
        ("Win rate", f"{result.win_rate:.1%}"),
        ("Avg return/trade", f"{result.avg_return_pct:+.2f}%"),
        ("Total return", f"{result.total_return_pct:+.2f}%"),
        ("Sharpe ratio", f"{result.sharpe_ratio:.2f}"),
        ("SPY benchmark", f"{result.benchmark_return_pct:+.2f}%"),
        ("Alpha", f"{result.alpha_pct:+.2f}%"),
    ]
    for label, value in rows:
        table += (
            f"\n\u2502 {label:<16} "
            f"\u2502 {value:<24} \u2502"
        )
    table += (
        "\n\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2534\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
        "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518"
        "\n```"
    )

    perf_embed = {
        "title": (
            f"\U0001f4ca QU100 Money Flow Ranking Backtest "
            f"({n_days} days)"
        ),
        "color": 0x2196F3 if result.alpha_pct >= 0 else 0xFF1744,
        "description": (
            f"Top {result.top_n} 'Long in' stocks, "
            f"{result.holding_days}-day hold\n"
            + table
        ),
    }

    # --- Embed 2: Top performers by total return ---
    symbol_stats: dict[str, list[float]] = {}
    for t in result.trades:
        symbol_stats.setdefault(t.symbol, []).append(t.return_pct)

    by_total_ret = sorted(
        symbol_stats.items(),
        key=lambda x: sum(x[1]),
        reverse=True,
    )[:10]

    perf_lines = []
    for sym, rets in by_total_ret:
        total = sum(rets) * 100
        perf_lines.append(f"{sym} ({total:+.0f}%)")
    performers_str = ", ".join(perf_lines)

    # --- Embed 3: Rank bucket performance ---
    rank_lines = []
    buckets = [
        (1, 5), (6, 10), (11, 20), (21, 30), (31, 50),
    ]
    rank_lines.append(
        f"{'Rank':<8} {'Trades':>6} {'Win%':>6} "
        f"{'AvgRet':>7} {'Sharpe':>7}"
    )
    rank_lines.append("-" * 38)
    for lo, hi in buckets:
        bucket_trades = [
            t for t in result.trades
            if lo <= t.rank <= hi
        ]
        if not bucket_trades:
            continue
        rets = [t.return_pct for t in bucket_trades]
        wins = sum(1 for r in rets if r > 0)
        avg = np.mean(rets) * 100
        std = np.std(rets)
        import math as _math
        sharpe = (
            np.mean(rets) / std * _math.sqrt(252 / result.holding_days)
            if std > 0 else 0
        )
        rank_lines.append(
            f"{lo:>2}-{hi:<4} {len(rets):>6} "
            f"{wins/len(rets):>5.0%} "
            f"{avg:>+6.1f}% {sharpe:>6.2f}"
        )

    rank_embed = {
        "title": "\U0001f3af Performance by Rank",
        "color": 0x2196F3,
        "description": f"```\n{chr(10).join(rank_lines)}\n```",
    }

    # --- Embed 4: Top sectors ---
    sorted_sectors = sorted(
        result.sector_returns.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:8]
    sector_lines = [f"{s}: {r:+.2f}%" for s, r in sorted_sectors]

    summary_embed = {
        "title": "\U0001f4cb Summary",
        "color": 0x2196F3,
        "fields": [
            {
                "name": "\U0001f525 Top Performers",
                "value": performers_str,
                "inline": False,
            },
            {
                "name": "\U0001f3ed Best Sectors",
                "value": "\n".join(sector_lines),
                "inline": False,
            },
        ],
    }

    return [perf_embed, rank_embed, summary_embed]


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------


@dataclass
class SweepRow:
    """One cell in the parameter sweep grid."""

    min_rank: int
    max_rank: int
    holding_days: int
    total_trades: int
    win_rate: float
    avg_return_pct: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    alpha_pct: float


def run_parameter_sweep(
    rank_ranges: list[tuple[int, int]] | None = None,
    hold_periods: list[int] | None = None,
) -> list[SweepRow]:
    """Run grid sweep over rank ranges x holding periods.

    Loads data once, then runs each combo. Returns list of SweepRow.
    """
    if rank_ranges is None:
        rank_ranges = [
            (1, 5), (6, 10), (1, 10), (11, 20),
            (1, 20), (6, 20), (1, 30), (1, 50),
        ]
    if hold_periods is None:
        hold_periods = [3, 5, 7, 10, 15, 20]

    rows: list[SweepRow] = []
    total = len(rank_ranges) * len(hold_periods)
    idx = 0

    for min_r, max_r in rank_ranges:
        for hold in hold_periods:
            idx += 1
            log.info("sweep_combo", combo=f"{idx}/{total}", ranks=f"{min_r}-{max_r}", hold=hold)
            try:
                result = run_qu100_backtest(
                    top_n=max_r,  # select all within rank range
                    holding_days=hold,
                    long_only=True,
                    min_rank=min_r,
                    max_rank=max_r,
                )
                rows.append(SweepRow(
                    min_rank=min_r,
                    max_rank=max_r,
                    holding_days=hold,
                    total_trades=result.total_trades,
                    win_rate=result.win_rate,
                    avg_return_pct=result.avg_return_pct,
                    total_return_pct=result.total_return_pct,
                    sharpe_ratio=result.sharpe_ratio,
                    max_drawdown_pct=result.max_drawdown_pct,
                    alpha_pct=result.alpha_pct,
                ))
            except Exception as e:
                log.warning("sweep_combo_failed", ranks=f"{min_r}-{max_r}", hold=hold, err=str(e))

    return rows


def format_sweep_table(rows: list[SweepRow]) -> str:
    """Format sweep results as a text table."""
    lines = [
        "=" * 90,
        "QU100 PARAMETER SWEEP",
        "=" * 90,
        f"  {'Ranks':<8} {'Hold':>5} {'Trades':>7} {'Win%':>6} "
        f"{'AvgRet':>8} {'TotalRet':>10} {'Sharpe':>7} {'MaxDD':>7} {'Alpha':>8}",
        "-" * 90,
    ]
    for r in sorted(rows, key=lambda x: x.sharpe_ratio, reverse=True):
        lines.append(
            f"  {r.min_rank}-{r.max_rank:<5} {r.holding_days:>5}d "
            f"{r.total_trades:>7} {r.win_rate:>5.1%} "
            f"{r.avg_return_pct:>+7.2f}% {r.total_return_pct:>+9.1f}% "
            f"{r.sharpe_ratio:>6.2f} {r.max_drawdown_pct:>6.1%} "
            f"{r.alpha_pct:>+7.1f}%"
        )
    lines.append("=" * 90)

    # Highlight best
    if rows:
        best_sharpe = max(rows, key=lambda x: x.sharpe_ratio)
        best_return = max(rows, key=lambda x: x.total_return_pct)
        lines.append("")
        lines.append(
            f"  Best Sharpe:  ranks {best_sharpe.min_rank}-{best_sharpe.max_rank}, "
            f"{best_sharpe.holding_days}d hold → {best_sharpe.sharpe_ratio:.2f}"
        )
        lines.append(
            f"  Best Return:  ranks {best_return.min_rank}-{best_return.max_rank}, "
            f"{best_return.holding_days}d hold → {best_return.total_return_pct:+.1f}%"
        )

    return "\n".join(lines)


def format_sweep_discord(rows: list[SweepRow]) -> list[dict]:
    """Format sweep results as Discord embeds."""
    sorted_rows = sorted(rows, key=lambda x: x.sharpe_ratio, reverse=True)

    # Table embed
    header = f"{'Ranks':<7} {'Hold':>4} {'Win%':>5} {'Ret':>7} {'Shrp':>5} {'DD':>5} {'Alpha':>7}"
    lines = [header, "-" * 46]
    for r in sorted_rows[:20]:
        lines.append(
            f"{r.min_rank}-{r.max_rank:<4} {r.holding_days:>4}d "
            f"{r.win_rate:>4.0%} {r.total_return_pct:>+6.0f}% "
            f"{r.sharpe_ratio:>5.2f} {r.max_drawdown_pct:>4.0%} "
            f"{r.alpha_pct:>+6.0f}%"
        )

    table_embed = {
        "title": "\U0001f50d QU100 Parameter Sweep (sorted by Sharpe)",
        "color": 0x9C27B0,
        "description": f"```\n{chr(10).join(lines)}\n```",
    }

    # Best combos summary
    if rows:
        best_sharpe = max(rows, key=lambda x: x.sharpe_ratio)
        best_return = max(rows, key=lambda x: x.total_return_pct)
        best_winrate = max(rows, key=lambda x: x.win_rate)
        lowest_dd = min(rows, key=lambda x: x.max_drawdown_pct)

        summary_embed = {
            "title": "\U0001f3c6 Best Combinations",
            "color": 0xFFD700,
            "fields": [
                {
                    "name": "Best Sharpe",
                    "value": (
                        f"Rank {best_sharpe.min_rank}-{best_sharpe.max_rank}, "
                        f"{best_sharpe.holding_days}d → "
                        f"**{best_sharpe.sharpe_ratio:.2f}** "
                        f"({best_sharpe.total_return_pct:+.0f}%)"
                    ),
                    "inline": True,
                },
                {
                    "name": "Best Return",
                    "value": (
                        f"Rank {best_return.min_rank}-{best_return.max_rank}, "
                        f"{best_return.holding_days}d → "
                        f"**{best_return.total_return_pct:+.0f}%** "
                        f"(Sharpe {best_return.sharpe_ratio:.2f})"
                    ),
                    "inline": True,
                },
                {
                    "name": "Best Win Rate",
                    "value": (
                        f"Rank {best_winrate.min_rank}-{best_winrate.max_rank}, "
                        f"{best_winrate.holding_days}d → "
                        f"**{best_winrate.win_rate:.1%}**"
                    ),
                    "inline": True,
                },
                {
                    "name": "Lowest Drawdown",
                    "value": (
                        f"Rank {lowest_dd.min_rank}-{lowest_dd.max_rank}, "
                        f"{lowest_dd.holding_days}d → "
                        f"**{lowest_dd.max_drawdown_pct:.1%}**"
                    ),
                    "inline": True,
                },
            ],
        }
        return [table_embed, summary_embed]

    return [table_embed]


# ---------------------------------------------------------------------------
# Signal tuning variations
# ---------------------------------------------------------------------------


def run_qu100_backtest_with_momentum(
    top_n: int = 20,
    holding_days: int = 5,
    min_rank: int = 1,
    max_rank: int = 50,
    rank_improve_days: int = 3,
) -> QU100BacktestResult:
    """Rank momentum filter: only buy stocks whose rank improved over N days."""
    rankings = load_rankings_from_db()
    top100 = rankings[rankings["ranking_type"] == "top100"].copy()
    top100 = top100[top100["long_short"] == "Long in"]
    top100 = top100[(top100["rank"] >= min_rank) & (top100["rank"] <= max_rank)]

    all_dates = sorted(top100["data_date"].unique())
    if len(all_dates) < rank_improve_days + 2:
        raise ValueError(f"Need at least {rank_improve_days + 2} dates")

    # Build rank history per symbol per date
    rank_lookup: dict[tuple[str, date], int] = {}
    for _, row in top100.iterrows():
        rank_lookup[(row["symbol"], row["data_date"])] = row["rank"]

    # Filter: only keep symbols whose rank improved
    filtered_rows = []
    for i, signal_date in enumerate(all_dates):
        if i < rank_improve_days:
            continue
        prev_date = all_dates[i - rank_improve_days]
        day_stocks = top100[top100["data_date"] == signal_date].nsmallest(top_n, "rank")

        for _, row in day_stocks.iterrows():
            sym = row["symbol"]
            prev_rank = rank_lookup.get((sym, prev_date))
            curr_rank = row["rank"]
            # Keep if rank improved (lower = better) or new to list
            if prev_rank is None or curr_rank < prev_rank:
                filtered_rows.append(row)

    if not filtered_rows:
        raise ValueError("No trades after momentum filter")

    # Build filtered DataFrame and run through standard engine
    filtered_df = pd.DataFrame(filtered_rows)
    log.info("momentum_filter", original=len(top100), filtered=len(filtered_df))

    # Use the standard backtest with the filtered data
    return _run_with_filtered_rankings(
        filtered_df, rankings, top_n, holding_days, min_rank, max_rank,
        label="momentum",
    )


def run_qu100_backtest_short(
    top_n: int = 20,
    holding_days: int = 5,
) -> QU100BacktestResult:
    """Short-side backtest: short bottom100 'Short in' stocks."""
    rankings = load_rankings_from_db()
    bottom100 = rankings[rankings["ranking_type"] == "bottom100"].copy()
    bottom100 = bottom100[bottom100["long_short"] == "Short in"]

    if bottom100.empty:
        raise ValueError("No 'Short in' bottom100 data found")

    all_dates = sorted(bottom100["data_date"].unique())
    start_date = all_dates[0]
    end_date = all_dates[-1]
    all_symbols = sorted(bottom100["symbol"].unique())

    symbols_with_bench = list(set(all_symbols + ["SPY"]))
    prices = fetch_prices(symbols_with_bench, start_date, end_date)

    if isinstance(prices.columns, pd.MultiIndex):
        open_prices = prices["Open"]
        close_prices = prices["Close"]
    else:
        open_prices = prices[["Open"]].rename(columns={"Open": all_symbols[0]})
        close_prices = prices[["Close"]].rename(columns={"Close": all_symbols[0]})

    price_dates = [d.date() for d in open_prices.index]

    trades: list[QU100Trade] = []
    for signal_date in all_dates[:-holding_days]:
        day_stocks = bottom100[bottom100["data_date"] == signal_date].nsmallest(top_n, "rank")

        entry_idx = None
        for i, d in enumerate(price_dates):
            if d > signal_date:
                entry_idx = i
                break
        if entry_idx is None:
            continue

        exit_idx = min(entry_idx + holding_days, len(price_dates) - 1)
        entry_date_actual = price_dates[entry_idx]
        exit_date_actual = price_dates[exit_idx]

        for _, row in day_stocks.iterrows():
            sym = row["symbol"]
            if sym not in open_prices.columns:
                continue
            entry_price = open_prices.iloc[entry_idx].get(sym)
            exit_price = close_prices.iloc[exit_idx].get(sym)
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue
            # Short return: profit when price goes down
            ret = (entry_price - exit_price) / entry_price
            trades.append(QU100Trade(
                symbol=sym,
                rank=int(row["rank"]),
                long_short="Short in",
                signal_date=signal_date,
                entry_date=entry_date_actual,
                entry_price=float(entry_price),
                exit_date=exit_date_actual,
                exit_price=float(exit_price),
                return_pct=float(ret),
                holding_days=holding_days,
            ))

    if not trades:
        raise ValueError("No short trades generated")

    result = _compute_metrics(trades, top_n, holding_days, start_date, end_date, rankings)

    if "SPY" in open_prices.columns:
        spy_start_idx = next(
            (i for i, d in enumerate(price_dates) if d >= start_date), 0,
        )
        spy_start = open_prices["SPY"].iloc[spy_start_idx]
        spy_end = close_prices["SPY"].iloc[-1]
        if not pd.isna(spy_start) and spy_start > 0:
            result.benchmark_return_pct = float((spy_end - spy_start) / spy_start * 100)
            result.alpha_pct = result.total_return_pct - result.benchmark_return_pct

    result.equity_curve, result.dates, result.max_drawdown_pct = _build_equity_curve(
        trades, price_dates, open_prices, close_prices,
    )
    return result


def _run_with_filtered_rankings(
    filtered_df: pd.DataFrame,
    rankings: pd.DataFrame,
    top_n: int,
    holding_days: int,
    min_rank: int,
    max_rank: int,
    label: str = "",
) -> QU100BacktestResult:
    """Run backtest on a pre-filtered set of rankings."""
    all_dates = sorted(filtered_df["data_date"].unique())
    start_date = all_dates[0]
    end_date = all_dates[-1]
    all_symbols = sorted(filtered_df["symbol"].unique())

    symbols_with_bench = list(set(all_symbols + ["SPY"]))
    prices = fetch_prices(symbols_with_bench, start_date, end_date)

    if isinstance(prices.columns, pd.MultiIndex):
        open_prices = prices["Open"]
        close_prices = prices["Close"]
    else:
        open_prices = prices[["Open"]].rename(columns={"Open": all_symbols[0]})
        close_prices = prices[["Close"]].rename(columns={"Close": all_symbols[0]})

    price_dates = [d.date() for d in open_prices.index]

    trades: list[QU100Trade] = []
    for signal_date in all_dates[:-holding_days]:
        day_stocks = (
            filtered_df[filtered_df["data_date"] == signal_date]
            .nsmallest(top_n, "rank")
        )

        entry_idx = None
        for i, d in enumerate(price_dates):
            if d > signal_date:
                entry_idx = i
                break
        if entry_idx is None:
            continue

        exit_idx = min(entry_idx + holding_days, len(price_dates) - 1)
        entry_date_actual = price_dates[entry_idx]
        exit_date_actual = price_dates[exit_idx]

        for _, row in day_stocks.iterrows():
            sym = row["symbol"]
            if sym not in open_prices.columns:
                continue
            entry_price = open_prices.iloc[entry_idx].get(sym)
            exit_price = close_prices.iloc[exit_idx].get(sym)
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue
            ret = (exit_price - entry_price) / entry_price
            trades.append(QU100Trade(
                symbol=sym,
                rank=int(row["rank"]),
                long_short=row["long_short"],
                signal_date=signal_date,
                entry_date=entry_date_actual,
                entry_price=float(entry_price),
                exit_date=exit_date_actual,
                exit_price=float(exit_price),
                return_pct=float(ret),
                holding_days=holding_days,
            ))

    if not trades:
        raise ValueError(f"No trades generated after {label} filter")

    result = _compute_metrics(trades, top_n, holding_days, start_date, end_date, rankings)

    if "SPY" in open_prices.columns:
        spy_start_idx = next(
            (i for i, d in enumerate(price_dates) if d >= start_date), 0,
        )
        spy_start = open_prices["SPY"].iloc[spy_start_idx]
        spy_end = close_prices["SPY"].iloc[-1]
        if not pd.isna(spy_start) and spy_start > 0:
            result.benchmark_return_pct = float((spy_end - spy_start) / spy_start * 100)
            result.alpha_pct = result.total_return_pct - result.benchmark_return_pct

    result.equity_curve, result.dates, result.max_drawdown_pct = _build_equity_curve(
        trades, price_dates, open_prices, close_prices,
    )
    return result


# ---------------------------------------------------------------------------
# Variation comparison report
# ---------------------------------------------------------------------------


@dataclass
class VariationResult:
    """Summary of one backtest variation for comparison."""

    name: str
    total_trades: int
    win_rate: float
    avg_return_pct: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    alpha_pct: float


def format_variation_comparison(variations: list[VariationResult]) -> str:
    """Format side-by-side comparison of backtest variations."""
    lines = [
        "=" * 95,
        "QU100 SIGNAL VARIATION COMPARISON",
        "=" * 95,
        f"  {'Variation':<30} {'Trades':>7} {'Win%':>6} "
        f"{'AvgRet':>8} {'TotalRet':>10} {'Sharpe':>7} {'MaxDD':>7} {'Alpha':>8}",
        "-" * 95,
    ]
    for v in sorted(variations, key=lambda x: x.sharpe_ratio, reverse=True):
        lines.append(
            f"  {v.name:<30} {v.total_trades:>7} {v.win_rate:>5.1%} "
            f"{v.avg_return_pct:>+7.2f}% {v.total_return_pct:>+9.1f}% "
            f"{v.sharpe_ratio:>6.2f} {v.max_drawdown_pct:>6.1%} "
            f"{v.alpha_pct:>+7.1f}%"
        )
    lines.append("=" * 95)
    return "\n".join(lines)


def format_variation_discord(variations: list[VariationResult]) -> list[dict]:
    """Format variation comparison as Discord embeds."""
    sorted_vars = sorted(variations, key=lambda x: x.sharpe_ratio, reverse=True)

    header = f"{'Strategy':<28} {'Win%':>5} {'Ret':>7} {'Shrp':>5} {'DD':>5} {'Alpha':>7}"
    lines = [header, "-" * 62]
    for v in sorted_vars:
        name = v.name[:27]
        lines.append(
            f"{name:<28} {v.win_rate:>4.0%} {v.total_return_pct:>+6.0f}% "
            f"{v.sharpe_ratio:>5.2f} {v.max_drawdown_pct:>4.0%} "
            f"{v.alpha_pct:>+6.0f}%"
        )

    embed = {
        "title": "\U0001f9ea QU100 Signal Tuning Variations",
        "color": 0x00BCD4,
        "description": f"```\n{chr(10).join(lines)}\n```",
    }

    # Best variation callout
    if sorted_vars:
        best = sorted_vars[0]
        callout = {
            "title": "\U0001f947 Best Variation",
            "color": 0xFFD700,
            "description": (
                f"**{best.name}**\n"
                f"Sharpe: {best.sharpe_ratio:.2f} | "
                f"Return: {best.total_return_pct:+.1f}% | "
                f"Win: {best.win_rate:.1%} | "
                f"MaxDD: {best.max_drawdown_pct:.1%} | "
                f"Alpha: {best.alpha_pct:+.1f}%"
            ),
        }
        return [embed, callout]

    return [embed]


def result_to_variation(name: str, r: QU100BacktestResult) -> VariationResult:
    """Convert a QU100BacktestResult to a VariationResult summary."""
    return VariationResult(
        name=name,
        total_trades=r.total_trades,
        win_rate=r.win_rate,
        avg_return_pct=r.avg_return_pct,
        total_return_pct=r.total_return_pct,
        sharpe_ratio=r.sharpe_ratio,
        max_drawdown_pct=r.max_drawdown_pct,
        alpha_pct=r.alpha_pct,
    )


# ---------------------------------------------------------------------------
# Pattern-filtered backtest
# ---------------------------------------------------------------------------

# Best-performing patterns from backtest analysis
BEST_PATTERNS = ["false_breakdown_w_bottom", "false_breakdown", "bull_flag"]


@dataclass
class PatternMatch:
    """A pattern detected on a specific symbol at a specific bar date."""

    symbol: str
    pattern_type: str
    confidence: float
    signal_date: date  # date the pattern was confirmed (pattern_end bar date)


def run_qu100_pattern_backtest(
    pattern_matches: list[PatternMatch],
    top_n: int = 5,
    holding_days: int = 5,
    allowed_patterns: list[str] | None = None,
) -> QU100BacktestResult:
    """Run QU100 backtest filtering by Caisen pattern matches.

    Only trades stocks that have a qualifying pattern confirmed on or before
    the signal date. Ranks candidates by pattern confidence, picks top N.

    Args:
        pattern_matches: Pre-computed pattern detections with signal dates.
            Typically produced by CLI wiring (detect_patterns on daily data).
        top_n: Number of top pattern-matched stocks to trade per day.
        holding_days: Days to hold each position.
        allowed_patterns: Pattern types to allow. Defaults to BEST_PATTERNS.

    Returns:
        QU100BacktestResult with all trades and aggregate metrics.
    """
    if allowed_patterns is None:
        allowed_patterns = BEST_PATTERNS

    # Filter to allowed patterns
    matches = [m for m in pattern_matches if m.pattern_type in allowed_patterns]
    if not matches:
        raise ValueError(
            f"No pattern matches for {allowed_patterns}. "
            f"Got {len(pattern_matches)} total matches."
        )

    # Build index: date → [(symbol, pattern_type, confidence)]
    by_date: dict[date, list[PatternMatch]] = {}
    for m in matches:
        by_date.setdefault(m.signal_date, []).append(m)

    all_dates = sorted(by_date.keys())
    if len(all_dates) < 2:
        raise ValueError(f"Need at least 2 dates with pattern matches, got {len(all_dates)}")

    start_date = all_dates[0]
    end_date = all_dates[-1]

    # Collect all symbols
    all_symbols = sorted({m.symbol for m in matches})
    log.info(
        "pattern_backtest_setup",
        dates=len(all_dates),
        symbols=len(all_symbols),
        patterns=allowed_patterns,
        top_n=top_n,
        holding_days=holding_days,
    )

    # Fetch prices
    symbols_with_bench = list(set(all_symbols + ["SPY"]))
    prices = fetch_prices(symbols_with_bench, start_date, end_date)

    if isinstance(prices.columns, pd.MultiIndex):
        open_prices = prices["Open"]
        close_prices = prices["Close"]
    else:
        open_prices = prices[["Open"]].rename(columns={"Open": all_symbols[0]})
        close_prices = prices[["Close"]].rename(columns={"Close": all_symbols[0]})

    price_dates = [d.date() for d in open_prices.index]

    # Run the backtest
    trades: list[QU100Trade] = []

    for signal_date in all_dates:
        # Get top N matches by confidence for this date
        day_matches = sorted(by_date[signal_date], key=lambda m: m.confidence, reverse=True)
        top_matches = day_matches[:top_n]

        # Find next trading day for entry
        entry_idx = None
        for i, d in enumerate(price_dates):
            if d > signal_date:
                entry_idx = i
                break
        if entry_idx is None:
            continue

        # Find exit day
        exit_idx = min(entry_idx + holding_days, len(price_dates) - 1)
        if exit_idx <= entry_idx:
            continue
        entry_date_actual = price_dates[entry_idx]
        exit_date_actual = price_dates[exit_idx]

        for match in top_matches:
            sym = match.symbol
            if sym not in open_prices.columns:
                continue

            entry_price = open_prices.iloc[entry_idx].get(sym)
            exit_price = close_prices.iloc[exit_idx].get(sym)

            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue

            ret = (exit_price - entry_price) / entry_price

            trades.append(QU100Trade(
                symbol=sym,
                rank=0,  # not rank-based
                long_short="Long in",
                signal_date=signal_date,
                entry_date=entry_date_actual,
                entry_price=float(entry_price),
                exit_date=exit_date_actual,
                exit_price=float(exit_price),
                return_pct=float(ret),
                holding_days=holding_days,
                pattern_type=match.pattern_type,
            ))

    if not trades:
        raise ValueError("No trades generated from pattern matches")

    # Load rankings for sector breakdown (best effort)
    try:
        rankings = load_rankings_from_db()
    except Exception:
        rankings = pd.DataFrame(columns=[
            "data_date", "symbol", "rank", "ranking_type",
            "long_short", "sector", "industry",
        ])

    result = _compute_metrics(trades, top_n, holding_days, start_date, end_date, rankings)

    # Benchmark
    if "SPY" in open_prices.columns:
        spy_start_idx = next(
            (i for i, d in enumerate(price_dates) if d >= start_date), 0,
        )
        spy_start = open_prices["SPY"].iloc[spy_start_idx]
        spy_end = close_prices["SPY"].iloc[-1]
        if not pd.isna(spy_start) and spy_start > 0:
            result.benchmark_return_pct = float((spy_end - spy_start) / spy_start * 100)
            result.alpha_pct = result.total_return_pct - result.benchmark_return_pct

    result.equity_curve, result.dates, result.max_drawdown_pct = _build_equity_curve(
        trades, price_dates, open_prices, close_prices,
    )

    return result


def format_pattern_report(result: QU100BacktestResult, allowed_patterns: list[str]) -> str:
    """Format pattern-filtered backtest results."""
    pattern_names = ", ".join(allowed_patterns)
    lines = [
        "=" * 70,
        "QU100 PATTERN-FILTERED BACKTEST",
        "=" * 70,
        "",
        f"  Period: {result.start_date} to {result.end_date}",
        f"  Patterns: {pattern_names}",
        f"  Strategy: Buy top {result.top_n} pattern-matched stocks,",
        f"            hold {result.holding_days} trading days",
        "",
        "PERFORMANCE",
        "-" * 70,
        f"  Total trades:       {result.total_trades:,}",
        f"  Win rate:           {result.win_rate:.1%}",
        f"  Avg return/trade:   {result.avg_return_pct:+.2f}%",
        f"  Median return:      {result.median_return_pct:+.2f}%",
        f"  Total return:       {result.total_return_pct:+.2f}%",
        f"  Sharpe ratio:       {result.sharpe_ratio:.2f}",
        f"  Max drawdown:       {result.max_drawdown_pct:.2%}",
        "",
        "BENCHMARK (SPY buy & hold)",
        "-" * 70,
        f"  SPY return:         {result.benchmark_return_pct:+.2f}%",
        f"  Alpha:              {result.alpha_pct:+.2f}%",
        "",
    ]

    # Per-pattern breakdown
    pattern_trades: dict[str, list[float]] = {}
    for t in result.trades:
        pt = t.pattern_type or "unknown"
        pattern_trades.setdefault(pt, []).append(t.return_pct)

    if pattern_trades:
        lines.append("PER-PATTERN BREAKDOWN")
        lines.append("-" * 70)
        lines.append(
            f"  {'Pattern':<28} {'Trades':>7} {'WinRate':>8} {'AvgRet':>8}"
        )
        lines.append("-" * 70)
        for pt, rets in sorted(pattern_trades.items()):
            wins = sum(1 for r in rets if r > 0)
            lines.append(
                f"  {pt:<28} {len(rets):>7} {wins / len(rets):>7.1%} "
                f"{np.mean(rets) * 100:>+7.2f}%"
            )
        lines.append("")

    # Top symbols
    symbol_counts: dict[str, list[float]] = {}
    for t in result.trades:
        symbol_counts.setdefault(t.symbol, []).append(t.return_pct)

    top_symbols = sorted(
        symbol_counts.items(), key=lambda x: len(x[1]), reverse=True,
    )[:15]

    lines.append("TOP SYMBOLS (by frequency)")
    lines.append("-" * 70)
    lines.append(
        f"  {'Symbol':<8} {'Trades':>7} {'WinRate':>8} "
        f"{'AvgRet':>8} {'TotalRet':>10}"
    )
    lines.append("-" * 70)
    for sym, rets in top_symbols:
        wins = sum(1 for r in rets if r > 0)
        lines.append(
            f"  {sym:<8} {len(rets):>7} {wins / len(rets):>7.1%} "
            f"{np.mean(rets) * 100:>+7.2f}% "
            f"{sum(rets) * 100:>+9.2f}%"
        )

    lines.append("=" * 70)
    return "\n".join(lines)
