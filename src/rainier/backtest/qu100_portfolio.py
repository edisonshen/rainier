"""QU100 portfolio backtest — day-by-day simulation with capital management.

Strategy:
  - Start with fixed capital (default $100).
  - Max 5 concurrent positions, each allocated 20% of current portfolio value.
  - Entry: top 2 QU100 stocks matching false_breakdown or false_breakdown_w_bottom.
  - Exit: stop loss hit, target price hit, or pattern invalidated.

Dependency rule: imports only from core/ (pattern detection passed in from CLI).
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import structlog
import yfinance as yf
from sqlalchemy import select

from rainier.core.database import get_session
from rainier.core.models import BacktestTradingLog, MoneyFlowSnapshot

log = structlog.get_logger()

ALLOWED_PATTERNS = ["false_breakdown", "false_breakdown_w_bottom"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """An open position in the portfolio."""

    symbol: str
    pattern_type: str
    entry_date: date
    entry_price: float
    shares: float
    allocated_amount: float
    stop_loss: float
    target_price: float
    confidence: float
    qu100_rank: int


@dataclass
class ClosedTrade:
    """A completed trade with exit info."""

    symbol: str
    pattern_type: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    shares: float
    allocated_amount: float
    stop_loss: float
    target_price: float
    confidence: float
    exit_reason: str  # "stop_loss", "target_hit", "pattern_invalidated"
    return_pct: float
    pnl: float
    qu100_rank: int


@dataclass
class PortfolioBacktestResult:
    """Results from portfolio backtest."""

    trades: list[ClosedTrade] = field(default_factory=list)
    total_trades: int = 0
    win_rate: float = 0.0
    avg_return_pct: float = 0.0
    median_return_pct: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    alpha_pct: float = 0.0
    start_capital: float = 100.0
    final_capital: float = 100.0
    start_date: date | None = None
    end_date: date | None = None
    max_positions: int = 5
    top_n: int = 2
    equity_curve: list[float] = field(default_factory=list)
    equity_dates: list[date] = field(default_factory=list)
    # Strategy params (for reporting)
    max_hold_days: int = 0
    hard_stop_pct: float = 0.0
    use_close_price: bool = False
    use_stop_limit: bool = False
    # Exit reason breakdown
    exit_by_stop_loss: int = 0
    exit_by_target: int = 0
    exit_by_pattern: int = 0
    exit_by_max_hold: int = 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rankings_from_db() -> pd.DataFrame:
    """Load all QU100 rankings from the database."""
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
    return df


def fetch_all_prices(
    symbols: list[str], start: date, end: date,
) -> pd.DataFrame:
    """Fetch daily OHLCV — loads from DB first, fetches missing from yfinance.

    On first run, downloads from yfinance and persists to stock_prices table.
    Subsequent runs load from DB instantly. Only fetches new symbols/dates.
    """
    from rainier.core.models import Stock, StockPrice

    end_buffered = end + timedelta(days=30)

    # Step 1: Try loading from DB
    log.info("loading_prices_from_db", symbols=len(symbols))
    with get_session() as db:
        rows = db.execute(
            select(
                StockPrice.symbol,
                StockPrice.date,
                StockPrice.open,
                StockPrice.high,
                StockPrice.low,
                StockPrice.close,
                StockPrice.volume,
            ).where(
                StockPrice.symbol.in_(symbols),
                StockPrice.date >= start.isoformat(),
                StockPrice.date <= end_buffered.isoformat(),
            )
        ).all()

    if rows:
        db_df = pd.DataFrame(rows, columns=[
            "symbol", "date", "open", "high", "low", "close", "volume",
        ])
        db_symbols = set(db_df["symbol"].unique())
        log.info("db_prices_loaded", symbols=len(db_symbols), rows=len(db_df))
    else:
        db_df = pd.DataFrame()
        db_symbols = set()

    # Step 2: Find symbols not in DB
    missing = [s for s in symbols if s not in db_symbols]

    if missing:
        batch_size = 20
        total_batches = math.ceil(len(missing) / batch_size)
        log.info(
            "fetching_missing_from_yfinance",
            missing=len(missing),
            cached=len(db_symbols),
            batches=total_batches,
        )
        for bi in range(0, len(missing), batch_size):
            batch = missing[bi : bi + batch_size]
            batch_num = bi // batch_size + 1
            log.info("yf_batch", batch=batch_num, total=total_batches, symbols=len(batch))
            if batch_num > 1:
                time.sleep(10)  # Avoid rate limiting between batches

            yf_df = yf.download(
                " ".join(batch),
                start=start.isoformat(),
                end=end_buffered.isoformat(),
                auto_adjust=True,
                progress=True,
                threads=True,
            )

            if not yf_df.empty:
                # Ensure MultiIndex for single-symbol batches
                if not isinstance(yf_df.columns, pd.MultiIndex) and len(batch) == 1:
                    yf_df.columns = pd.MultiIndex.from_product(
                        [yf_df.columns, batch]
                    )
                # Persist to DB
                _save_prices_to_db(yf_df, batch)

                # Merge with DB data
                yf_long = _yf_to_long(yf_df, batch)
                db_df = pd.concat([db_df, yf_long], ignore_index=True)

    if db_df.empty:
        raise ValueError("No price data available")

    # Convert long-form DB data to yfinance-style MultiIndex DataFrame
    return _long_to_multiindex(db_df)


def _yf_to_long(
    yf_df: pd.DataFrame, symbols: list[str],
) -> pd.DataFrame:
    """Convert yfinance MultiIndex DataFrame to long-form rows."""
    if yf_df.empty:
        return pd.DataFrame(
            columns=["symbol", "date", "open", "high", "low", "close", "volume"]
        )

    frames = []
    if isinstance(yf_df.columns, pd.MultiIndex):
        available = set(yf_df["Close"].columns)
        for sym in symbols:
            if sym not in available:
                continue
            sym_df = pd.DataFrame({
                "open": yf_df["Open"][sym],
                "high": yf_df["High"][sym],
                "low": yf_df["Low"][sym],
                "close": yf_df["Close"][sym],
                "volume": yf_df["Volume"][sym],
            }).dropna(subset=["close"])
            sym_df["symbol"] = sym
            sym_df["date"] = sym_df.index
            frames.append(sym_df)
    else:
        sym = symbols[0] if symbols else "UNKNOWN"
        sym_df = yf_df[["Open", "High", "Low", "Close", "Volume"]].copy()
        sym_df.columns = ["open", "high", "low", "close", "volume"]
        sym_df = sym_df.dropna(subset=["close"])
        sym_df["symbol"] = sym
        sym_df["date"] = sym_df.index
        frames.append(sym_df)

    if not frames:
        return pd.DataFrame(
            columns=["symbol", "date", "open", "high", "low", "close", "volume"]
        )
    return pd.concat(frames, ignore_index=True)


def _save_prices_to_db(
    yf_df: pd.DataFrame, symbols: list[str],
) -> None:
    """Persist yfinance price data to stock_prices table."""
    from rainier.core.models import Stock, StockPrice

    long_df = _yf_to_long(yf_df, symbols)
    if long_df.empty:
        return

    with get_session() as db:
        # Ensure stocks exist
        existing = {
            row[0] for row in db.execute(
                select(Stock.symbol).where(Stock.symbol.in_(symbols))
            ).all()
        }
        new_symbols = set(long_df["symbol"].unique()) - existing
        for sym in new_symbols:
            db.add(Stock(symbol=sym))
        if new_symbols:
            db.flush()

        # Batch insert prices using INSERT ... ON CONFLICT DO NOTHING
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        rows_to_insert = []
        for _, row in long_df.iterrows():
            if pd.isna(row["close"]):
                continue
            rows_to_insert.append({
                "symbol": row["symbol"],
                "date": row["date"],
                "open": row["open"] if pd.notna(row["open"]) else None,
                "high": row["high"] if pd.notna(row["high"]) else None,
                "low": row["low"] if pd.notna(row["low"]) else None,
                "close": row["close"],
                "volume": int(row["volume"]) if pd.notna(row["volume"]) else None,
            })

        if rows_to_insert:
            # Insert in chunks to avoid huge queries
            chunk_size = 1000
            count = 0
            for ci in range(0, len(rows_to_insert), chunk_size):
                chunk = rows_to_insert[ci : ci + chunk_size]
                stmt = pg_insert(StockPrice).values(chunk)
                stmt = stmt.on_conflict_do_nothing(
                    constraint="uq_stock_price_symbol_date"
                )
                db.execute(stmt)
                count += len(chunk)
            db.commit()
            log.info("prices_saved_to_db", records=count)


def _long_to_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form price data to yfinance-style MultiIndex DataFrame."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df = df.set_index("date")

    symbols = sorted(df["symbol"].unique())
    fields = ["Open", "High", "Low", "Close", "Volume"]
    field_map = {"open": "Open", "high": "High", "low": "Low",
                 "close": "Close", "volume": "Volume"}

    frames = {}
    for sym in symbols:
        sym_data = df[df["symbol"] == sym].copy()
        sym_data = sym_data.rename(columns=field_map)
        sym_data = sym_data[fields]
        sym_data.columns = pd.MultiIndex.from_product(
            [fields, [sym]]
        )
        # Flatten to individual (field, sym) columns
        for field in fields:
            frames[(field, sym)] = sym_data[(field, sym)]

    result = pd.DataFrame(frames)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    result = result.sort_index()
    return result


def _extract_symbol_ohlcv(
    price_data: pd.DataFrame, symbol: str, end_date: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """Extract lowercase OHLCV DataFrame for a single symbol, up to end_date."""
    try:
        if isinstance(price_data.columns, pd.MultiIndex):
            sym_df = pd.DataFrame({
                "open": price_data["Open"][symbol],
                "high": price_data["High"][symbol],
                "low": price_data["Low"][symbol],
                "close": price_data["Close"][symbol],
                "volume": price_data["Volume"][symbol],
            }).dropna()
        else:
            sym_df = pd.DataFrame({
                "open": price_data["Open"],
                "high": price_data["High"],
                "low": price_data["Low"],
                "close": price_data["Close"],
                "volume": price_data["Volume"],
            }).dropna()
    except (KeyError, TypeError):
        return None

    if end_date is not None:
        sym_df = sym_df[sym_df.index <= end_date]

    return sym_df if len(sym_df) > 0 else None


# ---------------------------------------------------------------------------
# Portfolio backtest engine
# ---------------------------------------------------------------------------


def run_qu100_portfolio_backtest(
    detect_patterns_fn,
    start_capital: float = 100.0,
    max_positions: int = 5,
    top_n: int = 2,
    start_date_str: str | None = None,
    max_hold_days: int = 0,
    hard_stop_pct: float = 0.0,
    use_close_price: bool = False,
    use_stop_limit: bool = False,
) -> PortfolioBacktestResult:
    """Run day-by-day portfolio backtest.

    Args:
        detect_patterns_fn: Pattern detection function (injected from CLI to respect
            dependency rule — backtest/ does not import from analysis/).
        start_capital: Starting capital in USD.
        max_positions: Maximum concurrent positions.
        top_n: Number of top pattern matches to buy per day.
        start_date_str: Start date as YYYY-MM-DD string (default: earliest data).
        max_hold_days: Max trading days to hold (0 = unlimited).
        hard_stop_pct: Hard stop loss as fraction (e.g. 0.05 = 5%). 0 = use pattern SL.
        use_close_price: If True, buy and sell at close price instead of open/SL/TP.
        use_stop_limit: If True (with use_close_price), stop-limit order triggers
            intraday at exact stop price. If False, SL checked at close only.

    Returns:
        PortfolioBacktestResult with all trades and metrics.
    """
    from rainier.core.config import StockScreenerConfig
    config = StockScreenerConfig()

    # Load QU100 rankings
    rankings = load_rankings_from_db()
    top100 = rankings[rankings["ranking_type"] == "top100"]
    top100 = top100[top100["long_short"] == "Long in"]

    # Pre-filter: only keep stocks that were ever in top 20 (candidates for entry)
    top100 = top100[top100["rank"] <= 20]

    all_dates = sorted(top100["data_date"].unique())
    all_symbols = sorted(top100["symbol"].unique())

    if start_date_str:
        from datetime import datetime as dt
        start_dt = dt.strptime(start_date_str, "%Y-%m-%d").date()
        all_dates = [d for d in all_dates if d >= start_dt]

    if len(all_dates) < 2:
        raise ValueError(f"Need at least 2 dates, got {len(all_dates)}")

    start_date = all_dates[0]
    end_date = all_dates[-1]

    log.info(
        "portfolio_backtest_setup",
        dates=len(all_dates),
        symbols=len(all_symbols),
        start=str(start_date),
        end=str(end_date),
        capital=start_capital,
        max_positions=max_positions,
    )

    # Fetch prices for all symbols + SPY (with extra history for pattern detection)
    symbols_with_bench = list(set(all_symbols + ["SPY"]))
    price_start = start_date - timedelta(days=180)
    price_data = fetch_all_prices(symbols_with_bench, price_start, end_date)

    # Build date-indexed price lookups
    if isinstance(price_data.columns, pd.MultiIndex):
        open_prices = price_data["Open"]
        high_prices = price_data["High"]
        low_prices = price_data["Low"]
        close_prices = price_data["Close"]
    else:
        sym = all_symbols[0]
        open_prices = price_data[["Open"]].rename(columns={"Open": sym})
        high_prices = price_data[["High"]].rename(columns={"High": sym})
        low_prices = price_data[["Low"]].rename(columns={"Low": sym})
        close_prices = price_data[["Close"]].rename(columns={"Close": sym})

    price_dates = [d.date() for d in open_prices.index]
    date_to_idx = {d: i for i, d in enumerate(price_dates)}

    # State
    cash = start_capital
    positions: list[Position] = []
    closed_trades: list[ClosedTrade] = []
    equity_curve = [start_capital]
    equity_dates: list[date] = []

    # Pending entries: signal on day T → buy at day T+1 open
    pending_entries: list[dict] = []

    log.info("starting_simulation", trading_days=len(all_dates))

    for day_i, current_date in enumerate(all_dates):
        idx = date_to_idx.get(current_date)
        if idx is None:
            # Not a trading day in price data, skip
            equity_curve.append(equity_curve[-1])
            equity_dates.append(current_date)
            continue

        # --- EXECUTE PENDING ENTRIES (normal mode: buy at today's open) ---
        if not use_close_price:
            for entry in pending_entries:
                sym = entry["symbol"]
                if sym not in open_prices.columns:
                    continue
                entry_price = open_prices.iloc[idx].get(sym)
                if pd.isna(entry_price) or entry_price <= 0:
                    continue
                if entry["target_price"] <= entry_price:
                    continue
                if entry["stop_loss"] >= entry_price:
                    continue
                if len(positions) >= max_positions:
                    break
                portfolio_value = cash + _positions_value(positions, close_prices, idx)
                alloc = portfolio_value / max_positions
                if alloc > cash:
                    alloc = cash
                if alloc <= 0:
                    continue
                shares = alloc / entry_price
                sl = entry_price * (1 - hard_stop_pct) if hard_stop_pct > 0 else entry["stop_loss"]
                positions.append(Position(
                    symbol=sym, pattern_type=entry["pattern_type"],
                    entry_date=current_date, entry_price=entry_price,
                    shares=shares, allocated_amount=alloc,
                    stop_loss=sl, target_price=entry["target_price"],
                    confidence=entry["confidence"], qu100_rank=entry["qu100_rank"],
                ))
                cash -= alloc
            pending_entries = []

        # --- CHECK EXITS for open positions (skip same-day entries) ---
        exits_today: list[int] = []  # indices to remove
        for pos_i, pos in enumerate(positions):
            if pos.entry_date == current_date:
                continue  # Don't exit on entry day
            if pos.symbol not in close_prices.columns:
                continue

            day_low = low_prices.iloc[idx].get(pos.symbol)
            day_high = high_prices.iloc[idx].get(pos.symbol)
            day_close = close_prices.iloc[idx].get(pos.symbol)

            if pd.isna(day_low) or pd.isna(day_high) or pd.isna(day_close):
                continue

            exit_price = None
            exit_reason = None

            if use_close_price:
                if use_stop_limit and hard_stop_pct > 0 and day_low <= pos.stop_loss:
                    # Stop-limit order: triggers intraday at exact stop price
                    exit_price = pos.stop_loss
                    exit_reason = "stop_loss"
                elif not use_stop_limit and hard_stop_pct > 0:
                    # Close-only: check close price against hard stop
                    close_ret = (day_close - pos.entry_price) / pos.entry_price
                    if close_ret <= -hard_stop_pct:
                        exit_price = day_close
                        exit_reason = "stop_loss"
                if exit_price is None and day_close >= pos.target_price:
                    exit_price = day_close
                    exit_reason = "target_hit"
            else:
                # Intraday mode: SL on low, TP on high
                if day_low <= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = "stop_loss"
                elif day_high >= pos.target_price:
                    exit_price = pos.target_price
                    exit_reason = "target_hit"

            # Check max hold period
            if exit_price is None and max_hold_days > 0:
                entry_idx = date_to_idx.get(pos.entry_date, 0)
                days_held = idx - entry_idx
                if days_held >= max_hold_days:
                    exit_price = day_close
                    exit_reason = "max_hold"

            if exit_price is None:
                # Check pattern invalidation: re-run detection on recent data
                sym_df = _extract_symbol_ohlcv(
                    price_data, pos.symbol,
                    end_date=open_prices.index[idx],
                )
                if sym_df is not None and len(sym_df) >= config.min_pattern_bars:
                    detected = detect_patterns_fn(
                        pos.symbol, sym_df, config,
                        pattern_filter=ALLOWED_PATTERNS,
                    )
                    still_valid = any(
                        p.pattern_type == pos.pattern_type for p in detected
                    )
                    if not still_valid:
                        exit_price = day_close
                        exit_reason = "pattern_invalidated"

            if exit_price is not None and exit_reason is not None:
                ret = (exit_price - pos.entry_price) / pos.entry_price
                pnl = pos.shares * (exit_price - pos.entry_price)

                closed_trades.append(ClosedTrade(
                    symbol=pos.symbol,
                    pattern_type=pos.pattern_type,
                    entry_date=pos.entry_date,
                    entry_price=pos.entry_price,
                    exit_date=current_date,
                    exit_price=exit_price,
                    shares=pos.shares,
                    allocated_amount=pos.allocated_amount,
                    stop_loss=pos.stop_loss,
                    target_price=pos.target_price,
                    confidence=pos.confidence,
                    exit_reason=exit_reason,
                    return_pct=ret,
                    pnl=pnl,
                    qu100_rank=pos.qu100_rank,
                ))
                cash += pos.shares * exit_price
                exits_today.append(pos_i)

                log.debug(
                    "position_closed",
                    symbol=pos.symbol,
                    date=str(current_date),
                    reason=exit_reason,
                    return_pct=f"{ret:.2%}",
                    pnl=f"{pnl:.2f}",
                )

        # Remove exited positions (reverse order to preserve indices)
        for i in sorted(exits_today, reverse=True):
            positions.pop(i)

        # --- CHECK ENTRIES (if we have room) ---
        open_slots = max_positions - len(positions) - len(pending_entries)
        if open_slots > 0:
            # Get today's QU100 top100 Long in stocks
            day_stocks = top100[top100["data_date"] == current_date].copy()
            if not day_stocks.empty:
                # Exclude stocks we already hold
                held_symbols = {p.symbol for p in positions}
                pending_symbols = {e["symbol"] for e in pending_entries}
                day_stocks = day_stocks[
                    ~day_stocks["symbol"].isin(held_symbols | pending_symbols)
                ]

                # Run pattern detection on candidates (sorted by rank, top candidates first)
                candidates: list[dict] = []
                for _, row in day_stocks.nsmallest(50, "rank").iterrows():
                    sym = row["symbol"]
                    sym_df = _extract_symbol_ohlcv(
                        price_data, sym,
                        end_date=open_prices.index[idx],
                    )
                    if sym_df is None or len(sym_df) < config.min_pattern_bars:
                        continue

                    detected = detect_patterns_fn(
                        sym, sym_df, config,
                        pattern_filter=ALLOWED_PATTERNS,
                    )
                    if not detected:
                        continue

                    # Take the best pattern for this symbol
                    best = detected[0]
                    candidates.append({
                        "symbol": sym,
                        "pattern_type": best.pattern_type,
                        "confidence": best.confidence,
                        "stop_loss": best.stop_loss,
                        "target_price": best.target_wave1,
                        "qu100_rank": int(row["rank"]),
                    })

                # Rank by confidence, take top_n
                candidates.sort(key=lambda c: c["confidence"], reverse=True)
                for entry in candidates[:min(top_n, open_slots)]:
                    if use_close_price:
                        # Buy at today's close immediately
                        sym = entry["symbol"]
                        ep = close_prices.iloc[idx].get(sym)
                        if pd.isna(ep) or ep <= 0:
                            continue
                        if entry["target_price"] <= ep:
                            continue
                        if len(positions) >= max_positions:
                            break
                        pv = cash + _positions_value(positions, close_prices, idx)
                        alloc = pv / max_positions
                        if alloc > cash:
                            alloc = cash
                        if alloc <= 0:
                            continue
                        shares = alloc / ep
                        sl = ep * (1 - hard_stop_pct) if hard_stop_pct > 0 else entry["stop_loss"]
                        positions.append(Position(
                            symbol=sym, pattern_type=entry["pattern_type"],
                            entry_date=current_date, entry_price=ep,
                            shares=shares, allocated_amount=alloc,
                            stop_loss=sl, target_price=entry["target_price"],
                            confidence=entry["confidence"],
                            qu100_rank=entry["qu100_rank"],
                        ))
                        cash -= alloc
                    else:
                        pending_entries.append(entry)
                    log.debug(
                        "entry_signal",
                        symbol=entry["symbol"],
                        date=str(current_date),
                        pattern=entry["pattern_type"],
                        confidence=f"{entry['confidence']:.2f}",
                        rank=entry["qu100_rank"],
                    )

        # --- UPDATE EQUITY ---
        portfolio_value = cash + _positions_value(positions, close_prices, idx)
        equity_curve.append(portfolio_value)
        equity_dates.append(current_date)

        if day_i % 100 == 0:
            log.info(
                "simulation_progress",
                day=day_i,
                total=len(all_dates),
                portfolio=f"${portfolio_value:.2f}",
                positions=len(positions),
                trades=len(closed_trades),
            )

    # Close any remaining positions at last close
    last_idx = date_to_idx.get(all_dates[-1])
    if last_idx is not None:
        for pos in positions:
            if pos.symbol not in close_prices.columns:
                continue
            last_close = close_prices.iloc[last_idx].get(pos.symbol)
            if pd.isna(last_close):
                continue
            ret = (last_close - pos.entry_price) / pos.entry_price
            pnl = pos.shares * (last_close - pos.entry_price)
            closed_trades.append(ClosedTrade(
                symbol=pos.symbol,
                pattern_type=pos.pattern_type,
                entry_date=pos.entry_date,
                entry_price=pos.entry_price,
                exit_date=all_dates[-1],
                exit_price=last_close,
                shares=pos.shares,
                allocated_amount=pos.allocated_amount,
                stop_loss=pos.stop_loss,
                target_price=pos.target_price,
                confidence=pos.confidence,
                exit_reason="end_of_backtest",
                return_pct=ret,
                pnl=pnl,
                qu100_rank=pos.qu100_rank,
            ))
            cash += pos.shares * last_close
        positions.clear()

    # Compute metrics
    result = _compute_result(
        closed_trades, equity_curve, equity_dates,
        start_capital, start_date, end_date,
        max_positions, top_n,
    )
    result.max_hold_days = max_hold_days
    result.hard_stop_pct = hard_stop_pct
    result.use_close_price = use_close_price
    result.use_stop_limit = use_stop_limit

    # SPY benchmark
    if "SPY" in open_prices.columns:
        spy_start_idx = date_to_idx.get(start_date)
        if spy_start_idx is not None:
            spy_start = open_prices["SPY"].iloc[spy_start_idx]
            spy_end_idx = date_to_idx.get(end_date, len(price_dates) - 1)
            spy_end = close_prices["SPY"].iloc[spy_end_idx]
            if not pd.isna(spy_start) and spy_start > 0:
                result.benchmark_return_pct = float((spy_end - spy_start) / spy_start * 100)
                result.alpha_pct = result.total_return_pct - result.benchmark_return_pct

    return result


def _positions_value(
    positions: list[Position], close_prices: pd.DataFrame, idx: int,
) -> float:
    """Compute mark-to-market value of all open positions."""
    total = 0.0
    for pos in positions:
        if pos.symbol not in close_prices.columns:
            continue
        price = close_prices.iloc[idx].get(pos.symbol)
        if not pd.isna(price):
            total += pos.shares * price
    return total


def _compute_result(
    trades: list[ClosedTrade],
    equity_curve: list[float],
    equity_dates: list[date],
    start_capital: float,
    start_date: date,
    end_date: date,
    max_positions: int,
    top_n: int,
) -> PortfolioBacktestResult:
    """Compute aggregate metrics from closed trades."""
    result = PortfolioBacktestResult(
        trades=trades,
        total_trades=len(trades),
        start_capital=start_capital,
        final_capital=equity_curve[-1] if equity_curve else start_capital,
        start_date=start_date,
        end_date=end_date,
        max_positions=max_positions,
        top_n=top_n,
        equity_curve=equity_curve,
        equity_dates=equity_dates,
    )

    if not trades:
        return result

    returns = [t.return_pct for t in trades]
    winners = [r for r in returns if r > 0]

    result.win_rate = len(winners) / len(returns)
    result.avg_return_pct = float(np.mean(returns) * 100)
    result.median_return_pct = float(np.median(returns) * 100)
    result.total_return_pct = float(
        (result.final_capital - start_capital) / start_capital * 100
    )

    # Sharpe from equity curve daily returns
    if len(equity_curve) > 2:
        eq = np.array(equity_curve[1:])  # skip initial capital
        daily_returns = np.diff(eq) / eq[:-1]
        valid = daily_returns[~np.isnan(daily_returns)]
        if len(valid) > 1 and np.std(valid) > 0:
            result.sharpe_ratio = float(
                np.mean(valid) / np.std(valid) * math.sqrt(252)
            )

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
    result.max_drawdown_pct = max_dd

    # Exit reason breakdown
    result.exit_by_stop_loss = sum(1 for t in trades if t.exit_reason == "stop_loss")
    result.exit_by_target = sum(1 for t in trades if t.exit_reason == "target_hit")
    result.exit_by_pattern = sum(1 for t in trades if t.exit_reason == "pattern_invalidated")
    result.exit_by_max_hold = sum(1 for t in trades if t.exit_reason == "max_hold")

    return result


# ---------------------------------------------------------------------------
# Trading log persistence
# ---------------------------------------------------------------------------


def save_trading_log(result: PortfolioBacktestResult) -> str:
    """Save all trades to the backtest_trading_log table. Returns run_id."""
    run_id = uuid.uuid4().hex[:12]

    with get_session() as db:
        for t in result.trades:
            record = BacktestTradingLog(
                backtest_run_id=run_id,
                symbol=t.symbol,
                pattern_type=t.pattern_type,
                entry_date=t.entry_date,
                entry_price=float(t.entry_price),
                exit_date=t.exit_date,
                exit_price=float(t.exit_price) if t.exit_price is not None else None,
                shares=float(t.shares),
                allocated_amount=float(t.allocated_amount),
                stop_loss=float(t.stop_loss),
                target_price=float(t.target_price),
                confidence=float(t.confidence),
                exit_reason=t.exit_reason,
                return_pct=float(t.return_pct) if t.return_pct is not None else None,
                pnl=float(t.pnl) if t.pnl is not None else None,
                qu100_rank=int(t.qu100_rank),
            )
            db.add(record)
        db.commit()

    log.info("trading_log_saved", run_id=run_id, trades=len(result.trades))
    return run_id


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def save_trade_log_csv(result: PortfolioBacktestResult, path: str) -> None:
    """Save trade log to a CSV file."""
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "#", "symbol", "pattern_type", "entry_date", "entry_price",
            "exit_date", "exit_price", "return_pct", "pnl",
            "stop_loss", "target_price", "confidence",
            "exit_reason", "qu100_rank", "shares", "allocated_amount",
        ])
        for i, t in enumerate(result.trades, 1):
            writer.writerow([
                i, t.symbol, t.pattern_type,
                t.entry_date, f"{t.entry_price:.2f}",
                t.exit_date, f"{t.exit_price:.2f}",
                f"{t.return_pct:.4f}", f"{t.pnl:.2f}",
                f"{t.stop_loss:.2f}", f"{t.target_price:.2f}",
                f"{t.confidence:.4f}",
                t.exit_reason, t.qu100_rank,
                f"{t.shares:.4f}", f"{t.allocated_amount:.2f}",
            ])
    log.info("trade_log_csv_saved", path=path, trades=len(result.trades))


def format_portfolio_report(result: PortfolioBacktestResult) -> str:
    """Format portfolio backtest results as readable text."""
    lines = [
        "=" * 70,
        "QU100 PORTFOLIO BACKTEST — PATTERN-BASED ENTRY/EXIT",
        "=" * 70,
        "",
        f"  Period:         {result.start_date} to {result.end_date}",
        f"  Start capital:  ${result.start_capital:.2f}",
        f"  Final capital:  ${result.final_capital:.2f}",
        f"  Max positions:  {result.max_positions} (20% each)",
        f"  Entry:          Top {result.top_n} by pattern confidence",
        f"  Patterns:       {', '.join(ALLOWED_PATTERNS)}",
    ]
    if result.max_hold_days > 0:
        lines.append(f"  Max hold:       {result.max_hold_days} trading days")
    if result.hard_stop_pct > 0:
        lines.append(f"  Hard stop:      {result.hard_stop_pct:.0%} max loss")
    if result.use_close_price:
        mode = "close + stop-limit order" if result.use_stop_limit else "close price only"
        lines.append(f"  Price mode:     {mode}")
    lines += [
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
        f"  SPY benchmark:      {result.benchmark_return_pct:+.2f}%",
        f"  Alpha:              {result.alpha_pct:+.2f}%",
        "",
        "EXIT REASONS",
        "-" * 70,
        f"  Stop loss:          {result.exit_by_stop_loss:,}",
        f"  Target hit:         {result.exit_by_target:,}",
        f"  Max hold:           {result.exit_by_max_hold:,}",
        f"  Pattern invalid:    {result.exit_by_pattern:,}",
        f"  End of backtest:    "
        f"{sum(1 for t in result.trades if t.exit_reason == 'end_of_backtest'):,}",
    ]

    # Top / worst trades
    if result.trades:
        sorted_by_pnl = sorted(result.trades, key=lambda t: t.pnl, reverse=True)
        lines.extend([
            "",
            "TOP 5 TRADES",
            "-" * 70,
            f"  {'Symbol':<8} {'Pattern':<28} {'Entry':>8} {'Exit':>8} "
            f"{'Return':>8} {'PnL':>8} {'Reason':<20}",
        ])
        for t in sorted_by_pnl[:5]:
            lines.append(
                f"  {t.symbol:<8} {t.pattern_type:<28} "
                f"${t.entry_price:>7.2f} ${t.exit_price:>7.2f} "
                f"{t.return_pct:>+7.1%} ${t.pnl:>+7.2f} {t.exit_reason:<20}"
            )

        lines.extend([
            "",
            "WORST 5 TRADES",
            "-" * 70,
        ])
        for t in sorted_by_pnl[-5:]:
            lines.append(
                f"  {t.symbol:<8} {t.pattern_type:<28} "
                f"${t.entry_price:>7.2f} ${t.exit_price:>7.2f} "
                f"{t.return_pct:>+7.1%} ${t.pnl:>+7.2f} {t.exit_reason:<20}"
            )

    lines.append("=" * 70)
    return "\n".join(lines)
