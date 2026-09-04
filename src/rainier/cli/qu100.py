"""QU100 commands: stock backtests, scraping, coverage audits, fear & greed ingest."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import click
import pandas as pd

if TYPE_CHECKING:
    from rainier.backtest.qu100_backtest import PatternMatch
    from rainier.core.config import StockScreenerConfig

from rainier.cli import (
    _get_discord_backtest_webhook,
    _legacy_db_for_config,
    _run_qu_scrape,
    _send_discord_embeds,
    backtest_group,
    cli,
)


@backtest_group.command(name="portfolio")
@click.option("--capital", default=100.0, type=float, help="Starting capital in USD")
@click.option("--max-positions", default=5, type=int, help="Max concurrent positions")
@click.option("--top-n", default=2, type=int, help="Top N pattern matches to buy per day")
@click.option("--max-hold", default=0, type=int, help="Max hold days (0=unlimited)")
@click.option("--hard-stop", default=0.0, type=float,
              help="Hard stop loss pct (e.g. 0.05 = 5%%)")
@click.option("--close-price", is_flag=True, default=False,
              help="Buy/sell at close price only")
@click.option("--stop-limit", is_flag=True, default=False,
              help="Stop-limit order (intraday trigger at stop price)")
@click.option("--start-date", default=None, type=str,
              help="Start date YYYY-MM-DD (default: earliest)")
@click.option("--discord", is_flag=True, default=False, help="Send to Discord")
@click.pass_context
def backtest_portfolio(ctx, capital, max_positions, top_n, max_hold,
                       hard_stop, close_price, stop_limit, start_date, discord):
    """Portfolio backtest: QU100 + pattern entry, dynamic SL/TP/invalidation exit."""
    from rainier.analysis.stock_patterns import detect_patterns
    from rainier.backtest.qu100_portfolio import (
        format_portfolio_report,
        run_qu100_portfolio_backtest,
        save_trade_log_csv,
        save_trading_log,
    )

    hold_msg = f", max hold {max_hold}d" if max_hold > 0 else ""
    stop_msg = f", hard stop {hard_stop:.0%}" if hard_stop > 0 else ""
    close_msg = (
        ", close+stop-limit" if close_price and stop_limit
        else (", close-only" if close_price else "")
    )
    click.echo(
        f"Running portfolio backtest: ${capital:.0f} capital, "
        f"max {max_positions} positions, top {top_n} patterns"
        f"{hold_msg}{stop_msg}{close_msg}..."
    )

    result = run_qu100_portfolio_backtest(
        detect_patterns_fn=detect_patterns,
        start_capital=capital,
        max_positions=max_positions,
        top_n=top_n,
        start_date_str=start_date,
        max_hold_days=max_hold,
        hard_stop_pct=hard_stop,
        use_close_price=close_price,
        use_stop_limit=stop_limit,
    )

    report = format_portfolio_report(result)
    click.echo(report)

    # Save trading log to DB
    run_id = save_trading_log(result)
    click.echo(
        f"\nTrading log saved to DB "
        f"(run_id: {run_id}, {len(result.trades)} trades)"
    )

    # Build tag from strategy params
    from datetime import datetime as dt
    ds = dt.now().strftime("%Y%m%d")
    tag_parts = []
    if max_hold > 0:
        tag_parts.append(f"hold{max_hold}d")
    if hard_stop > 0:
        tag_parts.append(f"stop{int(hard_stop * 100)}pct")
    if close_price and stop_limit:
        tag_parts.append("stoplimit")
    elif close_price:
        tag_parts.append("close")
    if not tag_parts:
        tag_parts.append("default")
    tag = "_".join(tag_parts)

    csv_path = f"reports/pf_btest_{ds}_{tag}_trades.csv"
    save_trade_log_csv(result, csv_path)
    click.echo(f"Trade log saved to {csv_path}")

    report_path = f"reports/pf_btest_{ds}_{tag}.md"
    with open(report_path, "w") as f:
        f.write(report)
    click.echo(f"Report saved to {report_path}")

    if discord:
        settings = ctx.obj["settings"]
        webhook = _get_discord_backtest_webhook(settings)
        if webhook:
            from rainier.alerts.discord import send_discord_message
            send_discord_message(webhook, report[:1900])
            click.echo("Sent to Discord.")


@backtest_group.command(name="qu100")
@click.option("--top-n", default=20, type=int, help="Top N stocks per day")
@click.option("--hold", default=5, type=int, help="Holding period in days")
@click.option("--min-rank", default=1, type=int, help="Min rank to include")
@click.option("--max-rank", default=50, type=int, help="Max rank to include")
@click.option("--entry-delay", default=0, type=int, help="Extra days to wait before entry")
@click.option("--discord", is_flag=True, default=False, help="Send to Discord")
@click.option("--sweep", is_flag=True, default=False, help="Run full parameter sweep")
@click.option("--variations", is_flag=True, default=False, help="Run signal tuning variations")
@click.option("--short", "short_side", is_flag=True, default=False, help="Short bottom100 stocks")
@click.option("--momentum", default=0, type=int, help="Rank momentum filter (N days improvement)")
@click.option("--patterns", is_flag=True, default=False,
              help="Pattern-filtered: only trade best 3 patterns (False Breakdown W Bottom, "
                   "False Breakdown, Bull Flag)")
@click.option("--pattern-top-n", default=5, type=int,
              help="Top N pattern-matched stocks per day (with --patterns)")
@click.pass_context
def backtest_qu100(ctx, top_n, hold, min_rank, max_rank, entry_delay,
                   discord, sweep, variations, short_side, momentum,
                   patterns, pattern_top_n):
    """Backtest QU100 money flow ranking strategy."""
    settings = ctx.obj["settings"]
    webhook = _get_discord_backtest_webhook(settings) if discord else None

    if discord and not webhook:
        click.echo("No Discord webhook configured")
        return

    if sweep:
        _run_qu100_sweep(webhook)
        return

    if variations:
        _run_qu100_variations(webhook)
        return

    if short_side:
        _run_qu100_short(top_n, hold, webhook)
        return

    if momentum > 0:
        _run_qu100_momentum(top_n, hold, min_rank, max_rank, momentum, webhook)
        return

    if patterns:
        _run_qu100_pattern_backtest(pattern_top_n, hold, webhook)
        return

    from rainier.backtest.qu100_backtest import (
        format_discord_report,
        format_qu100_report,
        run_qu100_backtest,
    )

    click.echo(
        f"Running QU100 backtest: rank {min_rank}-{max_rank}, "
        f"hold {hold}d, entry_delay={entry_delay}d..."
    )
    result = run_qu100_backtest(
        top_n=top_n, holding_days=hold,
        min_rank=min_rank, max_rank=max_rank,
        entry_delay=entry_delay,
    )

    report_text = format_qu100_report(result)
    click.echo(report_text)

    if webhook:
        embeds = format_discord_report(result)
        _send_discord_embeds(webhook, embeds)
        click.echo("Report sent to Discord")


@backtest_group.command(name="audit")
@click.option(
    "--symbols", default=None,
    help="Comma-separated symbols (default: all in money_flow_snapshots)",
)
@click.option(
    "--report", "report_path", default=None,
    help="Markdown report output path (default: canonical for a full run, "
         "a -scoped report for a filtered/short-window run)",
)
@click.option(
    "--window-days", default=365, show_default=True, type=int,
    help="Trailing as-of window in calendar days (0 = all history)",
)
@click.option(
    "--window-label", default=None,
    help="Override the report window label (default: derived from corpus dates)",
)
@click.option(
    "--workers", "n_workers", default=None, type=int,
    help="Process-pool size for the per-symbol replay (default: os.cpu_count())",
)
@click.pass_context
def pattern_audit(ctx, symbols, report_path, window_days, window_label, n_workers):
    """Pattern forward-return audit over `stock_prices` (WS B).

    Faithfully replays the LIVE pattern layer as-of each trading day, attaches
    5/10/20d forward returns + a regime tag, writes a regenerable Parquet
    corpus, and renders a per-(pattern, regime, horizon) hit-rate report.
    """

    from rainier.paper.pattern_audit import render_report_markdown, run_pattern_audit

    settings = ctx.obj["settings"]
    sym_list = (
        [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None
    )
    # 0 means "all history"; map to None so the corpus spans every date.
    win_days = window_days if window_days and window_days > 0 else None

    # A SCOPED run (filtered symbols or a non-default window) writes a distinct
    # corpus file AND a distinct report (unless --report is explicit) so it
    # doesn't silently clobber the canonical full-universe cache + checked-in
    # report that later consumers read.
    scoped = sym_list is not None or window_days != 365
    corpus_filename = "corpus-scoped.parquet" if scoped else None
    if report_path is None:
        report_path = (
            "docs/REPORT-qu100-pattern-hit-rate-scoped.md"
            if scoped
            else "docs/REPORT-qu100-pattern-hit-rate.md"
        )

    click.echo("Running QU100 pattern forward-return audit over stock_prices...")
    corpus, agg, corpus_file, derived_label = run_pattern_audit(
        config=settings.stock_screener, symbols=sym_list, window_days=win_days,
        corpus_filename=corpus_filename, n_workers=n_workers,
    )
    click.echo(f"Corpus: {len(corpus)} emissions → {corpus_file}")

    # Use the REQUESTED-window label from run_pattern_audit (states the true
    # scan cutoff even when the early window is emission-free); --window-label
    # only overrides it when given explicitly.
    label = window_label if window_label is not None else derived_label

    md = render_report_markdown(corpus, agg, window_label=label)
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Explicit UTF-8: the report carries non-ASCII glyphs (⚠ → —); the locale
    # default would UnicodeEncodeError on cp1252 / non-UTF-8 CI shells.
    out.write_text(md, encoding="utf-8")
    click.echo(f"Report → {out}")


def _run_qu100_sweep(webhook: str | None) -> None:
    """Run full parameter sweep and optionally send to Discord."""
    from rainier.backtest.qu100_backtest import (
        format_sweep_discord,
        format_sweep_table,
        run_parameter_sweep,
    )

    click.echo("Running QU100 parameter sweep (rank ranges x hold periods)...")
    rows = run_parameter_sweep()

    click.echo(format_sweep_table(rows))

    if webhook:
        embeds = format_sweep_discord(rows)
        _send_discord_embeds(webhook, embeds)
        click.echo("Sweep results sent to Discord")


def _run_qu100_variations(webhook: str | None) -> None:
    """Run all signal tuning variations and compare."""
    from rainier.backtest.qu100_backtest import (
        format_variation_comparison,
        format_variation_discord,
        result_to_variation,
        run_qu100_backtest,
        run_qu100_backtest_short,
        run_qu100_backtest_with_momentum,
    )

    variations = []

    # Baseline: top 20, 5d hold, rank 1-50
    click.echo("[1/8] Baseline: top 20, rank 1-50, 5d hold...")
    try:
        r = run_qu100_backtest(top_n=20, holding_days=5, min_rank=1, max_rank=50)
        variations.append(result_to_variation("Baseline (1-50, 5d)", r))
    except Exception as e:
        click.echo(f"  Failed: {e}")

    # Sweet spot from prior analysis: rank 6-10
    click.echo("[2/8] Sweet spot: rank 6-10, 5d hold...")
    try:
        r = run_qu100_backtest(top_n=10, holding_days=5, min_rank=6, max_rank=10)
        variations.append(result_to_variation("Sweet spot (6-10, 5d)", r))
    except Exception as e:
        click.echo(f"  Failed: {e}")

    # Tight top: rank 1-10, 7d hold
    click.echo("[3/8] Top 10, 7d hold...")
    try:
        r = run_qu100_backtest(top_n=10, holding_days=7, min_rank=1, max_rank=10)
        variations.append(result_to_variation("Top 10, 7d hold", r))
    except Exception as e:
        click.echo(f"  Failed: {e}")

    # Delayed entry: skip 1 day
    click.echo("[4/8] Delayed entry (skip 1 day), rank 1-20, 5d...")
    try:
        r = run_qu100_backtest(
            top_n=20, holding_days=5, min_rank=1, max_rank=20, entry_delay=1,
        )
        variations.append(result_to_variation("Delay 1d (1-20, 5d)", r))
    except Exception as e:
        click.echo(f"  Failed: {e}")

    # Delayed entry: skip 2 days
    click.echo("[5/8] Delayed entry (skip 2 days), rank 1-20, 5d...")
    try:
        r = run_qu100_backtest(
            top_n=20, holding_days=5, min_rank=1, max_rank=20, entry_delay=2,
        )
        variations.append(result_to_variation("Delay 2d (1-20, 5d)", r))
    except Exception as e:
        click.echo(f"  Failed: {e}")

    # Rank momentum: only stocks improving over 3 days
    click.echo("[6/8] Rank momentum (3d improving), rank 1-20, 5d...")
    try:
        r = run_qu100_backtest_with_momentum(
            top_n=20, holding_days=5, min_rank=1, max_rank=20, rank_improve_days=3,
        )
        variations.append(result_to_variation("Momentum 3d (1-20, 5d)", r))
    except Exception as e:
        click.echo(f"  Failed: {e}")

    # Longer hold: 10d
    click.echo("[7/8] Longer hold: rank 1-20, 10d...")
    try:
        r = run_qu100_backtest(top_n=20, holding_days=10, min_rank=1, max_rank=20)
        variations.append(result_to_variation("Long hold (1-20, 10d)", r))
    except Exception as e:
        click.echo(f"  Failed: {e}")

    # Short side
    click.echo("[8/8] Short bottom100, 5d hold...")
    try:
        r = run_qu100_backtest_short(top_n=20, holding_days=5)
        variations.append(result_to_variation("Short bottom100 (5d)", r))
    except Exception as e:
        click.echo(f"  Failed: {e}")

    if not variations:
        click.echo("All variations failed!")
        return

    click.echo(format_variation_comparison(variations))

    if webhook:
        embeds = format_variation_discord(variations)
        _send_discord_embeds(webhook, embeds)
        click.echo("Variation comparison sent to Discord")


def _run_qu100_short(top_n: int, hold: int, webhook: str | None) -> None:
    """Run short-side backtest."""
    from rainier.backtest.qu100_backtest import (
        format_discord_report,
        format_qu100_report,
        run_qu100_backtest_short,
    )

    click.echo(f"Running short-side backtest: top {top_n}, hold {hold}d...")
    result = run_qu100_backtest_short(top_n=top_n, holding_days=hold)

    click.echo(format_qu100_report(result))

    if webhook:
        embeds = format_discord_report(result)
        _send_discord_embeds(webhook, embeds)
        click.echo("Short report sent to Discord")


def _run_qu100_momentum(
    top_n: int, hold: int, min_rank: int, max_rank: int,
    momentum_days: int, webhook: str | None,
) -> None:
    """Run momentum-filtered backtest."""
    from rainier.backtest.qu100_backtest import (
        format_discord_report,
        format_qu100_report,
        run_qu100_backtest_with_momentum,
    )

    click.echo(
        f"Running momentum backtest: rank {min_rank}-{max_rank}, "
        f"hold {hold}d, momentum {momentum_days}d..."
    )
    result = run_qu100_backtest_with_momentum(
        top_n=top_n, holding_days=hold,
        min_rank=min_rank, max_rank=max_rank,
        rank_improve_days=momentum_days,
    )

    click.echo(format_qu100_report(result))

    if webhook:
        embeds = format_discord_report(result)
        _send_discord_embeds(webhook, embeds)
        click.echo("Momentum report sent to Discord")


def _extract_symbol_frame(price_data: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Extract one symbol's OHLCV frame from a (possibly MultiIndex) yf download."""
    if isinstance(price_data.columns, pd.MultiIndex):
        return pd.DataFrame({
            "open": price_data["Open"][sym],
            "high": price_data["High"][sym],
            "low": price_data["Low"][sym],
            "close": price_data["Close"][sym],
            "volume": price_data["Volume"][sym],
        }).dropna()
    return pd.DataFrame({
        "open": price_data["Open"],
        "high": price_data["High"],
        "low": price_data["Low"],
        "close": price_data["Close"],
        "volume": price_data["Volume"],
    }).dropna()


def _detect_symbol_patterns(
    args: tuple[str, pd.DataFrame, StockScreenerConfig],
) -> tuple[str, list[PatternMatch], str | None]:
    """Process-pool worker: detect BEST_PATTERNS matches for one symbol.

    Returns ``(symbol, matches, error)`` — exceptions are captured per symbol
    so one bad ticker never aborts the pool, matching the sequential behavior.
    """
    sym, sym_df, config = args
    from rainier.analysis.stock_patterns import detect_patterns
    from rainier.backtest.qu100_backtest import BEST_PATTERNS, PatternMatch

    matches: list[PatternMatch] = []
    try:
        for p in detect_patterns(sym, sym_df, config):
            if p.pattern_type not in BEST_PATTERNS:
                continue
            end_idx = p.pattern_end_idx or p.pattern_start_idx
            if end_idx is not None and end_idx < len(sym_df):
                matches.append(PatternMatch(
                    symbol=sym,
                    pattern_type=p.pattern_type,
                    confidence=p.confidence,
                    signal_date=sym_df.index[end_idx].date(),
                ))
    except Exception as exc:
        return sym, [], str(exc)
    return sym, matches, None


def _run_qu100_pattern_backtest(
    top_n: int, hold: int, webhook: str | None,
) -> None:
    """Run pattern-filtered QU100 backtest (composition root wiring)."""
    from concurrent.futures import ProcessPoolExecutor

    import yfinance as yf

    from rainier.backtest.qu100_backtest import (
        BEST_PATTERNS,
        format_discord_report,
        format_pattern_report,
        load_rankings_from_db,
        run_qu100_pattern_backtest,
    )
    from rainier.core.config import StockScreenerConfig

    click.echo(
        f"Running pattern-filtered QU100 backtest: "
        f"top {top_n}, hold {hold}d, patterns={BEST_PATTERNS}..."
    )

    # Step 1: Load QU100 universe from DB
    rankings = load_rankings_from_db()
    top100 = rankings[rankings["ranking_type"] == "top100"]
    top100 = top100[top100["long_short"] == "Long in"]
    all_symbols = sorted(top100["symbol"].unique())
    all_dates = sorted(top100["data_date"].unique())

    click.echo(f"  Universe: {len(all_symbols)} symbols, {len(all_dates)} dates")

    # Step 2: Fetch daily price data for pattern detection
    from datetime import timedelta
    start = all_dates[0] - timedelta(days=180)  # extra history for pattern detection
    end = all_dates[-1] + timedelta(days=30)

    click.echo(f"  Fetching daily prices for {len(all_symbols)} symbols...")
    price_data = yf.download(
        " ".join(all_symbols),
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
    )

    # Step 3: Run pattern detection per symbol on a process pool. Detection is
    # pure CPU work over an already-downloaded frame, so symbols fan out safely;
    # results are collected in submission (sorted-symbol) order, keeping the
    # match list — and thus the backtest — deterministic.
    config = StockScreenerConfig()
    pattern_matches: list[PatternMatch] = []

    tasks = []
    for sym in all_symbols:
        sym_df = _extract_symbol_frame(price_data, sym)
        if len(sym_df) >= config.min_pattern_bars:
            tasks.append((sym, sym_df, config))

    click.echo(f"  Detecting patterns on {len(tasks)} symbols...")
    with ProcessPoolExecutor() as pool:
        for sym, matches, error in pool.map(_detect_symbol_patterns, tasks):
            if error is not None:
                click.echo(f"  Warning: {sym} pattern detection failed: {error}")
            pattern_matches.extend(matches)

    click.echo(f"  Found {len(pattern_matches)} pattern matches across {BEST_PATTERNS}")

    if not pattern_matches:
        click.echo("No pattern matches found. Cannot run backtest.")
        return

    # Step 4: Run the pattern-filtered backtest
    result = run_qu100_pattern_backtest(
        pattern_matches=pattern_matches,
        top_n=top_n,
        holding_days=hold,
        allowed_patterns=BEST_PATTERNS,
    )

    report_text = format_pattern_report(result, BEST_PATTERNS)
    click.echo(report_text)

    if webhook:
        embeds = format_discord_report(result)
        _send_discord_embeds(webhook, embeds)
        click.echo("Pattern backtest report sent to Discord")




# ---------------------------------------------------------------------------
# Scraping commands (from rainier)
# ---------------------------------------------------------------------------


@cli.group()
def scrape():
    """Data collection commands (QuantUnicorn, etc.)."""


CDP_OPTION = click.option(
    "--cdp", default=None,
    help="Connect to existing Chrome via CDP (e.g., http://localhost:9222)",
)


@scrape.command()
@click.option(
    "--session",
    type=click.Choice(["morning", "midday", "afternoon", "close"]),
    required=True,
    help="Which QU100 update session to scrape",
)
@click.option("--detail-top", default=0, help="Also scrape detail pages for top N stocks")
@click.option("--dates", default=None, help="Comma-separated dates (e.g., 2026-03-10)")
@click.option("--days-back", default=0, type=int, help="Scrape last N trading days")
@click.option("--start-date", default=None, help="Start from this date (e.g., 2024-08-05), scrape to yesterday")
@click.option("--delay", default=None, type=float, help="Seconds between fetches")
@click.option("--headed", is_flag=True, default=False, help="Run browser in headed mode")
@CDP_OPTION
@click.pass_context
def qu(ctx, session, detail_top, dates, days_back, start_date, delay, headed, cdp):
    """Scrape QuantUnicorn QU100 money flow rankings."""
    import asyncio
    import sys

    try:
        asyncio.run(_run_qu_scrape(
            session, detail_top, dates, days_back, start_date, delay, headed, cdp,
        ))
    except Exception:
        sys.exit(1)



@scrape.command(name="qu-detail")
@click.option("--symbols", required=True, help="Comma-separated list of symbols")
@click.option("--headed", is_flag=True, default=False)
@CDP_OPTION
@click.pass_context
def qu_detail(ctx, symbols, headed, cdp):
    """Scrape QuantUnicorn detail pages for specific tickers."""
    import asyncio
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    asyncio.run(_run_qu_detail(symbol_list, headed, cdp))


async def _run_qu_detail(symbols, headed, cdp):
    from rainier.scrapers import get_scraper
    from rainier.scrapers.browser import BrowserManager

    async with BrowserManager(headless=not headed, cdp_url=cdp) as browser:
        scraper = get_scraper("qu", browser)
        result = await scraper.execute(symbols=symbols)

    click.echo(f"Scrape complete: {result.records_created} records created")
    if result.duration_seconds is not None:
        click.echo(f"  Duration: {result.duration_seconds:.1f}s")


# ---------------------------------------------------------------------------
# QU DB-hygiene commands (qu-money-flow-backfill-ea9d / DESIGN v2 §3.1)
# ---------------------------------------------------------------------------


@cli.group(name="qu")
def qu_group() -> None:
    """QuantUnicorn DB-hygiene + coverage audits (read-only)."""


@qu_group.command(name="money-flow-coverage")
@click.option(
    "--asof",
    "asof_str",
    default=None,
    help="Audit anchor date (YYYY-MM-DD). Defaults to today.",
)
@click.option(
    "--lookback-days",
    default=30,
    show_default=True,
    type=int,
    help="Calendar days of history to audit.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit JSON instead of the textual report (for cron + structured logs).",
)
@click.pass_context
def qu_money_flow_coverage(
    ctx, asof_str: str | None, lookback_days: int, as_json: bool,
) -> None:
    """Audit money_flow_snapshots coverage; exit non-zero on alarm.

    Reads the DB selected by the root ``--config`` flag (we thread
    ``ctx.obj["settings"]`` into ``coverage._open_session`` so
    ``rainier --config staging.yaml qu money-flow-coverage`` hits staging,
    not the default DB). Tests monkey-patch ``coverage._open_session`` to run
    the same code path against an in-memory sqlite session.
    """
    import json as _json
    import sys

    from rainier.scrapers.qu import coverage

    if asof_str:
        try:
            asof = date.fromisoformat(asof_str)
        except ValueError:
            click.echo(f"invalid --asof: {asof_str!r}", err=True)
            sys.exit(2)
    else:
        asof = date.today()

    settings = (getattr(ctx, "obj", None) or {}).get("settings")
    with coverage._open_session(settings) as session:
        report = coverage.compute_report(
            session=session, asof=asof, lookback_days=lookback_days,
        )

    if as_json:
        click.echo(_json.dumps(coverage.report_to_dict(report), indent=2))
    else:
        click.echo(coverage.render_text_report(report), nl=False)

    sys.exit(report.exit_code())


# ---------------------------------------------------------------------------
# Fear & Greed ingest
# ---------------------------------------------------------------------------


@cli.group(name="fear-greed")
def fear_greed_group() -> None:
    """CNN Fear & Greed Index ingest (point-in-time, append-on-change)."""


@fear_greed_group.command(name="backfill")
@click.option(
    "--start",
    "start_str",
    default="2020-09-21",
    show_default=True,
    help="Backfill start date (YYYY-MM-DD); earliest CNN serves is 2020-09-21.",
)
@click.pass_context
def fear_greed_backfill(ctx, start_str: str) -> None:
    """Backfill the F&G index from --start to today (source_version=backfill)."""
    from datetime import date as _date

    from rainier.data.fear_greed import backfill

    with _legacy_db_for_config(ctx):
        inserted = backfill(start=_date.fromisoformat(start_str))
    click.echo(f"fear-greed backfill: {inserted} observation(s) inserted")


@fear_greed_group.command(name="fetch")
@click.pass_context
def fear_greed_fetch(ctx) -> None:
    """Append today's F&G observation (source_version=daily, append-on-change)."""
    from rainier.data.fear_greed import fetch

    with _legacy_db_for_config(ctx):
        inserted = fetch()
    click.echo(f"fear-greed fetch: {inserted} observation(s) inserted")


# ---------------------------------------------------------------------------
# Scheduler service command
# ---------------------------------------------------------------------------

