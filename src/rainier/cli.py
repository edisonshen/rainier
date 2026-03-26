"""CLI interface: rainier — trading analysis platform."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
import pandas as pd

from rainier.core.config import load_settings
from rainier.core.types import Timeframe


@click.group()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
@click.pass_context
def cli(ctx, config_path):
    """Rainier — trading analysis platform."""
    ctx.ensure_object(dict)
    path = Path(config_path) if config_path else Path("config/settings.yaml")
    ctx.obj["settings"] = load_settings(path)


@cli.command()
@click.option("--symbol", default="MES", help="Symbol to fetch (MES, NQ, ES, GC)")
@click.option("--data-dir", default="data/csv", help="Output directory for CSV files")
@click.option(
    "--provider", "provider_type", default="auto",
    type=click.Choice(["auto", "ibkr", "yfinance"]),
    help="Data source: auto (IBKR→yfinance fallback), ibkr, or yfinance",
)
@click.option("--plot/--no-plot", default=False, help="Run daytrade analysis + chart after fetch")
@click.pass_context
def fetch(ctx, symbol, data_dir, provider_type, plot):
    """Fetch latest data and merge with existing CSVs."""
    from rainier.data import get_provider
    from rainier.data.persistence import save_candles

    data_path = Path(data_dir)
    tfs = [Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M5]

    source_label = {"auto": "yfinance (IBKR fallback)", "ibkr": "IBKR", "yfinance": "yfinance"}
    click.echo(f"Fetching {symbol} data via {source_label[provider_type]}...")

    provider = get_provider(provider_type)
    for tf in tfs:
        try:
            df = provider.get_candles(symbol, tf)
            if df.empty:
                click.echo(f"  {tf.value}: no data")
                continue
            count = save_candles(df, symbol, tf, data_path)
            click.echo(f"  {tf.value}: {count} candles")
        except Exception as e:
            click.echo(f"  {tf.value}: ERROR - {e}")

    if plot:
        click.echo()
        ctx.invoke(daytrade, symbol=symbol, data_dir=data_dir, output_path=None)


@cli.command()
@click.option("--symbol", required=True, help="Instrument symbol (MES, NQ, GC)")
@click.option("--timeframe", "tf", required=True, help="Timeframe (1D, 4H, 1H, 15m, 5m)")
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True), help="CSV file")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.pass_context
def scan(ctx, symbol, tf, csv_path, start, end):
    """Scan a single CSV file for pin bar setups."""
    from rainier.analysis.analyzer import analyze
    from rainier.signals.generator import generate_signals

    settings = ctx.obj["settings"]
    timeframe = Timeframe(tf)
    start_dt = datetime.strptime(start, "%Y-%m-%d") if start else None
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else None

    from rainier.data.csv_provider import CSVProvider

    provider = CSVProvider(Path(csv_path).parent)
    df = provider._read_csv(Path(csv_path), start_dt, end_dt)

    click.echo(f"Loaded {len(df)} candles for {symbol} {tf}")

    result = analyze(df, symbol, timeframe, settings.analysis)
    click.echo(f"Found {len(result.pivots)} pivots, {len(result.sr_levels)} S/R levels, "
               f"{len(result.pin_bars)} pin bars, {len(result.inside_bars)} inside bars")

    if result.bias:
        click.echo(f"Bias: {result.bias.value}")

    signals = generate_signals(result, df, settings.signal)
    click.echo(f"\nSignals: {len(signals)}")

    for sig in signals:
        side = "BUY" if sig.direction.value == "LONG" else "SELL"
        click.echo(
            f"  {side} @ {sig.entry_price:.2f} | "
            f"SL {sig.stop_loss:.2f} | TP {sig.take_profit:.2f} | "
            f"R:R {sig.rr_ratio:.1f} | Conf {sig.confidence:.0%}"
        )


@cli.command()
@click.option("--symbol", required=True, help="Instrument symbol")
@click.option("--data-dir", required=True, type=click.Path(exists=True),
              help="Directory with CSV files (MES_1D.csv, MES_1H.csv, MES_5m.csv)")
@click.option("--output", "output_path", default=None, help="Output HTML path")
@click.pass_context
def daytrade(ctx, symbol, data_dir, output_path):
    """Multi-TF day trading analysis: 1D + 1H pin bar lines applied to 5m chart."""
    from rainier.analysis.analyzer import analyze_multi_tf
    from rainier.core.config import load_watchlist
    from rainier.data.csv_provider import CSVProvider
    from rainier.signals.generator import generate_signals

    settings = ctx.obj["settings"]
    data_path = Path(data_dir)
    provider = CSVProvider(data_path)

    # Load per-symbol config from watchlist
    watchlist = load_watchlist()
    instrument = watchlist.get(symbol)
    min_touches = instrument.min_touches if instrument else 3
    click.echo(f"  {symbol}: min_touches={min_touches}")

    # Load all available timeframes
    data: dict[Timeframe, pd.DataFrame] = {}
    tf_files = {
        Timeframe.D1: f"{symbol}_1D.csv",
        Timeframe.H4: f"{symbol}_4H.csv",
        Timeframe.H1: f"{symbol}_1H.csv",
        Timeframe.M5: f"{symbol}_5m.csv",
    }

    for tf, filename in tf_files.items():
        csv_file = data_path / filename
        if csv_file.exists():
            df = provider._read_csv(csv_file, None, None)
            data[tf] = df
            click.echo(f"  Loaded {tf.value}: {len(df)} candles")

    if Timeframe.M5 not in data:
        click.echo("Error: 5m CSV required for day trading analysis")
        return

    click.echo(f"\nRunning multi-TF analysis for {symbol}...")
    result = analyze_multi_tf(data, symbol, Timeframe.M5, settings.analysis, min_touches=min_touches)

    # Count levels by source TF
    from collections import Counter
    tf_counts = Counter(
        l.source_tf.value if l.source_tf else "5m"
        for l in result.sr_levels if l.sr_type.value == "horizontal"
    )
    click.echo(f"S/R levels: {dict(tf_counts)}")
    click.echo(f"Pin bars on 5m: {len(result.pin_bars)}")
    click.echo(f"Bias: {result.bias.value if result.bias else 'neutral'}")

    # Signals
    signals = generate_signals(result, data[Timeframe.M5], settings.signal)
    click.echo(f"\nHigh-confidence signals: {len(signals)}")

    for sig in signals:
        side = "BUY" if sig.direction.value == "LONG" else "SELL"
        click.echo(
            f"  {side} @ {sig.entry_price:.2f} | "
            f"SL {sig.stop_loss:.2f} | TP {sig.take_profit:.2f} | "
            f"R:R {sig.rr_ratio:.1f} | Conf {sig.confidence:.0%}"
        )

    # Chart — tabbed view with TF switcher (5m / 1H / 4H / 1D)
    from rainier.analysis.analyzer import analyze
    from rainier.viz.charts import create_tabbed_chart

    htf_results: dict[Timeframe, AnalysisResult] = {Timeframe.M5: result}
    for tf in [Timeframe.H1, Timeframe.H4, Timeframe.D1]:
        if tf in data:
            htf_results[tf] = analyze(data[tf], symbol, tf, settings.analysis, min_touches=min_touches)

    out = Path(output_path) if output_path else Path(f"charts/{symbol}_daytrade.html")
    create_tabbed_chart(data, htf_results, Timeframe.M5, signals, out)
    click.echo(f"\nChart saved to {out}")


@cli.command()
@click.option("--symbol", required=True)
@click.option("--timeframe", "tf", required=True)
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True))
@click.option("--start", default=None)
@click.option("--end", default=None)
@click.option("--output", "output_path", default=None)
@click.pass_context
def chart(ctx, symbol, tf, csv_path, start, end, output_path):
    """Generate an interactive chart with S/R lines and pin bars."""
    from rainier.analysis.analyzer import analyze
    from rainier.signals.generator import generate_signals
    from rainier.viz.charts import create_chart

    settings = ctx.obj["settings"]
    timeframe = Timeframe(tf)
    start_dt = datetime.strptime(start, "%Y-%m-%d") if start else None
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else None

    from rainier.data.csv_provider import CSVProvider

    provider = CSVProvider(Path(csv_path).parent)
    df = provider._read_csv(Path(csv_path), start_dt, end_dt)

    result = analyze(df, symbol, timeframe, settings.analysis)
    signals = generate_signals(result, df, settings.signal)

    out = Path(output_path) if output_path else Path(f"charts/{symbol}_{tf}.html")
    create_chart(df, result, signals, out)
    click.echo(f"Chart saved to {out}")


@cli.command()
@click.option("--symbol", required=True)
@click.option("--timeframe", "tf", required=True)
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True))
@click.option("--start", default=None)
@click.option("--end", default=None)
@click.option("--capital", default=100_000.0)
@click.option("--export", "export_path", default=None, help="Export trades to CSV/Parquet (ext determines format)")
@click.option("--sweep", is_flag=True, default=False, help="Run parameter sweep instead of single backtest")
@click.option("--slippage", default=None, type=float, help="Override slippage pct (e.g. 0.0005)")
@click.option("--commission", default=None, type=float, help="Override commission per side")
@click.option("--trades", "show_trades", is_flag=True, default=False, help="Show per-trade log")
@click.option("--walk-forward", "walk_forward", is_flag=True, default=False,
              help="Run walk-forward cross-validation")
@click.option("--wf-train-bars", default=500, type=int, help="Walk-forward training window size")
@click.option("--wf-test-bars", default=100, type=int, help="Walk-forward test window size")
@click.option("--wf-step-bars", default=100, type=int, help="Walk-forward step between folds")
@click.option("--wf-mode", default="anchored", type=click.Choice(["anchored", "rolling"]),
              help="Walk-forward window mode")
@click.option("--regime-filter", "regime_filter", default=None,
              help="Comma-separated regimes: trending_up,trending_down,range_bound,high_volatility")
@click.option("--symbols", default=None, help="Comma-separated symbols for portfolio backtest")
@click.option("--data-dir", "data_dir", default=None, type=click.Path(exists=True),
              help="Directory with CSV files for portfolio mode")
@click.pass_context
def backtest(ctx, symbol, tf, csv_path, start, end, capital, export_path, sweep,
             slippage, commission, show_trades, walk_forward, wf_train_bars, wf_test_bars,
             wf_step_bars, wf_mode, regime_filter, symbols, data_dir):
    """Run a backtest on historical data."""
    from rainier.backtest.engine import run_backtest
    from rainier.backtest.report import format_report, format_trade_log, plot_equity_curve
    from rainier.signals.emitter import PinBarSignalEmitter

    settings = ctx.obj["settings"]
    timeframe = Timeframe(tf)
    start_dt = datetime.strptime(start, "%Y-%m-%d") if start else None
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else None

    from rainier.data.csv_provider import CSVProvider

    provider = CSVProvider(Path(csv_path).parent)
    df = provider._read_csv(Path(csv_path), start_dt, end_dt)

    # Build backtest config with optional overrides
    bt_config = settings.backtest
    if capital != 100_000.0:
        bt_config.initial_capital = capital
    if slippage is not None:
        bt_config.slippage_pct = slippage
    if commission is not None:
        bt_config.commission_per_trade = commission

    # Parse regime filter
    regime_set = None
    if regime_filter:
        from rainier.core.types import MarketRegime
        regime_set = {MarketRegime(r.strip()) for r in regime_filter.split(",")}
        click.echo(f"Regime filter: {[r.value for r in regime_set]}")

    def _wrap_with_regime(emitter):
        if regime_set is None:
            return emitter
        from rainier.analysis.regime import RegimeDetector
        from rainier.signals.regime_filter import RegimeFilter
        return RegimeFilter(emitter, RegimeDetector(), regime_set)

    def emitter_factory(min_conf: float, min_rr: float):
        from rainier.core.config import ScorerConfig, SignalConfig
        sig_config = SignalConfig(
            scorer=ScorerConfig(min_confidence=min_conf),
            min_rr_ratio=min_rr,
        )
        return _wrap_with_regime(
            PinBarSignalEmitter(settings.analysis, sig_config)
        )

    if symbols:
        # Portfolio backtest mode
        from rainier.backtest.portfolio import (
            format_portfolio_report,
            run_portfolio_backtest,
        )
        from rainier.data.csv_provider import CSVProvider as CSVProv

        sym_list = [s.strip() for s in symbols.split(",")]
        dir_path = Path(data_dir) if data_dir else Path(csv_path).parent

        port_data: dict[str, pd.DataFrame] = {}
        port_tfs: dict[str, Timeframe] = {}
        prov = CSVProv(dir_path)
        for sym in sym_list:
            csv_file = dir_path / f"{sym}_{tf}.csv"
            if not csv_file.exists():
                click.echo(f"Warning: {csv_file} not found, skipping {sym}")
                continue
            port_data[sym] = prov._read_csv(csv_file, start_dt, end_dt)
            port_tfs[sym] = timeframe

        if not port_data:
            click.echo("No data files found for portfolio backtest.")
            return

        emitter = _wrap_with_regime(
            PinBarSignalEmitter(settings.analysis, settings.signal)
        )
        click.echo(
            f"Running portfolio backtest: {list(port_data.keys())}, "
            f"{tf}, {sum(len(d) for d in port_data.values())} total candles..."
        )

        port_result = run_portfolio_backtest(
            port_data, port_tfs, emitter, bt_config,
        )
        click.echo(format_portfolio_report(port_result))
        return

    if walk_forward:
        # Walk-forward cross-validation mode
        from rainier.backtest.walk_forward import format_walk_forward_report, run_walk_forward
        from rainier.core.config import WalkForwardConfig

        wf_cfg = WalkForwardConfig(
            train_bars=wf_train_bars,
            test_bars=wf_test_bars,
            step_bars=wf_step_bars,
            mode=wf_mode,
        )

        click.echo(
            f"Running walk-forward: {symbol} {tf}, {len(df)} candles, "
            f"mode={wf_mode}, train={wf_train_bars}, test={wf_test_bars}, step={wf_step_bars}..."
        )

        wf_result = run_walk_forward(
            df, symbol, timeframe, emitter_factory, bt_config, wf_cfg,
        )
        click.echo(format_walk_forward_report(wf_result))

    elif sweep:
        # Parameter sweep mode
        from rainier.backtest.sweep import format_sweep_table, run_sweep

        click.echo(f"Running parameter sweep: {symbol} {tf}, {len(df)} candles...")

        sweep_result = run_sweep(
            df, symbol, timeframe, emitter_factory, bt_config,
        )
        click.echo(format_sweep_table(sweep_result))

        if export_path:
            out = Path(export_path)
            sweep_result.to_dataframe().to_csv(out, index=False)
            click.echo(f"\nSweep results saved to {out}")
    else:
        # Single backtest mode
        emitter = _wrap_with_regime(
            PinBarSignalEmitter(settings.analysis, settings.signal)
        )
        click.echo(f"Running backtest: {symbol} {tf}, {len(df)} candles...")

        metrics = run_backtest(df, symbol, timeframe, emitter, bt_config)
        click.echo(format_report(metrics))

        if show_trades:
            click.echo()
            click.echo(format_trade_log(metrics))

        eq_path = Path(f"charts/{symbol}_{tf}_equity.html")
        plot_equity_curve(metrics, eq_path)
        click.echo(f"\nEquity curve saved to {eq_path}")

        if export_path:
            from rainier.backtest.export import export_trades_csv, export_trades_parquet
            out = Path(export_path)
            if out.suffix == ".parquet":
                export_trades_parquet(metrics, out)
            else:
                export_trades_csv(metrics, out)
            click.echo(f"Trades exported to {out}")


@cli.command()
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True))
@click.option("--symbol", required=True)
@click.option("--timeframe", "tf", required=True)
@click.pass_context
def report(ctx, csv_path, symbol, tf):
    """Generate a daily report for a symbol."""
    from rainier.analysis.analyzer import analyze
    from rainier.reports.daily import generate_daily_report
    from rainier.signals.generator import generate_signals

    settings = ctx.obj["settings"]
    timeframe = Timeframe(tf)

    from rainier.data.csv_provider import CSVProvider

    provider = CSVProvider(Path(csv_path).parent)
    df = provider._read_csv(Path(csv_path), None, None)

    result = analyze(df, symbol, timeframe, settings.analysis)
    signals = generate_signals(result, df, settings.signal)

    report_text = generate_daily_report({symbol: result}, {symbol: signals})
    click.echo(report_text)


@cli.group()
def jobs():
    """Manage scheduled jobs (config/cron.yaml → system crontab)."""
    pass


@jobs.command(name="list")
def jobs_list():
    """Show all jobs from cron.yaml and their crontab status."""
    from rainier.scheduler.jobs import list_active, load_config

    config_jobs = load_config()
    active = {j["name"] for j in list_active()}

    for job in config_jobs:
        name = job["name"]
        enabled = job.get("enabled", True)
        status = "ACTIVE" if name in active else ("DISABLED" if not enabled else "NOT SYNCED")
        click.echo(f"  {name:20s} {job['schedule']:20s} {status}")
        click.echo(f"    {job.get('description', '')}")
        click.echo(f"    cmd: {job['command']}")
        click.echo()


@jobs.command(name="sync")
def jobs_sync():
    """Sync cron.yaml jobs to system crontab."""
    from rainier.scheduler.jobs import sync

    actions = sync(project_dir=Path.cwd())
    for name, action in actions.items():
        click.echo(f"  {name}: {action}")
    click.echo("Done.")


@jobs.command(name="stop")
@click.option("--name", required=True, help="Job name to remove from crontab")
def jobs_stop(name):
    """Remove a job from system crontab (keeps it in cron.yaml)."""
    from rainier.scheduler.jobs import _remove_job, list_active

    active = {j["name"] for j in list_active()}
    if name not in active:
        click.echo(f"Job '{name}' not found in crontab.")
        return
    _remove_job(name)
    click.echo(f"Removed '{name}' from crontab.")


# ---------------------------------------------------------------------------
# Alert commands
# ---------------------------------------------------------------------------


@cli.group()
def alert():
    """Send alerts manually (Discord, etc.)."""


@alert.command(name="discord")
@click.option("--dry-run", is_flag=True, default=False, help="Format and print without sending")
@click.option("--top-n", default=20, help="Max candidates to include")
@click.pass_context
def alert_discord(ctx, dry_run, top_n):
    """Send latest QU100 screening results to Discord."""
    from rainier.alerts.discord import format_stock_candidates_json, send_stock_candidates

    settings = ctx.obj["settings"]

    # TODO: Replace with real screener call once stock_screener module is built
    # from rainier.analysis.stock_screener import screen_stocks
    # candidates = screen_stocks(settings)[:top_n]
    sample_candidates = _make_sample_candidates(top_n)

    if dry_run:
        click.echo(format_stock_candidates_json(sample_candidates))
        click.echo(f"\n({len(sample_candidates)} candidates formatted, not sent)")
        return

    discord_config = settings.alerts.discord
    if not discord_config.enabled:
        click.echo("Discord alerts are disabled. Set alerts.discord.enabled=true in settings.yaml")
        return

    send_stock_candidates(sample_candidates, discord_config)
    click.echo(f"Sent {len(sample_candidates)} candidates to Discord.")


def _make_sample_candidates(n: int = 20) -> list:
    """Generate sample candidates for testing Discord formatting."""
    from rainier.core.types import StockCandidate

    # (sym, rank, chg, sector, pat, dir, status, conf,
    #  entry, sl, tp, rr, vol)
    samples = [
        ("NVDA", 1, 3, "Technology", "w_bottom",
         "bullish", "confirmed", 0.85,
         142.50, 135.00, 165.00, 3.0, True),
        ("TSLA", 5, -2, "Consumer Cyclical", "bull_flag",
         "bullish", "forming", 0.72,
         285.00, 270.00, 320.00, 2.3, False),
        ("AAPL", 8, 1, "Technology", "hs_bottom",
         "bullish", "confirmed", 0.90,
         198.00, 190.00, 220.00, 2.75, True),
        ("AMD", 12, 5, "Technology", "false_breakdown",
         "bullish", "confirmed", 0.78,
         165.00, 158.00, 185.00, 2.86, True),
        ("AMZN", 15, 0, "Consumer Cyclical",
         None, None, None, None,
         None, None, None, None, False),
        ("META", 18, -1, "Communication Services",
         "bull_flag", "bullish", "forming", 0.65,
         520.00, 500.00, 570.00, 2.5, False),
        ("MSFT", 22, 2, "Technology", "w_bottom",
         "bullish", "forming", 0.70,
         430.00, 415.00, 465.00, 2.33, False),
        ("GOOG", 25, -3, "Communication Services",
         None, None, None, None,
         None, None, None, None, False),
        ("AVGO", 3, 7, "Technology", "hs_bottom",
         "bullish", "confirmed", 0.88,
         185.00, 175.00, 210.00, 2.5, True),
        ("CRM", 30, 1, "Technology", "false_breakdown",
         "bullish", "forming", 0.62,
         310.00, 298.00, 340.00, 2.5, False),
    ]

    candidates = []
    for i, s in enumerate(samples[:n]):
        (sym, rank, chg, sector, pat,
         pat_dir, pat_status, conf,
         entry, sl, tp, rr, vol) = s
        candidates.append(StockCandidate(
            symbol=sym, rank=rank, rank_change=chg,
            long_short="Long in",
            capital_flow_direction="+", sector=sector,
            signal_strength=0.9 - i * 0.03,
            pattern_type=pat, pattern_direction=pat_dir,
            pattern_status=pat_status,
            pattern_confidence=conf, entry_price=entry,
            stop_loss=sl, target_price=tp,
            rr_ratio=rr, volume_confirmed=vol,
        ))

    # Pad with generic candidates if needed
    while len(candidates) < n:
        idx = len(candidates)
        candidates.append(StockCandidate(
            symbol=f"SYM{idx}", rank=30 + idx, rank_change=0, long_short="Long in",
            capital_flow_direction="+", sector="Technology", signal_strength=0.5,
        ))

    return candidates


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
@click.option("--headed", is_flag=True, default=False, help="Run browser in headed mode")
@CDP_OPTION
@click.pass_context
def qu(ctx, session, detail_top, dates, days_back, headed, cdp):
    """Scrape QuantUnicorn QU100 money flow rankings."""
    import asyncio
    asyncio.run(_run_qu_scrape(session, detail_top, dates, days_back, headed, cdp))


async def _run_qu_scrape(session, detail_top, dates, days_back, headed, cdp):
    from datetime import date, timedelta

    from rainier.scrapers import get_scraper
    from rainier.scrapers.browser import BrowserManager

    date_list = None
    if dates:
        date_list = [d.strip() for d in dates.split(",")]
    elif days_back > 0:
        date_list = []
        d = date.today()
        while len(date_list) < days_back:
            d -= timedelta(days=1)
            if d.weekday() < 5:
                date_list.append(d.isoformat())
        date_list.reverse()

    if date_list:
        click.echo(f"Scraping {len(date_list)} dates: {', '.join(date_list)}")

    async with BrowserManager(headless=not headed, cdp_url=cdp) as browser:
        scraper = get_scraper("qu", browser)
        result = await scraper.execute(
            session=session, top_n=detail_top, dates=date_list,
        )

    click.echo(f"Scrape complete: {result.records_created} records created")
    if result.errors:
        for err in result.errors:
            click.echo(f"  - {err}")
    if result.duration_seconds is not None:
        click.echo(f"  Duration: {result.duration_seconds:.1f}s")


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
# Scheduler service command
# ---------------------------------------------------------------------------


@cli.command(name="run")
@click.option(
    "--once", default=None,
    type=click.Choice(["morning", "midday", "afternoon", "close"]),
    help="Run a single scrape immediately instead of starting the scheduler",
)
@click.pass_context
def run_scheduler(ctx, once):
    """Start the scraper scheduler (long-running service)."""
    import asyncio

    if once:
        click.echo(f"Running one-off scrape: {once}")
        asyncio.run(_run_once(once))
    else:
        click.echo("Starting Rainier scheduler (Ctrl+C to stop)...")
        asyncio.run(_start_scheduler())


async def _start_scheduler():
    from rainier.scheduler.service import start_scheduler
    await start_scheduler()


async def _run_once(session):
    from rainier.scheduler.service import run_qu_scrape
    await run_qu_scrape(session)


# ---------------------------------------------------------------------------
# Database commands
# ---------------------------------------------------------------------------


@cli.group()
def db():
    """Database management commands."""


@db.command(name="init")
@click.pass_context
def db_init(ctx):
    """Initialize database tables and hypertables."""
    from rainier.core.database import init_db

    click.echo("Initializing database...")
    init_db()
    click.echo("Database initialized successfully.")
