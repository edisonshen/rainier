"""CLI interface: rainier — trading analysis platform."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd

from rainier.core.config import load_settings
from rainier.core.types import Timeframe
from rainier.pipeline.post_scrape import run_post_scrape_pipeline


@click.group()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
@click.pass_context
def cli(ctx, config_path):
    """Rainier — trading analysis platform."""
    ctx.ensure_object(dict)
    path = Path(config_path) if config_path else Path("config/settings.yaml")
    ctx.obj["settings"] = load_settings(path)
    # Codex P1: thesis subcommands need to honor `--config staging.yaml` so
    # they don't silently fall back to config/settings.yaml in non-default
    # environments. We stash the resolved path here; load_settings_fresh()
    # call sites pull it from ctx.obj via _settings_path(ctx).
    ctx.obj["settings_path"] = str(path)


def _settings_path(ctx) -> str:
    """Return the YAML path the user selected at the CLI root.

    Defaults to `config/settings.yaml` when ctx.obj is missing or empty
    (callsites that build a click.Context just for unit tests).
    """
    obj = getattr(ctx, "obj", None) or {}
    return obj.get("settings_path") or "config/settings.yaml"


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
    from rainier.core.types import AnalysisResult
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


@cli.command(name="backtest-pattern")
@click.option("--symbol", required=True, help="Stock ticker (AAPL, NVDA, etc.)")
@click.option("--csv", "csv_path", default=None, type=click.Path(exists=True),
              help="CSV file with daily OHLCV (fetches via yfinance if omitted)")
@click.option("--start", default=None)
@click.option("--end", default=None)
@click.option("--capital", default=100_000.0)
@click.option("--min-confidence", default=None, type=float)
@click.option("--min-rr", default=None, type=float)
@click.option("--wave-target", default="wave1",
              type=click.Choice(["wave1", "wave2"]))
@click.option("--export", "export_path", default=None)
@click.option("--trades", "show_trades", is_flag=True, default=False)
@click.pass_context
def backtest_pattern(ctx, symbol, csv_path, start, end, capital,
                     min_confidence, min_rr, wave_target,
                     export_path, show_trades):
    """Backtest 蔡森 chart patterns on daily stock data."""
    from rainier.backtest.engine import run_backtest
    from rainier.backtest.report import format_report, format_trade_log
    from rainier.core.config import BacktestConfig, PatternEmitterConfig
    from rainier.signals.pattern_emitter import PatternSignalEmitter

    settings = ctx.obj["settings"]
    timeframe = Timeframe.D1
    start_dt = datetime.strptime(start, "%Y-%m-%d") if start else None
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else None

    # Load data
    if csv_path:
        from rainier.data.csv_provider import CSVProvider
        provider = CSVProvider(Path(csv_path).parent)
        df = provider._read_csv(Path(csv_path), start_dt, end_dt)
    else:
        from rainier.data.yfinance_provider import YFinanceProvider
        provider = YFinanceProvider()
        df = provider.fetch(symbol, timeframe)
        if start_dt:
            df = df[df["timestamp"] >= start_dt]
        if end_dt:
            df = df[df["timestamp"] <= end_dt]

    # Build emitter config with overrides
    emitter_cfg = PatternEmitterConfig(wave_target=wave_target)
    if min_confidence is not None:
        emitter_cfg.min_confidence = min_confidence
    if min_rr is not None:
        emitter_cfg.min_rr_ratio = min_rr

    emitter = PatternSignalEmitter(settings.stock_screener, emitter_cfg)

    # Daily bars: recompute every bar
    bt_config = BacktestConfig(
        initial_capital=capital,
        sr_recompute_interval=1,
        max_open_positions=3,
    )

    click.echo(
        f"Running pattern backtest: {symbol} D1, {len(df)} candles..."
    )
    metrics = run_backtest(df, symbol, timeframe, emitter, bt_config)
    click.echo(format_report(metrics))

    if show_trades:
        click.echo()
        click.echo(format_trade_log(metrics))

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


def _get_discord_webhook(settings) -> str | None:
    """Get Discord webhook URL from settings (stock/scrape alerts)."""
    return settings.discord_stock_webhook_url or settings.discord_webhook_url or None


def _get_discord_backtest_webhook(settings) -> str | None:
    """Get Discord webhook URL for backtest notifications."""
    return settings.discord_backtest_webhook_url or _get_discord_webhook(settings)


def _send_discord_embeds(webhook: str, embeds: list[dict]) -> None:
    """Send embeds to Discord, splitting into batches of 10."""
    import httpx

    for i in range(0, len(embeds), 10):
        batch = embeds[i : i + 10]
        resp = httpx.post(webhook, json={"embeds": batch}, timeout=10)
        resp.raise_for_status()


@cli.command(name="backtest-portfolio")
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


@cli.command(name="backtest-qu100")
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


def _run_qu100_pattern_backtest(
    top_n: int, hold: int, webhook: str | None,
) -> None:
    """Run pattern-filtered QU100 backtest (composition root wiring)."""
    import yfinance as yf

    from rainier.analysis.stock_patterns import detect_patterns
    from rainier.backtest.qu100_backtest import (
        BEST_PATTERNS,
        PatternMatch,
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

    # Step 3: Run pattern detection on each symbol
    config = StockScreenerConfig()
    pattern_matches: list[PatternMatch] = []

    click.echo(f"  Detecting patterns on {len(all_symbols)} symbols...")
    for sym in all_symbols:
        try:
            # Extract single-symbol OHLCV
            if isinstance(price_data.columns, pd.MultiIndex):
                sym_df = pd.DataFrame({
                    "open": price_data["Open"][sym],
                    "high": price_data["High"][sym],
                    "low": price_data["Low"][sym],
                    "close": price_data["Close"][sym],
                    "volume": price_data["Volume"][sym],
                }).dropna()
            else:
                # Single symbol
                sym_df = pd.DataFrame({
                    "open": price_data["Open"],
                    "high": price_data["High"],
                    "low": price_data["Low"],
                    "close": price_data["Close"],
                    "volume": price_data["Volume"],
                }).dropna()

            if len(sym_df) < config.min_pattern_bars:
                continue

            detected = detect_patterns(sym, sym_df, config)

            # Convert to PatternMatch with dates
            for p in detected:
                if p.pattern_type not in BEST_PATTERNS:
                    continue

                # Use the pattern end bar's date as signal date
                end_idx = p.pattern_end_idx or p.pattern_start_idx
                if end_idx is not None and end_idx < len(sym_df):
                    signal_date = sym_df.index[end_idx].date()
                    pattern_matches.append(PatternMatch(
                        symbol=sym,
                        pattern_type=p.pattern_type,
                        confidence=p.confidence,
                        signal_date=signal_date,
                    ))
        except Exception as exc:
            click.echo(f"  Warning: {sym} pattern detection failed: {exc}")

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
    from rainier.alerts.discord import (
        _build_payloads,
        format_stock_candidates_json,
    )
    from rainier.analysis.stock_screener import screen_stocks

    settings = ctx.obj["settings"]

    click.echo("Running QU100 stock screener (3-layer pipeline)...")
    candidates, _ohlcv = screen_stocks(settings)
    candidates = candidates[:top_n]

    if not candidates:
        click.echo("No candidates found from screener.")
        return

    # Show summary
    with_pattern = sum(1 for c in candidates if c.pattern_type)
    click.echo(
        f"Screener returned {len(candidates)} candidates "
        f"({with_pattern} with pattern match)"
    )

    if dry_run:
        click.echo(format_stock_candidates_json(candidates))
        click.echo(f"\n({len(candidates)} candidates formatted, not sent)")
        return

    webhook = _get_discord_webhook(settings)
    if not webhook:
        click.echo("No Discord webhook configured")
        return

    import httpx
    payloads = _build_payloads(candidates)
    for payload in payloads:
        resp = httpx.post(webhook, json=payload, timeout=10)
        resp.raise_for_status()
    click.echo(f"Sent {len(candidates)} candidates to Discord.")


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


async def _run_qu_scrape(session, detail_top, dates, days_back, start_date, delay, headed, cdp):
    import asyncio
    from datetime import date, datetime, timedelta

    from rainier.core.config import get_settings
    from rainier.scrapers import get_scraper
    from rainier.scrapers.browser import BrowserManager

    settings = get_settings()

    # Override backfill delay if --delay is specified
    if delay is not None:
        settings.scraping.quantunicorn.backfill_delay_seconds = delay

    date_list = None
    if dates:
        date_list = [d.strip() for d in dates.split(",")]
    elif start_date:
        import exchange_calendars as xcals
        nyse = xcals.get_calendar("XNYS")
        start = date.fromisoformat(start_date)
        end = date.today() - timedelta(days=1)
        sessions = nyse.sessions_in_range(start.isoformat(), end.isoformat())
        date_list = [s.date().isoformat() for s in sessions]
    elif days_back > 0:
        import exchange_calendars as xcals
        nyse = xcals.get_calendar("XNYS")
        end = date.today() - timedelta(days=1)
        # Go back far enough to find N trading days
        start = end - timedelta(days=int(days_back * 1.6))
        sessions = nyse.sessions_in_range(
            start.isoformat(), end.isoformat(),
        )
        date_list = [
            s.date().isoformat() for s in sessions[-days_back:]
        ]

    if date_list:
        effective_delay = settings.scraping.quantunicorn.backfill_delay_seconds
        est_minutes = len(date_list) * effective_delay / 60
        click.echo(
            f"Scraping {len(date_list)} dates "
            f"(~{est_minutes:.0f} min at {effective_delay}s/date)"
        )
        click.echo(f"  First: {date_list[0]}, Last: {date_list[-1]}")

    try:
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

        # Notify Discord on success with errors
        if result.errors:
            _notify_scrape_discord(
                settings, session,
                title=f"QU100 Scrape Warning ({session})",
                message=(
                    f"Scraped {result.records_created} records "
                    f"with {len(result.errors)} error(s):\n"
                    + "\n".join(f"- {e}" for e in result.errors[:5])
                ),
                color=0xFFA500,  # orange
            )

        # Post-scrape: delegate to the shared pipeline so the cron CLI path
        # emits the same Discord output as the long-running ``rainier run``
        # scheduler — including the LLM thesis embeds for sessions in
        # settings.llm_thesis.enabled_sessions. Skip during backfill runs
        # (date_list is set) since those replay historical scans and would
        # spam the channel.
        #
        # Hand off via asyncio.to_thread because compute_theses_and_persist
        # wraps its async work in asyncio.run(); calling it directly from this
        # already-running event loop raises RuntimeError. The shared pipeline
        # is sync (mirrors the scheduler hand-off in scheduler/service.py).
        #
        # Wrap in try/except so a screener / pipeline failure POST scrape
        # surfaces as a partial-success warning rather than morphing the
        # already-successful scrape into a red "Scrape FAILED" alert. The
        # outer except is for actual scrape errors only.
        if result.records_created > 0 and not date_list:
            try:
                await asyncio.to_thread(
                    run_post_scrape_pipeline, settings, session,
                )
            except Exception as pipeline_exc:
                err_msg = str(pipeline_exc)
                if len(err_msg) > 400:
                    err_msg = err_msg[:400] + "..."
                click.echo(
                    f"  Post-scrape pipeline failed: {err_msg}", err=True,
                )
                _notify_scrape_discord(
                    settings, session,
                    title=f"QU100 Post-Scrape FAILED ({session})",
                    message=(
                        f"Scrape succeeded but post-scrape pipeline failed:\n"
                        f"{err_msg}"
                    ),
                    color=0xFFA500,  # orange — partial success
                )

    except Exception as exc:
        error_msg = str(exc)
        # Truncate long playwright tracebacks
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."

        click.echo(f"Scrape FAILED: {error_msg}", err=True)

        # Send failure alert to Discord
        _notify_scrape_discord(
            settings, session,
            title=f"QU100 Scrape FAILED ({session})",
            message=(
                f"**Session:** {session}\n"
                f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M PT')}\n"
                f"**Error:** {error_msg}"
            ),
            color=0xFF1744,  # red
        )
        raise


def _notify_scrape_discord(
    settings, session: str, title: str, message: str, color: int,
) -> None:
    """Send a scrape status notification to Discord."""
    import httpx

    webhook = _get_discord_webhook(settings)
    if not webhook:
        click.echo("  (no Discord webhook configured, skipping notification)")
        return

    embed = {
        "title": title,
        "description": message,
        "color": color,
    }
    try:
        resp = httpx.post(webhook, json={"embeds": [embed]}, timeout=10)
        resp.raise_for_status()
        click.echo(f"  Discord notification sent: {title}")
    except Exception as notify_exc:
        click.echo(f"  Failed to send Discord notification: {notify_exc}")


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


@cli.command()
@click.option("--dry-run", is_flag=True, default=False, help="Show what would be done without doing it")
@click.pass_context
def recover(ctx, dry_run):
    """Recover after a restart: check services and re-run missed jobs."""
    import asyncio
    asyncio.run(_recover(ctx.obj["settings"], dry_run))


RECOVER_WEBHOOK = (
    "https://discord.com/api/webhooks/1486760877867794503/"
    "A7-DOUrsQMmJfzxaZ2GRlqIPSJHrA3KXRxTUXLFOc4K4_cNRZVnwPdwdnBLPvAIkVSAk"
)


def _notify_recover(title: str, description: str, color: int = 0x3498DB):
    """Send a recovery event notification to Discord."""
    import httpx

    payload = {
        "embeds": [{
            "title": f"🔧 Recovery: {title}",
            "description": description,
            "color": color,
            "timestamp": datetime.now().astimezone().isoformat(),
        }],
    }
    try:
        httpx.post(RECOVER_WEBHOOK, json=payload, timeout=10)
    except Exception:
        pass  # Don't let notification failures block recovery


async def _recover(settings, dry_run: bool):
    """Check Chrome CDP, scheduler, and missed scrape sessions."""
    import subprocess
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import httpx

    tz = ZoneInfo(settings.app.timezone)
    now = datetime.now(tz)
    issues = []
    actions = []

    # --- 1. Check Chrome CDP ---
    click.echo("Checking Chrome CDP...")
    cdp_ok = False
    try:
        resp = httpx.get("http://127.0.0.1:9222/json/version", timeout=3)
        cdp_ok = resp.status_code == 200
    except Exception:
        pass

    if cdp_ok:
        click.echo("  Chrome CDP: running")
    else:
        issues.append("Chrome CDP not running")
        actions.append("start_cdp")
        click.echo("  Chrome CDP: DOWN")

    # --- 2. Check scheduler ---
    click.echo("Checking scheduler...")
    result = subprocess.run(
        ["pgrep", "-f", "rainier run"], capture_output=True, text=True,
    )
    scheduler_ok = result.returncode == 0

    if scheduler_ok:
        click.echo("  Scheduler: running")
    else:
        issues.append("Scheduler not running")
        actions.append("start_scheduler")
        click.echo("  Scheduler: DOWN")

    # --- 3. Check missed scrape sessions ---
    click.echo("Checking missed scrape sessions...")
    schedule = settings.scraping.schedule
    sessions_config = {
        "morning": schedule.morning,
        "midday": schedule.midday,
        "afternoon": schedule.afternoon,
        "close": schedule.close,
    }

    # Only check weekdays
    if now.weekday() >= 5:
        click.echo("  Weekend — no scrape sessions to check")
    else:
        from rainier.core.database import get_session
        from rainier.core.models import MoneyFlowSnapshot

        today = now.date()
        missed_sessions = []

        for session_name, time_str in sessions_config.items():
            hour, minute = map(int, time_str.split(":"))
            scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # Skip future sessions
            if now < scheduled_time:
                click.echo(f"  {session_name} ({time_str}): upcoming")
                continue

            # Check if data exists for this session today
            with get_session() as db:
                count = (
                    db.query(MoneyFlowSnapshot)
                    .filter(
                        MoneyFlowSnapshot.capture_session == session_name,
                        MoneyFlowSnapshot.data_date == today,
                    )
                    .count()
                )

            if count > 0:
                click.echo(f"  {session_name} ({time_str}): OK ({count} rows)")
            else:
                missed_sessions.append(session_name)
                issues.append(f"Missed {session_name} scrape")
                actions.append(f"scrape_{session_name}")
                click.echo(f"  {session_name} ({time_str}): MISSED")

    # --- 4. Check missed QU100 Discord report ---
    click.echo("Checking QU100 Discord report...")
    # The backtest-qu100 --discord runs from scheduler after morning scrape
    # We can't easily tell if it was sent, so if morning scrape was missed, re-send
    if "scrape_morning" in actions:
        actions.append("discord_report")
        issues.append("Morning Discord report likely missed")
        click.echo("  Discord report: likely MISSED (morning scrape was missed)")
    else:
        click.echo("  Discord report: likely OK")

    # --- Summary ---
    click.echo()
    if not issues:
        click.echo("All systems healthy — nothing to recover.")
        _notify_recover(
            "Health Check — All Clear",
            "All services running, no missed jobs.",
            color=0x2ECC71,
        )
        return

    click.echo(f"Found {len(issues)} issue(s):")
    for issue in issues:
        click.echo(f"  - {issue}")
    click.echo()

    if dry_run:
        click.echo("Dry run — would take these actions:")
        for action in actions:
            click.echo(f"  - {action}")
        return

    # --- Execute recovery ---
    click.echo("Recovering...")
    _notify_recover(
        "Recovery Started",
        "Issues detected:\n" + "\n".join(f"• {i}" for i in issues),
        color=0xE67E22,
    )

    uid = subprocess.getoutput("id -u")

    if "start_cdp" in actions:
        click.echo("  Starting Chrome CDP via launchd...")
        subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/com.rainier.chrome-cdp"],
            capture_output=True,
        )
        # Wait for CDP to be ready
        import asyncio
        for _ in range(15):
            await asyncio.sleep(2)
            try:
                resp = httpx.get("http://127.0.0.1:9222/json/version", timeout=2)
                if resp.status_code == 200:
                    click.echo("  Chrome CDP: started")
                    _notify_recover("Chrome CDP", "Started successfully", color=0x2ECC71)
                    break
            except Exception:
                pass
        else:
            click.echo("  Chrome CDP: FAILED to start — skipping scrapes")
            _notify_recover("Chrome CDP", "FAILED to start — aborting recovery", color=0xE74C3C)
            return

    if "start_scheduler" in actions:
        click.echo("  Starting scheduler via launchd...")
        subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/com.rainier.scheduler"],
            capture_output=True,
        )
        click.echo("  Scheduler: restarted")
        _notify_recover("Scheduler", "Restarted via launchd", color=0x2ECC71)

    # Re-run missed scrapes via CDP (Chrome is already running)
    recovered_scrapes = []
    for action in actions:
        if action.startswith("scrape_"):
            session_name = action.replace("scrape_", "")
            click.echo(f"  Running missed {session_name} scrape...")
            try:
                await _run_qu_scrape(
                    session=session_name,
                    detail_top=0,
                    dates=None,
                    days_back=0,
                    start_date=None,
                    delay=None,
                    headed=False,
                    cdp="http://127.0.0.1:9222",
                )
                click.echo(f"  {session_name} scrape: done")
                recovered_scrapes.append(session_name)
                _notify_recover(
                    f"Scrape: {session_name}",
                    f"Re-ran missed {session_name} scrape successfully",
                    color=0x2ECC71,
                )
            except Exception as exc:
                click.echo(f"  {session_name} scrape: FAILED ({exc})")
                _notify_recover(
                    f"Scrape: {session_name}",
                    f"FAILED: {exc}",
                    color=0xE74C3C,
                )

    if "discord_report" in actions:
        click.echo("  Sending QU100 Discord report...")
        try:
            from rainier.backtest.qu100_backtest import (
                format_discord_report,
                run_qu100_backtest,
            )
            result = run_qu100_backtest()
            webhook = _get_discord_webhook(settings)
            if webhook:
                embeds = format_discord_report(result)
                _send_discord_embeds(webhook, embeds)
                click.echo("  Discord report: sent")
                _notify_recover("QU100 Report", "Sent to Discord", color=0x2ECC71)
            else:
                click.echo("  Discord report: no webhook configured")
        except Exception as exc:
            click.echo(f"  Discord report: FAILED ({exc})")
            _notify_recover("QU100 Report", f"FAILED: {exc}", color=0xE74C3C)

    # --- Final summary ---
    summary_parts = []
    if "start_cdp" in actions:
        summary_parts.append("Chrome CDP restarted")
    if "start_scheduler" in actions:
        summary_parts.append("Scheduler restarted")
    if recovered_scrapes:
        summary_parts.append(f"Scrapes recovered: {', '.join(recovered_scrapes)}")
    if "discord_report" in actions:
        summary_parts.append("QU100 report re-sent")

    _notify_recover(
        "Recovery Complete",
        "\n".join(f"✓ {p}" for p in summary_parts),
        color=0x2ECC71,
    )

    click.echo()
    click.echo("Recovery complete.")


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


@db.command(name="backfill-prices")
@click.option("--years", default=5, type=int, help="Years of history to fetch")
@click.option("--batch-size", default=20, type=int, help="Symbols per yfinance batch")
@click.option("--dry-run", is_flag=True, help="Show what would be fetched without fetching")
def db_backfill_prices(years, batch_size, dry_run):
    """Backfill historical daily OHLCV for all QU100 stocks via yfinance."""
    import math
    import time

    import yfinance as yf
    from sqlalchemy import func, select

    from rainier.core.database import get_session
    from rainier.core.models import MoneyFlowSnapshot, StockPrice

    end = datetime.now()
    start = datetime(end.year - years, end.month, end.day)

    # Find QU100 symbols missing price data
    with get_session() as session:
        qu_symbols = set(
            session.execute(
                select(func.distinct(MoneyFlowSnapshot.symbol))
            ).scalars().all()
        )
        symbols_with_prices = set(
            session.execute(
                select(func.distinct(StockPrice.symbol)).where(
                    StockPrice.date >= start.isoformat()
                )
            ).scalars().all()
        )

    missing = sorted(qu_symbols - symbols_with_prices)
    has_prices = sorted(qu_symbols & symbols_with_prices)

    click.echo(f"QU100 symbols: {len(qu_symbols)}")
    click.echo(f"Already have prices: {len(has_prices)}")
    click.echo(f"Missing prices: {len(missing)}")
    click.echo(f"Date range: {start.date()} to {end.date()} ({years} years)")

    if dry_run:
        if missing:
            click.echo(f"\nWould fetch: {missing[:50]}{'...' if len(missing) > 50 else ''}")
        return

    if not missing:
        click.echo("All QU100 symbols have price data. Nothing to do.")
        return

    total_batches = math.ceil(len(missing) / batch_size)
    click.echo(f"\nFetching {len(missing)} symbols in {total_batches} batches...")

    from rainier.backtest.qu100_portfolio import _save_prices_to_db

    fetched = 0
    failed = 0
    for bi in range(0, len(missing), batch_size):
        batch = missing[bi : bi + batch_size]
        batch_num = bi // batch_size + 1
        click.echo(
            f"  Batch {batch_num}/{total_batches}: {batch[0]}..{batch[-1]} "
            f"({len(batch)} symbols)"
        )

        if batch_num > 1:
            time.sleep(2)

        try:
            yf_df = yf.download(
                " ".join(batch),
                start=str(start.date()),
                end=str(end.date()),
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if not yf_df.empty:
                if not isinstance(yf_df.columns, pd.MultiIndex) and len(batch) == 1:
                    yf_df.columns = pd.MultiIndex.from_product(
                        [yf_df.columns, batch]
                    )
                _save_prices_to_db(yf_df, batch)
                fetched += len(batch)
            else:
                failed += len(batch)
                click.echo("    No data returned for batch")
        except Exception as exc:
            failed += len(batch)
            click.echo(f"    Error: {exc}")

    click.echo(f"\nDone. Fetched: {fetched}, Failed: {failed}")


# ---------------------------------------------------------------------------
# Feature store commands
# ---------------------------------------------------------------------------


@cli.group()
def features():
    """ML feature store commands."""


@features.command(name="export")
@click.option("--start", "start_date", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", "end_date", default=None, help="End date (YYYY-MM-DD)")
@click.option("--symbols", default=None, help="Comma-separated symbols (default: all with prices)")
@click.option("--output-dir", default="data/features", help="Output directory for Parquet files")
@click.option("--min-bars", default=100, type=int, help="Minimum bars per symbol")
@click.option("--dry-run", is_flag=True, help="Show what would be exported")
def features_export(start_date, end_date, symbols, output_dir, min_bars, dry_run):
    """Export ML features + labels to Parquet for model training."""
    from rainier.ml.feature_store import (
        export_training_data,
        get_symbols_with_prices,
    )

    end = (
        datetime.strptime(end_date, "%Y-%m-%d").date()
        if end_date
        else datetime.now().date()
    )
    start = (
        datetime.strptime(start_date, "%Y-%m-%d").date()
        if start_date
        else date(end.year - 5, end.month, end.day)
    )

    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        symbol_list = get_symbols_with_prices(start, end, min_bars=min_bars)

    click.echo(f"Symbols: {len(symbol_list)}")
    click.echo(f"Date range: {start} to {end}")
    click.echo(f"Output: {output_dir}/")

    if dry_run:
        click.echo(f"\nWould export: {symbol_list[:30]}{'...' if len(symbol_list) > 30 else ''}")
        return

    if not symbol_list:
        click.echo("No symbols with sufficient price data. Run `rainier db backfill-prices` first.")
        return

    output_path = export_training_data(
        symbols=symbol_list,
        start=start,
        end=end,
        output_dir=Path(output_dir),
    )
    click.echo(f"\nExported to: {output_path}")

    # Validate
    from rainier.ml.feature_store import validate_parquet

    stats = validate_parquet(output_path)
    click.echo("\nValidation:")
    click.echo(f"  Rows: {stats['rows']:,}")
    click.echo(f"  Symbols: {stats['symbols']}")
    click.echo(f"  Features: {stats['features']}")
    click.echo(f"  Date range: {stats['date_range']}")
    click.echo(f"  Feature NaN count: {stats['feature_nan_total']}")
    for col, rate in stats["label_positive_rate"].items():
        click.echo(f"  {col} positive rate: {rate}")


@features.command(name="validate")
@click.argument("path", type=click.Path(exists=True))
def features_validate(path):
    """Validate an exported Parquet file."""
    from rainier.ml.feature_store import validate_parquet

    stats = validate_parquet(Path(path))
    for key, val in stats.items():
        if isinstance(val, dict):
            click.echo(f"{key}:")
            for k, v in val.items():
                click.echo(f"  {k}: {v}")
        else:
            click.echo(f"{key}: {val}")


# ---------------------------------------------------------------------------
# ML commands
# ---------------------------------------------------------------------------


@cli.group()
def ml():
    """Machine learning model commands."""


@ml.command(name="train")
@click.argument("features_path", type=click.Path(exists=True))
@click.option("--output-dir", default="models", help="Directory to save model")
@click.option("--label", default="label_5d", help="Label column to train on")
@click.option("--folds", default=3, type=int, help="Walk-forward CV folds")
def ml_train(features_path, output_dir, label, folds):
    """Train XGBoost pattern scorer on feature store data."""
    from rainier.ml.pattern_scorer import TrainConfig, train_model

    config = TrainConfig(label_col=label, n_folds=folds)
    click.echo(f"Training on: {features_path}")
    click.echo(f"Label: {label}, Folds: {folds}")

    model, result = train_model(
        parquet_path=Path(features_path),
        config=config,
        output_dir=Path(output_dir),
    )

    click.echo("\n--- Evaluation ---")
    click.echo(f"Accuracy:      {result.accuracy:.3f}")
    click.echo(f"Precision:     {result.precision:.3f}")
    click.echo(f"Recall:        {result.recall:.3f}")
    click.echo(f"F1:            {result.f1:.3f}")
    click.echo(f"Profit Factor: {result.profit_factor:.2f}")
    click.echo(f"Test samples:  {result.n_test} ({result.n_positive} positive)")
    click.echo(f"Fold scores:   {[f'{s:.3f}' for s in result.fold_scores]}")

    click.echo("\n--- Top 10 Features ---")
    for i, (feat, imp) in enumerate(list(result.feature_importance.items())[:10], 1):
        click.echo(f"  {i:2d}. {feat:30s} {imp:.4f}")

    click.echo(f"\nModel saved to: {output_dir}/pattern_scorer.json")


@ml.command(name="evaluate")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("features_path", type=click.Path(exists=True))
@click.option("--label", default="label_5d", help="Label column")
def ml_evaluate(model_path, features_path, label):
    """Evaluate a trained model on feature store data."""
    import xgboost as xgb_lib
    from sklearn.metrics import accuracy_score as acc_score
    from sklearn.metrics import classification_report as cls_report

    from rainier.ml.pattern_scorer import get_feature_columns

    model = xgb_lib.XGBClassifier()
    model.load_model(model_path)

    df = pd.read_parquet(features_path)
    df = df.dropna(subset=[label])
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    y = df[label].values.astype(int)
    y_pred = model.predict(X)

    click.echo(cls_report(y, y_pred, target_names=["bearish", "bullish"]))
    click.echo(f"Accuracy: {acc_score(y, y_pred):.3f}")


@ml.command(name="explain")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("features_path", type=click.Path(exists=True))
@click.option("--output", default=None, help="Save SHAP results to JSON")
def ml_explain(model_path, features_path, output):
    """Generate SHAP explanations for a trained model."""
    import xgboost as xgb_lib

    from rainier.ml.pattern_scorer import explain_model

    model = xgb_lib.XGBClassifier()
    model.load_model(model_path)

    output_path = Path(output) if output else None
    result = explain_model(model, Path(features_path), output_path)

    click.echo("--- SHAP Feature Importance (Top 10) ---")
    for feat in result["top_10"]:
        click.echo(f"  {feat:30s} {result['mean_shap'][feat]:.4f}")

    if output_path:
        click.echo(f"\nSaved to: {output_path}")


@ml.command(name="regime")
@click.argument("features_path", type=click.Path(exists=True))
@click.option("--symbols", default=None, help="Comma-separated symbols to analyze")
@click.option("--save-model", default=None, help="Save HMM model to path")
def ml_regime(features_path, symbols, save_model):
    """Fit HMM regime detector and show regime distribution."""
    from rainier.ml.regime import HMMRegimeDetector

    df = pd.read_parquet(features_path)

    if symbols:
        sym_list = [s.strip().upper() for s in symbols.split(",")]
        df = df[df["symbol"].isin(sym_list)]

    if "close" not in df.columns:
        click.echo("Error: Parquet file doesn't contain 'close' column")
        return

    # Use first symbol for fitting
    sym = df["symbol"].iloc[0]
    sym_df = df[df["symbol"] == sym].sort_values("date").reset_index(drop=True)

    # Reconstruct minimal OHLCV from features
    ohlcv = pd.DataFrame({
        "open": sym_df["close"].shift(1).fillna(sym_df["close"]),
        "high": sym_df["close"] * (1 + sym_df.get("range", 0.01).clip(lower=0.001)),
        "low": sym_df["close"] * (1 - sym_df.get("range", 0.01).clip(lower=0.001)),
        "close": sym_df["close"],
        "volume": sym_df.get("volume", 1_000_000),
    })

    detector = HMMRegimeDetector(n_states=3)
    regimes = detector.fit_predict(ohlcv)
    summary = detector.regime_summary(regimes)

    click.echo(f"\n--- HMM Regime Detection ({sym}, {len(ohlcv)} bars) ---")
    click.echo("Distribution:")
    for regime, pct in summary["pct"].items():
        count = summary["distribution"][regime]
        dur = summary["avg_duration"].get(regime, "N/A")
        click.echo(f"  {regime:20s} {pct:>6s} ({count:>4d} bars, avg duration: {dur})")

    if save_model:
        detector.save(Path(save_model))
        click.echo(f"\nModel saved to: {save_model}")


@ml.command(name="select-features")
@click.argument("features_path", type=click.Path(exists=True))
@click.option("--label", default="label_5d", help="Label column to optimize for")
@click.option("--folds", default=5, type=int, help="Walk-forward CV folds")
@click.option("--min-features", default=5, type=int, help="Minimum features to test")
@click.option("--max-features", default=40, type=int, help="Maximum features to test")
@click.option(
    "--stability", default=0.6, type=float,
    help="Stability threshold (fraction of folds a feature must appear in)",
)
@click.option(
    "--methods", default="mdi,mda",
    help="Importance methods: mdi,mda,shap (comma-separated)",
)
@click.option("--output", default=None, help="Save results to JSON")
def ml_select_features(
    features_path, label, folds, min_features, max_features, stability, methods, output,
):
    """Select optimal feature set using nested walk-forward CV.

    Runs feature importance inside each training fold (no data leakage),
    then ranks features by stability across folds. Sweeps feature counts
    to find the set that maximizes out-of-sample profit factor.
    """
    from rainier.ml.feature_selector import SelectionConfig, select_features

    method_list = [m.strip() for m in methods.split(",")]
    config = SelectionConfig(
        label_col=label,
        n_folds=folds,
        min_features=min_features,
        max_features=max_features,
        stability_threshold=stability,
        methods=method_list,
    )
    output_path = Path(output) if output else None

    click.echo(f"Feature selection: {features_path}")
    click.echo(f"Label: {label}, Folds: {folds}, Methods: {method_list}")
    click.echo(f"Stability threshold: {stability:.0%}")
    click.echo()

    result = select_features(
        parquet_path=Path(features_path),
        config=config,
        output_path=output_path,
    )

    # --- Sweep results table ---
    click.echo("--- Feature Count Sweep ---")
    click.echo(f"{'Features':>10s}  {'Accuracy':>10s}  {'Acc Std':>10s}  {'Profit Factor':>14s}")
    for s in result.sweep_results:
        pf_str = f"{s['profit_factor']:.2f}" if np.isfinite(s["profit_factor"]) else "inf"
        click.echo(
            f"{s['n_features']:>10d}  {s['accuracy']:>10.3f}  "
            f"{s['accuracy_std']:>10.3f}  {pf_str:>14s}"
        )

    # --- Stability rankings ---
    click.echo("\n--- Feature Stability (top 20) ---")
    click.echo(f"{'Feature':30s}  {'Stability':>10s}  {'Mean Rank':>10s}")
    for r in result.rankings[:20]:
        click.echo(f"{r.name:30s}  {r.stability:>10.0%}  {r.mean_rank:>10.1f}")

    # --- Selected features ---
    click.echo(f"\n--- Optimal Feature Set ({result.optimal_n} features) ---")
    for i, feat in enumerate(result.selected_features, 1):
        click.echo(f"  {i:2d}. {feat}")

    click.echo(
        f"\nTotal: {result.total_features} -> {result.optimal_n} features "
        f"({result.total_features - result.optimal_n} dropped)"
    )

    if output_path:
        click.echo(f"Results saved to: {output_path}")


@ml.command(name="compare")
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True),
              help="OHLCV CSV file")
@click.option("--symbol", required=True, help="Instrument symbol (e.g. MES)")
@click.option("--timeframe", "tf", default="1H",
              type=click.Choice([t.value for t in Timeframe]),
              help="Bar timeframe")
@click.option("--model", "model_path", default=None, type=click.Path(exists=True),
              help="Trained ML model path (XGBoost JSON). When omitted, only "
                   "the BookScorer baseline is evaluated.")
@click.option("--start", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD)")
@click.option("--walk-forward", "walk_forward", is_flag=True, default=False,
              help="Run walk-forward backtest comparison instead of full-sample.")
@click.option("--wf-train-bars", default=500, type=int,
              help="Walk-forward training window size (bars).")
@click.option("--wf-test-bars", default=100, type=int,
              help="Walk-forward test window size (bars).")
@click.option("--wf-step-bars", default=None, type=int,
              help="Walk-forward step between folds (default: --wf-test-bars).")
@click.pass_context
def ml_compare(ctx, csv_path, symbol, tf, model_path, start, end,
               walk_forward, wf_train_bars, wf_test_bars, wf_step_bars):
    """Compare BookScorer vs MLScorer side-by-side on the same data.

    Closes the ML feedback loop: takes a trained XGBoost model and runs it
    through the backtest engine alongside the rule-based BookScorer so the
    operator can answer "would this model make more money?".

    Use --walk-forward to slide a (train, test) window through the data and
    aggregate per-fold out-of-sample metrics. NOTE: this CLI does NOT refit
    the ML model per fold; the model from --model is loaded once and
    evaluated on each test window. Model retraining is offline via
    ``rainier ml train``. Lookahead leakage is therefore only as good as the
    training cut-off used to produce the model file — pass a model trained
    strictly on data before the CSV's earliest test bar.
    """
    from rainier.core.config import ScorerConfig, SignalConfig
    from rainier.data.csv_provider import CSVProvider
    from rainier.features.extractor import FeatureExtractor
    from rainier.ml.compare import (
        compare_emitters,
        format_comparison_table,
        format_walkforward_table,
        run_walkforward_compare,
    )
    from rainier.ml.scorers import MLScorer
    from rainier.signals.emitter import PinBarSignalEmitter
    from rainier.signals.ml_emitter import MLSignalEmitter

    settings = ctx.obj["settings"]
    timeframe = Timeframe(tf)
    start_dt = datetime.strptime(start, "%Y-%m-%d") if start else None
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else None

    provider = CSVProvider(Path(csv_path).parent)
    df = provider._read_csv(Path(csv_path), start_dt, end_dt)

    if df is None or df.empty:
        click.echo(f"No data loaded from {csv_path}")
        return

    bt_config = settings.backtest
    sig_config = SignalConfig(
        scorer=ScorerConfig(min_confidence=settings.signal.scorer.min_confidence),
        min_rr_ratio=settings.signal.min_rr_ratio,
    )
    extractor = FeatureExtractor()

    # Always include BookScorer baseline; ML scorer only if --model given.
    def make_book_emitter(_train_df=None) -> PinBarSignalEmitter:
        return PinBarSignalEmitter(settings.analysis, sig_config)

    ml_factory = None
    if model_path is not None:
        ml_scorer = MLScorer(model_path=Path(model_path))

        def _make_ml(_train_df=None) -> MLSignalEmitter:
            # MLScorer is loaded once and reused across folds (the model
            # itself doesn't refit; the caller refits offline via `ml train`).
            return MLSignalEmitter(
                scorer=ml_scorer,
                feature_extractor=extractor,
                analysis_config=settings.analysis,
                signal_config=sig_config,
            )

        ml_factory = _make_ml

    if walk_forward:
        factories: dict = {"BookScorer": make_book_emitter}
        if ml_factory is not None:
            factories["MLScorer"] = ml_factory

        click.echo(
            f"Walk-forward compare: train={wf_train_bars}, test={wf_test_bars}, "
            f"step={wf_step_bars or wf_test_bars}, n_bars={len(df)}"
        )
        wf_result = run_walkforward_compare(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            emitter_factories=factories,
            train_bars=wf_train_bars,
            test_bars=wf_test_bars,
            step_bars=wf_step_bars,
            config=bt_config,
        )
        click.echo(format_walkforward_table(wf_result))
        return

    emitters: dict = {"BookScorer": make_book_emitter()}
    if ml_factory is not None:
        emitters["MLScorer"] = ml_factory()

    click.echo(f"Single-window compare: n_bars={len(df)}, emitters={list(emitters)}")
    cmp_result = compare_emitters(
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        emitters=emitters,
        config=bt_config,
    )
    click.echo(format_comparison_table(cmp_result))


# ---------------------------------------------------------------------------
# `rainier thesis` — LLM thesis layer (PR1)
# ---------------------------------------------------------------------------


@cli.group()
def thesis():
    """LLM thesis layer (signals, daily run, single-ticker debug, log, signals registry)."""


@thesis.command("daily")
@click.option(
    "--session",
    "session_name",
    default="afternoon",
    type=click.Choice(["morning", "midday", "afternoon", "close"]),
    help="Which scan to attribute this run to.",
)
@click.option("--top-n", default=5, type=int, help="How many top candidates to LLM-thesis.")
@click.option("--discord/--no-discord", default=False, help="Post results to Discord.")
@click.option("--dry-run", is_flag=True, help="Skip Discord, print theses to stdout.")
@click.option("--max-usd", default=1.0, type=float, help="Hard kill switch on cumulative spend.")
@click.pass_context
def thesis_daily(ctx, session_name, top_n, discord, dry_run, max_usd):
    """Manual scheduler-hook trigger — runs the same pipeline as `rainier run`."""
    from datetime import date as _date

    from rainier.alerts.discord import send_stock_candidates
    from rainier.analysis.stock_screener import screen_stocks
    from rainier.core.config import load_settings_fresh
    from rainier.llm_thesis.persistence import persist_screened_stocks
    from rainier.llm_thesis.service import compute_theses_and_persist

    settings = load_settings_fresh(_settings_path(ctx))
    # Override the kill switch via CLI.
    settings.llm_thesis.max_usd_per_scan = float(max_usd)

    click.echo(f"Running screener for session={session_name}...")
    all_candidates, ohlcv_by_symbol = screen_stocks(settings)
    # PR2 carry-over P2 #3: persist top-50 for unbiased shadow validation;
    # display set capped at top-20; LLM thesis on top-N (default 5).
    scan_candidates = all_candidates[:50]
    candidates = all_candidates[:20]
    if not candidates:
        click.echo("No candidates from screener.")
        return

    scan_date = _date.today()
    persist_screened_stocks(
        scan_candidates, scan_date=scan_date, session_name=session_name
    )

    click.echo(f"Running LLM thesis on top {top_n} (max_usd={max_usd:.2f})...")
    theses = compute_theses_and_persist(
        candidates[:top_n],
        ohlcv_by_symbol,
        scan_date=scan_date,
        session_name=session_name,
        settings=settings,
    )

    if dry_run or not discord:
        import json as _json
        click.echo(_json.dumps(theses, indent=2, default=str))
        click.echo(f"\n({len(theses)} theses generated, Discord skipped)")
        return

    send_stock_candidates(
        candidates,
        settings.alerts.discord,
        theses=theses or None,
        # PR5: pipe the dashboard base URL through so embed titles get a
        # clickable deep-link. None disables the link.
        dashboard_base_url=settings.llm_thesis.dashboard_base_url,
        # iter-2: forward session so the summary embed title carries the
        # `— Morning/Midday/Afternoon/Close` suffix when multiple scans
        # land in the same channel on the same day.
        session=session_name,
    )
    click.echo(f"Sent {len(theses)} theses + summary to Discord.")


@thesis.command("ticker")
@click.argument("symbol")
@click.option(
    "--session", "session_name", default="afternoon",
    type=click.Choice(["morning", "midday", "afternoon", "close"]),
)
@click.option("--max-usd", default=1.0, type=float)
@click.pass_context
def thesis_ticker(ctx, symbol, session_name, max_usd):
    """Single-ticker debug pipeline against the latest QU100 snapshot."""
    from datetime import date as _date

    from rainier.analysis.stock_screener import screen_stocks
    from rainier.core.config import load_settings_fresh
    from rainier.llm_thesis.service import compute_theses_and_persist

    settings = load_settings_fresh(_settings_path(ctx))
    settings.llm_thesis.max_usd_per_scan = float(max_usd)

    click.echo(f"Running screener (will filter to {symbol})...")
    candidates, ohlcv = screen_stocks(settings)
    target = next((c for c in candidates if c.symbol.upper() == symbol.upper()), None)
    if target is None:
        click.echo(f"{symbol} not in screener output.")
        return

    theses = compute_theses_and_persist(
        [target], ohlcv,
        scan_date=_date.today(),
        session_name=session_name,
        settings=settings,
    )
    import json as _json
    click.echo(_json.dumps(theses, indent=2, default=str))


@thesis.command("log")
@click.option("--ticker", required=True)
@click.option("--date", "scan_date_s", required=True, help="Scan date YYYY-MM-DD.")
@click.option(
    "--session",
    "session_name",
    required=True,
    type=click.Choice(["morning", "midday", "afternoon", "close"]),
    help="Which scan session row to log against (afternoon, close, etc.).",
)
@click.option(
    "--action", required=True, type=click.Choice(["took", "skipped", "watched"])
)
@click.option("--outcome", "outcome", default=None, help="e.g. +2.3% or -1.1%.")
@click.option("--notes", default=None)
def thesis_log(ticker, scan_date_s, session_name, action, outcome, notes):
    """Manually record action_taken + outcome on a ScreenedStockRecord.

    Codex P1: scoped to a single (scan_date, session_name, symbol) row.
    Without the session filter, the same ticker showing up in morning and
    afternoon scans would have its outcome stamped onto every row, which
    corrupts the per-scan history PR2's outcome backfill consumes.
    """
    from datetime import date as _date
    from datetime import datetime as _datetime
    from datetime import timezone as _tz

    from sqlalchemy import update

    from rainier.core.database import get_session
    from rainier.core.models import ScreenedStockRecord

    scan_date = _date.fromisoformat(scan_date_s)

    pct: float | None = None
    if outcome is not None:
        try:
            pct = float(outcome.strip().rstrip("%"))
        except ValueError as exc:
            raise click.BadParameter(f"Bad --outcome: {exc}") from exc

    with get_session() as session:
        result = session.execute(
            update(ScreenedStockRecord)
            .where(
                ScreenedStockRecord.scan_date == scan_date,
                ScreenedStockRecord.session_name == session_name,
                ScreenedStockRecord.symbol == ticker.upper(),
            )
            .values(
                action_taken=action,
                outcome_pct=pct,
                outcome_recorded_at=_datetime.now(_tz.utc),
                notes=notes,
            )
        )
    affected = int(result.rowcount or 0)
    if affected == 0:
        raise click.ClickException(
            "No ScreenedStockRecord row found for "
            f"symbol={ticker} scan_date={scan_date_s} session={session_name}"
        )
    click.echo(
        f"Updated {affected} row(s) for {ticker} on {scan_date_s} ({session_name})."
    )


@thesis.group("signals")
def thesis_signals():
    """Inspect / toggle / test individual signals in the LLM thesis registry."""


@thesis_signals.command("list")
@click.pass_context
def thesis_signals_list(ctx):
    """Print every signal in the registry + its enabled flag from settings.yaml."""
    from rainier.core.config import load_settings_fresh
    from rainier.llm_thesis.signals import REGISTRY

    settings = load_settings_fresh(_settings_path(ctx))
    cfg_map = settings.llm_thesis.signals
    click.echo(f"{'Name':<24} {'Enabled':<8} {'Version':<8} {'Cost(ms)':<10} Weight")
    click.echo("-" * 64)
    for name, sig_cls in REGISTRY.items():
        instance = sig_cls()
        cfg = cfg_map.get(name)
        enabled = "yes" if (cfg and cfg.enabled) else "no"
        weight = cfg.weight if cfg else 1.0
        click.echo(
            f"{name:<24} {enabled:<8} {instance.version:<8} "
            f"{instance.cost_estimate_ms:<10} {weight:.2f}"
        )


def _set_signal_enabled_yaml(name: str, enabled: bool, config_path: str) -> None:
    """Toggle a signal in `config_path` via plain PyYAML.

    PR1 uses PyYAML which does not preserve comments/order — that's intentional
    for PR1 minimality. PR3 will swap in ruamel.yaml when the auto-research
    accept flow needs to mutate config without nuking layout.

    Codex P1: targets the same YAML the caller's `--config` selected so that
    `rainier --config staging.yaml thesis signals enable X` doesn't surprise-
    edit the production `config/settings.yaml`.
    """
    import yaml as _yaml
    path = Path(config_path)
    if not path.exists():
        raise click.ClickException(f"Missing {path}")
    with path.open("r") as f:
        data = _yaml.safe_load(f) or {}
    section = data.setdefault("llm_thesis", {}).setdefault("signals", {})
    if name not in section:
        section[name] = {"enabled": enabled, "params": {}, "weight": 1.0}
    else:
        section[name]["enabled"] = enabled
    with path.open("w") as f:
        _yaml.safe_dump(data, f, sort_keys=False)


@thesis_signals.command("enable")
@click.argument("name")
@click.pass_context
def thesis_signals_enable(ctx, name):
    """Flip the signal to enabled=true in settings.yaml."""
    from rainier.llm_thesis.signals import REGISTRY
    if name not in REGISTRY:
        raise click.ClickException(
            f"Unknown signal {name!r}. Known: {', '.join(REGISTRY)}"
        )
    _set_signal_enabled_yaml(name, True, _settings_path(ctx))
    click.echo(f"Enabled signal: {name}")


@thesis_signals.command("disable")
@click.argument("name")
@click.pass_context
def thesis_signals_disable(ctx, name):
    """Flip the signal to enabled=false in settings.yaml."""
    from rainier.llm_thesis.signals import REGISTRY
    if name not in REGISTRY:
        raise click.ClickException(
            f"Unknown signal {name!r}. Known: {', '.join(REGISTRY)}"
        )
    _set_signal_enabled_yaml(name, False, _settings_path(ctx))
    click.echo(f"Disabled signal: {name}")


@thesis_signals.command("test")
@click.argument("name")
@click.option("--symbol", required=True, help="Symbol to dry-run the signal against.")
@click.pass_context
def thesis_signals_test(ctx, name, symbol):
    """Dry-run a single signal's compute() against the latest QU100 snapshot."""
    from datetime import date as _date

    from rainier.analysis.stock_screener import screen_stocks
    from rainier.core.config import load_settings_fresh
    from rainier.core.types import StockCandidate
    from rainier.llm_thesis.signals import REGISTRY
    from rainier.llm_thesis.signals.base import SignalContext

    if name not in REGISTRY:
        raise click.ClickException(
            f"Unknown signal {name!r}. Known: {', '.join(REGISTRY)}"
        )

    settings = load_settings_fresh(_settings_path(ctx))
    candidates, ohlcv = screen_stocks(settings)
    target: StockCandidate | None = next(
        (c for c in candidates if c.symbol.upper() == symbol.upper()), None
    )
    if target is None:
        click.echo(f"{symbol} not in screener output — building stub candidate.")
        target = StockCandidate(
            symbol=symbol.upper(), rank=0, rank_change=0, long_short="Long in",
            capital_flow_direction="N", sector="Unknown", signal_strength=0.0,
        )

    cfg = settings.llm_thesis.signals.get(name)
    params = dict(cfg.params) if cfg else {}
    ctx = SignalContext(
        symbol=target.symbol, scan_date=_date.today(),
        session_name="manual", candidate=target,
        ohlcv_df=ohlcv.get(target.symbol), params=params,
    )
    sig = REGISTRY[name]()
    value = sig.compute(ctx)
    click.echo(f"value: {value}")
    if value is not None:
        click.echo(f"render_for_prompt: {sig.render_for_prompt(value)}")


@thesis.command("eval")
@click.option(
    "--date", "eval_date_s", default=None,
    help="Evaluation date YYYY-MM-DD; defaults to today.",
)
@click.option(
    "--horizon", "horizon", default=None,
    type=click.Choice(["1d", "5d", "10d"]),
    help="Single horizon to evaluate; default runs all three.",
)
@click.pass_context
def thesis_eval(ctx, eval_date_s, horizon):
    """Run the daily-eval job manually.

    Backfills ThesisEvaluation rows + posts the Discord eval report.
    Idempotent: only inserts missing rows.
    """
    import asyncio as _asyncio
    from datetime import date as _date

    from rainier.scheduler.service import run_daily_eval

    if horizon is None:
        # Full pass — invoke the scheduler entry which runs all three plus
        # composes the Discord report.
        _asyncio.run(run_daily_eval(eval_date_iso=eval_date_s))
        click.echo("Daily eval finished.")
        return

    # Single-horizon: skip the Discord report, just run the backfill so the
    # operator can see the insert count for one horizon at a time.
    from rainier.llm_thesis.eval import evaluate_horizon

    eval_date = (
        _date.fromisoformat(eval_date_s) if eval_date_s else _date.today()
    )
    n = evaluate_horizon(eval_date, horizon)  # type: ignore[arg-type]
    click.echo(
        f"Evaluated horizon={horizon} on {eval_date.isoformat()}: {n} rows inserted."
    )
    _ = ctx  # ctx unused for single-horizon path; kept for API symmetry


# ---------------------------------------------------------------------------
# `rainier thesis research` — auto-research loop (PR3)
# ---------------------------------------------------------------------------


@thesis.group("research")
def thesis_research():
    """Weekly auto-research loop — produce, browse, accept, reject insights."""


@thesis_research.command("run")
@click.option(
    "--eval-date", "eval_date_s", default=None,
    help="Logical 'today' for the rolling window, YYYY-MM-DD; defaults to today.",
)
@click.option("--days", "days", default=30, type=int)
def thesis_research_run(eval_date_s, days):
    """Manually trigger the weekly research job.

    Same code path the Friday 09:00 PT scheduler entry uses — runs all 6
    check classes, posts the Discord report, idempotent re-runs UPSERT
    pending insights instead of duplicating.
    """
    import asyncio as _asyncio

    from rainier.scheduler.service import run_research_job

    _ = days  # currently fixed at 30 in run_research_job; param kept for symmetry
    _asyncio.run(run_research_job(eval_date_iso=eval_date_s))
    click.echo("Research job finished.")


@thesis_research.group("insights")
def thesis_research_insights():
    """Browse / accept / reject ResearchInsight rows."""


@thesis_research_insights.command("list")
@click.option(
    "--status",
    "status_filter",
    default="pending",
    type=click.Choice(["pending", "accepted", "rejected", "stale", "auto_applied", "all"]),
    help="Filter rows by status; default 'pending'.",
)
def thesis_research_insights_list(status_filter):
    """Print the ResearchInsight queue."""
    from sqlalchemy import select as _select

    from rainier.core.database import get_session
    from rainier.core.models import ResearchInsight

    with get_session() as session:
        stmt = _select(ResearchInsight).order_by(ResearchInsight.id.desc())
        if status_filter != "all":
            stmt = stmt.where(ResearchInsight.status == status_filter)
        rows = session.execute(stmt).scalars().all()

    if not rows:
        click.echo(f"No {status_filter} insights.")
        return

    click.echo(
        f"{'ID':<6} {'Severity':<10} {'Kind':<24} {'Subject':<24} "
        f"{'Recur':<6} {'Action':<22} Status"
    )
    click.echo("-" * 110)
    for r in rows:
        action_kind = "?"
        if isinstance(r.action, dict):
            action_kind = str(r.action.get("kind", "?"))
        click.echo(
            f"{r.id:<6} {r.severity:<10} {r.kind:<24} {(r.subject or '')[:23]:<24} "
            f"{r.recurrence_count:<6} {action_kind:<22} {r.status}"
        )
        rationale = (r.rationale or "").strip().replace("\n", " ")
        if rationale:
            if len(rationale) > 100:
                rationale = rationale[:97] + "..."
            click.echo(f"       -> {rationale}")


@thesis_research_insights.command("accept")
@click.argument("insight_id", type=int)
@click.pass_context
def thesis_research_insights_accept(ctx, insight_id):
    """Apply the suggested action and mark the insight accepted.

    Looks up the row, dispatches `insight.action.kind` through
    `research.ACTION_EXECUTORS`, mutates `config/settings.yaml` via ruamel.yaml,
    then UPDATEs the DB row to `status='accepted'` with the diff stored in
    `applied_change`. Errors out cleanly if the row is non-pending or the
    action kind is unknown.
    """
    from datetime import datetime as _datetime
    from datetime import timezone as _tz

    from rainier.core.database import get_session
    from rainier.core.models import ResearchInsight
    from rainier.llm_thesis.research import apply_action

    settings_path = Path(_settings_path(ctx))

    # Review iter-1 [P2]: order matters. Validate the action kind FIRST (no
    # side-effects), then take the DB row + apply YAML inside one
    # transaction so a YAML-write failure rolls back the DB row update via
    # the contextmanager. Previously YAML was written before the DB commit
    # — a transient DB error left settings.yaml mutated while the row stayed
    # `pending`, allowing a second accept to re-apply the action.
    with get_session() as session:
        row = session.get(ResearchInsight, insight_id)
        if row is None:
            raise click.ClickException(f"No ResearchInsight with id={insight_id}")
        if row.status != "pending":
            raise click.ClickException(
                f"Insight {insight_id} has status={row.status!r}, not pending. "
                "Only pending insights can be accepted."
            )
        action = row.action or {}
        try:
            diff = apply_action(action, settings_path)
        except ValueError as exc:
            # Bad action shape — never wrote anything. Surface clearly.
            raise click.ClickException(f"Could not apply action: {exc}") from exc

        row.status = "accepted"
        row.decided_at = _datetime.now(_tz.utc)
        row.applied_change = diff
        session.flush()
        # contextmanager will COMMIT on clean exit; if anything below this
        # point raised inside the `with`, it would rollback. The YAML write
        # already happened atomically (temp-file rename) — operator can
        # always reconcile by editing YAML by hand and accepting again.

    click.echo(
        f"Accepted insight #{insight_id}: action={action.get('kind')} "
        f"target={action.get('target')!r}"
    )
    click.echo(f"Applied change: {diff}")


@thesis_research_insights.command("reject")
@click.argument("insight_id", type=int)
@click.option("--reason", required=True, help="Free-text reason stored on the row.")
def thesis_research_insights_reject(insight_id, reason):
    """Dismiss the insight without applying its action.

    Sets status='rejected' and stores the reason in `decided_by`. The
    settings.yaml is not touched.
    """
    from datetime import datetime as _datetime
    from datetime import timezone as _tz

    from rainier.core.database import get_session
    from rainier.core.models import ResearchInsight

    with get_session() as session:
        row = session.get(ResearchInsight, insight_id)
        if row is None:
            raise click.ClickException(f"No ResearchInsight with id={insight_id}")
        if row.status != "pending":
            raise click.ClickException(
                f"Insight {insight_id} has status={row.status!r}, not pending."
            )
        row.status = "rejected"
        row.decided_at = _datetime.now(_tz.utc)
        row.decided_by = reason[:200]
        session.flush()

    click.echo(f"Rejected insight #{insight_id} with reason: {reason}")


@thesis_research.command("signals")
@click.option("--signal", "signal_name", default=None,
              help="Filter to a single signal name; default lists all.")
@click.option("--days", "days", default=30, type=int)
@click.option("--horizon", "horizon", default="5d",
              type=click.Choice(["1d", "5d", "10d"]))
def thesis_research_signals(signal_name, days, horizon):
    """Ad-hoc per-signal contribution dump (Mann-Whitney U on used vs absent)."""
    from rainier.llm_thesis.eval import compute_signal_contribution

    contribs = compute_signal_contribution(days=days, horizon=horizon)
    if signal_name:
        contribs = [c for c in contribs if c.name == signal_name]
    if not contribs:
        click.echo("No signal contribution rows found.")
        return

    click.echo(f"{'Signal':<24} {'lift':<10} {'p-value':<10} {'n_used':<8} {'n_absent':<10}")
    click.echo("-" * 72)
    for c in contribs:
        p = f"{c.p_value:.3f}" if c.p_value is not None else "n/a"
        click.echo(
            f"{c.name:<24} {c.lift:+.4f}   {p:<10} {c.n_used:<8} {c.n_absent:<10}"
        )


@thesis_research.command("verdicts")
@click.option("--days", "days", default=30, type=int)
def thesis_research_verdicts(days):
    """Ad-hoc per-verdict hit-rate dump."""
    from rainier.llm_thesis.eval import compute_verdict_hit_rate

    rates = compute_verdict_hit_rate(days=days)
    click.echo(f"{'Verdict':<12} {'Horizon':<8} {'n':<6} {'win-rate':<10} avg-return")
    click.echo("-" * 56)
    for verdict, hits in rates.items():
        for hr in hits:
            click.echo(
                f"{verdict:<12} {hr.horizon:<8} {hr.n:<6} {hr.win_rate:<10.2%} "
                f"{hr.avg_return_pct:+.4f}"
            )


# ---------------------------------------------------------------------------
# `rainier debug ...` — test utilities. NOT for normal operations.
# ---------------------------------------------------------------------------
#
# This group hosts probes that exercise live integration points without
# their usual upstream dependencies (scrape, screener, LLM). Today: one
# command that POSTs a synthetic per-ticker thesis embed to Discord so the
# operator can verify the PR #73 routing (llm_webhook_url vs
# stock_webhook_url) end-to-end without waiting on a fresh scan.

@cli.group()
def debug():
    """Test utilities — synthetic probes, no DB / LLM side effects."""


# Verdicts allowed by `core.llm_thesis.schemas.TradeThesis.verdict`. Derived
# from the Literal annotation at module load so a schema change (new verdict,
# rename, removal) automatically flows through to the click.Choice without a
# manual edit here. The import does pull pydantic into the cli import path,
# but that's already true via core.config → BaseSettings, so the cost is zero.
def _trade_thesis_verdicts() -> tuple[str, ...]:
    from typing import get_args

    from rainier.llm_thesis.schemas import TradeThesis

    return tuple(get_args(TradeThesis.model_fields["verdict"].annotation))


_FAKE_THESIS_VERDICTS = _trade_thesis_verdicts()


def _mask_webhook_url(url: str | None) -> str:
    """Redact a Discord webhook URL for stdout logging.

    Discord webhook URLs are bearer credentials: anyone with the full URL
    can post into the channel. The routing probe needs to tell the
    operator WHICH channel was resolved (so they can verify a config
    change took effect) without leaking the credential into shared
    shells, CI logs, or debugging transcripts.

    Format: ``https://<host>/api/webhooks/<channel_id>/****<last6>``
    Channel ID is non-secret (visible in Discord's UI); the token tail
    gives the operator a stable 6-char fingerprint they can compare
    against their .env to confirm the right URL was picked without
    exposing the full secret.

    Returns ``"(none)"`` for empty / missing URLs, and the literal input
    (with a fallback redaction) for malformed URLs that don't match the
    Discord webhook shape — we never echo a non-empty URL verbatim.
    """
    if not url:
        return "(none)"
    # Discord webhook URL: https://discord.com/api/webhooks/<channel_id>/<token>
    # Token is the secret. Channel ID is public-equivalent (visible in UI).
    marker = "/api/webhooks/"
    idx = url.find(marker)
    if idx == -1:
        # Not a Discord webhook URL — could be a proxy / smee.io / custom
        # relay. Don't trust the format; redact aggressively.
        return f"<non-discord webhook, {len(url)} chars, ...{url[-6:]}>"
    prefix = url[: idx + len(marker)]
    tail = url[idx + len(marker) :]
    parts = tail.split("/", 1)
    if len(parts) != 2 or not parts[1]:
        return f"{prefix}<malformed>"
    channel_id, token = parts
    token_tail = token[-6:] if len(token) > 6 else "****"
    return f"{prefix}{channel_id}/****{token_tail}"


@debug.command("post-fake-thesis")
@click.option(
    "--symbol",
    default="TSLA",
    help="Ticker symbol stamped on the synthetic candidate + thesis.",
)
@click.option(
    "--verdict",
    default="setup_long",
    type=click.Choice(_FAKE_THESIS_VERDICTS),
    help="TradeThesis.verdict on the synthetic post.",
)
@click.option(
    "--llm-webhook/--stock-webhook",
    "use_llm_webhook",
    default=True,
    help=(
        "--llm-webhook (default): natural routing — thesis embed goes to "
        "llm_webhook_url. --stock-webhook: clear llm_webhook_url in-memory "
        "so the thesis falls back to stock_webhook_url (verifies the "
        "_resolve_llm_webhook_url fallback path)."
    ),
)
@click.pass_context
def debug_post_fake_thesis(ctx, symbol, verdict, use_llm_webhook):
    """POST a synthetic thesis embed to Discord — routing-only probe.

    Builds an in-memory StockCandidate + TradeThesis tagged with the
    literal `[FAKE TEST POST]` marker, then calls send_stock_candidates
    to exercise the same code path the scheduler uses. No DB I/O, no LLM
    call, no chart attachment. Prints the resolved webhook URLs to stdout
    so the operator can confirm routing without parsing Discord.

    Exit code: 0 on successful POST(s). Non-zero on (a) missing
    summary-channel webhook config, (b) ``send_stock_candidates``
    raising synchronously, or (c) a Discord HTTP failure (4xx/5xx,
    timeout, connection error) that ``send_stock_candidates`` would
    otherwise swallow as ``log.exception(...)``. The probe attaches a
    logging captor to ``rainier.alerts.discord`` so swallowed httpx
    failures surface as a ClickException, preserving the routing
    probe's value as a real end-to-end diagnostic.
    """
    from rainier.alerts.discord import (
        _resolve_llm_webhook_url,
        _resolve_webhook_url,
        send_stock_candidates,
    )
    from rainier.core.config import DiscordConfig, load_settings_fresh
    from rainier.core.types import StockCandidate
    from rainier.llm_thesis.schemas import TradeThesis

    settings = load_settings_fresh(_settings_path(ctx))

    # We never mutate the operator's live settings — clone the DiscordConfig
    # so the --stock-webhook override stays in-process only. Pydantic v2's
    # model_copy keeps validation invariants intact.
    discord_cfg: DiscordConfig = settings.alerts.discord.model_copy()
    # The probe always wants Discord on, even if the operator's config
    # has alerts.discord.enabled=False (e.g. local dev where notifications
    # are normally muted). The probe IS the notification.
    discord_cfg.enabled = True
    if not use_llm_webhook:
        # --stock-webhook: clear llm_webhook_url so _resolve_llm_webhook_url
        # falls back to stock_webhook_url (or webhook_url).
        discord_cfg.llm_webhook_url = ""

    # Resolve both URLs ahead of time so we can print + sanity-check before
    # POSTing. Stays in sync with send_stock_candidates' own resolution.
    stock_url = _resolve_webhook_url(discord_cfg)
    thesis_url = _resolve_llm_webhook_url(discord_cfg)
    # send_stock_candidates bails at its first line when stock_url is empty
    # (it needs the summary channel to fire even if only the thesis is
    # actually interesting to us). Catch that here so the probe doesn't
    # exit 0 having sent nothing — the canonical false-success failure
    # mode for routing diagnostics. A config with ONLY llm_webhook_url
    # set is supported by the dataclass but unusable by the renderer; the
    # operator wants to know that immediately.
    if not stock_url:
        raise click.ClickException(
            "No summary Discord webhook configured (stock_webhook_url + "
            "webhook_url both empty). send_stock_candidates requires a "
            "summary channel even when llm_webhook_url is set; the thesis "
            "embed would never POST. Set DISCORD_STOCK_WEBHOOK_URL (or "
            "DISCORD_WEBHOOK_URL) in .env or alerts.discord.* in "
            "settings.yaml."
        )

    # ---- synthetic candidate ------------------------------------------------
    # Plausible-looking but obviously test values; the [FAKE TEST POST]
    # marker is the load-bearing safety signal once this lands in Discord.
    fake_symbol = symbol.upper()
    candidate = StockCandidate(
        symbol=fake_symbol,
        rank=1,
        rank_change=0,
        long_short="Long in",
        capital_flow_direction="+",
        sector="[FAKE TEST POST] Synthetic",
        signal_strength=0.85,
        money_flow_score=70.0,
        pattern_type="bull_flag",
        pattern_direction="bullish",
        pattern_status="confirmed",
        pattern_confidence=0.85,
        entry_price=100.0,
        stop_loss=95.0,
        target_price=115.0,
        rr_ratio=3.0,
        volume_confirmed=True,
        current_price=100.0,
        distance_to_entry_pct=0.0,
        bars_since_breakout=0,
    )

    # ---- synthetic thesis ---------------------------------------------------
    thesis_model = TradeThesis(
        verdict=verdict,
        setup_quality=8,
        llm_confidence=7,
        paragraph_radar="[FAKE TEST POST] synthetic",
        paragraph_evidence="[FAKE TEST POST] synthetic",
        paragraph_invalidation="[FAKE TEST POST] synthetic",
        risks=["[FAKE TEST POST] synthetic risk"],
        watch_items=["[FAKE TEST POST] synthetic watch"],
        evidence_used=["rank_trajectory"],
        signals_used=["rank_trajectory", "capital_flow_streak"],
        patterns_in_chart_not_in_indicators="none",
    )
    thesis_dict = thesis_model.model_dump()

    click.echo(f"Posting [FAKE TEST POST] thesis for {fake_symbol}...")
    click.echo(f"  summary webhook (stock_webhook_url) : {_mask_webhook_url(stock_url)}")
    click.echo(f"  thesis  webhook (llm_webhook_url)   : {_mask_webhook_url(thesis_url)}")

    # send_stock_candidates() catches httpx exceptions internally
    # (`log.exception(...)`) and returns normally, so a 4xx/5xx Discord
    # response would otherwise let this probe exit 0 with "done." — the
    # exact false-success path operators use this command to diagnose.
    # Attach a captor handler to the discord logger so we can detect
    # those swallowed failures and surface them as a non-zero exit.
    import logging as _stdlib_logging

    captured_failures: list[_stdlib_logging.LogRecord] = []

    class _DiscordFailureCaptor(_stdlib_logging.Handler):
        def emit(self, record: _stdlib_logging.LogRecord) -> None:
            if record.levelno >= _stdlib_logging.ERROR:
                captured_failures.append(record)

    captor = _DiscordFailureCaptor(level=_stdlib_logging.ERROR)
    discord_logger = _stdlib_logging.getLogger("rainier.alerts.discord")
    # Logger.isEnabledFor() is the gate BEFORE handlers run — if the
    # caller (CI wrapper, structlog config, etc.) raised the discord
    # logger to CRITICAL, ERROR records get dropped at the logger and
    # the captor never sees them. Temporarily lower the level to ERROR
    # so log.exception(...) calls inside send_stock_candidates reach
    # our handler, then restore on the way out so we don't perturb the
    # operator's logging config beyond this call.
    saved_level = discord_logger.level
    saved_disabled = discord_logger.disabled
    saved_propagate = discord_logger.propagate
    discord_logger.setLevel(_stdlib_logging.ERROR)
    discord_logger.disabled = False
    # propagate=False so our captured records don't also fire on parent
    # handlers (which might surface stack traces the operator doesn't
    # need for the routing diagnostic). The captor is our only sink.
    discord_logger.propagate = False
    discord_logger.addHandler(captor)
    try:
        send_stock_candidates(
            [candidate],
            discord_cfg,
            theses={fake_symbol: thesis_dict},
            dashboard_base_url=None,
        )
    except Exception as exc:  # noqa: BLE001 — synthetic-payload bugs surface here
        raise click.ClickException(f"Discord POST failed: {exc}") from exc
    finally:
        discord_logger.removeHandler(captor)
        discord_logger.setLevel(saved_level)
        discord_logger.disabled = saved_disabled
        discord_logger.propagate = saved_propagate

    if captured_failures:
        # Render a terse one-line summary of each captured failure so the
        # operator can tell which endpoint failed (summary vs thesis)
        # without having to grep the structured log. We do NOT echo the
        # full webhook URL — only the logger event name + exception
        # class. The URL has already been printed in masked form above.
        summaries = []
        for rec in captured_failures:
            exc_type = (
                rec.exc_info[0].__name__ if rec.exc_info and rec.exc_info[0] else "?"
            )
            summaries.append(f"{rec.getMessage()} ({exc_type})")
        raise click.ClickException(
            "Discord POST failed (send_stock_candidates swallowed the "
            "error internally; the probe surfaced it): "
            + "; ".join(summaries)
        )

    click.echo("done.")


# ---------------------------------------------------------------------------
# sma-sweep — TQQQ/SQQQ rotation backtest grid
# ---------------------------------------------------------------------------


@cli.command("sma-sweep")
@click.option("--phase", type=click.IntRange(1, 2), default=1, show_default=True,
              help="1 = trend-following (sell >= buy). 2 = full grid (adds anti-trend).")
@click.option("--max-window", type=int, default=60, show_default=True,
              help="Largest SMA window to sweep over.")
@click.option("--refresh-data", is_flag=True, default=False,
              help="Force a fresh yfinance download even if the parquet cache is valid.")
@click.option("--n-workers", type=int, default=None,
              help="Pool size. Defaults to os.cpu_count().")
@click.option("--flush-every", type=int, default=50_000, show_default=True,
              help="Flush results to parquet every N completed combos (crash safety).")
@click.option("--slippage-bp", type=float, default=5.0, show_default=True,
              help="Round-trip slippage per state transition, basis points.")
@click.option("--report/--no-report", default=True, show_default=True,
              help="Render the HTML report after the sweep completes.")
@click.option("--report-path", type=click.Path(), default=None,
              help="Output path for the HTML report. Defaults: Phase 1 → "
                   "docs/tqqq-sma-backtest-report.html, Phase 2 → "
                   "docs/tqqq-sma-backtest-phase2-report.html.")
@click.option("--top-n-walkforward", type=int, default=100, show_default=True,
              help="How many top-by-final_value combos to walk-forward.")
def sma_sweep(
    phase: int,
    max_window: int,
    refresh_data: bool,
    n_workers: int | None,
    flush_every: int,
    slippage_bp: float,
    report: bool,
    report_path: str | None,
    top_n_walkforward: int,
) -> None:
    """Sweep the TQQQ/SQQQ rotation strategy over the QQQ-SMA grid.

    Phase 1 enforces sell >= buy on both legs (trend-following). Phase 2
    drops the constraint and explores the full 4-D grid (60^4 = 12.96M combos
    at default max_window), adding the anti-trend regions on top of Phase 1.

    Phase 2 extends an existing Phase-1 parquet in place — the resumability
    skip-set in run_sweep ensures already-completed trend-following combos
    aren't recomputed.
    """
    import time as _time

    from rainier.backtest.tqqq_sma_sweep import (
        RESULTS_CACHE_PATH,
        fetch_prices,
        run_sweep,
        walk_forward_top_n,
    )

    # Default report paths differ by phase so a Phase-2 run doesn't clobber
    # the Phase-1 report.
    if report_path is None:
        report_path = (
            "docs/tqqq-sma-backtest-report.html" if phase == 1
            else "docs/tqqq-sma-backtest-phase2-report.html"
        )

    click.echo("Fetching prices (QQQ/TQQQ/SQQQ)…")
    prices = fetch_prices(refresh=refresh_data)
    click.echo(f"  rows: {len(prices)}  range: {prices.index[0].date()} → {prices.index[-1].date()}")

    label = "trend-following" if phase == 1 else "full grid (incl. anti-trend)"
    click.echo(f"Running Phase-{phase} sweep — {label} "
               f"(max_window={max_window}, slippage={slippage_bp} bp)…")
    t0 = _time.time()
    results_path = run_sweep(
        prices,
        results_path=RESULTS_CACHE_PATH,
        max_window=max_window,
        n_workers=n_workers,
        slippage_bp=slippage_bp,
        flush_every=flush_every,
        progress=True,
        phase=phase,
    )
    elapsed = _time.time() - t0
    click.echo(f"Sweep done in {elapsed/60:.1f} min → {results_path}")

    click.echo(f"Walk-forward top-{top_n_walkforward}…")
    top_wf = walk_forward_top_n(
        prices,
        results_path=results_path,
        top_n=top_n_walkforward,
        slippage_bp=slippage_bp,
        max_window=max_window,
        phase=phase,
    )
    # Phase 2 walk-forward goes to a sibling parquet so the Phase-1 walkforward
    # remains intact for the Phase-1 report.
    wf_name = "top_walkforward.parquet" if phase == 1 else "top_walkforward_phase2.parquet"
    top_wf_path = results_path.parent / wf_name
    top_wf.to_parquet(top_wf_path, index=False)
    click.echo(f"  → {top_wf_path}")

    if report:
        from rainier.backtest.tqqq_sma_report import render_report

        click.echo(f"Rendering report → {report_path}")
        # Phase-2 report adds the Phase 1 vs Phase 2 comparison section. The
        # path-1 parquet (if it exists) is loaded internally for the headline
        # comparison; no extra CLI args needed.
        render_report(
            prices=prices,
            results_path=results_path,
            walkforward_path=top_wf_path,
            output_path=Path(report_path),
            sweep_wall_seconds=elapsed,
            slippage_bp=slippage_bp,
            max_window=max_window,
            phase=phase,
        )
        click.echo("done.")


# ---------------------------------------------------------------------------
# thematic — Industry/Thematic ranks substrate (Phase A: data only)
# ---------------------------------------------------------------------------
#
# Phase A surface: `backfill` and `snapshot-universe`. The compute / render /
# run-daily subcommands ship in Phase B (`thematic-ranks-dashboard-d245`).
#
# ASCII flow:
#     thematic backfill         -> load YAML -> yfinance OHLCV -> parquet cache
#                                  + seed ticker_registry + sector_registry
#     thematic snapshot-universe -> SHA-diff YAML vs log -> append row if changed


@cli.group()
def thematic() -> None:
    """Industry / Thematic ranks dashboard substrate (DESIGN §5)."""


@thematic.command("backfill")
@click.option(
    "--yaml",
    "yaml_path",
    type=click.Path(exists=True),
    default="config/thematic_universe.yaml",
    show_default=True,
    help="Path to the thematic universe YAML.",
)
@click.option("--start", default=None, help="Window start (YYYY-MM-DD).")
@click.option("--end", default=None, help="Window end (YYYY-MM-DD, inclusive).")
@click.option(
    "--out",
    "out_path",
    type=click.Path(),
    default="data/cache/thematic_universe.parquet",
    show_default=True,
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="When the cache exists, write a timestamped sibling cohort.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the planned ticker x date matrix; do not fetch or write.",
)
@click.option(
    "--allow-empty",
    default="",
    help="Comma-separated symbols permitted to return zero rows.",
)
@click.option(
    "--allow-gaps",
    default="",
    help="Comma-separated symbols permitted to return partial coverage.",
)
@click.option(
    "--min-coverage",
    type=float,
    default=None,
    help="Per-symbol minimum row count as a fraction of business days.",
)
@click.option(
    "--ticker-registry",
    type=click.Path(),
    default="data/cache/ticker_registry.parquet",
    show_default=True,
)
@click.option(
    "--sector-registry",
    type=click.Path(),
    default="data/cache/sector_registry.parquet",
    show_default=True,
)
def thematic_backfill(
    yaml_path: str,
    start: str | None,
    end: str | None,
    out_path: str,
    force: bool,
    dry_run: bool,
    allow_empty: str,
    allow_gaps: str,
    min_coverage: float | None,
    ticker_registry: str,
    sector_registry: str,
) -> None:
    """Backfill the OHLCV cache + seed ticker/sector registries.

    Operator-run, not CI-run. Hits the yfinance network.
    """
    import importlib.util
    from datetime import date as _date

    from rainier.research.breadth import registry as _reg
    from rainier.research.breadth import universe_loader as _ul

    # Load the script as a module — it lives in `scripts/`, not under `src/`,
    # because it's a one-off operator tool (same pattern as macro_context).
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "backfill_thematic_universe",
        root / "scripts" / "backfill_thematic_universe.py",
    )
    if spec is None or spec.loader is None:
        raise click.ClickException(
            "could not load scripts/backfill_thematic_universe.py"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    yaml_p = Path(yaml_path)
    symbols = module.load_symbols_from_yaml(yaml_p)
    spec_obj = _ul.load_universe(yaml_p)

    start_eff = start or module.DEFAULT_START
    end_eff = end or module.DEFAULT_END
    coverage = (
        min_coverage if min_coverage is not None else module.DEFAULT_MIN_COVERAGE_RATIO
    )

    result = module.backfill(
        symbols=symbols,
        start=start_eff,
        end=end_eff,
        out_path=Path(out_path),
        force=force,
        dry_run=dry_run,
        allow_empty=[s.strip() for s in allow_empty.split(",") if s.strip()],
        allow_gaps=[s.strip() for s in allow_gaps.split(",") if s.strip()],
        min_coverage=coverage,
    )

    if dry_run:
        plan = result
        click.echo(
            f"DRY-RUN: would fetch {len(symbols)} symbols "
            f"{start_eff}..{end_eff} -> {plan['planned_out']}"
        )
        for sym in symbols:
            click.echo(f"  {sym}")
        return

    written_path = result
    click.echo(f"wrote OHLCV cache -> {written_path}")

    # Seed registries with the just-fetched universe. Idempotent — re-running
    # backfill does not change existing IDs (per [D-015]).
    today = _date.today()
    _reg.seed_registries_from_universe(
        spec_obj.sectors,
        asof=today,
        ticker_registry_path=Path(ticker_registry),
        sector_registry_path=Path(sector_registry),
    )
    click.echo(f"seeded ticker registry  -> {ticker_registry}")
    click.echo(f"seeded sector registry  -> {sector_registry}")


@thematic.command("snapshot-universe")
@click.option(
    "--yaml",
    "yaml_path",
    type=click.Path(exists=True),
    default="config/thematic_universe.yaml",
    show_default=True,
)
@click.option(
    "--log",
    "log_path",
    type=click.Path(),
    default="data/cache/thematic_universe_log.parquet",
    show_default=True,
)
@click.option(
    "--effective-from",
    default=None,
    help="First asof_date the universe applies to (YYYY-MM-DD). Default: today.",
)
@click.option("--note", default="", help="Free-form change note.")
def thematic_snapshot_universe(
    yaml_path: str,
    log_path: str,
    effective_from: str | None,
    note: str,
) -> None:
    """Append a row to thematic_universe_log if the YAML SHA changed."""
    from datetime import date as _date

    from rainier.research.breadth import universe_loader as _ul

    eff = _date.fromisoformat(effective_from) if effective_from else _date.today()
    appended = _ul.snapshot_universe(
        yaml_path=Path(yaml_path),
        log_path=Path(log_path),
        effective_from=eff,
        note=note,
    )
    if appended:
        click.echo(f"appended new row -> {log_path}")
    else:
        click.echo(f"no-op: yaml SHA unchanged; {log_path} untouched")


# Composition root — wire the research engine's `llm-research` subgroup
# onto the root `rainier` command. Per project CLAUDE.md the research
# package never reaches into production CLI plumbing; we register it here
# so research/cli.py stays free of imports from production modules.
from rainier.research.cli import register as _register_llm_research  # noqa: E402

_register_llm_research(cli)


# ---------------------------------------------------------------------------
# thematic — Phase B: compute, backfill-labels, render, run-daily
# ---------------------------------------------------------------------------
#
# These build on Phase A's thematic_universe.parquet + ticker/sector
# registries + universe log to produce:
#   - Layer A features  -> data/cache/thematic_features_daily.parquet
#   - Layer B labels    -> data/cache/thematic_labels_daily.parquet
#   - Dashboard HTML    -> docs/thematic-ranks-<YYYY-MM-DD>.html
#
# Design ref: docs/DESIGN-thematic-ranks-dashboard.md §6 / §7.


def _load_universe_for_compute(yaml_path: Path):
    """Return ``(sector_map, ticker_registry, sector_registry, yaml_sha,
    all_symbols)`` from the YAML at ``yaml_path``.

    Registries are loaded if their parquets exist, else seeded from the YAML
    so first-run compute always succeeds even before the operator has run
    `thematic backfill` (which usually seeds them).
    """
    from rainier.research.breadth import universe_loader as _ul

    spec = _ul.load_universe(yaml_path)
    sector_map: dict[str, str] = {}
    for sec, syms in spec.sectors.items():
        for s in syms:
            sector_map[s] = sec
    return spec, sector_map


@thematic.command("compute")
@click.option("--asof", required=True, help="Compute date (YYYY-MM-DD).")
@click.option(
    "--ohlcv",
    "ohlcv_path",
    type=click.Path(exists=True),
    default="data/cache/thematic_universe.parquet",
    show_default=True,
    help="Path to thematic_universe.parquet (Phase A OHLCV cache).",
)
@click.option(
    "--yaml",
    "yaml_path",
    type=click.Path(exists=True),
    default="config/thematic_universe.yaml",
    show_default=True,
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(),
    default="data/cache/thematic_features_daily.parquet",
    show_default=True,
)
@click.option(
    "--ticker-registry",
    type=click.Path(),
    default="data/cache/ticker_registry.parquet",
    show_default=True,
)
@click.option(
    "--sector-registry",
    type=click.Path(),
    default="data/cache/sector_registry.parquet",
    show_default=True,
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Recompute even if asof_date already present in the output.",
)
def thematic_compute(
    asof: str,
    ohlcv_path: str,
    yaml_path: str,
    out_path: str,
    ticker_registry: str,
    sector_registry: str,
    force: bool,
) -> None:
    """Compute Layer A features for one ``asof`` and insert into the parquet.

    Idempotent: re-running for the same asof_date is a no-op unless ``--force``.
    """
    from datetime import date as _date

    from rainier.research.breadth import registry as _reg
    from rainier.research.breadth.ranks import compute_thematic_features

    asof_dt = _date.fromisoformat(asof)
    spec, sector_map = _load_universe_for_compute(Path(yaml_path))

    # Seed registries from the YAML if they don't exist yet.
    ticker_registry_path = Path(ticker_registry)
    sector_registry_path = Path(sector_registry)
    _reg.seed_registries_from_universe(
        spec.sectors,
        asof=asof_dt,
        ticker_registry_path=ticker_registry_path,
        sector_registry_path=sector_registry_path,
    )
    ticker_reg_map = _reg.load_ticker_registry(ticker_registry_path)
    sector_reg_map = _reg.load_sector_registry(sector_registry_path)

    out_p = Path(out_path)
    # Idempotency check.
    if out_p.exists() and not force:
        existing = pd.read_parquet(out_p)
        if "asof_date" in existing.columns:
            existing_dates = pd.to_datetime(existing["asof_date"]).dt.date
            if (existing_dates == asof_dt).any():
                click.echo(f"no-op: asof={asof_dt} already in {out_p}")
                return

    panel = pd.read_parquet(ohlcv_path)
    if "date" in panel.columns:
        panel["date"] = pd.to_datetime(panel["date"]).dt.date

    # Pull previous asof's row (if any) so deltas + streak chain correctly.
    prev_features = None
    if out_p.exists():
        existing = pd.read_parquet(out_p)
        if not existing.empty and "asof_date" in existing.columns:
            existing["asof_date"] = pd.to_datetime(existing["asof_date"]).dt.date
            prior = existing.loc[existing["asof_date"] < asof_dt]
            if not prior.empty:
                latest = prior["asof_date"].max()
                prev_features = prior.loc[prior["asof_date"] == latest]

    out_df = compute_thematic_features(
        panel=panel,
        asof=asof_dt,
        sector_map=sector_map,
        ticker_registry=ticker_reg_map,
        sector_registry=sector_reg_map,
        universe_yaml_sha=spec.yaml_sha,
        prev_features=prev_features,
    )
    if out_df.empty:
        click.echo(f"no rows computed for asof={asof_dt}; aborting write")
        return

    _append_features(out_df, out_p, asof_dt, force=force)
    click.echo(f"wrote {len(out_df)} rows -> {out_p}")


def _append_features(new_df: pd.DataFrame, path: Path, asof: date, force: bool) -> None:
    """Append new_df to the parquet at path, replacing any existing rows for
    the same asof_date if ``force``. Atomic write via tmp + os.replace.
    """
    import os


    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        if not existing.empty and "asof_date" in existing.columns:
            existing["asof_date"] = pd.to_datetime(existing["asof_date"]).dt.date
            if force:
                existing = existing.loc[existing["asof_date"] != asof]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(["symbol", "asof_date"]).reset_index(drop=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    combined.to_parquet(tmp, index=False)
    os.replace(tmp, path)


@thematic.command("backfill-labels")
@click.option(
    "--ohlcv",
    "ohlcv_path",
    type=click.Path(exists=True),
    default="data/cache/thematic_universe.parquet",
    show_default=True,
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(),
    default="data/cache/thematic_labels_daily.parquet",
    show_default=True,
)
def thematic_backfill_labels(ohlcv_path: str, out_path: str) -> None:
    """Compute Layer B forward-return labels for every (asof_date, symbol)."""
    import os

    from rainier.research.breadth.ranks import compute_forward_labels

    panel = pd.read_parquet(ohlcv_path)
    if "date" in panel.columns:
        panel["date"] = pd.to_datetime(panel["date"]).dt.date
    out_df = compute_forward_labels(panel=panel)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_p.with_suffix(out_p.suffix + ".tmp")
    out_df.to_parquet(tmp, index=False)
    os.replace(tmp, out_p)
    click.echo(f"wrote {len(out_df)} label rows -> {out_p}")


@thematic.command("render")
@click.option("--asof", required=True, help="Date to render (YYYY-MM-DD).")
@click.option(
    "--features",
    "features_path",
    type=click.Path(exists=True),
    default="data/cache/thematic_features_daily.parquet",
    show_default=True,
)
@click.option(
    "--yaml",
    "yaml_path",
    type=click.Path(exists=True),
    default="config/thematic_universe.yaml",
    show_default=True,
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(),
    default=None,
    help="HTML path. Default: docs/thematic-ranks-<asof>.html.",
)
def thematic_render(
    asof: str, features_path: str, yaml_path: str, out_path: str | None
) -> None:
    """Render the thematic ranks dashboard for ``asof`` to a self-contained HTML."""
    from datetime import date as _date

    from rainier.viz.thematic_dashboard import render_dashboard

    asof_dt = _date.fromisoformat(asof)
    features = pd.read_parquet(features_path)
    if "asof_date" in features.columns:
        features["asof_date"] = pd.to_datetime(features["asof_date"]).dt.date
    sub = features.loc[features["asof_date"] == asof_dt]
    if sub.empty:
        raise click.ClickException(
            f"no features for asof={asof_dt} in {features_path}; "
            f"run `rainier thematic compute --asof {asof_dt}` first."
        )

    # Join sector_name from the YAML universe.
    spec, sector_map = _load_universe_for_compute(Path(yaml_path))
    sub = sub.copy()
    sub["sector_name"] = sub["symbol"].map(sector_map).fillna("unknown")

    target = Path(out_path) if out_path else Path(f"docs/thematic-ranks-{asof_dt}.html")
    written = render_dashboard(sub, out_path=target, asof=asof_dt)
    click.echo(f"wrote dashboard -> {written}")


@thematic.command("run-daily")
@click.option(
    "--asof",
    default=None,
    help="Date to run for (YYYY-MM-DD). Default: today.",
)
@click.option(
    "--ohlcv",
    "ohlcv_path",
    type=click.Path(exists=True),
    default="data/cache/thematic_universe.parquet",
    show_default=True,
)
@click.option(
    "--yaml",
    "yaml_path",
    type=click.Path(exists=True),
    default="config/thematic_universe.yaml",
    show_default=True,
)
@click.option(
    "--features-out",
    type=click.Path(),
    default="data/cache/thematic_features_daily.parquet",
    show_default=True,
)
@click.option(
    "--labels-out",
    type=click.Path(),
    default="data/cache/thematic_labels_daily.parquet",
    show_default=True,
)
@click.option(
    "--ticker-registry",
    type=click.Path(),
    default="data/cache/ticker_registry.parquet",
    show_default=True,
)
@click.option(
    "--sector-registry",
    type=click.Path(),
    default="data/cache/sector_registry.parquet",
    show_default=True,
)
@click.option(
    "--html-out",
    type=click.Path(),
    default=None,
    help="HTML path. Default: docs/thematic-ranks-<asof>.html.",
)
def thematic_run_daily(
    asof: str | None,
    ohlcv_path: str,
    yaml_path: str,
    features_out: str,
    labels_out: str,
    ticker_registry: str,
    sector_registry: str,
    html_out: str | None,
) -> None:
    """One-shot daily flow: compute Layer A + Layer B + render dashboard.

    Idempotent: if today's Layer A row already exists, the compute step is
    a no-op. Label backfill always runs (it's cheap and the freshest rows
    drift forward as new days arrive).
    """
    from datetime import date as _date

    from rainier.research.breadth import registry as _reg
    from rainier.research.breadth.ranks import (
        compute_forward_labels,
        compute_thematic_features,
    )
    from rainier.viz.thematic_dashboard import render_dashboard

    asof_dt = _date.fromisoformat(asof) if asof else _date.today()
    spec, sector_map = _load_universe_for_compute(Path(yaml_path))

    # Seed registries.
    ticker_registry_path = Path(ticker_registry)
    sector_registry_path = Path(sector_registry)
    _reg.seed_registries_from_universe(
        spec.sectors,
        asof=asof_dt,
        ticker_registry_path=ticker_registry_path,
        sector_registry_path=sector_registry_path,
    )
    ticker_reg_map = _reg.load_ticker_registry(ticker_registry_path)
    sector_reg_map = _reg.load_sector_registry(sector_registry_path)

    panel = pd.read_parquet(ohlcv_path)
    if "date" in panel.columns:
        panel["date"] = pd.to_datetime(panel["date"]).dt.date

    # Stale-OHLCV guard. Per DESIGN §7: "If OHLCV is stale, backfill
    # incrementally first." We don't auto-fetch (yfinance side effect inside
    # a cron run is too magical). Instead surface clearly with the exact
    # next-step command per the operator's surface-don't-silo discipline.
    #
    # The Phase A backfill script is revision-immutable: pointing it at an
    # existing `--out` raises FileExistsError, and `--force` writes a
    # timestamped sibling cohort (it deliberately never overwrites). The
    # recovery flow is therefore three steps: backfill --force to a sibling,
    # then atomically swap the new cohort into the cache path.
    if not panel.empty and "date" in panel.columns:
        panel_max = panel["date"].max()
        if panel_max < asof_dt:
            raise click.ClickException(
                f"OHLCV cache stale: max(date)={panel_max} < asof={asof_dt}. "
                f"Refresh with:\n"
                f"  1. uv run python scripts/backfill_thematic_universe.py "
                f"--start 2024-10-01 --end {asof_dt} --force\n"
                f"  2. mv $(ls -t {Path(ohlcv_path).parent}/"
                f"{Path(ohlcv_path).stem}_*{Path(ohlcv_path).suffix} | head -1) "
                f"{ohlcv_path}\n"
                f"  3. uv run rainier thematic run-daily  # retry"
            )

        # Partial-coverage guard. Ranks are cross-sectional: if a partial
        # backfill leaves the panel without an asof close for most of the
        # YAML universe, `compute_thematic_features` silently drops missing
        # symbols and ranks over a shrunken universe. Surface the gap so the
        # operator runs a full re-backfill rather than rendering a misleading
        # dashboard (codex iter-6 [P1] + memory feedback_surface_dont_silo).
        expected_syms = {
            sym for syms in spec.sectors.values() for sym in syms
        }
        asof_rows = panel.loc[panel["date"] == asof_dt]
        present = set(asof_rows["symbol"].dropna().unique()) if not asof_rows.empty else set()
        missing = expected_syms - present
        # Threshold: warn when >10% missing; fail when >25% missing. 10%
        # absorbs typical NYSE-holiday + late-listing noise without false
        # alarms; >25% indicates a botched backfill that should not silently
        # ship a dashboard.
        if expected_syms:
            missing_frac = len(missing) / len(expected_syms)
            if missing_frac > 0.25:
                example = sorted(missing)[:8]
                raise click.ClickException(
                    f"OHLCV cache has partial coverage on asof={asof_dt}: "
                    f"{len(missing)}/{len(expected_syms)} YAML symbols missing "
                    f"({missing_frac:.0%}). Examples: {example}. "
                    f"Re-run the backfill --force flow above before computing."
                )
            if missing_frac > 0.10:
                example = sorted(missing)[:8]
                click.echo(
                    f"warning: {len(missing)}/{len(expected_syms)} YAML "
                    f"symbols missing on asof={asof_dt} ({missing_frac:.0%}). "
                    f"Examples: {example}. Proceeding; consider refreshing OHLCV.",
                    err=True,
                )

    # Layer A: idempotent compute.
    features_path = Path(features_out)
    skip_compute = False
    if features_path.exists():
        existing = pd.read_parquet(features_path)
        if not existing.empty and "asof_date" in existing.columns:
            existing["asof_date"] = pd.to_datetime(existing["asof_date"]).dt.date
            if (existing["asof_date"] == asof_dt).any():
                click.echo(f"layer A: no-op (asof={asof_dt} already present)")
                skip_compute = True

    if not skip_compute:
        prev_features = None
        if features_path.exists():
            existing = pd.read_parquet(features_path)
            if not existing.empty and "asof_date" in existing.columns:
                existing["asof_date"] = pd.to_datetime(existing["asof_date"]).dt.date
                prior = existing.loc[existing["asof_date"] < asof_dt]
                if not prior.empty:
                    latest = prior["asof_date"].max()
                    prev_features = prior.loc[prior["asof_date"] == latest]

        out_df = compute_thematic_features(
            panel=panel,
            asof=asof_dt,
            sector_map=sector_map,
            ticker_registry=ticker_reg_map,
            sector_registry=sector_reg_map,
            universe_yaml_sha=spec.yaml_sha,
            prev_features=prev_features,
        )
        if out_df.empty:
            click.echo(f"layer A: no rows for asof={asof_dt}; skipping")
        else:
            _append_features(out_df, features_path, asof_dt, force=False)
            click.echo(f"layer A: wrote {len(out_df)} rows -> {features_path}")

    # Layer B: always recompute (cheap + freshness drifts forward).
    import os

    labels_path = Path(labels_out)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    label_df = compute_forward_labels(panel=panel)
    tmp = labels_path.with_suffix(labels_path.suffix + ".tmp")
    label_df.to_parquet(tmp, index=False)
    os.replace(tmp, labels_path)
    click.echo(f"layer B: wrote {len(label_df)} rows -> {labels_path}")

    # Render dashboard. Guard against a missing features parquet (would only
    # happen if the OHLCV panel had zero rows AND no prior features had ever
    # been written — surface a useful diagnostic instead of FileNotFoundError).
    if not features_path.exists():
        click.echo(
            f"render: features parquet missing at {features_path}; nothing "
            f"to render. Confirm OHLCV cache contains rows for asof={asof_dt}."
        )
        return
    features = pd.read_parquet(features_path)
    if "asof_date" in features.columns:
        features["asof_date"] = pd.to_datetime(features["asof_date"]).dt.date
    sub = features.loc[features["asof_date"] == asof_dt]
    if sub.empty:
        click.echo(f"render: no features for asof={asof_dt}; skipping HTML")
        return
    sub = sub.copy()
    sub["sector_name"] = sub["symbol"].map(sector_map).fillna("unknown")
    target = Path(html_out) if html_out else Path(f"docs/thematic-ranks-{asof_dt}.html")
    written = render_dashboard(sub, out_path=target, asof=asof_dt)
    click.echo(f"render: wrote dashboard -> {written}")
