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


@cli.command(name="pattern-audit")
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
@click.pass_context
def pattern_audit(ctx, symbols, report_path, window_days, window_label):
    """Pattern forward-return audit over `stock_prices` (WS B).

    Faithfully replays the LIVE pattern layer as-of each trading day, attaches
    5/10/20d forward returns + a regime tag, writes a regenerable Parquet
    corpus, and renders a per-(pattern, regime, horizon) hit-rate report.
    """
    from pathlib import Path

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
        corpus_filename=corpus_filename,
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
    from datetime import date

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


# Ranking types the QU scraper persists every slot. The day is only "fresh" when
# BOTH books are at/after the latest due slot — a partial scrape (e.g. top100 lands
# but bottom100 returns an empty no-op) must read as STALE so recover re-fires.
_QU100_RANKING_TYPES = ("top100", "bottom100")


def _recover_trading_day(now: datetime) -> date:
    """Anchor recover's "today" to the APP-LOCAL calendar date of ``now`` — the
    same timezone the schedule slots fire in.

    Recover must compare today's snapshots against the schedule, and the schedule
    slots are app-tz (the scheduler runs ``AsyncIOScheduler(timezone=app.tz)``).
    The day key MUST therefore be the app-local date so the latest-due-slot scan
    (``_latest_due_slot``) and the snapshot day-filter agree.

    This also matches the STORED ``data_date = market_date(captured_at)`` (ET) for
    every legitimately-scraped slot: scrapes fire only during US market hours,
    when the ET calendar date equals the app-local date (market hours are daytime
    in any reasonable app tz). So a close scrape at 1pm PT stores ``data_date`` =
    that Monday (ET) and an evening recover at 9pm PT resolves ``today`` = the same
    Monday (app-local) — they match.

    Using the ET date here (``market_date(now)``) instead would BREAK late-evening
    recovery: after ~9pm PT the ET clock has rolled to the next calendar day, so
    ``today`` would point at a not-yet-traded date while the latest due slot still
    belongs to the day that just ended — a missed Monday close would be checked
    against Tuesday, and on Friday night it would fall into the weekend fast-path
    and skip recovery entirely.
    """
    return now.date()


def _latest_due_slot(schedule: dict, now):
    """Return ``(name, slot_time)`` of the most-recent scheduled slot already due
    at ``now`` (``slot_time <= now``), or ``(None, None)`` when none is due yet.

    ``schedule`` maps slot name -> ``"HH:MM"`` in the app timezone; ``now`` is a
    tz-aware datetime in that zone. Single source of truth for "which slot should
    have run by now" — both the freshness check and the recovery-scrape label read
    from it so they can never drift.
    """
    latest_name = None
    latest_time = None
    for name, time_str in schedule.items():
        hour, minute = map(int, time_str.split(":"))
        slot_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if slot_time <= now and (latest_time is None or slot_time > latest_time):
            latest_time = slot_time
            latest_name = name
    return latest_name, latest_time


def _is_qu100_fresh(latest_captured_at, now, schedule: dict, tz) -> bool:
    """Is ONE book's ``captured_at`` present and fresh as of the latest due slot?

    Under the rebuild-the-day fix a day holds ONE ``captured_at`` per
    ``(data_date, ranking_type)`` and rows carry the LATEST scrape's
    ``capture_session`` — so the old per-``capture_session`` row count can no
    longer tell which slot ran. Freshness is detected PER DAY via ``captured_at``:

      * if no slot is due yet, nothing was expected -> fresh (no scrape needed);
      * otherwise this book is fresh iff a snapshot exists AND its ``captured_at``
        is at/after the most-recent-due slot.

    This is the per-ranking_type primitive; ``_qu100_day_is_fresh`` requires it to
    hold for EVERY book so a partial scrape never reads as fresh. ``now`` and
    ``latest_captured_at`` are tz-aware datetimes.
    """
    _, latest_due = _latest_due_slot(schedule, now)
    if latest_due is None:
        return True  # nothing due yet today
    if latest_captured_at is None:
        return False  # a slot is due but no data landed
    # captured_at is a timestamptz on Postgres (aware); a naive value (e.g. a
    # SQLite-backed read) is assumed UTC so `.astimezone` never reinterprets it as
    # local wall-clock. Compare in the app timezone.
    if latest_captured_at.tzinfo is None:
        from datetime import timezone as _timezone

        latest_captured_at = latest_captured_at.replace(tzinfo=_timezone.utc)
    return latest_captured_at.astimezone(tz) >= latest_due


def _qu100_day_is_fresh(db, today, now, schedule: dict, tz) -> bool:
    """Is today's QU100 snapshot fresh for EVERY ranking_type?

    Reads ``max(captured_at)`` per ranking_type for ``data_date == today`` and
    requires each expected book (``top100`` AND ``bottom100``) to be fresh. A
    single global ``max(captured_at)`` would let a fresh top100 mask a stale/empty
    bottom100 (partial-failure scrape) and report the day fresh when half the book
    is frozen. Used both at detection AND post-scrape to gate the Discord report.
    """
    from sqlalchemy import func

    from rainier.core.models import MoneyFlowSnapshot

    latest_by_type = dict(
        db.query(
            MoneyFlowSnapshot.ranking_type,
            func.max(MoneyFlowSnapshot.captured_at),
        )
        .filter(MoneyFlowSnapshot.data_date == today)
        .group_by(MoneyFlowSnapshot.ranking_type)
        .all()
    )
    return all(
        _is_qu100_fresh(latest_by_type.get(rt), now, schedule, tz)
        for rt in _QU100_RANKING_TYPES
    )


def _latest_due_session(schedule: dict, now) -> str:
    """The session name of the most-recent scheduled slot already due at ``now``.

    Used to label the single recovery scrape when today's data is stale. Falls
    back to the first slot if (defensively) none is due — the caller only invokes
    this when a slot IS due, so the fallback is just a safety net.
    """
    name, _ = _latest_due_slot(schedule, now)
    return name if name is not None else next(iter(schedule))


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

    # Detect today's QU100 freshness via the latest captured_at vs the schedule
    # (per-DAY, not per-capture_session — under the rebuild-the-day fix a day
    # holds one captured_at and the rows carry the LATEST scrape's session, so
    # per-session counts can no longer tell which slot ran). ``qu100_stale`` drives
    # both the recovery scrape AND whether the daily outlook is re-sent.
    qu100_stale = False
    # Anchor the day key to the APP-LOCAL date (the schedule's timezone) so the
    # latest-due-slot scan and the snapshot day-filter agree, and so late-evening
    # recovery still targets the day that just ended — see _recover_trading_day.
    today = _recover_trading_day(now)
    if today.weekday() >= 5:
        click.echo("  Weekend — no scrape sessions to check")
    else:
        from rainier.core.database import get_session

        with get_session() as db:
            day_fresh = _qu100_day_is_fresh(db, today, now, sessions_config, tz)

        if day_fresh:
            click.echo("  QU100 today: fresh (every book at/after the latest due slot)")
        else:
            qu100_stale = True
            issues.append("QU100 data stale (a book is missing today's latest slot)")
            # One recovery scrape targeting the most-recent due slot's session.
            recover_session = _latest_due_session(sessions_config, now)
            actions.append(f"scrape_{recover_session}")
            click.echo("  QU100 today: STALE — re-scraping the latest due slot")

    # --- 4. QU100 Discord report ---
    # Decoupled from scrape-action queueing: the daily outlook is re-sent ONLY
    # when freshness is RESTORED (re-read post-scrape, per ranking_type — see the
    # execution stage). A scrape that returns without raising but lands no fresh
    # data must not fire a report off a stale snapshot — that would reintroduce the
    # frozen-data bug. So we do NOT queue a "discord_report" action here; the
    # decision is made post-scrape from the DB, not from "the coroutine returned".
    click.echo("Checking QU100 Discord report...")
    if qu100_stale:
        click.echo("  Discord report: deferred — gated on restored freshness")
    else:
        click.echo("  Discord report: likely OK (data fresh)")

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

    # Re-send the daily outlook ONLY if the day was stale AND freshness was
    # actually RESTORED this run. We re-read the DB rather than trust "the scrape
    # coroutine returned": an empty/partial scrape (the documented empty-slot slip
    # -> _persist_qu100 no-op) returns without raising yet leaves the snapshot
    # stale. Gating on a post-scrape freshness re-read (per ranking_type) is the
    # only signal that won't fire a report off a frozen snapshot — the exact bug
    # this PR fixes. Data already fresh -> treated as already sent.
    report_sent = False
    freshness_restored = False
    if qu100_stale and recovered_scrapes:
        from datetime import time

        from rainier.core.database import get_session

        # Re-evaluate against the CURRENT clock, but CLAMPED to the day we set out
        # to recover (`today`). A recover that crossed a slot boundary (15:29 ->
        # 15:31) must judge against the now-overdue 15:30 slot, so the clock must
        # advance. But if the rerun crosses local MIDNIGHT, `now_after` lands on a
        # NEW day where no slot is due yet -> `_latest_due_slot` returns None ->
        # `_qu100_day_is_fresh` returns True and we'd resend the outlook off the
        # still-stale prior day (Codex). Clamp the clock to end-of-`today` so the
        # recovered day is always judged against its OWN last due slot, never a
        # fresh next-day clock. The day is fixed; the clock advances within it only.
        now_after = datetime.now(tz)
        if _recover_trading_day(now_after) > today:
            now_after = datetime.combine(today, time.max, tzinfo=tz)
        with get_session() as db:
            freshness_restored = _qu100_day_is_fresh(
                db, today, now_after, sessions_config, tz
            )
        if not freshness_restored:
            click.echo(
                "  Discord report: skipped — recovery scrape landed no fresh data"
            )
    if qu100_stale and freshness_restored:
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
                report_sent = True
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
    if report_sent:
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
    """Database management commands.

    Owns BOTH the legacy core/database.py singleton commands (``init``,
    ``backfill-prices``) AND the new Postgres canonical-store commands
    (``ping``, ``migrate``) defined further down in this file. There must
    be exactly ONE ``@cli.group() def db()`` in this module — declaring a
    second one shadows this group in click's registry and breaks the
    legacy subcommands (see CI #102 regression and the
    ``test_db_group_does_not_shadow_legacy_subcommands`` guard).
    """


@db.command(name="init")
@click.pass_context
def db_init(ctx):
    """Initialize database tables and hypertables."""
    from rainier.core.database import init_db

    click.echo("Initializing database...")
    init_db()
    click.echo("Database initialized successfully.")


@db.command(name="gc-test-schemas")
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Drop the leaked test schemas. Without this, dry-run lists them only.",
)
def db_gc_test_schemas(apply: bool) -> None:
    """Reap leaked throwaway test schemas from the LEGACY database.

    The paper-tracker test fixtures build disposable Postgres schemas
    (``rainier_paper_test*`` etc). A SIGKILL'd run can leave one behind in the
    live local TimescaleDB. This lists them (dry-run) and, with ``--apply``,
    drops only those matching the anchored allowlist regex — NEVER
    ``public`` / ``market`` / the active schema. Targets the legacy
    ``core.database`` engine (``LEGACY_DATABASE_URL``), never canonical Neon
    (the 2026-06-01 two-engine trap).
    """
    from rainier.core.database import get_engine
    from rainier.core.test_schema_gc import gc_test_schemas

    engine = get_engine()
    result = gc_test_schemas(engine, apply=apply)
    candidates = result["candidates"]
    if apply:
        dropped = result["dropped"]
        failed = result["failed"]
        click.echo(f"gc-test-schemas: dropped {len(dropped)} leaked schema(s).")
        for name in dropped:
            click.echo(f"  dropped {name}")
        if failed:
            for name, err in failed:
                click.echo(f"  FAILED to drop {name}: {err}", err=True)
            raise click.ClickException(
                f"gc-test-schemas: {len(failed)} schema(s) could not be "
                f"dropped (see above). Resolve the error and re-run --apply."
            )
    else:
        click.echo(
            f"DRY-RUN gc-test-schemas: would drop {len(candidates)} leaked "
            f"schema(s). Re-run with --apply to drop."
        )
        for name in candidates:
            click.echo(f"  {name}")


@db.command(name="backfill-prices")
@click.option(
    "--years",
    default=5,
    type=int,
    help="Lower-bound years of history (a floor; the fetch never starts LATER "
    "than the sweep-window start derived from the QU100 rankings).",
)
@click.option("--batch-size", default=20, type=int, help="Symbols per yfinance batch")
@click.option("--dry-run", is_flag=True, help="Show what would be fetched without fetching")
def db_backfill_prices(years, batch_size, dry_run):
    """Backfill historical daily OHLCV for all QU100 stocks via yfinance.

    Selection is COVERAGE-based, not presence-based: a symbol is re-fetched
    unless it already has a bar near BOTH ends of the sweep window (the start
    derived from the QU100 rankings, and today). A thin recent sliver (the AMZN
    case) or a stale tail no longer masks a multi-year gap. The download window
    starts at the sweep-window start so a re-selected symbol repairs its full
    history — not just the trailing ``--years``. After the run a 100%
    current-cohort coverage check reports any remaining shortfall loudly.
    """
    import math
    import time
    from datetime import timedelta

    import yfinance as yf
    from sqlalchemy import func, select

    from rainier.backtest.qu100_portfolio import (
        _save_prices_to_db,
        select_symbols_needing_backfill,
        sweep_window_start,
    )
    from rainier.core.database import get_session
    from rainier.core.models import MoneyFlowSnapshot
    from rainier.paper.calendar import DEFAULT_CALENDAR

    end = datetime.now()
    today = end.date()
    # Anchor the right boundary at the LAST COMPLETED trading session — the most
    # recent session whose bar yfinance reliably publishes. Today's session may be
    # in progress (its bar isn't published until after the close), so step back to
    # the previous session: requiring an as-yet-unpublished bar would make an
    # intraday run fail for a same-day top100 entrant. (Trade-off: after-close runs
    # lag one session; the next run catches today's bar. The "use today after the
    # close" optimisation needs the operator's run schedule — see Implementation
    # notes / BLOCKED.) Used for the coverage right edge, download end, cohort
    # lookup.
    end_date = DEFAULT_CALENDAR.prev_session(today)
    # yfinance `end` is EXCLUSIVE — pass the day after end_date so its bar is
    # actually fetched (codex).
    download_end = end_date + timedelta(days=1)

    # Empty rankings: no top100 snapshot yet (fresh/staging DB, pre-first-scrape).
    # Preserve the old no-op instead of raising from sweep_window_start() — a
    # bootstrap/smoke run of `db backfill-prices` before any rankings load must
    # exit cleanly (codex).
    with get_session() as session:
        has_top100 = session.execute(
            select(MoneyFlowSnapshot.symbol)
            .where(MoneyFlowSnapshot.ranking_type == "top100")
            .limit(1)
        ).first()
    if has_top100 is None:
        click.echo("No QU100 top100 rankings in the database yet. Nothing to do.")
        return

    # Two distinct windows, do NOT conflate them:
    #   cov_start (coverage gate) = sweep_start, the EXACT window the miss-sweep
    #     consumes. Selection + the post-run assertion key on THIS — a symbol is
    #     judged covered only against what the sweep reads.
    #   download_start (fetch) = min(sweep_start, years_floor). `--years` is a
    #     FLOOR that may WIDEN the download earlier (extra history is harmless),
    #     but the coverage gate must NOT require history older than the sweep —
    #     else a current constituent that IPOed after years_floor could never be
    #     "covered" and the gate would raise on every run (codex P1).
    sweep_start = sweep_window_start()
    cov_start = sweep_start
    years_floor = date(end.year - years, end.month, end.day)
    download_start = min(sweep_start, years_floor)

    with get_session() as session:
        # Restrict the universe to ranking_type='top100' — the EXACT slice the
        # miss-sweep consumes. Auxiliary bottom100/other-type-only names are
        # never read by the sweep; including them would feed the stricter sweep-
        # window coverage check symbols with no first-top100-ranking date and
        # schedule wasted re-downloads (codex).
        qu_symbols = sorted(
            session.execute(
                select(func.distinct(MoneyFlowSnapshot.symbol)).where(
                    MoneyFlowSnapshot.ranking_type == "top100"
                )
            ).scalars().all()
        )

    # SELECTION (what to fetch): full sweep window per symbol, NO ranking clamp —
    # fetch generously so pre-signal lookback and post-signal tails the backtest
    # reads are pulled. The post-run GATE below clamps to ranked life so it does
    # not fail forever on IPO/delisted names.
    missing = select_symbols_needing_backfill(qu_symbols, cov_start, end_date)
    has_prices = sorted(set(qu_symbols) - set(missing))

    click.echo(f"QU100 symbols: {len(qu_symbols)}")
    click.echo(f"Covered (sweep window {cov_start}..{end_date}): {len(has_prices)}")
    click.echo(f"Needing backfill (incomplete/absent): {len(missing)}")
    click.echo(
        f"Fetch range: {download_start} to {end_date} "
        f"(sweep/coverage start {sweep_start}, --years floor {years_floor})"
    )

    if dry_run:
        if missing:
            click.echo(f"\nWould fetch: {missing[:50]}{'...' if len(missing) > 50 else ''}")
        return

    if missing:
        total_batches = math.ceil(len(missing) / batch_size)
        click.echo(f"\nFetching {len(missing)} symbols in {total_batches} batches...")

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
                    # Full (possibly --years-widened) download window, never
                    # capped below the sweep start: a re-selected symbol repairs
                    # its entire history.
                    start=str(download_start),
                    end=str(download_end),  # exclusive → day after end_date
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                if not yf_df.empty:
                    if not isinstance(yf_df.columns, pd.MultiIndex) and len(batch) == 1:
                        yf_df.columns = pd.MultiIndex.from_product(
                            [yf_df.columns, batch]
                        )
                    # _save_prices_to_db → _yf_to_long logs yf_batch_dropped_symbols
                    # for any requested symbol yfinance omitted (no silent drop).
                    _save_prices_to_db(yf_df, batch)
                    fetched += len(batch)
                else:
                    failed += len(batch)
                    click.echo("    No data returned for batch")
            except Exception as exc:
                failed += len(batch)
                click.echo(f"    Error: {exc}")

        click.echo(f"\nDone. Fetched: {fetched}, Failed: {failed}")
    else:
        click.echo("All QU100 symbols span the sweep window. Nothing to fetch.")

    # Post-run gate: validate the SAME universe the command just attempted to
    # repair (every historical top100 symbol — `qu_symbols`), NOT just today's
    # cohort. The backtest reads `load_rankings_from_db()` (all historical top100
    # constituents, current OR former), so a former member yfinance dropped would
    # otherwise exit 0 here yet stay silently broken downstream (codex). Report
    # NON-silently AND FAIL (non-zero exit) — a warn-and-exit-0 makes an
    # incomplete repair indistinguishable from success in cron/CI. Per the plan's
    # blocker: a symbol still uncovered after it WAS requested (a genuine
    # upstream omission or a post-start listing) is a STOP-and-raise for the
    # operator, never a silently lowered bar.
    still_missing = select_symbols_needing_backfill(
        qu_symbols, cov_start, end_date, clamp_to_ranking_life=True
    )
    if still_missing:
        preview = still_missing[:50]
        raise click.ClickException(
            f"{len(still_missing)} top100 symbol(s) still lack full sweep-window "
            f"coverage after the run: {preview}"
            f"{'...' if len(still_missing) > 50 else ''}. "
            "Investigate (genuine yfinance omission or post-start listing) — "
            "do not lower the coverage bar to make this pass."
        )
    click.echo(
        "\nCoverage check: 100% of the historical top100 universe is covered."
    )


@db.command(name="ingest-prices")
@click.option(
    "--universe",
    type=click.Choice(["qu100", "active", "screened"]),
    default="active",
    help="qu100=full universe (weekly); active=pending/open paper symbols; "
    "screened=today's top-50",
)
@click.option("--date", "as_of_iso", default=None, help="As-of date (YYYY-MM-DD)")
@click.option("--window-days", default=10, type=int, help="Recent gap window (sessions)")
def db_ingest_prices(universe, as_of_iso, window_days):
    """Gap-aware daily price ingest (Phase 0, design D9).

    Per-(symbol,date) gap detection over the recent window + (symbol,date)
    upsert (DO UPDATE) so split-adjusted values self-heal. Idempotent.
    """
    from datetime import date as _date

    from rainier.paper.ingest import (
        _yfinance_fetch_fn,
        active_symbols,
        ingest_prices,
        screened_symbols,
    )

    as_of = _date.fromisoformat(as_of_iso) if as_of_iso else _date.today()

    if universe == "active":
        symbols = active_symbols()
    elif universe == "screened":
        symbols = screened_symbols(as_of)
    else:
        from sqlalchemy import func as _func
        from sqlalchemy import select as _select

        from rainier.core.database import get_session
        from rainier.core.models import MoneyFlowSnapshot

        with get_session() as session:
            symbols = sorted(
                session.execute(
                    _select(_func.distinct(MoneyFlowSnapshot.symbol))
                ).scalars().all()
            )

    click.echo(f"Ingesting {len(symbols)} {universe} symbols (as_of={as_of})...")
    if not symbols:
        click.echo("No symbols to ingest.")
        return
    res = ingest_prices(
        symbols, as_of=as_of, fetch_fn=_yfinance_fetch_fn, window_days=window_days
    )
    click.echo(f"Done. Upserted {res['upserted']} bars.")


# ---------------------------------------------------------------------------
# Paper-trade tracker commands (design DESIGN-qu100-llm-feedback-loop)
# ---------------------------------------------------------------------------


@cli.group(name="paper")
def paper_group() -> None:
    """QU100 paper-trade tracker — open positions, update exits, report."""


@paper_group.command(name="shadow-replay")
@click.option(
    "--start", "start_iso", required=True, help="Window start (YYYY-MM-DD)"
)
@click.option("--end", "end_iso", required=True, help="Window end (YYYY-MM-DD)")
@click.option(
    "--thresholds",
    default="4,5,6",
    show_default=True,
    help="Comma-separated WATCH confidence thresholds to sweep.",
)
@click.option(
    "--benchmark", default="SPY", show_default=True, help="Benchmark symbol."
)
def paper_shadow_replay(start_iso, end_iso, thresholds, benchmark):
    """WS A — replay the shadow WATCH-buy book over a window per threshold T.

    Drives the REAL fill/exit engine day by day over historical theses, opening
    SHADOW positions (excluded from the live book) at each T, and prints the
    shadow book's return vs cash and vs the benchmark. Use the two review
    windows (2026-05-29→06-12 and 2026-05-22→06-12) to compare arms.

    NOTE: shadow rows persist in the DB. Run against a throwaway/replay schema
    or `rainier db gc-test-schemas` afterward — this command does not isolate
    arms from each other; it is a measurement tool, not a live mutation.
    """
    from datetime import date as _date

    from rainier.paper.calendar import DEFAULT_CALENDAR
    from rainier.paper.replay import replay_threshold

    start = _date.fromisoformat(start_iso)
    end = _date.fromisoformat(end_iso)
    ts = [int(t) for t in thresholds.split(",") if t.strip()]
    days = DEFAULT_CALENDAR.sessions_between(start, end)

    click.echo(f"Shadow replay {start}..{end} ({len(days)} sessions), bench={benchmark}")
    click.echo(f"{'T':>3} {'fired':>6} {'book%':>8} {'cash%':>7} {'bench%':>8}")
    for t in ts:
        arm = replay_threshold(
            threshold=t, trading_days=days, benchmark_symbol=benchmark
        )
        click.echo(
            f"{arm.threshold:>3} {arm.fired:>6} "
            f"{arm.book_return_pct * 100:>7.2f}% "
            f"{arm.cash_return_pct * 100:>6.2f}% "
            f"{arm.benchmark_return_pct * 100:>7.2f}%"
        )


@paper_group.command(name="open")
@click.option("--date", "as_of_iso", default=None, help="Fill as-of date (YYYY-MM-DD)")
@click.pass_context
def paper_open(ctx, as_of_iso):
    """Fill pending positions at their T+1 trading-session open."""
    from datetime import date as _date

    from rainier.paper.positions import fill_pending_positions

    as_of = _date.fromisoformat(as_of_iso) if as_of_iso else _date.today()
    # Snapshot the learned time-stop at fill, same as the scheduled daily path
    # (codex iter-2 P3). Read from the Click-loaded settings (honors --config,
    # codex iter-7) — NOT get_settings(), which would ignore a non-default root.
    settings = ctx.obj["settings"]
    learned_ts = settings.llm_thesis.learned_time_stop_days
    res = fill_pending_positions(as_of=as_of, learned_time_stop_days=learned_ts)
    click.echo(f"Filled {res['filled']}, expired {res['expired']}.")


@paper_group.command(name="update")
@click.option("--date", "as_of_iso", default=None, help="As-of date (YYYY-MM-DD)")
def paper_update(as_of_iso):
    """Walk open positions' OHLC and close any that triggered an exit."""
    from datetime import date as _date

    from rainier.paper.positions import update_open_positions

    as_of = _date.fromisoformat(as_of_iso) if as_of_iso else _date.today()
    res = update_open_positions(as_of=as_of)
    click.echo(
        f"Closed {res['closed']} "
        f"(same-bar ambiguous: {res['same_bar_ambiguous_exits']})."
    )


@paper_group.command(name="report")
@click.option("--date", "as_of_iso", default=None, help="Report as-of date (YYYY-MM-DD)")
@click.option("--week", is_flag=True, help="Weekly missed-winner report (Phase 3)")
@click.option(
    "--regenerate",
    is_flag=True,
    help="Recompute from raw inputs + upsert (else plain re-render from snapshot)",
)
def paper_report(as_of_iso, week, regenerate):
    """Render a paper-book report.

    Plain (default) re-renders from the persisted ``paper_report_snapshot`` only.
    ``--regenerate`` recomputes from raw inputs and upserts — for ``--week``
    that's money_flow_snapshots/stock_prices/screened_stocks/analysis_results/
    paper_trade (never the mutable ResearchInsight queue).
    """
    from datetime import date as _date

    from rainier.paper.report import (
        REPORT_TYPE_DAILY,
        REPORT_TYPE_WEEKLY,
        compute_daily_payload,
        load_snapshot,
        persist_daily_snapshot,
        render_payload,
    )

    as_of = _date.fromisoformat(as_of_iso) if as_of_iso else _date.today()

    if week:
        from rainier.paper.sweep import (
            compute_weekly_payload,
            persist_weekly_snapshot,
            render_weekly_payload,
        )

        if regenerate:
            # Recompute from the raw period inputs (historical as_of selects
            # the historical cohort: max data_date <= as_of). No price ingest,
            # no insight emission, no Discord — just compute + upsert + render.
            payload = compute_weekly_payload(as_of)
            persist_weekly_snapshot(as_of, payload)
            click.echo(render_weekly_payload(payload))
            return
        payload = load_snapshot(REPORT_TYPE_WEEKLY, as_of)
        if payload is None:
            raise click.ClickException(
                f"No weekly snapshot for {as_of}. "
                "Run with --regenerate to compute one."
            )
        click.echo(render_weekly_payload(payload))
        return

    if regenerate:
        payload = compute_daily_payload(as_of)
        persist_daily_snapshot(as_of, payload)
        click.echo(render_payload(payload))
        return

    # Plain: snapshot-only.
    payload = load_snapshot(REPORT_TYPE_DAILY, as_of)
    if payload is None:
        raise click.ClickException(
            f"No daily snapshot for {as_of}. Run with --regenerate to compute one."
        )
    click.echo(render_payload(payload))


@paper_group.command(name="appearances")
@click.argument("symbol")
@click.option(
    "--as-of", "as_of_iso", default=None, help="As-of date (YYYY-MM-DD, default today)"
)
@click.option(
    "--window",
    default=None,
    type=int,
    help="Trailing trading-session window (default: chart_lookback_days config)",
)
@click.option(
    "--all-sessions",
    is_flag=True,
    help="Every intraday top100 capture (default: one row/day, latest capture)",
)
@click.pass_context
def paper_appearances(ctx, symbol, as_of_iso, window, all_sessions):
    """Days SYMBOL was in the QU100 top-100 (and its rank each day)."""
    from datetime import date as _date

    from rainier.paper.ingest import get_qu100_appearances

    symbol = symbol.strip().upper()
    as_of = _date.fromisoformat(as_of_iso) if as_of_iso else _date.today()
    if window is None:
        window = ctx.obj["settings"].llm_thesis.chart_lookback_days

    apps = get_qu100_appearances(
        symbol, as_of=as_of, window=window, all_sessions=all_sessions
    )
    if not apps:
        click.echo(
            f"No QU100 top-100 appearances for {symbol} in the last "
            f"{window} sessions ending {as_of}."
        )
        return
    for a in apps:
        line = f"{a.data_date}  #{a.rank}"
        if all_sessions:
            line += f"  {a.capture_session}  {a.captured_at.isoformat()}"
        click.echo(line)


@paper_group.command(name="chart")
@click.argument("symbol")
@click.option(
    "--as-of", "as_of_iso", default=None, help="As-of date (YYYY-MM-DD, default today)"
)
@click.option(
    "--window",
    default=None,
    type=int,
    help="Daily-bar window (default: chart_lookback_days config)",
)
@click.option(
    "--out",
    "out_path",
    default=None,
    help="Output PNG path (default ./<SYMBOL>_<as-of>.png)",
)
@click.pass_context
def paper_chart(ctx, symbol, as_of_iso, window, out_path):
    """Render-on-demand archive chart (zero stored pixels, design App. C).

    Rebuilds the annotated chart for SYMBOL at --as-of from stored inputs
    (stock_prices + money_flow_snapshots + paper_trade). On unchanged data
    this reproduces a stored trade-close chart byte-identically.
    """
    from datetime import date as _date
    from pathlib import Path as _Path

    from rainier.paper.chart_archive import regenerate_chart

    symbol = symbol.strip().upper()
    as_of = _date.fromisoformat(as_of_iso) if as_of_iso else _date.today()
    if window is None:
        window = ctx.obj["settings"].llm_thesis.chart_lookback_days

    try:
        png, sha = regenerate_chart(symbol, as_of=as_of, window=window)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    out = _Path(out_path) if out_path else _Path(f"{symbol}_{as_of}.png")
    out.write_bytes(png)
    click.echo(f"Wrote {out} ({len(png):,} bytes, sha256 {sha[:12]})")


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

    # send_stock_candidates() catches httpx exceptions internally so a
    # Discord hiccup never crashes the scrape pipeline — but it now RETURNS a
    # DiscordSendResult whose per-endpoint failure counts reflect what landed.
    # This probe inspects those counts and exits non-zero on any failed POST,
    # surfacing the "false success" path (rotated webhook, 4xx/5xx, timeout)
    # operators use this command to diagnose. Detection is return-value based,
    # so it works regardless of the logger's level/config and never perturbs
    # the caller's logging setup.
    try:
        send_result = send_stock_candidates(
            [candidate],
            discord_cfg,
            theses={fake_symbol: thesis_dict},
            dashboard_base_url=None,
        )
    except Exception as exc:  # noqa: BLE001 — synthetic-payload bugs surface here
        raise click.ClickException(f"Discord POST failed: {exc}") from exc

    failures: list[str] = []
    if send_result.candidate_payloads_failed:
        failures.append(
            f"summary channel: {send_result.candidate_payloads_failed} "
            "payload(s) failed"
        )
    if send_result.thesis_failed:
        failures.append(
            f"thesis channel: {send_result.thesis_failed} embed(s) failed"
        )
    if failures:
        raise click.ClickException(
            "Discord POST failed (send_stock_candidates caught the error "
            "internally; the probe surfaced it via return counts): "
            + "; ".join(failures)
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
@click.option("--results-cache/--no-results-cache", default=True, show_default=True,
              help="Keep results.parquet (~570 MB at Phase 2) on disk after the "
                   "run. --no-results-cache deletes it (and its fingerprint) once "
                   "the walk-forward + report have been derived from it — the "
                   "computation and report output are unchanged.")
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
    results_cache: bool,
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
        results_cache_companions,
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

    # --no-results-cache: drop the giant results.parquet now that the
    # walk-forward set + report have been derived from it. The report embeds
    # the parquet's SHA-256 and the walk-forward parquet is its own file, so
    # the run's output is unchanged; only the regenerable ~570 MB cache goes.
    # Skipped (with a warning) when --no-report, since a later report render
    # would have nothing to read.
    if not results_cache:
        if not report:
            click.echo(
                "  --no-results-cache ignored: --no-report leaves no report to "
                "derive from the cache, so results.parquet is kept."
            )
        else:
            removed = results_cache_companions(results_path)
            for path in removed:
                path.unlink()
            if removed:
                click.echo(
                    f"  --no-results-cache: removed {len(removed)} cache file(s) "
                    f"({results_path.name} + companions)."
                )


# ---------------------------------------------------------------------------
# cache — manage regenerable on-disk caches
# ---------------------------------------------------------------------------


@cli.group()
def cache() -> None:
    """Manage regenerable on-disk caches under data/cache/."""


@cache.command("clean")
@click.option("--yes", is_flag=True, default=False,
              help="Skip the confirmation prompt.")
def cache_clean(yes: bool) -> None:
    """Delete the TQQQ/SQQQ SMA sweep cache (data/cache/tqqq_sma/).

    The cache (prices + results parquets, ~570 MB at Phase 2) is pure sweep
    output and is fully regenerated by the next `rainier sma-sweep` run.
    """
    from rainier.backtest.tqqq_sma_sweep import CACHE_DIR, clean_cache

    if not CACHE_DIR.exists():
        click.echo(f"Nothing to clean — {CACHE_DIR} does not exist.")
        return

    if not yes:
        click.confirm(
            f"Delete regenerable sweep cache at {CACHE_DIR}?",
            abort=True,
        )

    removed = clean_cache()
    click.echo(f"Removed {len(removed)} item(s) from {CACHE_DIR}.")


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
    "--adopt",
    "--in-place",
    "adopt",
    is_flag=True,
    default=False,
    help=(
        "With --force: atomically replace the (stale) canonical --out with the "
        "fresh cohort in place (no orphan sibling, no manual mv) AND mirror the "
        "fresh OHLCV + registries to Neon. The sanctioned cohort -> canonical "
        "bridge for a stale-canonical self-heal."
    ),
)
@click.option(
    "--incremental",
    is_flag=True,
    default=False,
    help=(
        "Daily-cron refresh: ignore --start/--end, fetch only the last few "
        "calendar days, and upsert on (symbol, date) into the existing cache "
        "in place (no --force / cohort). Mirrors "
        "`market-breadth backfill-ohlcv --incremental`."
    ),
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
    adopt: bool,
    incremental: bool,
    dry_run: bool,
    allow_empty: str,
    allow_gaps: str,
    min_coverage: float | None,
    ticker_registry: str,
    sector_registry: str,
) -> None:
    """Backfill the OHLCV cache + seed ticker/sector registries.

    Operator-run for the one-shot mode; cron-driven for ``--incremental``.
    Hits the yfinance network.
    """
    import importlib.util
    from datetime import date as _date

    from rainier.breadth import registry as _reg
    from rainier.breadth import universe_loader as _ul

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
        adopt=adopt,
        incremental=incremental,
        dry_run=dry_run,
        allow_empty=[s.strip() for s in allow_empty.split(",") if s.strip()],
        allow_gaps=[s.strip() for s in allow_gaps.split(",") if s.strip()],
        min_coverage=coverage,
        # Pass the universe YAML so backfill() mirrors the OHLCV frame +
        # ticker/sector registries into market.* (Neon). Without this the
        # daily cron's --incremental refresh advances the PARQUET only and
        # leaves market.thematic_ohlcv STALE — exactly the gap a live
        # catch-up found 2026-05-30 (features/labels reached today but
        # thematic_ohlcv was stuck days behind). _dual_write_pg fires only
        # when yaml_path is set and is a no-op when DATABASE_URL is unset, so
        # this is safe on both the one-shot and incremental paths.
        yaml_path=yaml_p,
        ticker_registry_path=Path(ticker_registry),
        sector_registry_path=Path(sector_registry),
    )

    if dry_run:
        plan = result
        # backfill() resolves the incremental window internally, so the dry-run
        # plan reflects the actual window that would be fetched.
        click.echo(
            f"DRY-RUN: would fetch {len(symbols)} symbols "
            f"{plan['start']}..{plan['end']} -> {plan['planned_out']}"
        )
        for sym in symbols:
            click.echo(f"  {sym}")
        return

    written_path = result
    if incremental:
        mode = "incremental"
    elif force and adopt:
        mode = "adopt (canonical replaced in place)"
    elif force:
        mode = "force (sibling cohort)"
    else:
        mode = "one-shot"
    click.echo(f"wrote OHLCV cache ({mode}) -> {written_path}")

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


@thematic.command("gc-cohorts")
@click.option(
    "--out",
    "out_path",
    type=click.Path(),
    default="data/cache/thematic_universe.parquet",
    show_default=True,
    help="Canonical parquet path; siblings next to it are gc candidates.",
)
@click.option(
    "--keep",
    type=int,
    default=2,
    show_default=True,
    help="Number of most-recent sibling cohorts to retain (newest by mtime).",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Delete the reap candidates. Without this, dry-run lists them only.",
)
def thematic_gc_cohorts(out_path: str, keep: int, apply: bool) -> None:
    """Reap orphan sibling cohort parquets, keeping canonical + N latest.

    ``--force`` (without ``--adopt``) writes dated sibling cohorts next to the
    canonical ``thematic_universe.parquet``; these accumulate. This reaps them,
    keeping the canonical (NEVER deleted) plus the ``--keep`` most-recent
    cohorts. Dry-run by default — pass ``--apply`` to delete. Mirrors
    ``fleet gc``'s surface-then-apply ethos.
    """
    import importlib.util

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

    canonical = Path(out_path)
    plan = module.gc_cohorts(out_path=canonical, keep=keep, apply=apply)

    candidates = plan["delete"]
    kept = plan["keep"]
    if apply:
        deleted = plan["deleted"]
        failed = plan.get("failed", [])
        click.echo(
            f"gc-cohorts: deleted {len(deleted)} orphan cohort(s); "
            f"kept canonical {canonical.name} + {len(kept)} latest."
        )
        for p in deleted:
            click.echo(f"  deleted {p}")
        # Surface unlink failures as a hard error so automation/operators don't
        # think gc completed when an orphan is still on disk (codex iter-2 [P2]).
        if failed:
            for p, err in failed:
                click.echo(f"  FAILED to delete {p}: {err}", err=True)
            raise click.ClickException(
                f"gc-cohorts: {len(failed)} orphan cohort(s) could not be "
                f"deleted (see above). Resolve the error and re-run --apply."
            )
    else:
        click.echo(
            f"DRY-RUN gc-cohorts: would delete {len(candidates)} orphan "
            f"cohort(s); would keep canonical {canonical.name} + {len(kept)} "
            f"latest. Re-run with --apply to delete."
        )
        for p in candidates:
            click.echo(f"  would delete {p}")
    for p in kept:
        click.echo(f"  keep {p}")


@thematic.command("backfill-names")
@click.option(
    "--universe-yaml",
    "yaml_path",
    type=click.Path(exists=True),
    default="config/thematic_universe.yaml",
    show_default=True,
    help="Path to the thematic universe YAML — flattens to the symbol list.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default="data/cache/etf_names.parquet",
    show_default=True,
    help="Destination parquet cache for yfinance long/short names.",
)
@click.option(
    "--refresh-stale",
    is_flag=True,
    default=False,
    help=(
        "When set, skip the backfill if the parquet exists and is newer "
        "than 30 days. Cron-friendly: ETF names rarely change so daily "
        "refetch is wasteful."
    ),
)
def thematic_backfill_names(
    yaml_path: str, output_path: str, refresh_stale: bool
) -> None:
    """One-shot backfill: fetch yfinance long/short names for the thematic universe.

    Writes ``data/cache/etf_names.parquet`` (schema:
    ``symbol, long_name, short_name, fetched_at``). The ETF dashboard
    renderer reads this on each invocation to populate the Name column.

    Idempotent — re-running upserts on ``symbol``. yfinance rate-limit
    triggers ONE retry, then surfaces a clean non-zero exit so the cron
    Discord alert fires.
    """
    from rainier.breadth import universe_loader
    from rainier.dashboard import etf_names_backfill

    out = Path(output_path)

    # Load the universe FIRST so --refresh-stale can carve out newly-added
    # tickers that aren't in the cache yet. Without this, the 30-day stale
    # gate short-circuits the whole run on a fresh parquet, so a just-added
    # symbol renders as a fallback for up to 30 days.
    #
    #   refresh-stale + stale parquet      -> full-universe backfill
    #   refresh-stale + fresh, new symbols -> fetch ONLY the new symbols
    #   refresh-stale + fresh, none new    -> skip (existing rows keep cadence)
    #   no refresh-stale                   -> full-universe backfill
    spec = universe_loader.load_universe(Path(yaml_path))
    symbols = list(spec.all_tickers)
    if not symbols:
        raise click.ClickException("thematic_universe.yaml flattened to zero tickers")

    if refresh_stale and not etf_names_backfill.is_stale(out):
        missing = etf_names_backfill.missing_from_cache(symbols, out)
        if not missing:
            click.echo(f"etf_names parquet is fresh (<30d); skipping -> {out}")
            return
        written = etf_names_backfill.backfill_names(symbols=missing, out_path=out)
        click.echo(
            f"etf_names parquet is fresh (<30d) but {len(missing)} new "
            f"symbol(s) missing; force-refreshed {missing} -> {written}"
        )
        return

    written = etf_names_backfill.backfill_names(symbols=symbols, out_path=out)
    click.echo(f"wrote etf_names ({len(symbols)} symbols) -> {written}")


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

    from rainier.breadth import universe_loader as _ul

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


# ---------------------------------------------------------------------------
# market-breadth — S&P 500 OHLCV substrate for the breadth-webpage
# ---------------------------------------------------------------------------


@cli.group(name="market-breadth")
def market_breadth() -> None:
    """S&P 500 breadth-webpage data substrate.

    Twin of the `thematic` group — same shape, different universe + parquet.
    Owns `config/sp500_universe.yaml` → `data/cache/sp500_universe.parquet`.
    """


@market_breadth.command("backfill-ohlcv")
@click.option(
    "--yaml",
    "yaml_path",
    type=click.Path(exists=True),
    default="config/sp500_universe.yaml",
    show_default=True,
    help="Path to the S&P 500 universe YAML.",
)
@click.option(
    "--since",
    default="2020-01-01",
    show_default=True,
    help="One-shot backfill anchor (YYYY-MM-DD). Ignored with --incremental.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default="data/cache/sp500_universe.parquet",
    show_default=True,
)
@click.option(
    "--incremental",
    is_flag=True,
    default=False,
    help="Fetch last 5 calendar days only, upsert into the existing parquet.",
)
@click.option(
    "--chunk-size",
    type=int,
    default=None,
    help="Symbols per yfinance call. Default: package constant (25).",
)
@click.option(
    "--min-coverage",
    type=float,
    default=None,
    help=(
        "Fraction of requested symbols that must return rows; below threshold → "
        "exit non-zero (Discord alert via cron-wrapper) and preserve prior parquet. "
        "Default: package constant (0.95). Pass 0.0 to disable the gate."
    ),
)
def market_breadth_backfill_ohlcv(
    yaml_path: str,
    since: str,
    output_path: str,
    incremental: bool,
    chunk_size: int | None,
    min_coverage: float | None,
) -> None:
    """Backfill the S&P 500 OHLCV parquet via yfinance.

    Operator-run for the one-shot mode; cron-driven for `--incremental`.
    The yfinance network call is real — tests mock it via a fetch_fn
    injection point that lives in the python package, not the CLI.
    """
    from rainier.market_breadth import ohlcv_backfill, universe_loader

    entries = universe_loader.load_sp500_universe(Path(yaml_path))
    symbols = [sym for sym, _sec in entries]

    kwargs: dict[str, object] = {
        "symbols": symbols,
        "since": since,
        "out_path": Path(output_path),
        "incremental": incremental,
    }
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    if min_coverage is not None:
        kwargs["min_coverage"] = min_coverage

    written = ohlcv_backfill.backfill(**kwargs)  # type: ignore[arg-type]
    mode = "incremental" if incremental else "one-shot"
    click.echo(f"wrote sp500 OHLCV ({mode}, {len(symbols)} symbols) -> {written}")


@market_breadth.command("backfill-spy")
@click.option(
    "--since",
    default="2018-01-01",
    show_default=True,
    help="One-shot backfill anchor (YYYY-MM-DD). SPY pane needs 5y for the toggle.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default="data/cache/spy_history.parquet",
    show_default=True,
)
@click.option(
    "--incremental",
    is_flag=True,
    default=False,
    help="Fetch last 5 calendar days only, upsert into the existing parquet.",
)
def market_breadth_backfill_spy(
    since: str, output_path: str, incremental: bool
) -> None:
    """One-shot SPY OHLCV backfill for the breadth dashboard's price pane.

    DESIGN-market-breadth-v0.1-canonical-layout §2.1. Single ticker via the
    same ``ohlcv_backfill`` plumbing the S&P 500 universe uses — atomic
    parquet write, retry-on-rate-limit, idempotent (upsert on
    ``(symbol, date)``). The dashboard renderer loads this parquet on
    ``--spy-path`` to render the top price pane.
    """
    from rainier.market_breadth import ohlcv_backfill

    written = ohlcv_backfill.backfill(
        symbols=["SPY"],
        since=since,
        out_path=Path(output_path),
        incremental=incremental,
        # Single-ticker run; coverage check is meaningless here.
        min_coverage=0.0,
    )
    mode = "incremental" if incremental else "one-shot"
    click.echo(f"wrote SPY OHLCV ({mode}) -> {written}")

    # Dual-write into market.benchmark_ohlcv (non-fatal mirror; parquet above is
    # load-bearing). Read back the parquet — for an incremental run this is the
    # merged full history, so PG mirrors the on-disk file exactly.
    _dual_write_benchmark_pg(pd.read_parquet(written))


@market_breadth.command("compute-indicators")
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    default="data/cache/sp500_universe.parquet",
    show_default=True,
    help="Path to the S&P 500 OHLCV parquet (long format).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default="data/cache/sp500_breadth_daily.parquet",
    show_default=True,
    help="Destination long-format breadth parquet.",
)
@click.option(
    "--epoch",
    default="2020-01-01",
    show_default=True,
    help=(
        "Cumulative-indicator anchor (YYYY-MM-DD). ad_cumulative and "
        "mcclellan_summation start the day strictly AFTER this date."
    ),
)
def market_breadth_compute_indicators(
    input_path: str, output_path: str, epoch: str
) -> None:
    """Compute the 12 S&P 500 breadth indicators and write long parquet.

    Re-computes from scratch every run — cheap because the universe is
    ~500 symbols × ~1400 trading days (~700k rows). Idempotent: same
    input parquet → byte-identical output parquet. Cron runs this after
    `backfill-ohlcv --incremental` so today's bar is already in place.
    """
    from datetime import date

    from rainier.market_breadth import compute

    epoch_d = date.fromisoformat(epoch)
    written = compute.compute_indicators(
        input_path=Path(input_path),
        output_path=Path(output_path),
        epoch=epoch_d,
    )
    click.echo(f"wrote sp500 breadth indicators -> {written}")

    # Dual-write the FULL breadth history into market.breadth_indicator_daily
    # (non-fatal mirror; parquet above is load-bearing). Read back the parquet
    # we just wrote so PG mirrors exactly what landed on disk.
    _dual_write_breadth_pg(pd.read_parquet(written))


@market_breadth.command("render-html")
@click.option(
    "--source",
    type=click.Choice(["neon", "parquet"]),
    default="neon",
    show_default=True,
    help=(
        "Data source. 'neon' (default) reads the canonical "
        "market.breadth_indicator_daily + market.benchmark_ohlcv store; "
        "'parquet' reads the local caches (--input / --spy-path). Passing "
        "--input or --spy-path implies parquet."
    ),
)
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    default=None,
    help=(
        "Long-format breadth parquet (parquet source only). "
        "[default: data/cache/sp500_breadth_daily.parquet]"
    ),
)
@click.option(
    "--asof",
    default=None,
    help=(
        "Display + filter date (YYYY-MM-DD). Defaults to the most recent "
        "asof_date present in the source (parquet last row / Neon max)."
    ),
)
@click.option(
    "--rendered-at-pt",
    required=True,
    help="Wall-clock HH:MM (Pacific) for the header timestamp. Caller-supplied "
    "to keep the renderer deterministic (no datetime.now inside the renderer).",
)
@click.option(
    "--window-days",
    type=int,
    default=504,
    show_default=True,
    help="Trailing-window cap for the chart series (~2y default).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default="out/dashboards/market-breadth.html",
    show_default=True,
)
@click.option(
    "--spy-path",
    "spy_path",
    type=click.Path(),
    default=None,
    help=(
        "Optional SPY OHLCV parquet (parquet source only; implies parquet). "
        "When present, the SPY price pane renders at the top of the page. "
        "Missing file → SPY pane is omitted (back-compat path). "
        "[default: data/cache/spy_history.parquet]"
    ),
)
def market_breadth_render_html(
    source: str,
    input_path: str | None,
    asof: str | None,
    rendered_at_pt: str,
    window_days: int,
    output_path: str,
    spy_path: str | None,
) -> None:
    """Render the S&P 500 market-breadth self-contained HTML dashboard.

    Default source is the canonical Neon ``market.breadth_indicator_daily``
    store (SPY pane from ``market.benchmark_ohlcv``); ``--source parquet`` (or
    passing ``--input`` / ``--spy-path``) reads the local caches for offline /
    back-compat use. Fails loud on a zero-row Neon result — never silently
    publishes a stale parquet.
    """
    import os
    import sys
    from datetime import date as _date

    import pandas as _pd

    from rainier.market_breadth.render import render_breadth_html

    # Passing an explicit parquet path implies the parquet source (back-compat).
    if input_path is not None or spy_path is not None:
        source = "parquet"

    if source == "parquet":
        eff_input = input_path or "data/cache/sp500_breadth_daily.parquet"
        if not Path(eff_input).exists():
            click.echo(f"error: breadth parquet not found: {eff_input}", err=True)
            sys.exit(1)
        breadth = _pd.read_parquet(eff_input)
        # Fail-loud on zero-row parquet. Otherwise `.max()` returns `pd.NaT`,
        # `NaT.date()` returns `NaT` again (not a real Python `date`), and
        # `render_breadth_html(asof=NaT).isoformat()` renders the literal
        # string "NaT" into the published HTML — silently overwriting the
        # last-good dashboard with a stale-data page. The cron chain's
        # `&&` propagates this non-zero exit to cron-wrapper.sh which fires
        # a Discord alert (discord_on_failure: true).
        if breadth.empty:
            click.echo(f"error: input parquet has no rows: {eff_input}", err=True)
            sys.exit(1)
        spy_ohlcv: _pd.DataFrame | None = None
        eff_spy = spy_path or "data/cache/spy_history.parquet"
        if Path(eff_spy).exists():
            spy_ohlcv = _pd.read_parquet(eff_spy)
        if asof is None:
            asof_max = breadth["asof_date"].max()
            # Catch `NaT` even on non-empty parquets where every `asof_date`
            # cell happens to be null. Same publish-stale rationale.
            if _pd.isna(asof_max):
                click.echo(
                    f"error: input parquet has no usable asof_date values: {eff_input}",
                    err=True,
                )
                sys.exit(1)
            if hasattr(asof_max, "date"):
                asof_dt = asof_max.date()
            elif isinstance(asof_max, _date):
                asof_dt = asof_max
            else:
                asof_dt = _date.fromisoformat(str(asof_max))
        else:
            asof_dt = _date.fromisoformat(asof)
    else:  # neon (default)
        from rainier.dashboard import neon_source as _ns
        from rainier.db.engine import get_engine

        # get_engine() reads os.environ["DATABASE_URL"] directly and does NOT
        # load .env; load it first so the cron path (uv run from PROJECT_DIR)
        # finds the var.
        _ns.ensure_env_loaded()
        engine = get_engine()
        try:
            asof_dt = (
                _date.fromisoformat(asof)
                if asof is not None
                else _ns.latest_breadth_asof(engine)
            )
            breadth = _ns.load_breadth_neon(engine, asof_dt)
            spy_ohlcv = _ns.load_spy_neon(engine)
        except _ns.EmptyNeonResultError as exc:
            click.echo(f"error: {exc}", err=True)
            sys.exit(1)
        finally:
            engine.dispose()
        if spy_ohlcv is not None and spy_ohlcv.empty:
            spy_ohlcv = None

    html = render_breadth_html(
        breadth=breadth,
        asof=asof_dt,
        rendered_at_pt=rendered_at_pt,
        window_days=window_days,
        spy_ohlcv=spy_ohlcv,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, out)
    click.echo(f"wrote market-breadth dashboard -> {out}")


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
    """Return ``(spec, sector_map)`` for the YAML at ``yaml_path``.

    ``spec`` is the parsed universe (with ``.sectors`` dict + ``.yaml_sha``);
    ``sector_map`` is the flattened ``symbol -> sector_name`` dict.

    Registries are loaded if their parquets exist, else seeded from the YAML
    so first-run compute always succeeds even before the operator has run
    `thematic backfill` (which usually seeds them).
    """
    from rainier.breadth import universe_loader as _ul

    spec = _ul.load_universe(yaml_path)
    sector_map: dict[str, str] = {}
    for sec, syms in spec.sectors.items():
        for s in syms:
            sector_map[s] = sec
    return spec, sector_map


def _check_ohlcv_freshness(
    panel: pd.DataFrame,
    asof_dt: date,
    ohlcv_path: str,
    spec,  # universe spec with .sectors dict
) -> None:
    """Raise click.ClickException if the OHLCV cache is stale or partial for
    ``asof_dt``. Shared by `thematic compute` and `thematic run-daily` so
    both paths surface the same diagnostic (codex iter-6/8 [P1]/[P2]).

    Stale: panel.max(date) < asof_dt.
    Shallow: fewer than MIN_HISTORY_TRADING_DAYS distinct dates <= asof_dt.
             Layer A's longest rolling window is 20 trading days (rel_20 +
             vol_20); with less history compute_thematic_features emits the
             RANK_SENTINEL for every symbol and the job exits 0 with unusable
             ranks. This catches the fresh-host / deleted-cache case where
             `thematic backfill --incremental` writes only the 5-day window
             (the incremental path is a refresh, not a substitute for the
             full-history seed). Fail loud so the operator runs the one-shot
             seed first.
    Partial: > 25% of YAML symbols missing on asof_dt (fail);
             > 10% missing (warning to stderr).
    """
    if panel.empty or "date" not in panel.columns:
        return

    panel_max = panel["date"].max()
    if panel_max < asof_dt:
        raise click.ClickException(
            f"OHLCV cache stale: max(date)={panel_max} < asof={asof_dt}. "
            f"Self-heal the canonical with the sanctioned adopt bridge "
            f"(atomic in-place replace + Neon mirror, no manual mv):\n"
            f"  1. uv run rainier thematic backfill --force --adopt "
            f"--start 2024-10-01 --end {asof_dt} --out {ohlcv_path}\n"
            f"  2. uv run rainier thematic run-daily  # retry"
        )

    # Shallow-history guard: Layer A's longest lookback is the 20-trading-day
    # rel_20 window. compute_thematic_features indexes the asof row at
    # asof_idx = (#dates <= asof) - 1 and reads prior_idx = asof_idx - 20; that
    # is only valid (non-negative) when asof_idx >= 20, i.e. there are >= 21
    # distinct dates at/before asof. With exactly 20, prior_idx = -1 and rel_20
    # /vol_20 stay NaN -> sentinel ranks (codex iter-5 off-by-one). Require 21.
    MIN_HISTORY_TRADING_DAYS = 21
    distinct_dates = panel.loc[panel["date"] <= asof_dt, "date"].nunique()
    if distinct_dates < MIN_HISTORY_TRADING_DAYS:
        # The shallow cache already exists (--incremental created it), so the
        # seed must replace it: the one-shot backfill refuses to overwrite
        # without --force. --force --adopt does the full re-fetch then atomically
        # replaces the shallow canonical in place (+ mirrors Neon) — the
        # sanctioned bridge, no manual mv (revision-immutability preserved by the
        # atomic os.replace inside backfill()).
        raise click.ClickException(
            f"OHLCV cache too shallow: only {distinct_dates} trading day(s) "
            f"<= asof={asof_dt}; Layer A needs >= {MIN_HISTORY_TRADING_DAYS} "
            f"(rel_20/vol_20 windows) or every rank is the no-data sentinel. "
            f"The --incremental refresh only fetches a few days and is NOT a "
            f"substitute for the full-history seed. Replace the shallow cache "
            f"with the sanctioned adopt bridge (no manual mv):\n"
            f"  1. uv run rainier thematic backfill --force --adopt "
            f"--start 2024-10-01 --end {asof_dt} --out {ohlcv_path}\n"
            f"  2. uv run rainier thematic run-daily  # retry"
        )

    expected_syms = {sym for syms in spec.sectors.values() for sym in syms}
    asof_rows = panel.loc[panel["date"] == asof_dt]
    present = (
        set(asof_rows["symbol"].dropna().unique()) if not asof_rows.empty else set()
    )
    missing = expected_syms - present
    if not expected_syms:
        return
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
            f"warning: {len(missing)}/{len(expected_syms)} YAML symbols "
            f"missing on asof={asof_dt} ({missing_frac:.0%}). "
            f"Examples: {example}. Proceeding; consider refreshing OHLCV.",
            err=True,
        )


# ---------------------------------------------------------------------------
# Phase 2 dual-write (task plan §2/§4): mirror the feature/label frames into
# market.thematic_features_daily / market.thematic_labels_daily alongside the
# parquet write. Additive + non-fatal: DATABASE_URL unset -> warn + skip.
#
#   compute -> registries (parents) -> features (FK child)
#   backfill-labels -> labels (no FK)
# ---------------------------------------------------------------------------


# Phase 3 factored these shared frame->rows helpers into rainier.db.rows so the
# dual-write call sites here, the one-shot backfill, and verify-coverage share
# one implementation (task plan §3). Re-exported under their original private
# names to keep these call sites (and existing tests that import
# ``rainier.cli._frame_to_pg_rows``) unchanged.
from rainier.db.rows import frame_to_pg_rows as _frame_to_pg_rows  # noqa: E402
from rainier.db.rows import pg_value as _pg_value  # noqa: E402


def _dual_write_features_pg(
    feat_df: pd.DataFrame,
    sector_map: dict[str, str],
    asof_dt: date,
    ticker_first_seen: dict[str, date] | None = None,
    sector_first_seen: dict[str, date] | None = None,
) -> None:
    """Mirror the computed feature frame into market.thematic_features_daily.

    Registries (market.tickers + market.sectors) are upserted FIRST so the
    feature FK references resolve. IDs come from the feature frame itself
    (ticker_id/sector_id columns), which the compute layer assigned from the
    same stable registry — so PG IDs match parquet.

    ``ticker_first_seen`` / ``sector_first_seen`` carry the registry's stored
    first-seen dates (``{symbol|sector_name: date}``); when supplied they stamp
    true provenance instead of the current ``asof_dt`` (matters when PG is
    enabled after the registry already exists — first_seen is insert-only).
    """
    if feat_df.empty:
        return
    from rainier.db import schema
    from rainier.db.dualwrite import mirror_guard
    from rainier.db.upsert import market_upsert

    ticker_first_seen = ticker_first_seen or {}
    sector_first_seen = sector_first_seen or {}

    # mirror_guard: None when DATABASE_URL unset; any SQLAlchemyError inside is
    # caught + warned so a broken mirror DB never aborts the parquet pipeline.
    with mirror_guard("thematic compute") as eng:
        if eng is None:
            return
        # Parents: derive (ticker_id, symbol) and (sector_id, sector_name) from
        # the frame + sector_map. Dedupe so a multi-row frame upserts each once.
        ticker_rows = {}
        sector_rows = {}
        for rec in feat_df.to_dict(orient="records"):
            sym = str(rec["symbol"])
            tid = _pg_value(rec["ticker_id"])
            sid = _pg_value(rec["sector_id"])
            sec_name = sector_map.get(sym, "unknown")
            ticker_rows[tid] = {
                "ticker_id": tid,
                "symbol": sym,
                "first_seen": ticker_first_seen.get(sym, asof_dt),
            }
            sector_rows[sid] = {
                "sector_id": sid,
                "sector_name": sec_name,
                "first_seen": sector_first_seen.get(sec_name, asof_dt),
            }
        # Registry identity is insert-only: sector_name/symbol AND first_seen
        # are immutable so a conflict on the stable id never remaps the name
        # (which would point existing feature FK rows at the wrong ticker —
        # [D-015]: IDs are never remapped) or re-stamp the date. All non-PK
        # cols immutable -> the upsert degrades to ON CONFLICT DO NOTHING.
        market_upsert(
            eng,
            schema.sectors,
            list(sector_rows.values()),
            ["sector_id"],
            immutable_cols=["sector_name", "first_seen"],
        )
        market_upsert(
            eng,
            schema.tickers,
            list(ticker_rows.values()),
            ["ticker_id"],
            immutable_cols=["symbol", "first_seen"],
        )

        feature_cols = list(schema.thematic_features_daily.columns.keys())
        rows = _frame_to_pg_rows(feat_df, feature_cols)
        market_upsert(
            eng, schema.thematic_features_daily, rows, ["asof_date", "symbol"]
        )


def _dual_write_labels_pg(label_df: pd.DataFrame) -> None:
    """Mirror the label frame into market.thematic_labels_daily (no FK)."""
    if label_df.empty:
        return
    from rainier.db import schema
    from rainier.db.dualwrite import mirror_guard
    from rainier.db.upsert import market_upsert

    # mirror_guard: None when DATABASE_URL unset; SQLAlchemyError inside is
    # caught + warned so a broken mirror DB never aborts the parquet pipeline.
    with mirror_guard("thematic backfill-labels") as eng:
        if eng is None:
            return
        label_cols = list(schema.thematic_labels_daily.columns.keys())
        rows = _frame_to_pg_rows(label_df, label_cols)
        market_upsert(
            eng, schema.thematic_labels_daily, rows, ["asof_date", "symbol"]
        )


def _dual_write_breadth_pg(breadth_df: pd.DataFrame) -> None:
    """Mirror the LONG breadth frame into market.breadth_indicator_daily.

    The table is LONG == the parquet (asof_date, indicator, value), so we
    project those three columns directly — NO pivot/transform (design D-1).

    FULL-HISTORY upsert (design D-3): ``compute_indicators`` recomputes the
    ENTIRE breadth history every run, and the cumulative indicators
    (ad_cumulative, mcclellan_summation) rewrite every subsequent row when a
    late OHLCV correction lands. So we upsert the whole frame, not just the
    latest asof_date — writing only today would leave PG stale on the corrected
    past dates. Full upsert is cheap at this volume (~12 indicators x N days).

    Values mirror the parquet byte-for-parity, including NaN warm-up rows
    (-> NULL via pg_value) — no divergent NULL logic (would break checksum
    parity, design D-4).
    """
    if breadth_df.empty:
        return
    from rainier.db import schema
    from rainier.db.dualwrite import mirror_guard
    from rainier.db.upsert import market_upsert

    # mirror_guard: None when DATABASE_URL unset; any SQLAlchemyError inside is
    # caught + warned so a broken mirror DB never aborts the parquet pipeline.
    with mirror_guard("breadth compute") as eng:
        if eng is None:
            return
        rows = _frame_to_pg_rows(breadth_df, ["asof_date", "indicator", "value"])
        market_upsert(
            eng,
            schema.breadth_indicator_daily,
            rows,
            ["asof_date", "indicator"],
        )


def _dual_write_benchmark_pg(spy_df: pd.DataFrame) -> None:
    """Mirror the SPY/benchmark OHLCV frame into market.benchmark_ohlcv.

    The parquet (symbol, date, open, high, low, close, volume, fetched_at,
    yfinance_version) mirrors the table 1:1.

    All columns (including ``fetched_at``/``yfinance_version``) are MUTABLE on
    conflict — exactly like thematic_ohlcv's dual-write (which passes no
    immutable_cols). The parquet ``ohlcv_backfill._upsert`` is latest-write-wins,
    so a ``backfill-spy --incremental`` run that overlaps existing dates (the
    default 5-day window) re-stamps those rows' provenance in the parquet. PG is
    a byte-for-parity mirror checked by ``verify-coverage`` (design D-5), so it
    MUST adopt the same new provenance — pinning it immutable here would leave PG
    holding stale fetched_at/version while the parquet moved on, failing the
    checksum. Provenance immutability belongs to the append-only registries
    (first_seen), not to a mirror of a mutable cache.
    """
    if spy_df.empty:
        return
    from rainier.db import schema
    from rainier.db.dualwrite import mirror_guard
    from rainier.db.upsert import market_upsert

    with mirror_guard("breadth backfill-spy") as eng:
        if eng is None:
            return
        cols = list(schema.benchmark_ohlcv.columns.keys())
        rows = _frame_to_pg_rows(spy_df, cols)
        market_upsert(
            eng,
            schema.benchmark_ohlcv,
            rows,
            ["symbol", "date"],
        )


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

    from rainier.breadth import registry as _reg
    from rainier.breadth.ranks import compute_thematic_features

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

    # Same stale/partial-coverage gate as run-daily (codex iter-8 [P2]):
    # the direct compute path must not silently rank over a shrunken
    # universe just because the operator invoked `thematic compute` instead
    # of `thematic run-daily`.
    _check_ohlcv_freshness(panel, asof_dt, ohlcv_path, spec)

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

    # Phase 2 dual-write: mirror features into market.* (additive; skips on
    # DATABASE_URL unset). Parquet above is the load-bearing write. Pass the
    # registry's stored first_seen so PG records true provenance, not asof_dt.
    _dual_write_features_pg(
        out_df,
        sector_map,
        asof_dt,
        ticker_first_seen=_reg.load_ticker_first_seen(ticker_registry_path),
        sector_first_seen=_reg.load_sector_first_seen(sector_registry_path),
    )


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

    from rainier.breadth.ranks import compute_forward_labels

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

    # Phase 2 dual-write: mirror labels into market.thematic_labels_daily.
    _dual_write_labels_pg(out_df)


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

    from rainier.breadth import registry as _reg
    from rainier.breadth.ranks import (
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

    # Stale-OHLCV / partial-coverage guard. Per DESIGN §7: "If OHLCV is
    # stale, backfill incrementally first." We don't auto-fetch (yfinance
    # side effect inside a cron run is too magical) — instead surface clearly
    # with the exact next-step command per `feedback_surface_dont_silo`.
    _check_ohlcv_freshness(panel, asof_dt, ohlcv_path, spec)

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
            # Phase 2 dual-write: mirror layer A features into market.* with
            # the registry's stored first_seen (true provenance, not asof_dt).
            _dual_write_features_pg(
                out_df,
                sector_map,
                asof_dt,
                ticker_first_seen=_reg.load_ticker_first_seen(ticker_registry_path),
                sector_first_seen=_reg.load_sector_first_seen(sector_registry_path),
            )

    # Layer B: always recompute (cheap + freshness drifts forward).
    import os

    labels_path = Path(labels_out)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    label_df = compute_forward_labels(panel=panel)
    tmp = labels_path.with_suffix(labels_path.suffix + ".tmp")
    label_df.to_parquet(tmp, index=False)
    os.replace(tmp, labels_path)
    click.echo(f"layer B: wrote {len(label_df)} rows -> {labels_path}")
    # Phase 2 dual-write: mirror layer B labels into market.*.
    _dual_write_labels_pg(label_df)

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


# ---------------------------------------------------------------------------
# dashboard — publish-friendly self-contained HTML renderers
# ---------------------------------------------------------------------------
#
# Distinct from `thematic render` (the legacy in-tree dashboard). The
# `dashboard` group exists for the public/static-publish path:
#   - Light + dark mode via fengshen-site CSS vars
#   - Three CSS-toggle tab views (All / Top 15 / Movers)
#   - Inline SVG sparklines
#   - Deterministic: caller supplies --rendered-at-pt (no wall-clock reads)
#
# Design ref: docs/DESIGN-etf-dashboard-publish.md.


@cli.group()
def dashboard() -> None:
    """Publish-friendly self-contained HTML dashboard renderers."""


@dashboard.command("render-etf-html")
@click.option(
    "--source",
    type=click.Choice(["neon", "parquet"]),
    default="neon",
    show_default=True,
    help=(
        "Data source for the features frame. 'neon' (default) reads the "
        "canonical market.thematic_features_daily store; 'parquet' reads the "
        "local cache (--features). Passing --features implies parquet."
    ),
)
@click.option(
    "--features",
    "features_path",
    type=click.Path(exists=True),
    default=None,
    help=(
        "Local features parquet (parquet source only). "
        "[default: data/cache/thematic_features_daily.parquet]"
    ),
)
@click.option(
    "--registry",
    "registry_path",
    type=click.Path(exists=True),
    default="data/cache/sector_registry.parquet",
    show_default=True,
)
@click.option(
    "--asof",
    default=None,
    help=(
        "Date to render (YYYY-MM-DD). The renderer slices features to this "
        "asof_date. Required for the parquet source; optional for neon "
        "(defaults to max(asof_date) in market.thematic_features_daily)."
    ),
)
@click.option(
    "--rendered-at-pt",
    required=True,
    help="Wall-clock HH:MM (Pacific) for the header. Caller-supplied to keep "
    "the renderer deterministic (no datetime.now inside the renderer).",
)
@click.option(
    "--history-days",
    type=int,
    default=30,
    show_default=True,
    help="Sparkline history window per ticker.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default="out/dashboards/etf-ranks.html",
    show_default=True,
)
@click.option(
    "--names-path",
    "names_path",
    type=click.Path(),
    default="data/cache/etf_names.parquet",
    show_default=True,
    help=(
        "Optional ETF names parquet (from `thematic backfill-names`). "
        "When present, the Name column renders yfinance longName; "
        "missing or unreadable → falls back to the symbol itself."
    ),
)
def dashboard_render_etf_html(
    source: str,
    features_path: str | None,
    registry_path: str,
    asof: str | None,
    rendered_at_pt: str,
    history_days: int,
    output_path: str,
    names_path: str,
) -> None:
    """Render the ETF-ranks self-contained HTML dashboard.

    Default source is the canonical Neon ``market.thematic_features_daily``
    store (kept fresh by other jobs); ``--source parquet`` (or passing
    ``--features``) reads the local cache for offline / back-compat use. The
    sector registry stays a parquet (human-edited config). Fails loud on a
    zero-row Neon result — never silently publishes a stale parquet.
    """
    import os
    import sys
    from datetime import date as _date

    import pandas as _pd

    from rainier.dashboard.render_etf import render_etf_html

    # Passing an explicit --features implies the parquet source (back-compat).
    if features_path is not None:
        source = "parquet"

    names_arg: str | None = names_path if Path(names_path).exists() else None
    registry = _pd.read_parquet(registry_path)

    if source == "parquet":
        eff_features = features_path or "data/cache/thematic_features_daily.parquet"
        if not Path(eff_features).exists():
            click.echo(f"error: features parquet not found: {eff_features}", err=True)
            sys.exit(1)
        if asof is None:
            click.echo(
                "error: --asof is required for the parquet source", err=True
            )
            sys.exit(1)
        features = _pd.read_parquet(eff_features)
        asof_dt = _date.fromisoformat(asof)
    else:  # neon (default)
        from rainier.dashboard import neon_source as _ns
        from rainier.db.engine import get_engine

        # get_engine() reads os.environ["DATABASE_URL"] directly and does NOT
        # load .env; load it first so the cron path (uv run from PROJECT_DIR)
        # finds the var.
        _ns.ensure_env_loaded()
        engine = get_engine()
        try:
            asof_dt = (
                _date.fromisoformat(asof)
                if asof is not None
                else _ns.latest_etf_asof(engine)
            )
            features = _ns.load_etf_features_neon(engine, asof_dt)
        except _ns.EmptyNeonResultError as exc:
            click.echo(f"error: {exc}", err=True)
            sys.exit(1)
        finally:
            engine.dispose()

    html = render_etf_html(
        features=features,
        registry=registry,
        asof=asof_dt,
        rendered_at_pt=rendered_at_pt,
        history_days=history_days,
        names_path=names_arg,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, out)
    click.echo(f"wrote ETF dashboard -> {out}")


@dashboard.command("render-combined")
@click.option(
    "--breadth-input",
    "breadth_path",
    type=click.Path(exists=True),
    default="data/cache/sp500_breadth_daily.parquet",
    show_default=True,
    help="Long-format breadth parquet (asof_date, indicator, value).",
)
@click.option(
    "--etf-features",
    "features_path",
    type=click.Path(exists=True),
    default="data/cache/thematic_features_daily.parquet",
    show_default=True,
    help="ETF features parquet (asof_date × symbol × rank/delta fields).",
)
@click.option(
    "--etf-registry",
    "registry_path",
    type=click.Path(exists=True),
    default="data/cache/sector_registry.parquet",
    show_default=True,
    help="Sector registry parquet (sector_id, sector_name).",
)
@click.option(
    "--spy-path",
    "spy_path",
    type=click.Path(),
    default="data/cache/spy_history.parquet",
    show_default=True,
    help=(
        "Optional SPY OHLCV parquet (from `market-breadth backfill-spy`). "
        "Missing file → SPY pane is omitted in the breadth section."
    ),
)
@click.option(
    "--asof",
    default=None,
    help=(
        "Display + filter date (YYYY-MM-DD). Defaults to the most recent "
        "asof_date in the breadth parquet (cron-friendly)."
    ),
)
@click.option(
    "--rendered-at-pt",
    required=True,
    help="Wall-clock HH:MM (Pacific) for the header. Caller-supplied so the "
    "renderer stays deterministic (no datetime.now inside).",
)
@click.option(
    "--history-days",
    type=int,
    default=30,
    show_default=True,
    help="ETF sparkline history window per ticker.",
)
@click.option(
    "--window-days",
    type=int,
    default=504,
    show_default=True,
    help="Trailing-window cap for breadth charts (~2y default).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default="out/dashboards/trading.html",
    show_default=True,
)
@click.option(
    "--names-path",
    "names_path",
    type=click.Path(),
    default="data/cache/etf_names.parquet",
    show_default=True,
    help=(
        "Optional ETF names parquet (from `thematic backfill-names`). "
        "When present, the ETF tab's Name column renders yfinance "
        "longName; missing or unreadable → falls back to the symbol "
        "itself (same fallback semantics as the standalone "
        "`/trading/etf-ranks/` page)."
    ),
)
def dashboard_render_combined(
    breadth_path: str,
    features_path: str,
    registry_path: str,
    spy_path: str,
    asof: str | None,
    rendered_at_pt: str,
    history_days: int,
    window_days: int,
    output_path: str,
    names_path: str,
) -> None:
    """Render the combined trading dashboard (breadth + ETF ranks) to HTML.

    DESIGN-trading-dashboard-combined-v1.md §2 + §4 D1 — one URL
    (`/trading/`) with shared header + top-level tabs over both
    the S&P 500 breadth view and the ETF ranks table. Standalone breadth +
    ETF render-html commands still ship the per-page URLs (D4 — keep all
    three URLs, no 301).
    """
    import sys
    from datetime import date as _date

    import pandas as _pd

    from rainier.dashboard.render_combined import write_combined_html

    if asof is None:
        # Same auto-asof resolution as `market-breadth render-html` —
        # latest row of the breadth parquet wins. Keeps the cron caller
        # agnostic to weekend / holiday timing + avoids the `%`-in-crontab
        # footgun (the script wrapper computes nothing date-related).
        breadth = _pd.read_parquet(breadth_path)
        if breadth.empty:
            click.echo(
                f"error: breadth parquet has no rows: {breadth_path}",
                err=True,
            )
            sys.exit(1)
        asof_max = breadth["asof_date"].max()
        if _pd.isna(asof_max):
            click.echo(
                f"error: breadth parquet has no usable asof_date: {breadth_path}",
                err=True,
            )
            sys.exit(1)
        if hasattr(asof_max, "date"):
            asof_dt = asof_max.date()
        elif isinstance(asof_max, _date):
            asof_dt = asof_max
        else:
            asof_dt = _date.fromisoformat(str(asof_max))
    else:
        asof_dt = _date.fromisoformat(asof)

    # Mirror the standalone `render-etf-html` CLI: the default points at
    # the cache path, but pass None to the renderer when the file doesn't
    # exist so the renderer's symbol-fallback kicks in cleanly (no
    # spurious "file not found" surprise).
    names_arg: str | None = names_path if Path(names_path).exists() else None

    written = write_combined_html(
        breadth_path=breadth_path,
        features_path=features_path,
        registry_path=registry_path,
        spy_path=spy_path,
        output_path=output_path,
        asof=asof_dt,
        rendered_at_pt=rendered_at_pt,
        history_days=history_days,
        window_days=window_days,
        names_path=names_arg,
    )
    click.echo(f"wrote trading dashboard -> {written}")


# ---------------------------------------------------------------------------
# db — canonical Postgres store (Phase 1 of the architecture pivot)
# ---------------------------------------------------------------------------
#
# New subcommands (per task plan §5):
#
#   rainier db ping                          connect, SELECT 1, exit 0 or fail loud
#   rainier db migrate                       alembic upgrade head
#   rainier db migrate --downgrade -1        alembic downgrade -1
#   rainier db migrate --downgrade base      alembic downgrade base
#
# Implementation uses Alembic's Python API (not subprocess) so the CLI surface
# is testable from pytest without spawning processes.
#
# This is the NEW `db/` package — separate from the legacy `core/database.py`
# singleton that backs LLM thesis persistence, monitors, etc. Both engines
# coexist for the duration of the pivot.
#
# IMPORTANT: ping + migrate decorate the EXISTING `db` group defined at the
# top of the legacy db block (above, around line 1766) which already owns
# `init` and `backfill-prices`. Do NOT re-declare `@cli.group() def db()`
# here — click's registry would replace the legacy group with this one and
# silently kill `rainier db init` / `db backfill-prices` (CI #102 regression).
# See tests/test_cli/test_db.py::test_db_group_does_not_shadow_legacy_subcommands.


def _resolve_alembic_config():
    """Build an Alembic Config bound to the packaged ``db/alembic.ini``.

    Resolves the config in two ways, in priority order:

    1. **Wheel install** — pulls ``alembic.ini`` + the ``alembic/`` migration
       tree via ``importlib.resources.files("rainier") / "_db_assets"``.
       Hatchling's ``force-include`` in pyproject.toml ships the top-level
       ``db/`` tree into the wheel at ``rainier/_db_assets/``, so wheel
       installs don't need a source checkout to run ``rainier db migrate``.

    2. **Editable / source checkout** — falls back to ``<repo>/db/alembic.ini``
       (resolved via ``__file__``). Editable installs of hatch projects place
       ``__file__`` inside the source tree, so we resolve the repo root via
       ``Path(__file__).resolve().parents[2]``.

    The .ini file leaves ``sqlalchemy.url`` empty on purpose — db/alembic/
    env.py reads DATABASE_URL from the environment so creds never land in
    git. We override ``script_location`` defensively after loading so the
    Config works even if a future ini edit drops the ``%(here)s`` prefix
    (the regression test ``test_alembic_ini_script_location_is_config_relative``
    in tests/test_cli/test_db.py guards the raw ini path too).
    """
    from importlib import resources
    from pathlib import Path

    from alembic.config import Config

    # 1. Wheel-friendly path via importlib.resources.
    try:
        anchor = resources.files("rainier") / "_db_assets"
        cfg_resource = anchor / "alembic.ini"
        script_resource = anchor / "alembic"
        with resources.as_file(cfg_resource) as cfg_path_obj:
            cfg_path = Path(cfg_path_obj)
        with resources.as_file(script_resource) as script_path_obj:
            script_path = Path(script_path_obj)
        if cfg_path.exists() and script_path.exists():
            cfg = Config(str(cfg_path))
            cfg.set_main_option("script_location", str(script_path))
            return cfg
    except (ModuleNotFoundError, FileNotFoundError):
        pass  # fall through to source-checkout path

    # 2. Editable / source-checkout fallback. cli.py at src/rainier/cli.py
    #    → repo root is parents[2], db/ lives at the repo root.
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "db" / "alembic.ini"
    script_path = repo_root / "db" / "alembic"
    if not cfg_path.exists():
        raise click.ClickException(
            f"alembic config not found at {cfg_path} and no packaged "
            "rainier/_db_assets/ in the installed package. Reinstall "
            "rainier (e.g. `uv sync`) or run from a source checkout."
        )
    cfg = Config(str(cfg_path))
    cfg.set_main_option("script_location", str(script_path))
    return cfg


@db.command("ping")
def db_ping() -> None:
    """Connect to ``DATABASE_URL``, run ``SELECT 1``, print ``ok`` or fail."""
    from sqlalchemy import text

    from rainier.db.engine import get_engine

    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
        engine.dispose()
    except Exception as exc:  # pragma: no cover — connection-time failure
        raise click.ClickException(f"db ping failed: {exc}") from exc

    if result != 1:
        raise click.ClickException(f"db ping returned unexpected value: {result!r}")
    click.echo("ok")


@db.command("migrate")
@click.option(
    "--downgrade",
    "downgrade_to",
    default=None,
    help=(
        "If set, downgrade to this revision (e.g. -1, base, 0001). "
        "Without this flag, the command upgrades to head."
    ),
)
def db_migrate(downgrade_to: str | None) -> None:
    """Run Alembic ``upgrade head`` (default) or ``downgrade <rev>``."""
    from alembic import command

    cfg = _resolve_alembic_config()

    # Mirror db_ping's wrapping idiom: misconfig (missing DATABASE_URL, an
    # unreachable host, or an Alembic config error) raises RuntimeError /
    # OperationalError / alembic.* exceptions from the upgrade/downgrade call.
    # Catch them all and re-raise as ClickException so the CLI prints a single
    # actionable `Error:` line instead of a raw traceback. (_resolve_alembic_config
    # already raises ClickException on its own failure path, so it stays outside.)
    try:
        if downgrade_to is None:
            command.upgrade(cfg, "head")
            click.echo("alembic upgrade head — ok")
        else:
            command.downgrade(cfg, downgrade_to)
            click.echo(f"alembic downgrade {downgrade_to} — ok")
    except click.ClickException:
        raise
    except Exception as exc:
        action = "upgrade head" if downgrade_to is None else f"downgrade {downgrade_to}"
        raise click.ClickException(f"db migrate ({action}) failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Phase 3 (task plan §2): one-shot parquet -> market.* backfill +
# verify-coverage parity gate. Both are EXPLICIT DB ops (unlike the Phase 2
# dual-write skip path) — DATABASE_URL must be set or we fail loud with a
# ClickException rather than a raw traceback.
# ---------------------------------------------------------------------------


def _require_db_engine(op_name: str):
    """Return a fresh Engine, or raise a clean ClickException if unconfigured.

    Phase 3 backfill/verify are explicit DB operations: a missing DATABASE_URL
    is an operator error, not a skip path. ``get_engine()`` raises RuntimeError
    in that case; we translate it to a ClickException so the CLI prints an
    actionable message instead of a traceback.
    """
    from rainier.db.engine import get_engine

    try:
        return get_engine()
    except RuntimeError as exc:
        raise click.ClickException(
            f"{op_name} requires DATABASE_URL to be set (e.g. "
            f"postgresql+psycopg://user:pass@host:5432/db). {exc}"
        ) from exc


def _parse_asof(value: str | None, flag: str) -> date | None:
    """Parse a YYYY-MM-DD option into a date, or None when unset."""
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise click.ClickException(f"{flag} must be YYYY-MM-DD, got {value!r}") from exc


def _asof_window(asof_start: str | None, asof_end: str | None) -> tuple[date | None, date | None]:
    """Parse + validate the inclusive [start, end] as-of window.

    A reversed range (start > end) is rejected loudly: it would filter every
    date-keyed row out, so backfill would write only the registries and
    verify-coverage would report a CLEAN empty window — silently passing the
    parity gate it exists to enforce. Fail fast instead.
    """
    start = _parse_asof(asof_start, "--asof-start")
    end = _parse_asof(asof_end, "--asof-end")
    if start is not None and end is not None and start > end:
        raise click.ClickException(
            f"--asof-start ({start}) must be <= --asof-end ({end}); "
            f"a reversed window filters out every row."
        )
    return start, end


@db.command("backfill-from-parquet")
@click.option(
    "--cache-dir",
    "cache_dir",
    type=click.Path(file_okay=False),
    default="data/cache",
    show_default=True,
    help="Directory holding the parquet caches to backfill from.",
)
@click.option("--asof-start", default=None, help="Inclusive start date (YYYY-MM-DD).")
@click.option("--asof-end", default=None, help="Inclusive end date (YYYY-MM-DD).")
@click.option(
    "--dry-run", is_flag=True, help="Report per-table row counts without writing."
)
def db_backfill_from_parquet(
    cache_dir: str, asof_start: str | None, asof_end: str | None, dry_run: bool
) -> None:
    """Backfill the full parquet history into ``market.*`` (registries first).

    Idempotent (UPSERT). ``--asof-start/--asof-end`` window the date-keyed
    tables; the ticker/sector registries always load fully (FK parents).
    """
    from rainier.db.backfill import backfill_from_parquet

    start, end = _asof_window(asof_start, asof_end)
    engine = _require_db_engine("db backfill-from-parquet")
    try:
        counts = backfill_from_parquet(
            engine, cache_dir, asof_start=start, asof_end=end, dry_run=dry_run
        )
    finally:
        engine.dispose()

    label = "would write" if dry_run else "wrote"
    for table, n in counts.items():
        click.echo(f"  {table}: {label} {n} rows")
    total = sum(counts.values())
    suffix = " (dry-run, no mutation)" if dry_run else ""
    click.echo(f"backfill-from-parquet — {label} {total} rows total{suffix}")


@db.command("verify-coverage")
@click.option(
    "--cache-dir",
    "cache_dir",
    type=click.Path(file_okay=False),
    default="data/cache",
    show_default=True,
    help="Directory holding the parquet caches to verify against.",
)
@click.option("--asof-start", default=None, help="Inclusive start date (YYYY-MM-DD).")
@click.option("--asof-end", default=None, help="Inclusive end date (YYYY-MM-DD).")
def db_verify_coverage(
    cache_dir: str, asof_start: str | None, asof_end: str | None
) -> None:
    """Verify parquet and Postgres agree per (asof_date, table).

    Compares row counts + an order-independent, float-tolerant checksum. Prints
    a per-table report; exits 0 if everything matches, nonzero (naming each
    offending (asof_date, table)) on any drift — so it is CI/cron-usable.
    """
    from rainier.db.verify import verify_coverage

    start, end = _asof_window(asof_start, asof_end)
    engine = _require_db_engine("db verify-coverage")
    try:
        report = verify_coverage(engine, cache_dir, asof_start=start, asof_end=end)
    finally:
        engine.dispose()

    # Per-table summary: matched groups / total groups + parquet/pg row totals.
    by_table: dict[str, list] = {}
    for table, _key, pq_n, pg_n, ok in report.rows:
        by_table.setdefault(table, []).append((pq_n, pg_n, ok))
    for table, groups in by_table.items():
        matched = sum(1 for _pq, _pg, ok in groups if ok)
        pq_total = sum(pq for pq, _pg, _ok in groups)
        pg_total = sum(pg for _pq, pg, _ok in groups)
        status = "OK" if matched == len(groups) else "DRIFT"
        click.echo(
            f"  {table}: {status} — {matched}/{len(groups)} date-groups match, "
            f"parquet={pq_total} pg={pg_total} rows"
        )

    if report.ok:
        click.echo("verify-coverage — all match")
        return

    click.echo("verify-coverage — DRIFT detected:", err=True)
    for d in report.drift:
        click.echo(f"  {d.table} asof={d.asof_date}: {d.reason}", err=True)
    raise click.ClickException(
        f"{len(report.drift)} (asof_date, table) group(s) drifted — "
        f"parquet and Postgres disagree."
    )


@db.command("backfill-screened-levels")
@click.option("--from", "from_date", required=True, help="Inclusive start scan_date (YYYY-MM-DD).")
@click.option("--to", "to_date", required=True, help="Inclusive end scan_date (YYYY-MM-DD).")
@click.option(
    "--apply",
    is_flag=True,
    help="Write the recovered levels. Omit for a dry-run (report only).",
)
@click.pass_context
def db_backfill_screened_levels(ctx, from_date: str, to_date: str, apply: bool) -> None:
    """One-time backfill of NULL screened_stocks trade levels (historical repair).

    Replays the pattern detector as-of each historical scan_date over a fresh
    yfinance 6-month window (the source the live screener used — the stored
    `stock_prices` corpus is too short to form a pattern), matches the actionable
    pattern whose `pattern_type` equals the row's stored type, and coalesce-upserts
    entry/stop/target/rr (fills NULL only, never clobbers a set value). Dry-run by
    default; pass `--apply` to write.

    Honors the root `--config` for BOTH the target database AND the detector knobs
    used in the replay. Because this reconstructs HISTORICAL screen output, point
    `--config` at the settings YAML whose `stock_screener` section matches what the
    live screen ran on those dates (e.g. `rainier --config config/settings.yaml db
    backfill-screened-levels ...`). If detector thresholds (swing_lookback,
    neckline_tolerance_pct, min_daily_bars, ...) were tuned after the scan window,
    a default config would replay with TODAY's knobs and write levels the live
    screen never produced — so pin the historical config.

    Only `close`-session rows are repaired: the replay uses the completed daily
    bar, which matches what the live screen saw only at close (earlier sessions
    that day lacked the final high/low/close). Non-close patterned-NULL rows, and
    rows whose stored pattern does not re-detect as-of, are LEFT NULL and reported
    in `still-NULL` — never given look-ahead or wrong-pattern levels.
    """
    from datetime import date as _date

    from rainier.core import config as _config_mod
    from rainier.core import database as _database_mod
    from rainier.core.config import load_settings_fresh
    from rainier.paper.backfill_screened_levels import backfill_screened_levels

    # Honor the root `--config` (codex P1). load_settings_fresh reads the YAML the
    # operator selected; we (a) seed the process settings singleton so the legacy
    # DB session + persist_screened_stocks target THAT database (not the default
    # config/settings.yaml), and (b) pass its stock_screener as the replay config
    # so the reconstruction uses the operator-pinned (historical) detector knobs.
    #
    # These are PROCESS globals. Snapshot them and RESTORE in finally (codex P2):
    # an in-process caller (CliRunner / programmatic reuse) must not have its later
    # commands silently inherit this backfill's --config DB. Without restore, the
    # next get_settings()/get_session() in the same interpreter would read/write
    # the wrong database.
    _prev_settings = _config_mod._settings
    _prev_engine = _database_mod._engine
    _prev_factory = _database_mod._session_factory

    settings = load_settings_fresh(_settings_path(ctx))
    _config_mod._settings = settings  # seed singleton before any get_session()
    # Reset the cached legacy engine/session factory so they REBIND against the
    # just-seeded settings. Seeding _settings alone is insufficient if a session
    # was already opened earlier in the same process — get_session() would keep the
    # stale module-level _engine pointed at the old DB.
    _database_mod._engine = None
    _database_mod._session_factory = None

    try:
        # Operator-facing validation failures (a bad date string, from>to, or
        # --apply on a pre-0012 DB) must surface as a clean Click error, not a raw
        # traceback that looks like the command crashed.
        try:
            start = _date.fromisoformat(from_date)
            end = _date.fromisoformat(to_date)
        except ValueError as exc:
            raise click.BadParameter(f"dates must be YYYY-MM-DD: {exc}") from exc

        try:
            result = backfill_screened_levels(
                from_date=start,
                to_date=end,
                apply=apply,
                config=settings.stock_screener,
            )
        except (ValueError, RuntimeError) as exc:
            # ValueError: from_date > to_date. RuntimeError: --apply preflight on a
            # pre-0012 DB (the message already names migration 0012).
            raise click.ClickException(str(exc)) from exc

        mode = "APPLY" if apply else "DRY-RUN"
        click.echo(
            f"backfill-screened-levels [{mode}] {start}..{end}: "
            f"scanned={result.scanned} recovered={result.recovered} "
            f"still_null={result.still_null} "
            f"no_price_data={result.no_price_data} "
            f"detector_errors={result.detector_errors} "
            f"skipped_non_close={result.skipped_non_close}"
        )
        if result.still_null_keys:
            click.echo(
                f"  still-NULL ({result.still_null} rows, had prices but no as-of "
                "pattern matched the stored type — permanent, re-run won't help):"
            )
            for symbol, scan_date in result.still_null_keys:
                click.echo(f"    {symbol} {scan_date}")
        if result.no_price_data_keys:
            click.echo(
                f"  no-price-data ({result.no_price_data} rows, yfinance returned "
                "no bars even after solo retry — TRANSIENT, re-run may recover):"
            )
            for symbol, scan_date in result.no_price_data_keys:
                click.echo(f"    {symbol} {scan_date}")
        if result.detector_error_keys:
            click.echo(
                f"  detector-errors ({result.detector_errors} rows, the detector "
                "raised — NOT a permanent no-match; a data re-pull or code fix may "
                "recover these; see logs for tracebacks):"
            )
            for symbol, scan_date in result.detector_error_keys:
                click.echo(f"    {symbol} {scan_date}")
        if result.skipped_non_close:
            click.echo(
                f"  skipped-non-close ({result.skipped_non_close} rows): non-close "
                "patterned-NULL rows are NOT repairable (look-ahead) and remain NULL."
            )
        if not apply and result.recovered:
            click.echo(
                "  (dry-run — re-run with --apply to write the recovered levels)"
            )
    finally:
        # Restore the process globals so a later in-process command uses its own
        # root --config, not this backfill's. Dispose the engine we (re)built.
        if _database_mod._engine is not None:
            _database_mod._engine.dispose()
        _config_mod._settings = _prev_settings
        _database_mod._engine = _prev_engine
        _database_mod._session_factory = _prev_factory


# ---------------------------------------------------------------------------
# money-flow-neon-backup-b613: nightly off-machine backup of the irreplaceable
# money_flow_snapshots into Neon (durability). Local TimescaleDB stays primary;
# Neon holds a managed-backup copy. Design: DESIGN-money-flow-neon-backup.md §2.
# ---------------------------------------------------------------------------


@db.command("backup-money-flow")
@click.option(
    "--verify",
    "do_verify",
    is_flag=True,
    help=(
        "After the copy, run a strong integrity check (max-id, missing-row, "
        "full-window canonicalized checksum, id-uniqueness). Non-zero on any drift."
    ),
)
@click.option(
    "--skip-if-unconfigured",
    is_flag=True,
    help=(
        "DEV ONLY: if DATABASE_URL is unset, warn and exit 0 instead of failing. "
        "The cron must NOT pass this — a missing Neon target is a real durability "
        "failure prod must alert on."
    ),
)
def db_backup_money_flow(do_verify: bool, skip_if_unconfigured: bool) -> None:
    """Back up ``money_flow_snapshots`` (local) -> ``backup.money_flow_snapshots`` (Neon).

    ``data_date``-aware reconcile (delete-changed-day + recopy), idempotent — so a
    same-day rebuild (QU scraper re-INSERTs a day with new ids) is mirrored, not
    orphaned. Reads the local TimescaleDB via the legacy engine and writes Neon via
    ``DATABASE_URL``. DATABASE_URL unset fails loud (non-zero) by default;
    ``--skip-if-unconfigured`` turns that into a warn + exit 0 for local dev (cron
    stays loud).
    """
    from rainier.core.database import get_engine as get_local_engine
    from rainier.db.engine import get_engine as get_neon_engine
    from rainier.db.money_flow_backup import backup_money_flow, verify_backup

    # Resolve the Neon target FIRST. get_neon_engine() raises RuntimeError when
    # DATABASE_URL is unset. By default that is a loud non-zero failure (a missing
    # backup target is a durability failure, not a skip); --skip-if-unconfigured
    # turns it into a warn + exit 0 for local dev. The cron must NOT pass it.
    try:
        dst = get_neon_engine()
    except RuntimeError as exc:
        if skip_if_unconfigured:
            click.echo(
                "backup-money-flow: DATABASE_URL unset — skipping (warn, "
                "--skip-if-unconfigured). Backup did NOT run."
            )
            return
        raise click.ClickException(
            "backup-money-flow requires DATABASE_URL to point at the Neon backup "
            "target (e.g. postgresql+psycopg://user:pass@host/db). A missing "
            "target is a durability failure, not a skip — pass "
            f"--skip-if-unconfigured only for local dev. {exc}"
        ) from exc

    # Local source via the legacy singleton engine (PR #115: bound to local).
    # A local-unreachable failure surfaces as a loud non-zero exit (no catch).
    src = get_local_engine()

    try:
        result = backup_money_flow(src, dst)
        click.echo(
            f"backed up {result.copied} rows "
            f"(reconciled up to id {result.run_max})"
        )
        if do_verify:
            report = verify_backup(src, dst, run_max=result.run_max)
            if not report.ok:
                click.echo("backup-money-flow — VERIFY FAILED:", err=True)
                for f in report.failures:
                    click.echo(f"  {f}", err=True)
                raise click.ClickException(
                    f"{len(report.failures)} integrity check(s) failed — the Neon "
                    f"backup does not match local. See the diagnostics above."
                )
            click.echo("backup-money-flow — verify OK")
    finally:
        dst.dispose()
