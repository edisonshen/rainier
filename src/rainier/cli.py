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

    import yfinance as yf

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
    candidates = screen_stocks(settings)[:top_n]

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
    asyncio.run(_run_qu_scrape(
        session, detail_top, dates, days_back, start_date, delay, headed, cdp,
    ))


async def _run_qu_scrape(session, detail_top, dates, days_back, start_date, delay, headed, cdp):
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

        # Post-scrape: run screener and send actionable setups to Discord
        if result.records_created > 0 and not date_list:
            _post_scrape_screener(settings, session)

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


def _post_scrape_screener(settings, session: str) -> None:
    """Run stock screener after successful scrape and send results to Discord."""
    from rainier.alerts.discord import _build_payloads
    from rainier.analysis.stock_screener import screen_stocks

    import httpx

    webhook = _get_discord_webhook(settings)
    if not webhook:
        click.echo("  (no Discord webhook, skipping screener alert)")
        return

    click.echo("Running post-scrape stock screener...")
    try:
        candidates = screen_stocks(settings)[:20]
    except Exception as exc:
        click.echo(f"  Screener failed: {exc}")
        _notify_scrape_discord(
            settings, session,
            title=f"QU100 Screener FAILED ({session})",
            message=f"Scrape succeeded but screener failed:\n{str(exc)[:400]}",
            color=0xFF1744,
        )
        return

    if not candidates:
        click.echo("  Screener returned 0 candidates")
        return

    with_pattern = sum(1 for c in candidates if c.pattern_type)
    click.echo(
        f"  Screener: {len(candidates)} candidates "
        f"({with_pattern} with pattern match)"
    )

    try:
        payloads = _build_payloads(candidates)
        for payload in payloads:
            resp = httpx.post(webhook, json=payload, timeout=10)
            resp.raise_for_status()
        click.echo(f"  Screener results sent to Discord ({session})")
    except Exception as exc:
        click.echo(f"  Failed to send screener results: {exc}")


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

    import httpx
    from zoneinfo import ZoneInfo

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
                await _run_qu_scrape(session_name, 0, None, 0, None, False, "http://127.0.0.1:9222")
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
