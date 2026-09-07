"""Futures price-action commands: fetch/scan/daytrade/chart/backtest/report, SMA sweep."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click
import pandas as pd

from rainier.cli import (
    cli,
)
from rainier.core.types import Timeframe


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

