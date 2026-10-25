"""Futures commands: fetch, S/R zones (sr-levels/sr-validate), pattern backtest, SMA sweep."""

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
@click.pass_context
def fetch(ctx, symbol, data_dir, provider_type):
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


MTF_TIMEFRAMES = (Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M5)


def _load_mtf_csvs(data_path: Path, symbol: str) -> dict[Timeframe, pd.DataFrame]:
    """Load {symbol}_{1D,4H,1H,5m}.csv from data_path (missing files are skipped)."""
    from rainier.data.csv_provider import CSVProvider

    provider = CSVProvider(data_path)
    data: dict[Timeframe, pd.DataFrame] = {}
    for tf in MTF_TIMEFRAMES:
        csv_file = data_path / f"{symbol}_{tf.value}.csv"
        if csv_file.exists():
            data[tf] = provider._read_csv(csv_file, None, None)
            click.echo(f"  Loaded {tf.value}: {len(data[tf])} candles")
    return data


@cli.command("sr-levels")
@click.option("--symbol", default="MES")
@click.option("--data-dir", default="data/csv", type=click.Path(exists=True))
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json", "pine"]))
@click.option("--output", "output_path", default=None, help="Write to file instead of stdout")
@click.option("--publish-kv", is_flag=True,
              help="Also push the JSON chart payload to Workers KV (fengshen.dev /trading/mes)")
@click.pass_context
def sr_levels(ctx, symbol, data_dir, fmt, output_path, publish_kv):
    """Current multi-TF S/R zones (1D -> 4H -> 1H, consolidated) as text, JSON or Pine v5."""
    import json

    from rainier.analysis.sr_zones import detect_mtf_zones, zones_to_levels
    from rainier.publish.cloudflare_kv import put_kv_if_changed
    from rainier.research.xiaojiang.chart import chart_setups
    from rainier.viz.chart_payload import sr_chart_payload
    from rainier.viz.pine import zones_to_pine

    settings = ctx.obj["settings"]
    pub = settings.chart_publish
    cfg = settings.analysis.sr_zones.model_copy(
        update={"max_distance_atr_mult": pub.max_distance_atr_mult,
                "keep_nearest_each_side": pub.keep_nearest_each_side})
    if publish_kv and not (settings.cloudflare_api_token and settings.cloudflare_account_id):
        raise click.ClickException("--publish-kv needs CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID")
    data = _load_mtf_csvs(Path(data_dir), symbol)
    htf = {tf: df for tf, df in data.items() if tf in cfg.timeframes}
    if not htf:
        raise click.ClickException(f"no {symbol} 1D/4H/1H CSVs in {data_dir}")
    latest = data.get(Timeframe.M5, data[next(iter(htf))])
    price = float(latest["close"].iloc[-1])
    asof = pd.Timestamp(latest["timestamp"].iloc[-1]).to_pydatetime()
    zones = detect_mtf_zones(htf, cfg, price)
    levels = zones_to_levels(zones, price)
    levels.sort(key=lambda lv: -lv.price)
    xj = []
    if pub.xj_setups and Timeframe.M5 in data and Timeframe.M5 in pub.bars:
        since = data[Timeframe.M5]["timestamp"].tail(pub.bars[Timeframe.M5]).iloc[0]
        xj = chart_setups(data, pub.xj_setups, zones, since, settings.analysis.sr_zones)
    payload = sr_chart_payload(symbol, asof, price, levels, data, pub.bars,
                               pub.vma_bars, pub.vma_offset, xj)
    if publish_kv:
        key = pub.kv_key.format(symbol=symbol.lower())
        # Same bars + zones (daily halt, holidays) -> identical payload, so skip the write.
        if put_kv_if_changed(settings.cloudflare_account_id, pub.kv_namespace_id, key,
                             json.dumps(payload, separators=(",", ":")),
                             settings.cloudflare_api_token):
            click.echo(f"Published {len(levels)} zones to KV {key}")
        else:
            click.echo(f"KV {key} unchanged (no new bar), skipped write")
    if fmt == "pine":
        out = zones_to_pine(symbol, levels, asof)
    elif fmt == "json":
        out = json.dumps(payload, separators=(",", ":"))
    else:
        out = "\n".join(
            f"{lv.role.value[0].upper()} {lv.price:9.2f}  [{lv.zone_low:.2f}-{lv.zone_high:.2f}]"
            f"  {'+'.join(t.value for t in lv.timeframes):10s} str={lv.strength:.2f}"
            for lv in levels)
    if output_path:
        Path(output_path).write_text(out)
        click.echo(f"Wrote {len(levels)} zones to {output_path}")
    else:
        click.echo(out)


@cli.command("xj-backtest")
@click.option("--symbol", default="MES")
@click.option("--data-dir", default="data/csv", type=click.Path(exists=True))
@click.option("--holdout-start", required=True, help="First holdout date (YYYY-MM-DD), PT")
@click.option("--setup", "keys", multiple=True,
              help="Key in chart_publish.xj_setups (repeatable; default: all)")
@click.option("--random", "n_random", default=20, show_default=True,
              help="Randomly shifted-zone controls")
@click.pass_context
def xj_backtest(ctx, symbol, data_dir, holdout_start, keys, n_random):
    """Xiaojiang setups on the chart's point-in-time zones, train vs holdout vs random zones."""
    from rainier.research.xiaojiang.sim import Market, metrics, simulate
    from rainier.research.xiaojiang.zones import ZoneBook

    settings = ctx.obj["settings"]
    setups = settings.chart_publish.xj_setups
    unknown = set(keys) - set(setups)
    if unknown:
        raise click.ClickException(f"unknown setup(s) {sorted(unknown)}; have {sorted(setups)}")
    data = _load_mtf_csvs(Path(data_dir), symbol)
    if Timeframe.M5 not in data:
        raise click.ClickException(f"{symbol}_5m.csv required")
    book = ZoneBook(data, settings.analysis.sr_zones)
    mkt = Market(data, book)
    controls = [Market(data, book.shifted(seed)) for seed in range(n_random)]
    hs = pd.Timestamp(holdout_start)
    rows = []
    for key in keys or setups:
        p = setups[key]
        real = simulate(mkt, p).trades
        rand = [simulate(m, p).trades for m in controls]
        for name, pick in (("train", lambda t: t[t.t_in < hs]),
                           ("holdout", lambda t: t[t.t_in >= hs])):
            m = metrics(pick(real))
            beat = [m["avg_usd"] > metrics(pick(r))["avg_usd"] for r in rand]
            rows.append({"setup": key, "period": name, **m,
                         "beats_random": sum(beat) / len(beat) if beat else float("nan")})
    click.echo(pd.DataFrame(rows).round(3).to_string(index=False))


@cli.command("sr-validate")
@click.option("--symbol", default="MES")
@click.option("--data-dir", default="data/csv", type=click.Path(exists=True))
@click.option("--holdout-start", required=True, help="First holdout date (YYYY-MM-DD), UTC")
@click.option("--min-timeframes", default=2, show_default=True)
@click.option("--step", default=48, show_default=True, help="5m bars between re-draws")
@click.pass_context
def sr_validate(ctx, symbol, data_dir, holdout_start, min_timeframes, step):
    """Walk-forward 5m touch/reaction scorecard of S/R zones vs randomly shifted zones."""
    from rainier.analysis.sr_eval import scorecard, walk_forward
    from rainier.analysis.sr_zones import detect_mtf_zones

    cfg = ctx.obj["settings"].analysis.sr_zones
    data = _load_mtf_csvs(Path(data_dir), symbol)
    if Timeframe.M5 not in data:
        raise click.ClickException(f"{symbol}_5m.csv required")
    htf = {tf: df for tf, df in data.items() if tf in cfg.timeframes}
    outcomes = walk_forward(
        htf, data[Timeframe.M5], lambda past, px: detect_mtf_zones(past, cfg, px),
        pd.Timestamp(holdout_start, tz="UTC"), reference_tf=cfg.reference_tf, step=step,
    )
    click.echo(scorecard(outcomes, min_timeframes).round(3).to_string())


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

