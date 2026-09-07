"""Thematic ranks and market-breadth commands."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import click
import pandas as pd

from rainier.cli import (
    cli,
)

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
    root = Path(__file__).resolve().parents[3]
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

    root = Path(__file__).resolve().parents[3]
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

