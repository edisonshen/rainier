"""Publish-friendly HTML dashboard renderers."""

from __future__ import annotations

from pathlib import Path

import click

from rainier.cli import (
    cli,
)

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

