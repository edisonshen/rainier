"""Coverage-based yfinance OHLCV backfill for QU100 stocks.

Extracted verbatim from ``rainier.cli`` (``db backfill-prices``); the CLI
keeps a thin click wrapper.
"""

from __future__ import annotations

from datetime import date, datetime

import click
import pandas as pd


def backfill_prices(years: int, batch_size: int, dry_run: bool) -> None:
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

