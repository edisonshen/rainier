"""Database management commands (legacy singleton + canonical Postgres store)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import click

from rainier.cli import (
    cli,
)


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
    """Initialize database tables and hypertables.

    After ``create_all`` (additive only — it never ALTERs an existing table to
    add a missing column), run ``check_schema_drift`` as a loud chokepoint. An
    existing-but-drifted table (a stub ``stocks``, an unapplied
    ``migrations/*.sql`` column) is invisible to ``create_all`` and otherwise
    surfaces only as a silent zero-row scrape. If drift is found we print every
    missing object and exit non-zero so it screams instead.
    """
    from rainier.core.database import get_engine, init_db
    from rainier.core.schema_check import KNOWN_BENIGN_DRIFT, check_schema_drift

    click.echo("Initializing database...")
    init_db()
    # Success banner is printed only AFTER the drift gate below passes —
    # printing it here would tell operators/scripts "success" on the exact
    # drifted state this chokepoint exists to catch (codex 43f3 [P2]).
    click.echo("Tables created (create_all complete).")

    # Split out the documented pre-existing benign drift (KNOWN_BENIGN_DRIFT):
    # the operator runbook blocks only on findings BEYOND it. Hard-failing on
    # the benign finding would make `db init` exit non-zero on the live legacy
    # DB forever, with no migrations/*.sql able to clear it.
    findings = check_schema_drift(get_engine())
    benign = [f for f in findings if f in KNOWN_BENIGN_DRIFT]
    real = [f for f in findings if f not in KNOWN_BENIGN_DRIFT]
    for finding in benign:
        click.echo(f"Known-benign drift (ignored): {finding}")
    if real:
        click.echo("")
        click.echo("SCHEMA DRIFT DETECTED — the live DB is behind the ORM:")
        for finding in real:
            click.echo(f"  - {finding}")
        click.echo("")
        click.echo(
            "Recovery: if this DB is already versioned (schema_migrations "
            "exists), apply the pending files via "
            "rainier.core.legacy_migrate.run_migrations. If it is "
            "UNVERSIONED, hand-apply the missing migrations/*.sql to the "
            "legacy engine first, then adopt with baseline_migrations "
            "(run_migrations refuses unversioned schemas, and baseline "
            "refuses while drift remains)."
        )
        raise click.ClickException(f"{len(real)} schema-drift finding(s); see above.")
    # Honest scope: the drift check verifies ORM-DECLARED objects (tables,
    # columns, named indexes/constraints by name). It cannot see DDL that
    # exists only in migrations/*.sql without an ORM mirror (e.g. 0001's
    # idx_llm_analysis_idempotent — absent on the live DB too: its expression
    # is not runnable on modern Postgres), and create_all never runs
    # migrations/*.sql. Surface the unrecorded legacy-migration state instead
    # of implying full health.
    click.echo("Schema drift check (ORM-declared objects): clean.")

    from rainier.core.legacy_migrate import applied_versions, pending_migrations

    engine = get_engine()
    pending = pending_migrations(engine)
    if pending:
        if applied_versions(engine):
            # Versioned DB: the pending files are new migrations — apply them.
            hint = (
                "Apply them via rainier.core.legacy_migrate.run_migrations."
            )
        else:
            # Unversioned (incl. the fresh db-init bootstrap): run_migrations
            # refuses unversioned schemas, so lead with baseline_migrations.
            hint = (
                "This DB is unversioned; adopt the file history with "
                "rainier.core.legacy_migrate.baseline_migrations "
                "(run_migrations refuses unversioned schemas)."
            )
        click.echo(
            f"NOTE: {len(pending)} legacy migration file(s) not recorded as applied "
            "in schema_migrations. `db init` (create_all) does NOT run "
            f"migration-only DDL such as indexes/constraints. {hint}"
        )
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
    from rainier.db.backfill_prices import backfill_prices

    backfill_prices(years, batch_size, dry_run)


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
       ``Path(__file__).resolve().parents[3]``.

    The .ini file leaves ``sqlalchemy.url`` empty on purpose — db/alembic/
    env.py reads DATABASE_URL from the environment so creds never land in
    git. We override ``script_location`` defensively after loading so the
    Config works even if a future ini edit drops the ``%(here)s`` prefix
    (the regression test ``test_alembic_ini_script_location_is_config_relative``
    in tests/test_cli/test_db.py guards the raw ini path too).
    """
    from importlib import resources

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
    #    → repo root is parents[3], db/ lives at the repo root.
    repo_root = Path(__file__).resolve().parents[3]
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
        # An EMPTY source is a misconfiguration (mispointed LEGACY_DATABASE_URL /
        # failed restore / empty local DB), NOT a legitimate "nothing to do". The
        # reconcile left the backup intact (non-destructive), but exiting 0 here
        # would let the nightly cron silently back up NOTHING from a dead source.
        # Fail loud (non-zero) so the cron alerts instead of masking the gap.
        if result.source_empty:
            raise click.ClickException(
                "backup-money-flow — local SOURCE is EMPTY (0 rows). The backup "
                "was left intact, but nothing was copied. This is a "
                "misconfiguration (check LEGACY_DATABASE_URL / local DB health), "
                "not a no-op — failing loudly rather than reporting success."
            )
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
