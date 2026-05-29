"""Tests for `rainier db` CLI subgroup — Phase 1 of Postgres canonical pivot.

Two behaviors covered without needing a live Postgres:

* CLI surface: the `db` group exists and exposes `ping` + `migrate`
  (with a `--downgrade` flag). Smoke check on the click tree.
* Missing DATABASE_URL: `db ping` exits non-zero with an actionable message.

Live-DB tests for migrate/downgrade are in test_db_migrations.py — gated
on `requires_postgres` so they only run when a real Postgres is available.
"""

from __future__ import annotations

from click.testing import CliRunner

from rainier.cli import cli


def test_db_group_registered():
    """`rainier db --help` lists the ping + migrate subcommands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["db", "--help"])
    assert result.exit_code == 0, result.output
    assert "ping" in result.output
    assert "migrate" in result.output


def test_db_group_does_not_shadow_legacy_subcommands():
    """Regression for CI #102: declaring a second `@cli.group() def db()` in
    cli.py replaces the first one in click's registry, silently killing the
    legacy `init` + `backfill-prices` subcommands. The fix is to register
    every db subcommand on a SINGLE `db` group.

    This test asserts the full surface — legacy + new — is reachable via
    `rainier db --help`. If anyone re-introduces a duplicate `@cli.group()
    def db()`, this test goes red before CI does.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["db", "--help"])
    assert result.exit_code == 0, result.output

    for subcmd in ("init", "backfill-prices", "ping", "migrate"):
        assert subcmd in result.output, (
            f"`rainier db --help` is missing `{subcmd}` — likely caused by a "
            f"duplicate `@cli.group() def db()` shadowing the original. "
            f"Help output was:\n{result.output}"
        )

    # And each legacy subcommand must actually invoke (--help is a no-op
    # that proves click can resolve the subcommand off the merged group).
    legacy_init = runner.invoke(cli, ["db", "init", "--help"])
    assert legacy_init.exit_code == 0, legacy_init.output
    legacy_backfill = runner.invoke(cli, ["db", "backfill-prices", "--help"])
    assert legacy_backfill.exit_code == 0, legacy_backfill.output


def test_db_migrate_help_exposes_downgrade_flag():
    """`rainier db migrate --help` shows the --downgrade option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["db", "migrate", "--help"])
    assert result.exit_code == 0, result.output
    assert "--downgrade" in result.output


def test_db_ping_fails_loud_without_database_url(monkeypatch):
    """`db ping` without DATABASE_URL exits non-zero with a clear message."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    runner = CliRunner()
    result = runner.invoke(cli, ["db", "ping"])
    assert result.exit_code != 0
    # Click wraps RuntimeError into a ClickException with our message.
    combined = (result.output or "") + (str(result.exception) if result.exception else "")
    assert "DATABASE_URL" in combined


def test_db_migrate_fails_loud_without_database_url(monkeypatch):
    """`db migrate` without DATABASE_URL exits non-zero with a clean message.

    Mirrors `test_db_ping_fails_loud_without_database_url`: alembic env.py
    raises RuntimeError("DATABASE_URL not set ...") the moment `command.upgrade`
    resolves the engine config. Without wrapping, that surfaced as a raw
    traceback (codex iter-5 / PR #102 deferred [P2]). The command must catch it
    and re-raise as a ClickException so the CLI prints a single `Error:` line.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    runner = CliRunner()
    result = runner.invoke(cli, ["db", "migrate"])
    assert result.exit_code != 0
    # A wrapped ClickException leaves no traceback frames on stdout/stderr.
    assert "Traceback" not in (result.output or "")
    # The ClickException carries the underlying RuntimeError text (DATABASE_URL).
    combined = (result.output or "") + (str(result.exception) if result.exception else "")
    assert "DATABASE_URL" in combined
    # And it must be a ClickException, not a bare RuntimeError leaking through.
    if result.exception is not None:
        import click

        assert isinstance(result.exception, click.ClickException), (
            f"db migrate leaked a {type(result.exception).__name__} instead of "
            f"wrapping it in click.ClickException: {result.exception!r}"
        )


def test_db_migrate_wraps_unreachable_db_in_click_exception(monkeypatch):
    """`db migrate` against an unreachable DB exits clean, not with a traceback.

    Stub `alembic.command.upgrade` to raise OperationalError (what a real
    unreachable host produces). The command must catch any migration-time
    failure and re-raise as ClickException — no traceback, non-zero exit.
    """
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@unreachable:5432/x")

    from sqlalchemy.exc import OperationalError

    from rainier import cli as cli_mod

    def _boom(*_args, **_kwargs):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    # Patch the alembic command surface the migrate body calls into. The body
    # does `from alembic import command` then `command.upgrade(...)`, so we
    # patch the attribute on the alembic.command module itself.
    import alembic.command as alembic_command

    monkeypatch.setattr(alembic_command, "upgrade", _boom)
    # Avoid touching the real filesystem-resolved config (irrelevant here).
    monkeypatch.setattr(cli_mod, "_resolve_alembic_config", lambda: object())

    runner = CliRunner()
    result = runner.invoke(cli, ["db", "migrate"])
    assert result.exit_code != 0
    assert "Traceback" not in (result.output or "")
    if result.exception is not None:
        import click

        assert isinstance(result.exception, click.ClickException), (
            f"db migrate leaked a {type(result.exception).__name__}: "
            f"{result.exception!r}"
        )


def test_wheel_packages_db_assets_under_rainier_db_assets():
    """Regression: pyproject.toml's `[tool.hatch.build.targets.wheel]` must
    `force-include` the top-level db/ tree at `rainier/_db_assets/` inside
    the wheel. Without this, non-editable wheel installs crash on
    `rainier db migrate` because alembic.ini isn't on disk.

    Codex review iter-2 (2026-05-28) caught this footgun before any wheel
    consumer hit it. We don't actually build a wheel here — that would be
    slow + flaky in CI — instead we assert the static config that controls
    wheel layout.
    """
    from pathlib import Path

    try:
        import tomllib  # py311+
    except ImportError:  # pragma: no cover
        import tomli as tomllib

    repo_root = Path(__file__).resolve().parents[2]
    pyproject_path = repo_root / "pyproject.toml"
    with pyproject_path.open("rb") as fh:
        pyproject = tomllib.load(fh)

    wheel_cfg = (
        pyproject.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
    )
    force_include = wheel_cfg.get("force-include", {})
    assert force_include.get("db") == "rainier/_db_assets", (
        "pyproject.toml's [tool.hatch.build.targets.wheel] must contain "
        "`force-include = { \"db\" = \"rainier/_db_assets\" }` so the wheel "
        "ships alembic.ini + alembic/. Got: " + repr(force_include)
    )

    # Sanity: the source dir referenced by force-include actually exists.
    assert (repo_root / "db" / "alembic.ini").exists()
    assert (repo_root / "db" / "alembic").is_dir()


def test_alembic_ini_script_location_is_config_relative(tmp_path, monkeypatch):
    """Regression: alembic.ini's `script_location` must resolve relative to
    the ini file (via `%(here)s`), NOT relative to cwd. Without this, running
    `alembic -c db/alembic.ini revision --autogenerate ...` from anywhere
    other than the repo root fails with `Path doesn't exist: alembic`.

    Codex review 2026-05-28 caught this footgun before it bit the next
    migration author.
    """
    from pathlib import Path

    from alembic.config import Config
    from alembic.script import ScriptDirectory

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "db" / "alembic.ini"

    # Change cwd to a tempdir so any `.`-relative resolution would break.
    monkeypatch.chdir(tmp_path)

    # Load the raw config WITHOUT overriding script_location — this is the
    # path Alembic CLI takes when invoked directly.
    cfg = Config(str(cfg_path))
    # If %(here)s is missing from alembic.ini, ScriptDirectory.from_config()
    # raises `Path doesn't exist: alembic`. With %(here)s, it resolves
    # correctly to <repo>/db/alembic regardless of cwd.
    script = ScriptDirectory.from_config(cfg)
    resolved = Path(script.dir).resolve()
    assert resolved == (repo_root / "db" / "alembic").resolve(), (
        f"script_location resolved to {resolved}; "
        f"expected {repo_root / 'db' / 'alembic'}. "
        "Did you drop the %(here)s prefix in db/alembic.ini?"
    )


class _NullCM:
    """Minimal context-manager for the alembic.context shim in tests."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_alembic_include_filter_scopes_to_market_schema(monkeypatch):
    """Regression: db/alembic/env.py's `_include_only_market` callback must
    accept everything in market + alembic's own version table, and reject
    everything else. Without this filter, `alembic revision --autogenerate`
    against a DB with legacy rainier tables in public would emit destructive
    drop_table operations for them.

    Codex review iter-3 (2026-05-28) caught this before the first
    autogenerated migration ran.
    """
    import importlib.util
    import sys
    import types
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / "db" / "alembic" / "env.py"

    # env.py imports `from alembic import context` and calls
    # `context.is_offline_mode()` at module top-level. Stub the import so
    # we can load the module just to extract `_include_only_market`.
    fake_context = types.SimpleNamespace(
        is_offline_mode=lambda: True,
        configure=lambda *a, **kw: None,
        begin_transaction=_NullCM,
        run_migrations=lambda: None,
        config=types.SimpleNamespace(
            config_file_name=None,
            config_ini_section="alembic",
            get_section=lambda *a, **kw: {},
        ),
    )
    fake_alembic_pkg = types.ModuleType("alembic")
    fake_alembic_pkg.context = fake_context

    monkeypatch.setitem(sys.modules, "alembic", fake_alembic_pkg)
    monkeypatch.setitem(sys.modules, "alembic.context", fake_context)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost/x")

    spec = importlib.util.spec_from_file_location("rainier_alembic_env", env_path)
    env_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_mod)

    inc = env_mod._include_only_market

    # Schema-level: only market is accepted.
    assert inc("market", "schema", None) is True
    assert inc("public", "schema", None) is False
    assert inc("timescaledb_internal", "schema", None) is False

    # Tables in market are accepted regardless of name.
    assert inc("tickers", "table", {"schema_name": "market"}) is True
    assert inc("any_future_table", "table", {"schema_name": "market"}) is True

    # Tables in public are rejected — EXCEPT alembic_version.
    assert inc("alembic_version", "table", {"schema_name": "public"}) is True
    assert inc("alembic_version", "table", {"schema_name": None}) is True
    assert inc("money_flow_snapshots", "table", {"schema_name": "public"}) is False
    assert inc("legacy_ohlcv", "table", {"schema_name": None}) is False

    # Indexes / FKs in market accepted; in public rejected.
    assert inc("ix_thematic_ohlcv_date", "index", {"schema_name": "market"}) is True
    assert inc("ix_legacy", "index", {"schema_name": "public"}) is False
