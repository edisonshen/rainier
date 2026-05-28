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
