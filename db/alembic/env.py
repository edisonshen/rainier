"""Alembic environment for the rainier canonical-store schema.

Three non-default settings are CRITICAL:

* ``include_schemas=True``
    Without this, autogenerate ignores non-default schemas. Since every
    table in this project lives under ``market.*``, autogenerate would
    silently produce no-op migrations. See task plan §3 acceptance + §5.

* ``version_table_schema="public"``
    The ``alembic_version`` bookkeeping table belongs in ``public``, not in
    ``market``. Otherwise ``downgrade base`` (which drops the market schema)
    also wipes alembic's own state, making re-upgrade impossible.

* ``include_name=_include_only_market``
    Pairs with ``include_schemas=True`` to keep autogenerate scoped to the
    ``market`` schema (plus alembic's own version table in ``public``).
    Without this filter, ``alembic revision --autogenerate`` against a DB
    that also contains legacy rainier tables in ``public`` (the existing
    ORM singleton + LLM thesis tables; design doc dual-write era) would
    interpret those tables as "removed" and emit destructive ``drop_table``
    operations in the generated migration. Codex review iter-3 flagged
    this footgun before any autogenerate migration shipped.

DATABASE_URL is read from the environment. There's no fallback — if the
operator forgot to set it, alembic should fail loudly, not connect to some
default local DB.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from rainier.db.schema import metadata as target_metadata

# Alembic Config object — provides access to .ini values.
config = context.config

# Set up Python logging from alembic.ini (if defined).
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _database_url() -> str:
    """Resolve DATABASE_URL with no silent fallback."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set — alembic env.py refuses to use a "
            "default. See docs/DESIGN-rainier-postgres-canonical-store.md."
        )
    return url


def _include_only_market(name, type_, parent_names):
    """Scope autogenerate to the ``market`` schema + alembic's version table.

    Alembic calls this for every schema, table, and index it discovers. We
    accept:

    * The ``market`` schema itself (``type_ == 'schema'``).
    * Anything whose parent schema is ``market`` (tables, indexes, FKs).
    * The ``alembic_version`` table in ``public`` (so alembic can manage
      its own bookkeeping; otherwise autogenerate would try to drop it).

    Everything else — legacy ORM tables in ``public``, hypothetical other
    schemas — is filtered out so autogenerate never emits destructive
    operations against tables it doesn't own.

    Signature matches Alembic's ``include_name`` callback contract.
    """
    if type_ == "schema":
        return name == "market"
    # For tables/indexes/etc, ``parent_names['schema_name']`` holds the
    # owning schema (may be None for default 'public').
    schema = parent_names.get("schema_name") if parent_names else None
    if schema == "market":
        return True
    if schema in (None, "public") and type_ == "table" and name == "alembic_version":
        return True
    return False


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout, no live DB)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        include_name=_include_only_market,
        version_table_schema="public",
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (live connection)."""
    # Inject DATABASE_URL into the engine config without persisting it to
    # alembic.ini (which stays empty so we don't leak creds in git).
    ini_section = config.get_section(config.config_ini_section, {}) or {}
    ini_section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        ini_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,             # market.* is non-default
            include_name=_include_only_market, # scope autogenerate to market+
            version_table_schema="public",    # alembic_version in public
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
