"""Tests for src/rainier/db/engine.py — Postgres canonical-store pivot Phase 1.

This is the new `db/` package (NOT the legacy `core/database.py` singleton).
The new engine reads `DATABASE_URL` directly from env, applies pool_size=2 +
max_overflow=2 + pool_pre_ping=True, and raises RuntimeError if the env var
is missing — see docs/DESIGN-rainier-postgres-canonical-store.md §D-3 and
docs/TASK-PLAN-rainier-pg-schema-bootst-966f.md §5.
"""

from __future__ import annotations

import pytest
from sqlalchemy.engine import Engine


def test_get_engine_raises_when_database_url_missing(monkeypatch):
    """get_engine() must raise RuntimeError with an actionable message
    when DATABASE_URL is not set. Don't silently fall back to a default."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from rainier.db.engine import get_engine

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        get_engine()


def test_get_engine_returns_engine_with_configured_url(monkeypatch):
    """get_engine() returns a SQLAlchemy Engine bound to DATABASE_URL."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost/x")
    from rainier.db.engine import get_engine

    eng = get_engine()
    assert isinstance(eng, Engine)
    # Render the URL with password hidden but driver + host preserved.
    url = eng.url
    assert url.drivername == "postgresql+psycopg"
    assert url.host == "localhost"
    assert url.database == "x"


def test_get_engine_pool_config(monkeypatch):
    """Pool config: size=2, max_overflow=2, pool_pre_ping=True (per task plan §5)."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost/x")
    from rainier.db.engine import get_engine

    eng = get_engine()
    # QueuePool exposes pool_size + overflow size via these attrs.
    assert eng.pool.size() == 2
    # `max_overflow` isn't exposed by name on QueuePool, but the configured
    # value is stored as `_max_overflow` (SQLAlchemy internal API). Use the
    # public `overflow()` semantics by checking the private slot — this is
    # the standard way to assert pool config in SQLAlchemy tests.
    assert eng.pool._max_overflow == 2
    assert eng.pool._pre_ping is True


def test_get_engine_returns_fresh_instance_per_call(monkeypatch):
    """The new db/ package's get_engine() is NOT a singleton — every call
    builds a new Engine. This is intentional: callers manage lifecycle.
    The legacy `core/database.py` keeps its singleton pattern; the new
    `db/` package mirrors the task plan §5 snippet, which constructs
    fresh each call."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost/x")
    from rainier.db.engine import get_engine

    e1 = get_engine()
    e2 = get_engine()
    assert e1 is not e2
