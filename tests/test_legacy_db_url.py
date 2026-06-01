"""Tests for the LEGACY_DATABASE_URL split (P0 QU100 restore).

Design: docs/DESIGN-legacy-database-url-split.md §5 (items 1, 3, 4, 6, 7).

The legacy engine (``core/database.py`` singleton) must read
``settings.legacy_database_url`` (env ``LEGACY_DATABASE_URL``, default local
TimescaleDB) so ``DATABASE_URL`` can stay pointed at the canonical Neon store.
``db/engine.py`` (canonical, ``DATABASE_URL``) is intentionally NOT touched here
— it has its own coverage in ``tests/test_db_engine.py`` (design §5 item 2).
"""

from __future__ import annotations

import pytest

LOCAL_DEFAULT = "postgresql://rainier:rainier_dev@localhost:5432/rainier"


@pytest.fixture
def reset_db_singletons(monkeypatch):
    """Reset ALL caches that hide env changes (design §5 item 7).

    Mutating ``LEGACY_DATABASE_URL``/``DATABASE_URL`` is useless if a cached
    ``Settings`` (``core.config._settings``) or a cached engine/session-factory
    (``core.database._engine`` / ``_session_factory``) survives across tests.
    Reset all three before AND after each test so neither stale config nor a
    pinned engine leaks in either direction.
    """
    from rainier.core import config, database

    def _clear() -> None:
        config._settings = None
        database._engine = None
        database._session_factory = None

    _clear()
    yield
    _clear()


def test_legacy_engine_uses_legacy_database_url(reset_db_singletons):
    """§5.1 — legacy engine binds to ``legacy_database_url``, NOT ``database_url``."""
    from rainier.core.config import Settings
    from rainier.core.database import get_engine

    settings = Settings(
        database_url="sqlite:///file:canonical?mode=memory&cache=shared&uri=true",
        legacy_database_url="sqlite:///file:legacy?mode=memory&cache=shared&uri=true",
    )
    engine = get_engine(settings)
    bound = str(engine.url)
    assert "legacy" in bound, f"legacy engine bound to wrong DB: {bound!r}"
    assert "canonical" not in bound, (
        f"legacy engine wrongly bound to the canonical database_url: {bound!r}"
    )


def test_default_fallback_to_local(reset_db_singletons, monkeypatch):
    """§5.3 — with ``LEGACY_DATABASE_URL`` unset, default is the local DSN."""
    from rainier.core.config import Settings

    monkeypatch.delenv("LEGACY_DATABASE_URL", raising=False)
    assert Settings().legacy_database_url == LOCAL_DEFAULT


def test_uppercase_env_var_populates_field(reset_db_singletons, monkeypatch):
    """§5.4 — uppercase ``LEGACY_DATABASE_URL`` env var maps to the field.

    Locks down pydantic's case-insensitive env mapping for this P0 field; do
    not assume the default silently masks a typo'd env name.
    """
    from rainier.core.config import Settings

    sentinel = "postgresql://u:p@example.invalid:5432/legacy_sentinel"
    monkeypatch.setenv("LEGACY_DATABASE_URL", sentinel)
    assert Settings().legacy_database_url == sentinel


def test_init_db_targets_legacy_engine(reset_db_singletons, monkeypatch):
    """§5.6 — ``init_db()`` operates on the legacy engine (``get_engine()``),
    never the canonical ``DATABASE_URL`` path.

    We capture the engine handed to ``Base.metadata.create_all`` and assert it
    is the legacy singleton bound to ``legacy_database_url``.
    """
    from rainier.core import database
    from rainier.core.config import Settings

    legacy_url = "sqlite:///file:initdb_legacy?mode=memory&cache=shared&uri=true"
    settings = Settings(
        database_url="postgresql://u:p@example.invalid/canonical",
        legacy_database_url=legacy_url,
    )
    # Pin the settings the no-arg get_engine() will use.
    monkeypatch.setattr(database, "get_settings", lambda: settings)

    captured: dict[str, object] = {}

    def _capture_create_all(engine, *args, **kwargs):
        captured["engine"] = engine
        # Don't actually create tables against a throwaway sqlite memory DB.
        return None

    monkeypatch.setattr(database.Base.metadata, "create_all", _capture_create_all)
    # Stub hypertable creation — TimescaleDB SQL isn't valid on sqlite.
    monkeypatch.setattr(database, "_create_hypertables", lambda engine: None)

    database.init_db()

    assert "engine" in captured, "init_db did not call create_all"
    bound = str(captured["engine"].url)  # type: ignore[union-attr]
    assert "initdb_legacy" in bound, (
        f"init_db targeted the wrong engine: {bound!r}"
    )
    assert "canonical" not in bound, (
        f"init_db wrongly targeted the canonical DATABASE_URL: {bound!r}"
    )
