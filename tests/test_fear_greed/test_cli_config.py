"""The `fear-greed` CLI commands must honor the root `--config` for their legacy
DB writes (codex P1). `_legacy_db_for_config` seeds the process settings
singleton from the operator-selected YAML and resets the cached legacy engine /
session factory so `get_session()` rebinds against THAT database, then restores
all three process globals on exit so an in-process caller does not inherit this
command's DB.
"""

from __future__ import annotations

from rainier import cli as cli_mod
from rainier.core import config as config_mod
from rainier.core import database as database_mod


def test_legacy_db_for_config_rebinds_then_restores(monkeypatch):
    sentinel_settings = object()
    # The helper imports load_settings_fresh from rainier.core.config at call
    # time, so patching it on the module is picked up.
    monkeypatch.setattr(config_mod, "load_settings_fresh", lambda _p: sentinel_settings)

    prev_settings = object()
    prev_engine = object()
    prev_factory = object()
    # Snapshot the true process globals so the test restores them regardless of
    # outcome (these are singletons shared across the interpreter).
    saved = (config_mod._settings, database_mod._engine, database_mod._session_factory)
    config_mod._settings = prev_settings
    database_mod._engine = prev_engine
    database_mod._session_factory = prev_factory

    class _Ctx:
        obj = {"settings_path": "config/staging.yaml"}

    try:
        with cli_mod._legacy_db_for_config(_Ctx()):
            # Inside: singleton seeded from --config; engine/factory reset so the
            # next get_session() rebinds against the selected DB.
            assert config_mod._settings is sentinel_settings
            assert database_mod._engine is None
            assert database_mod._session_factory is None
        # Outside: the prior globals are restored (no leak to later commands).
        assert config_mod._settings is prev_settings
        assert database_mod._engine is prev_engine
        assert database_mod._session_factory is prev_factory
    finally:
        config_mod._settings, database_mod._engine, database_mod._session_factory = saved


def test_legacy_db_for_config_disposes_engine_created_in_block(monkeypatch):
    """An engine created inside the block (via get_session) must be disposed on
    exit, not silently orphaned — its pooled connections would leak on every
    in-process/CliRunner reuse."""
    monkeypatch.setattr(config_mod, "load_settings_fresh", lambda _p: object())

    disposed = {"v": False}

    class _FakeEngine:
        def dispose(self):
            disposed["v"] = True

    prev_engine = object()
    saved = (config_mod._settings, database_mod._engine, database_mod._session_factory)
    config_mod._settings = object()
    database_mod._engine = prev_engine
    database_mod._session_factory = object()

    class _Ctx:
        obj = {"settings_path": "config/staging.yaml"}

    try:
        with cli_mod._legacy_db_for_config(_Ctx()):
            # Simulate get_session() lazily creating a fresh engine in the block.
            database_mod._engine = _FakeEngine()
        assert disposed["v"] is True, "block-created engine not disposed on exit"
        assert database_mod._engine is prev_engine  # prior engine restored
    finally:
        config_mod._settings, database_mod._engine, database_mod._session_factory = saved


def test_legacy_db_for_config_restores_on_exception(monkeypatch):
    """Restore must run on the failure path too (finally, not happy-path only)."""
    monkeypatch.setattr(config_mod, "load_settings_fresh", lambda _p: object())

    prev_settings = object()
    prev_engine = object()
    prev_factory = object()
    saved = (config_mod._settings, database_mod._engine, database_mod._session_factory)
    config_mod._settings = prev_settings
    database_mod._engine = prev_engine
    database_mod._session_factory = prev_factory

    class _Ctx:
        obj = {"settings_path": "config/staging.yaml"}

    try:
        try:
            with cli_mod._legacy_db_for_config(_Ctx()):
                raise RuntimeError("ingest blew up")
        except RuntimeError:
            pass
        assert config_mod._settings is prev_settings
        assert database_mod._engine is prev_engine
        assert database_mod._session_factory is prev_factory
    finally:
        config_mod._settings, database_mod._engine, database_mod._session_factory = saved
