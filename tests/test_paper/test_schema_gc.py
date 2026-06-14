"""WS C — leaked test-schema gc + fixture-leak regression.

Two lanes:
  * pure guard tests (no DB) — the allowlist regex + protected-name refusal;
  * postgres-lane tests (``requires_postgres`` via the shared pg url) — seed a
    leaked schema, prove dry-run lists it, ``--apply`` drops it, and protected
    schemas are refused.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text

from rainier.core.test_schema_gc import (
    gc_test_schemas,
    is_droppable_test_schema,
    list_test_schemas,
)

# --------------------------------------------------------------------------
# Pure guard lane (no DB)
# --------------------------------------------------------------------------

_NO_PROTECTED: frozenset[str] = frozenset()


@pytest.mark.parametrize(
    "name",
    [
        "rainier_paper_test",
        "rainier_chartmig_test",
        "rainier_mig0006_test",
        "rainier_paper_test_ab12",          # per-run suffix
        "rainier_paper_test_ab12_w3",       # per-run + per-worker suffix
    ],
)
def test_allowlist_matches_test_schemas(name):
    assert is_droppable_test_schema(name, protected=_NO_PROTECTED)


@pytest.mark.parametrize(
    "name",
    [
        "public",
        "market",
        "rainier",                  # not a test schema
        "rainier_paper",            # missing _test
        "rainier_other_test",       # not in {paper,chartmig,mig0006}
        "xrainier_paper_test",      # not anchored at start
        "rainier_paper_test_AB12",  # uppercase suffix not allowed
        "rainier_paper_testx",      # suffix not separated by underscore
    ],
)
def test_allowlist_refuses_non_test_schemas(name):
    assert not is_droppable_test_schema(name, protected=_NO_PROTECTED)


def test_protected_names_refused_even_if_regex_would_match():
    # A protected name that *also* matches the regex must still be refused.
    protected = frozenset({"rainier_paper_test"})
    assert not is_droppable_test_schema("rainier_paper_test", protected=protected)


# --------------------------------------------------------------------------
# Postgres lane — seed a leak, gc it. Skips cleanly without a pg url.
# --------------------------------------------------------------------------


def _pg_url() -> str | None:
    return os.environ.get("RAINIER_TEST_DATABASE_URL")


requires_pg_url = pytest.mark.skipif(
    _pg_url() is None,
    reason="RAINIER_TEST_DATABASE_URL not set",
)

_LEAK = "rainier_paper_test_gcseed"


@pytest.fixture
def seeded_leak_engine():
    """Engine on the shared pg url with one seeded leaked test schema.

    Always drops the seed in teardown (success AND failure) so this test never
    becomes the leak it is testing for.
    """
    url = _pg_url()
    engine = create_engine(url, future=True)
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{_LEAK}" CASCADE'))
        conn.execute(text(f'CREATE SCHEMA "{_LEAK}"'))
    try:
        yield engine
    finally:
        with engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{_LEAK}" CASCADE'))
        engine.dispose()


@requires_pg_url
def test_dry_run_lists_seeded_leak(seeded_leak_engine):
    listed = list_test_schemas(seeded_leak_engine)
    assert _LEAK in listed
    result = gc_test_schemas(seeded_leak_engine, apply=False)
    assert _LEAK in result["candidates"]
    assert result["dropped"] == []
    # Dry-run must not have dropped it.
    assert _LEAK in list_test_schemas(seeded_leak_engine)


@requires_pg_url
def test_apply_drops_seeded_leak(seeded_leak_engine):
    result = gc_test_schemas(seeded_leak_engine, apply=True)
    assert _LEAK in result["dropped"]
    assert _LEAK not in list_test_schemas(seeded_leak_engine)


@requires_pg_url
def test_gc_never_drops_public_or_market(seeded_leak_engine):
    # public always exists; market may or may not — neither may ever appear.
    result = gc_test_schemas(seeded_leak_engine, apply=True)
    assert "public" not in result["candidates"]
    assert "public" not in result["dropped"]
    assert "market" not in result["dropped"]
    # public still present after apply.
    with seeded_leak_engine.connect() as conn:
        present = conn.execute(
            text(
                "SELECT 1 FROM information_schema.schemata "
                "WHERE schema_name = 'public'"
            )
        ).scalar()
    assert present == 1
