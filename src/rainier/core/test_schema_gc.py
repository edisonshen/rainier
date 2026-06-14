"""Reap leaked throwaway test schemas from the LEGACY engine's database.

The paper-tracker test suite builds disposable Postgres schemas
(``rainier_paper_test``, ``rainier_chartmig_test``, ``rainier_mig0006_test``,
and — after this change — per-run suffixed variants like
``rainier_paper_test_ab12``). Teardown normally drops them, but a SIGKILL'd
test run, a crashed worker, or a pre-fix fixture can leave a schema behind in
the live local TimescaleDB. Those orphans accumulate silently.

This module lists and (on apply) drops ONLY schemas matching the anchored
allowlist regex, and NEVER ``public`` / ``market`` / the connection's current
schema — even if such a name somehow matched the regex. It targets the
**legacy** engine (``core.database`` / ``legacy_database_url``), never the
canonical Neon ``db.engine`` (the 2026-06-01 two-engine trap).

ASCII flow:

    list_test_schemas(engine)  -> [names matching regex, minus protected]
    gc_test_schemas(engine,
                    apply=False) -> {"candidates": [...], "dropped": []}
                    apply=True   -> drops each, {"candidates": [...],
                                                 "dropped": [...],
                                                 "failed": [(name, err)]}
"""

from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.engine import Engine

# Anchored allowlist. The trailing ``(_[a-z0-9]+)*`` permits per-run AND
# per-worker suffix tokens appended after the base name, e.g.
# ``rainier_paper_test_ab12`` or ``rainier_paper_test_ab12_w3``. A raw prefix
# match (``startswith("rainier_")``) is deliberately NOT used — it would sweep
# in unrelated namespaces.
TEST_SCHEMA_RE = re.compile(r"^rainier_(paper|chartmig|mig0006)_test(_[a-z0-9]+)*$")

# Never droppable, regardless of regex match. ``current_schema()`` is added
# dynamically at call time (the active schema must never be dropped out from
# under the live connection).
HARD_PROTECTED = frozenset({"public", "market"})


def is_droppable_test_schema(name: str, *, protected: frozenset[str]) -> bool:
    """True iff ``name`` matches the allowlist regex and is not protected.

    Protected names (``public``, ``market``, the current schema) are refused
    even when they match the regex — the regex can never authorise dropping
    them.
    """
    if name in protected:
        return False
    return TEST_SCHEMA_RE.match(name) is not None


def _protected_set(engine: Engine) -> frozenset[str]:
    with engine.connect() as conn:
        current = conn.execute(text("SELECT current_schema()")).scalar()
    extra = {current} if current else set()
    return HARD_PROTECTED | extra


def list_test_schemas(engine: Engine) -> list[str]:
    """Return sorted names of droppable leaked test schemas on ``engine``'s DB."""
    protected = _protected_set(engine)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT schema_name FROM information_schema.schemata")
        ).scalars().all()
    return sorted(
        n for n in rows if is_droppable_test_schema(n, protected=protected)
    )


def gc_test_schemas(engine: Engine, *, apply: bool = False) -> dict:
    """List (and on ``apply`` drop) leaked test schemas.

    Dry-run by default: returns the candidate list without touching the DB.
    With ``apply=True`` each candidate is ``DROP SCHEMA ... CASCADE``'d in its
    own transaction; a failure on one does not abort the rest. The candidate
    list is re-computed under the protected guard, so a protected schema can
    never be dropped even if a caller passes a hand-built list.
    """
    candidates = list_test_schemas(engine)
    result: dict = {"candidates": candidates, "dropped": [], "failed": []}
    if not apply:
        return result

    protected = _protected_set(engine)
    for name in candidates:
        # Defense in depth: re-check the guard right before the destructive op.
        if not is_droppable_test_schema(name, protected=protected):
            continue
        try:
            with engine.begin() as conn:
                # Identifier is regex-validated against the anchored allowlist
                # above, so interpolation here cannot inject (no quotes/spaces
                # possible in a matching name).
                conn.execute(text(f'DROP SCHEMA IF EXISTS "{name}" CASCADE'))
            result["dropped"].append(name)
        except Exception as exc:  # pragma: no cover - exercised via failed-drop test
            result["failed"].append((name, str(exc)))
    return result
