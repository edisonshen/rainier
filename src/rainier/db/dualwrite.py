"""Safe-rollout glue for Phase 2 dual-write (task plan §4).

The dual-write posture is *additive and non-fatal*: parquet is the load-bearing
output; the Postgres write is a mirror. So if ``DATABASE_URL`` is unset (local
dev, CI without a DB), the writer must log one warning, skip the PG write, and
let the parquet pipeline finish with exit 0 — never crash.

This module centralizes that decision so all three call sites share identical
behavior:

    eng = pg_engine_or_skip("backfill_thematic_universe")
    if eng is None:
        return            # parquet already written; PG skipped + warned
    try:
        ...market_upsert(eng, ...)
    finally:
        eng.dispose()     # fleet/rainier owns the lifecycle of what it opens
"""

from __future__ import annotations

import os
import sys

from sqlalchemy.engine import Engine


def pg_engine_or_skip(writer_name: str) -> Engine | None:
    """Return a fresh Engine, or None (with a warning) when PG is unconfigured.

    None means "DATABASE_URL is unset — skip the PG mirror". The caller must
    have already done its parquet write so the pipeline stays whole.
    """
    if not os.environ.get("DATABASE_URL"):
        print(
            f"warning: DATABASE_URL not set — {writer_name} skipping the "
            f"Postgres dual-write (parquet output unaffected). Set DATABASE_URL "
            f"to mirror into market.* (see "
            f"docs/DESIGN-rainier-postgres-canonical-store.md).",
            file=sys.stderr,
        )
        return None

    # Import here so modules that never dual-write don't pay the engine import.
    from rainier.db.engine import get_engine

    return get_engine()
