"""Safe-rollout glue for Phase 2 dual-write (task plan §4).

The dual-write posture is *additive and non-fatal*: parquet is the load-bearing
output; the Postgres write is a mirror. Two ways PG can be unusable, both must
leave the parquet pipeline whole (exit 0):

  1. DATABASE_URL unset (local dev, CI without a DB) -> skip + warn.
  2. DATABASE_URL set but PG is unreachable / unmigrated / missing migration
     0002 -> the connect or upsert raises a SQLAlchemyError. We catch it, warn,
     and continue. A flaky mirror DB must NOT abort `run-daily` (which would
     otherwise crash after writing labels parquet but before rendering).

``mirror_guard`` centralizes BOTH so all call sites share identical behavior:

    with mirror_guard("backfill_thematic_universe") as eng:
        if eng is None:
            ...            # parquet already written; PG skipped/failed + warned
        else:
            market_upsert(eng, ...)
    # engine disposed + SQLAlchemyError swallowed on the way out (rainier owns
    # the lifecycle of what it opens; a mirror failure is never fatal).
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator

from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


def pg_engine_or_skip(writer_name: str) -> Engine | None:
    """Return a fresh Engine, or None (with a warning) when PG is unconfigured.

    None means "DATABASE_URL is unset — skip the PG mirror". The caller must
    have already done its parquet write so the pipeline stays whole.

    NOTE: this only handles the *unset* case. A set-but-broken DATABASE_URL is
    handled by ``mirror_guard``, which catches the SQLAlchemyError the upsert
    raises. Prefer ``mirror_guard`` at call sites so broken-PG is non-fatal.
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


@contextlib.contextmanager
def mirror_guard(writer_name: str) -> Iterator[Engine | None]:
    """Context manager for an additive, non-fatal PG mirror write.

    Yields a fresh Engine (or None when DATABASE_URL is unset). The caller runs
    its ``market_upsert`` calls in the ``with`` body. On exit the engine is
    disposed, and ANY ``SQLAlchemyError`` raised inside the body (PG down,
    schema unmigrated, missing migration 0002, etc.) is caught, warned to
    stderr, and swallowed — so a broken mirror DB never aborts the load-bearing
    parquet pipeline. Non-SQLAlchemy errors (programmer bugs) propagate.
    """
    eng: Engine | None = None
    try:
        eng = pg_engine_or_skip(writer_name)
        yield eng
    except SQLAlchemyError as exc:
        print(
            f"warning: {writer_name} Postgres dual-write failed ({type(exc).__name__}: "
            f"{exc}); parquet output is unaffected. Check the mirror DB is "
            f"reachable and migrated to head "
            f"(see docs/DESIGN-rainier-postgres-canonical-store.md).",
            file=sys.stderr,
        )
    finally:
        if eng is not None:
            eng.dispose()
