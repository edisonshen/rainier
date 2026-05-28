"""SQLAlchemy Engine factory for the canonical Postgres store.

This is NOT a singleton — every `get_engine()` call returns a fresh Engine
bound to `DATABASE_URL`. Callers manage lifecycle (and typically dispose()
in tests). The legacy `rainier.core.database.get_engine` keeps its own
singleton pattern; the two engines coexist.

Pool config rationale (task plan §5):

    pool_size=2, max_overflow=2

rainier is single-process, single-threaded for batch work. Two connections
covers the cron + an ad-hoc query simultaneously. Neon free tier allows
~20 connections; small usage here leaves headroom for fengshen-site
(via Hyperdrive) + maunakea.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_engine() -> Engine:
    """Return a fresh SQLAlchemy Engine bound to ``DATABASE_URL``.

    Raises ``RuntimeError`` if the env var is missing — we never silently
    fall back to a default URL because that would make production
    misconfiguration silent.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set — see "
            "docs/DESIGN-rainier-postgres-canonical-store.md "
            "for the local-Postgres / Neon setup."
        )
    return create_engine(
        url,
        pool_pre_ping=True,  # cheap; detects dead connections after suspend/resume
        pool_size=2,
        max_overflow=2,
    )
