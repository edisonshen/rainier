"""Postgres canonical-store package — Phase 1 of the architecture pivot.

This package is intentionally separate from `rainier.core.database`:

* `core/database.py` — legacy ORM + Settings-driven singleton, TimescaleDB-aware.
  Keeps existing call sites (LLM thesis persistence, monitors, dashboards)
  working unchanged.

* `db/` — the new canonical-store layer. Reads DATABASE_URL directly from env,
  uses SQLAlchemy Core (not ORM), and ships the Alembic-managed `market.*`
  schema that fengshen-site + maunakea will read from.

Phase 1 ships read-only (DDL + connection). Phase 2 adds write paths.

See docs/DESIGN-rainier-postgres-canonical-store.md and
docs/TASK-PLAN-rainier-pg-schema-bootst-966f.md.
"""

from rainier.db.engine import get_engine
from rainier.db.schema import metadata
from rainier.db.upsert import market_upsert

__all__ = ["get_engine", "market_upsert", "metadata"]
