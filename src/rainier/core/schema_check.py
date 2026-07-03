"""Detect drift between the legacy ORM (``core/models.py`` Base) and a live DB.

Motivation — the 2026-06-03 QU100 P0: the shared local ``stocks`` table was
reduced to a 2-column stub (``id``, ``symbol``) by a buggy migration-test
fixture. The QU scraper's ``INSERT INTO stocks (symbol, name, sector, ...)``
then failed with ``column "name" ... does not exist`` and persisted 0 rows —
silently, because the failure Discord alert also errored.

``db init`` (``Base.metadata.create_all``) is **additive only**: it creates
MISSING tables but never ALTERs an existing table to add columns, so a stubbed
or drifted table is invisible to it. This module compares ``Base.metadata`` to
the live schema and reports missing tables/columns so drift is caught loudly
(e.g. a ``rainier db check`` preflight) instead of as a silent zero-row scrape.
"""

from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from rainier.core.models import Base

# Documented PRE-EXISTING benign drift the loud chokepoints must not block on.
# ``capital_flow_bars`` on the live legacy DB is an empty (0-row) table orphaned
# by the old stock_id-FK -> symbol refactor: no ``migrations/*.sql`` adds the
# column, the QU100 pipeline never reads the table, and the operator's runbook
# (memory ``project_prod_checkout_staleness``) says to block only on findings
# BEYOND it. Without this allowlist ``rainier db init`` would exit non-zero on
# the live DB forever, with ``db migrate-legacy`` unable to clear it.
KNOWN_BENIGN_DRIFT: frozenset[str] = frozenset(
    {
        "missing column: capital_flow_bars.symbol",
    }
)


def check_schema_drift(engine: Engine) -> list[str]:
    """Compare the legacy ORM to ``engine``'s live schema.

    Returns a list of human-readable findings, each one of:

      * ``"missing table: <name>"``
      * ``"missing column: <table>.<column>"``

    An empty list means the live schema satisfies every ORM-declared table and
    column. Extra (DB-only) tables/columns are intentionally NOT reported: the
    ORM is the contract, and a shared instance may legitimately carry tables the
    legacy models don't declare. Findings are ordered by ORM table-definition
    order (tables before their columns) for stable output.
    """
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    findings: list[str] = []
    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            findings.append(f"missing table: {table_name}")
            continue
        live_cols = {col["name"] for col in insp.get_columns(table_name)}
        for column in table.columns:
            if column.name not in live_cols:
                findings.append(f"missing column: {table_name}.{column.name}")
    return findings
