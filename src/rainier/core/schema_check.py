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

# Documented PRE-EXISTING benign drift the loud chokepoints must not block on
# (verified against the live legacy DB, 2026-07-03): ``capital_flow_bars``
# (+ its ``symbol`` index) is an empty (0-row) orphan of the old
# stock_id-FK -> symbol refactor. No ``migrations/*.sql`` adds the column, the
# QU100 pipeline never reads the table, and the operator's runbook (memory
# ``project_prod_checkout_staleness``) says to block only on findings BEYOND
# it. Without this allowlist ``rainier db init`` would exit non-zero on the
# live DB forever, with ``db migrate-legacy`` unable to clear it.
KNOWN_BENIGN_DRIFT: frozenset[str] = frozenset(
    {
        "missing column: capital_flow_bars.symbol",
        "missing index: capital_flow_bars.ix_capital_flow_bars_symbol",
    }
)

# NAME aliasing between the ORM's auto-names and migration-created names
# (verified live 2026-07-03): migrations/0012_reclaim_queue.sql created its
# indexes as ``ix_paper_reclaim_status`` / ``ix_paper_reclaim_symbol``, while
# the ORM's ``index=True`` auto-names expect ``ix_paper_reclaim_queue_<col>``.
# Same columns, same semantics — EITHER name satisfies the contract. This is
# deliberately NOT an allowlist entry (codex 43f3 [P2]): a DB where NEITHER
# name exists (partial/manual 0012 apply) is real drift and must be flagged.
INDEX_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "ix_paper_reclaim_queue_status": ("ix_paper_reclaim_status",),
    "ix_paper_reclaim_queue_symbol": ("ix_paper_reclaim_symbol",),
}


def check_schema_drift(engine: Engine) -> list[str]:
    """Compare the legacy ORM to ``engine``'s live ``public`` schema.

    Inspection targets ``public`` explicitly, independent of the engine's
    ``search_path`` — matching the migration runner, which pins its DDL and
    ``schema_migrations`` bookkeeping to ``public``.

    Returns a list of human-readable findings, each one of:

      * ``"missing table: <name>"``
      * ``"missing column: <table>.<column>"``
      * ``"missing index: <table>.<name>"``
      * ``"missing constraint: <table>.<name>"``

    An empty list means the live schema satisfies every ORM-declared table,
    column, and NAMED index/constraint. Index/constraint checking is by NAME
    presence only (codex 43f3 review: an index/constraint-only
    ``migrations/*.sql`` that never ran was invisible to a tables/columns
    check, so ``--baseline`` could stamp it as applied) — a live object whose
    name matches but whose definition drifted is NOT detected, and DDL that
    exists only in migration files (never mirrored in the ORM) cannot be
    checked at all. Unnamed ORM constraints (auto-named PKs/FKs) are skipped.

    Extra (DB-only) tables/columns/indexes are intentionally NOT reported: the
    ORM is the contract, and a shared instance may legitimately carry objects
    the legacy models don't declare. Findings are ordered by ORM
    table-definition order for stable output.
    """
    # Inspect ``public`` EXPLICITLY (codex 43f3 [P1]): the legacy ORM tables
    # and the migration runner's bookkeeping/DDL all live in (and are pinned
    # to) ``public``. Unqualified inspector calls would follow the engine's
    # ``search_path`` and could validate a DIFFERENT front schema — reporting
    # "clean" for a schema the runner never stamps.
    schema = "public"
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names(schema=schema))
    findings: list[str] = []
    for table_name, table in Base.metadata.tables.items():
        if table_name not in existing_tables:
            findings.append(f"missing table: {table_name}")
            continue
        live_cols = {col["name"] for col in insp.get_columns(table_name, schema=schema)}
        for column in table.columns:
            if column.name not in live_cols:
                findings.append(f"missing column: {table_name}.{column.name}")

        # Postgres exposes a UNIQUE declared as a constraint via
        # get_unique_constraints and one declared as an index via get_indexes
        # (and backs the former with a same-named index), so membership is
        # checked against the UNION of all live names to avoid classification
        # false-positives.
        live_names: set[str] = set()
        live_names.update(
            ix["name"]
            for ix in insp.get_indexes(table_name, schema=schema)
            if ix.get("name")
        )
        live_names.update(
            c["name"]
            for c in insp.get_unique_constraints(table_name, schema=schema)
            if c.get("name")
        )
        live_names.update(
            c["name"]
            for c in insp.get_check_constraints(table_name, schema=schema)
            if c.get("name")
        )
        pk = insp.get_pk_constraint(table_name, schema=schema)
        if pk and pk.get("name"):
            live_names.add(pk["name"])

        for index in sorted(table.indexes, key=lambda ix: str(ix.name)):
            if not index.name:
                continue
            name = str(index.name)
            aliases = INDEX_NAME_ALIASES.get(name, ())
            if name in live_names or any(a in live_names for a in aliases):
                continue
            findings.append(f"missing index: {table_name}.{name}")
        # table.constraints is a set — sort for stable finding order. Unnamed
        # constraints (None / SQLAlchemy's anonymous-name sentinel, which is
        # not a plain str) can't be checked by name and are skipped.
        declared_constraints = sorted(
            (
                str(c.name)
                for c in table.constraints
                if isinstance(c.name, str) and c.name
            ),
        )
        for name in declared_constraints:
            if name not in live_names:
                findings.append(f"missing constraint: {table_name}.{name}")
    return findings
