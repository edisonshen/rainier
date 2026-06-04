"""Shared in-memory ResearchInsight session for research unit tests.

`emit_insight(db_session=...)` only needs query().filter().first(), add(),
flush(). This minimal fake satisfies that so check-class tests stay
network/Postgres-free and deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rainier.core.models import ResearchInsight


class _Query:
    def __init__(self, rows):
        self._rows = rows
        self._filters: list = []

    def filter(self, *args):
        self._filters.extend(args)
        return self

    def first(self):
        candidates = list(self._rows)
        for f in self._filters:
            try:
                col_name = f.left.key
                expected = f.right.value
            except AttributeError:
                continue
            candidates = [
                r for r in candidates if getattr(r, col_name, None) == expected
            ]
        return candidates[0] if candidates else None

    def all(self):
        return list(self._rows)


class MemSession:
    def __init__(self):
        self.rows: list[ResearchInsight] = []
        self._next_id = 1
        self.committed = False

    def query(self, model):  # noqa: ARG002
        return _Query(self.rows)

    def add(self, row: ResearchInsight) -> None:
        if row.id is None:
            row.id = self._next_id
            self._next_id += 1
        if row.created_at is None:
            row.created_at = datetime.now(timezone.utc)
        if row.updated_at is None:
            row.updated_at = datetime.now(timezone.utc)
        if row.recurrence_count is None:
            row.recurrence_count = 1
        if row.status is None:
            row.status = "pending"
        self.rows.append(row)

    def flush(self) -> None:
        return None

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False
