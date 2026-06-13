"""Persistence helpers for ``ChartImage`` rows (PR5).

Why a dedicated module? PR1's ``ChartImage`` was filesystem-backed and only
referenced by analytics dashboards. PR5 needs the chart bytes that went to
the LLM to be readable later by the Discord renderer (multipart attachment)
and the Streamlit dashboard (deep-link). Persisting bytes inline (BYTEA on
Postgres) plus the sha256 digest gives us a deterministic idempotency key.

  ┌──────────────────┐    persist_chart_image
  │ render_chart_png │  ─►  ChartImage(symbol, scan_date, image_bytes,
  │  (kaleido / PNG) │      sha256, width, height)
  └──────────────────┘  ─►  partial-unique on (symbol, scan_date, sha256)

The ``persist_chart_image`` helper is idempotent: a re-run with identical
PNG bytes returns the existing row's id instead of inserting a duplicate.
This keeps Tier-2 cache misses from spamming chart_images with the same
bytes when the user manually re-fires a scan.
"""

from __future__ import annotations

import logging
from datetime import date as date_cls
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rainier.core.database import get_session
from rainier.core.models import ChartImage

log = logging.getLogger(__name__)


def persist_chart_image(
    *,
    symbol: str,
    scan_date: date_cls,
    image_bytes: bytes,
    sha256: str,
    width: int = 1280,
    height: int = 800,
    timeframe_days: int = 120,
) -> int | None:
    """Insert (or reuse) a ChartImage row for this (symbol, scan_date, sha256).

    R-D (migration 0010): thesis rows carry ``source='thesis'`` and KEEP
    ``as_of_date`` NULL forever — they live outside the archive's logical
    identity ``(symbol, as_of_date, source)`` partial unique, so the
    up-to-4-renders/day thesis pattern never collides (NULLs are distinct).
    ``timeframe_days`` follows the caller's actual render window (the 120
    default is the thesis-path constant; the archive path passes its own).

    Returns the row id, or ``None`` on persistence failure. The SELECT before
    INSERT is a deliberate two-step on the partial unique index: Postgres'
    ``ON CONFLICT`` on a partial index requires matching the partial WHERE
    in the conflict target, and SQLAlchemy's ``index_where`` bookkeeping
    differs across versions. The select probe is a single index seek and
    keeps the persist path version-portable.
    """
    if not image_bytes or not sha256:
        log.warning(
            "persist_chart_image_invalid_args symbol=%s sha=%s len=%s",
            symbol,
            sha256[:8] if sha256 else None,
            len(image_bytes) if image_bytes else 0,
        )
        return None

    with get_session() as session:
        existing = session.execute(
            select(ChartImage.id).where(
                ChartImage.symbol == symbol,
                ChartImage.scan_date == scan_date,
                ChartImage.sha256 == sha256,
            )
        ).first()
        if existing is not None:
            return int(existing[0])

        try:
            stmt = (
                pg_insert(ChartImage)
                .values(
                    symbol=symbol,
                    scan_date=scan_date,
                    image_bytes=image_bytes,
                    sha256=sha256,
                    file_size_bytes=len(image_bytes),
                    width=int(width),
                    height=int(height),
                    timeframe_days=int(timeframe_days),
                    # R-D: thesis source, as_of_date stays NULL (outside the
                    # archive logical-identity contract — see docstring).
                    source="thesis",
                )
                .returning(ChartImage.id)
            )
            row = session.execute(stmt).first()
            if row is None:
                # ON CONFLICT handled at the DB level — reread by the unique key.
                refetch = session.execute(
                    select(ChartImage.id).where(
                        ChartImage.symbol == symbol,
                        ChartImage.scan_date == scan_date,
                        ChartImage.sha256 == sha256,
                    )
                ).first()
                return int(refetch[0]) if refetch else None
            return int(row[0])
        except Exception:
            # PR5 review iter-1: race-loser path. The probe missed the row
            # (no concurrent insert visible yet), our INSERT lost the unique
            # constraint, so the row DOES exist now under the other call's
            # transaction. Re-probe in a fresh session so we surface the id
            # instead of silently returning None and missing the chart
            # attachment downstream.
            log.warning(
                "persist_chart_image_conflict_refetch symbol=%s scan_date=%s sha=%s",
                symbol,
                scan_date,
                sha256[:8] if sha256 else None,
            )

    # Fresh session to refetch — the original session was rolled back by the
    # context-manager exit when the INSERT raised.
    with get_session() as session:
        try:
            row = session.execute(
                select(ChartImage.id).where(
                    ChartImage.symbol == symbol,
                    ChartImage.scan_date == scan_date,
                    ChartImage.sha256 == sha256,
                )
            ).first()
            return int(row[0]) if row else None
        except Exception:
            log.exception(
                "persist_chart_image_refetch_failed symbol=%s scan_date=%s sha=%s",
                symbol,
                scan_date,
                sha256[:8] if sha256 else None,
            )
            return None


def attach_chart_id_to_thesis(*, thesis_id: int, chart_id: int) -> bool:
    """Append ``chart_id`` to LLMAnalysisRecord.chart_image_ids if not present.

    Returns True iff the array was mutated. Idempotent — a chart id that's
    already in the array is left alone (no duplicate ids).
    """
    from rainier.core.models import LLMAnalysisRecord

    with get_session() as session:
        record = session.get(LLMAnalysisRecord, int(thesis_id))
        if record is None:
            log.warning("attach_chart_thesis_missing thesis_id=%s", thesis_id)
            return False
        existing: list[Any] = list(record.chart_image_ids or [])
        if int(chart_id) in [int(x) for x in existing]:
            return False
        existing.append(int(chart_id))
        record.chart_image_ids = existing
    return True
