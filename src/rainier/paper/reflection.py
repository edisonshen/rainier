"""R-A — per-trade post-exit LLM reflections (design Appendix B, task plan
qu100-ra-reflections-587f).

WHAT THIS IS
------------
When a paper trade closes, the LLM writes a 2–3 sentence post-mortem — the
pattern that triggered entry, what price actually did versus the plan, and
whether the entry reasoning was validated (the FinAgent two-part framing).
The text is stored on the trade row (`paper_trade.reflection`); the last K=10
reflections are injected into the daily thesis prompt's calibration section so
written lessons — not just numbers — reach the model.

OUTCOME EMBARGO
---------------
Reflections only ever exist for RESOLVED trades. The schema enforces it:
`CHECK (reflection IS NULL OR exit_reason IS NOT NULL)` (migration 0009), and
the writer is a NULL-only UPDATE, so a reflection is written at most once and
never rewritten.

GENERATION CONTRACT (daily job, AFTER step (v) — scheduler/service.py)
----------------------------------------------------------------------
* Selection: all CLOSED trades with `reflection IS NULL` and
  `exit_date <= as_of`, bounded to the trailing 30 days — a failed generation
  is naturally retried on the next run, and the bound keeps a cold-started
  backlog from burning unbounded LLM spend.
* One LLM call per trade. A failure on one trade logs, leaves the reflection
  NULL, and moves on — it never blocks the pipeline or the batch.
* Chart attachment is FEATURE-DETECTED: when the `paper_trade.chart_id`
  column exists (chart-archive PR landed) and is set, the archived close-side
  PNG rides along as image input; otherwise the call is text-only. This module
  must work on a pre-chart-archive schema with zero crashes.

All reads/writes go through the LEGACY local-TimescaleDB engine
(`core.database.get_session`) — never Neon (memory:
project_two_database_url_engines).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Protocol

from sqlalchemy import text as sql_text

from rainier.core import database

log = logging.getLogger(__name__)


def _get_session():
    """Late-bound `core.database.get_session`.

    Resolved at CALL time through the module attribute, never captured at
    import time: this module is imported lazily from `generate_thesis`, so its
    first import can happen inside a test's
    `patch("rainier.core.database.get_session")` window — a `from ... import
    get_session` would freeze that ephemeral mock into this namespace forever
    (observed as cross-suite test pollution).
    """
    return database.get_session()

# Rolling number of reflections injected into the thesis prompt (task plan K).
REFLECTION_PROMPT_K = 10
# Generation selection window: closed trades exited within the trailing N days.
REFLECTION_LOOKBACK_DAYS = 30
# Hard cap on the stored reflection text (the prompt asks for 2–3 sentences;
# this is the defensive bound, not the target length).
REFLECTION_MAX_CHARS = 1000
# Per-reflection cap when rendering into the prompt section (keeps the whole
# K=10 block bounded at a few KB).
REFLECTION_RENDER_CHARS = 280
# Per-run cap on LLM calls — bounds a cold-start backlog (30-day window x N
# trades/day) to a fixed nightly spend; the overflow drains on later runs
# (selection is oldest-first + reflection-IS-NULL, so nothing is lost).
REFLECTION_MAX_PER_RUN = 25
# Original-thesis excerpt bound in the per-trade prompt.
REFLECTION_THESIS_EXCERPT_CHARS = 600
# Completion bounds: 2-3 sentences fit comfortably in 300 tokens; the timeout
# keeps one hung provider call from stalling the nightly pipeline (which runs
# the batch serially before step (vi) calibration).
REFLECTION_LLM_MAX_TOKENS = 300
REFLECTION_LLM_TIMEOUT_S = 120

REFLECTION_SYSTEM_PROMPT = """You are reviewing ONE closed paper trade after the fact.
Write a 2-3 sentence post-mortem in plain text (no JSON, no preamble, no headers):
1. What price actually did after entry versus the planned entry/stop/target — name the
   pattern that triggered the entry.
2. Whether the entry reasoning was validated by the outcome (did the thesis hold?), and
   the one lesson worth carrying forward.
Be concrete and terse. No hedging filler."""


class LLMFn(Protocol):
    def __call__(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes | None,
    ) -> str: ...


def _default_llm_fn(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_bytes: bytes | None,
) -> str:
    """Single LiteLLM completion returning the text content.

    Local twin of llm_thesis.service._call_llm (kept separate so paper/ does
    not import llm_thesis internals); lazy import so tests never load LiteLLM.
    """
    import base64

    import litellm

    content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    resp = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ],
        temperature=0.2,
        max_tokens=REFLECTION_LLM_MAX_TOKENS,
        timeout=REFLECTION_LLM_TIMEOUT_S,
    )
    return resp["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Feature detection + chart fetch (R-D integration point)
# ---------------------------------------------------------------------------


def _chart_id_column_exists(session) -> bool:
    """True iff `paper_trade.chart_id` exists in the connected database.

    Resolves `paper_trade` through the connection's search_path (pg_attribute
    + regclass), so a test schema and production `public` both work. Any
    failure (non-Postgres dialect, missing table) degrades to False — the
    text-only path, never a crash.
    """
    try:
        return bool(
            session.execute(
                sql_text(
                    "SELECT 1 FROM pg_attribute "
                    "WHERE attrelid = 'paper_trade'::regclass "
                    "AND attname = 'chart_id' AND NOT attisdropped"
                )
            ).scalar()
        )
    except Exception:
        return False


def _load_chart_bytes(session, trade_id: int, has_chart_col: bool) -> bytes | None:
    """Archived close-side chart PNG for the trade, when available.

    Best-effort: any failure returns None (text-only reflection). Raw SQL on
    purpose — `chart_id` is not on the ORM model until the chart-archive PR
    merges, and `image_bytes` is read by id only.
    """
    if not has_chart_col:
        return None
    try:
        chart_id = session.execute(
            sql_text("SELECT chart_id FROM paper_trade WHERE id = :id"),
            {"id": trade_id},
        ).scalar()
        if chart_id is None:
            return None
        raw = session.execute(
            sql_text("SELECT image_bytes FROM chart_images WHERE id = :cid"),
            {"cid": chart_id},
        ).scalar()
        return bytes(raw) if raw is not None else None
    except Exception:
        log.warning(
            "reflection_chart_fetch_failed trade_id=%s — falling back to text-only",
            trade_id,
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _fmt(v: Any, spec: str = "") -> str:
    return format(v, spec) if v is not None else "n/a"


def _build_reflection_prompt(t: dict[str, Any]) -> str:
    """Per-trade user prompt: the booked facts + the original thesis case."""
    thesis = (t.get("thesis_reasoning") or "").strip()
    if len(thesis) > REFLECTION_THESIS_EXCERPT_CHARS:
        thesis = thesis[:REFLECTION_THESIS_EXCERPT_CHARS] + "…"
    return_pct = t.get("return_pct")
    lines = [
        f"Symbol: {t['symbol']}",
        f"Pattern that triggered entry: {t.get('pattern_type') or 'unknown'}",
        f"Verdict / confidence at entry: {t.get('verdict') or 'n/a'} / "
        f"{t.get('llm_confidence') if t.get('llm_confidence') is not None else 'n/a'}",
        f"Plan: entry {_fmt(t.get('planned_entry_price'), '.2f')}, "
        f"stop {_fmt(t.get('stop_loss'), '.2f')}, "
        f"target {_fmt(t.get('target_price'), '.2f')}",
        f"Actual: filled {t.get('entry_date')} @ {_fmt(t.get('entry_price'), '.2f')}, "
        f"exited {t.get('exit_date')} @ {_fmt(t.get('exit_price'), '.2f')} "
        f"via {t.get('exit_reason')}",
        f"Outcome: return {_fmt(return_pct, '+.2%')}, "
        f"P&L ${_fmt(t.get('pnl'), ',.2f')}",
    ]
    if thesis:
        lines.append(f"Original entry thesis: {thesis}")
    if t.get("has_chart"):
        lines.append(
            "A chart of the trade window (entry/stop/target/exit marked) is attached."
        )
    lines.append("Write the 2-3 sentence post-mortem now.")
    return "\n".join(lines)


def select_reflection_candidates(as_of: date) -> list[dict[str, Any]]:
    """Closed, reflection-less trades with exit_date <= as_of, trailing 30 days.

    Materialized scalar rows (never live ORM entities) — session-scope safe.
    Oldest exit first so a backlog drains deterministically.
    """
    cutoff = as_of - timedelta(days=REFLECTION_LOOKBACK_DAYS)
    with _get_session() as session:
        rows = session.execute(
            sql_text(
                "SELECT pt.id, pt.symbol, pt.pattern_type, pt.verdict, "
                "       pt.llm_confidence, pt.planned_entry_price, pt.stop_loss, "
                "       pt.target_price, pt.entry_date, pt.entry_price, "
                "       pt.exit_date, pt.exit_price, pt.exit_reason, "
                "       pt.return_pct, pt.pnl, ar.reasoning AS thesis_reasoning "
                "FROM paper_trade pt "
                "LEFT JOIN analysis_results ar ON ar.id = pt.thesis_id "
                "WHERE pt.status = 'closed' "
                "  AND pt.reflection IS NULL "
                "  AND pt.exit_reason IS NOT NULL "
                "  AND pt.exit_date IS NOT NULL "
                "  AND pt.exit_date <= :as_of "
                "  AND pt.exit_date >= :cutoff "
                "ORDER BY pt.exit_date ASC, pt.id ASC"
            ),
            {"as_of": as_of, "cutoff": cutoff},
        ).mappings().all()
    return [dict(r) for r in rows]


def generate_reflections(
    as_of: date,
    *,
    model: str,
    llm_fn: LLMFn | None = None,
    max_per_run: int = REFLECTION_MAX_PER_RUN,
) -> dict[str, int]:
    """Write a post-exit reflection for every eligible closed trade.

    Idempotent: the selection (`reflection IS NULL`) plus the NULL-only UPDATE
    mean a re-run never re-calls the LLM for an already-reflected trade and
    never rewrites text. A per-trade failure logs and continues — the trade is
    naturally retried on the next run.

    `max_per_run` caps LLM spend per invocation: a cold-start backlog is
    processed oldest-first, `max_per_run` per night, until drained (deferred
    trades stay reflection-IS-NULL so they are re-selected next run).

    Returns {"candidates", "written", "failed", "deferred"} counts.
    """
    fn = llm_fn or _default_llm_fn
    candidates = select_reflection_candidates(as_of)
    deferred = max(0, len(candidates) - max_per_run)
    if deferred:
        log.info(
            "reflections_capped as_of=%s candidates=%s max_per_run=%s deferred=%s",
            as_of.isoformat(),
            len(candidates),
            max_per_run,
            deferred,
        )
        candidates = candidates[:max_per_run]
    written = 0
    failed = 0

    # One feature probe per run (schema doesn't change mid-batch).
    with _get_session() as session:
        has_chart_col = _chart_id_column_exists(session)

    for t in candidates:
        trade_id = int(t["id"])
        try:
            with _get_session() as session:
                image_bytes = _load_chart_bytes(session, trade_id, has_chart_col)
            t["has_chart"] = image_bytes is not None
            reply = fn(
                model=model,
                system_prompt=REFLECTION_SYSTEM_PROMPT,
                user_prompt=_build_reflection_prompt(t),
                image_bytes=image_bytes,
            )
            reflection = (reply or "").strip()[:REFLECTION_MAX_CHARS]
            if not reflection:
                raise ValueError("empty reflection from LLM")
            # NULL-only update: never rewrites, race-safe, and the schema CHECK
            # (exit_reason IS NOT NULL) backstops the embargo.
            # updated_at is set explicitly: the ORM-level onupdate does not
            # fire for raw-SQL UPDATEs, and change-detection consumers read it.
            with _get_session() as session:
                result = session.execute(
                    sql_text(
                        "UPDATE paper_trade "
                        "SET reflection = :r, updated_at = NOW() "
                        "WHERE id = :id AND reflection IS NULL "
                        "  AND exit_reason IS NOT NULL"
                    ),
                    {"r": reflection, "id": trade_id},
                )
            written += int(result.rowcount or 0)
        except Exception:
            failed += 1
            log.warning(
                "reflection_generation_failed trade_id=%s symbol=%s — stays NULL, "
                "retried next run",
                trade_id,
                t.get("symbol"),
                exc_info=True,
            )

    log.info(
        "reflections_generated as_of=%s candidates=%s written=%s failed=%s deferred=%s",
        as_of.isoformat(),
        len(candidates),
        written,
        failed,
        deferred,
    )
    return {
        "candidates": len(candidates),
        "written": written,
        "failed": failed,
        "deferred": deferred,
    }


# ---------------------------------------------------------------------------
# Prompt injection (consumed by llm_thesis.service.generate_thesis)
# ---------------------------------------------------------------------------


def load_recent_reflections(
    as_of: date, *, k: int = REFLECTION_PROMPT_K
) -> list[dict[str, Any]]:
    """Last `k` reflections from trades exited STRICTLY BEFORE `as_of`.

    Strictly-before mirrors the calibration block's no-hindsight contract: a
    scan on day D must never see a reflection for a trade that exited on day D
    (it is written end-of-day D, after the scan).
    """
    with _get_session() as session:
        rows = session.execute(
            sql_text(
                "SELECT symbol, exit_date, exit_reason, return_pct, reflection "
                "FROM paper_trade "
                "WHERE reflection IS NOT NULL "
                "  AND exit_date IS NOT NULL AND exit_date < :as_of "
                "ORDER BY exit_date DESC, id DESC "
                "LIMIT :k"
            ),
            {"as_of": as_of, "k": k},
        ).all()
    return [
        {
            "symbol": symbol,
            "exit_date": exit_date.isoformat() if exit_date else None,
            "exit_reason": exit_reason,
            "return_pct": float(return_pct) if return_pct is not None else None,
            "reflection": reflection,
        }
        for symbol, exit_date, exit_reason, return_pct, reflection in rows
    ]


def render_reflections_section(rows: list[dict[str, Any]]) -> str:
    """Bounded, labeled prompt block. "" when there is nothing to show."""
    if not rows:
        return ""
    # "reflected closed trades", not "closed trades": trades whose reflection
    # generation permanently failed (or aged out of the lookback window) are
    # absent, so the sample the model sees is reflected-only — say so.
    lines = [
        f"--- Recent trade reflections (last {len(rows)} reflected closed "
        "trades, post-exit post-mortems) ---"
    ]
    for r in rows:
        body = (r.get("reflection") or "").replace("\n", " ").strip()
        if len(body) > REFLECTION_RENDER_CHARS:
            body = body[: REFLECTION_RENDER_CHARS - 1] + "…"
        ret = r.get("return_pct")
        ret_s = f"{ret:+.2%}" if ret is not None else "n/a"
        lines.append(
            f"{r['symbol']} {r.get('exit_date') or '?'} "
            f"{r.get('exit_reason') or '?'} {ret_s}: {body}"
        )
    return "\n".join(lines)


def reflection_prompt_section(
    scan_date: date, *, k: int = REFLECTION_PROMPT_K
) -> str:
    """Convenience: load + render in one call (used by generate_thesis)."""
    return render_reflections_section(load_recent_reflections(scan_date, k=k))
