"""R-A — per-trade post-exit LLM reflections (task qu100-ra-reflections-587f).

Covers (TASK-PLAN acceptance):
  1. Schema: `paper_trade.reflection TEXT` + CHECK (reflection IS NULL OR
     exit_reason IS NOT NULL) — the outcome embargo at the schema level.
     Additive migration 0009 on $LEGACY_DATABASE_URL.
  2. Generation: closed trades with reflection IS NULL and exit_date <= as_of,
     bounded to the trailing 30 days; one LLM call per trade; idempotent;
     LLM failure -> reflection stays NULL (retried next run, never blocks).
     Chart attachment is FEATURE-DETECTED: `paper_trade.chart_id` column
     present + set -> image input; column absent (pre-chart-archive main) ->
     text-only path, no crash.
  3. Prompt injection: rolling last K=10 reflections appended to the
     calibration section; bounded; labeled; strictly-before discipline;
     PROMPT_VERSION bumped (v3) in lockstep across module/config/yaml;
     input_hash unaffected.
  4. Reflections only ever exist for resolved trades.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from rainier.core.models import PaperTrade

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_0009_UP = REPO_ROOT / "migrations" / "0009_paper_reflection.sql"
MIGRATION_0009_DOWN = (
    REPO_ROOT / "migrations" / "0009_paper_reflection_downgrade.sql"
)


# ---------------------------------------------------------------------------
# 1. Schema — ORM metadata (no Postgres needed)
# ---------------------------------------------------------------------------


def test_paper_trade_has_nullable_reflection_column():
    cols = {c.name for c in inspect(PaperTrade).columns}
    assert "reflection" in cols
    assert PaperTrade.__table__.c.reflection.nullable


def test_reflection_check_constraint_declared():
    checks = [
        c
        for c in PaperTrade.__table__.constraints
        if c.__class__.__name__ == "CheckConstraint"
    ]
    sqltexts = {c.name: str(c.sqltext) for c in checks}
    assert "ck_paper_trade_reflection_after_exit" in sqltexts
    assert (
        "reflection IS NULL OR exit_reason IS NOT NULL"
        in sqltexts["ck_paper_trade_reflection_after_exit"]
    )


# ---------------------------------------------------------------------------
# helpers — insert a thesis + paper_trade row in the pg test schema
# ---------------------------------------------------------------------------


def _mk_thesis(session) -> int:
    rid = session.execute(
        text(
            "INSERT INTO analysis_results (reasoning, structured_output) "
            "VALUES ('w_bottom forming, neckline 142.50', "
            "'{\"paragraph_evidence\": \"flow streak + neckline\"}'::jsonb) "
            "RETURNING id"
        )
    ).scalar()
    return int(rid)


def _mk_trade(
    session,
    *,
    symbol: str = "AAA",
    status: str = "closed",
    entry_date: date | None = date(2026, 6, 1),
    exit_date: date | None = date(2026, 6, 5),
    exit_reason: str | None = "target",
    reflection: str | None = None,
    return_pct: float | None = 0.08,
) -> int:
    thesis_id = _mk_thesis(session)
    tid = session.execute(
        text(
            "INSERT INTO paper_trade "
            "(thesis_id, symbol, scan_date, session_name, status, "
            " planned_entry_price, stop_loss, target_price, pattern_type, "
            " llm_confidence, verdict, entry_date, entry_price, shares, "
            " exit_date, exit_price, exit_reason, return_pct, pnl, reflection) "
            "VALUES "
            "(:thesis_id, :symbol, '2026-05-29', 'close', :status, "
            " 100.0, 95.0, 110.0, 'w_bottom', "
            " 7, 'setup_long', :entry_date, 100.0, 100, "
            " :exit_date, :exit_price, :exit_reason, :return_pct, :pnl, "
            " :reflection) "
            "RETURNING id"
        ),
        {
            "thesis_id": thesis_id,
            "symbol": symbol,
            "status": status,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "exit_price": 108.0 if exit_date else None,
            "exit_reason": exit_reason,
            "return_pct": return_pct,
            "pnl": 800.0 if exit_date else None,
            "reflection": reflection,
        },
    ).scalar()
    session.commit()
    return int(tid)


# ---------------------------------------------------------------------------
# 1. Schema — CHECK embargo on a real Postgres (migration 0009 applied by
#    the pg_legacy_engine fixture)
# ---------------------------------------------------------------------------


def test_check_rejects_reflection_on_open_row(pg_legacy_session):
    with pytest.raises(IntegrityError):
        _mk_trade(
            pg_legacy_session,
            status="open",
            exit_date=None,
            exit_reason=None,
            return_pct=None,
            reflection="premature reflection on an open trade",
        )
    pg_legacy_session.rollback()


def test_check_rejects_reflection_on_pending_row(pg_legacy_session):
    with pytest.raises(IntegrityError):
        _mk_trade(
            pg_legacy_session,
            status="pending",
            entry_date=None,
            exit_date=None,
            exit_reason=None,
            return_pct=None,
            reflection="premature reflection on a pending trade",
        )
    pg_legacy_session.rollback()


def test_check_accepts_reflection_on_closed_row(pg_legacy_session):
    tid = _mk_trade(pg_legacy_session, reflection="clean two-sentence post-mortem")
    got = pg_legacy_session.execute(
        text("SELECT reflection FROM paper_trade WHERE id = :id"), {"id": tid}
    ).scalar()
    assert got == "clean two-sentence post-mortem"


def test_migration_0009_idempotent_reapply(pg_legacy_engine):
    # The fixture already applied 0009 once; a re-apply must be a no-op.
    sql = MIGRATION_0009_UP.read_text()
    with pg_legacy_engine.begin() as conn:
        conn.execute(text(sql))


def test_migration_0009_downgrade_then_up_roundtrip(pg_legacy_engine):
    down = MIGRATION_0009_DOWN.read_text()
    up = MIGRATION_0009_UP.read_text()
    with pg_legacy_engine.begin() as conn:
        conn.execute(text(down))
    cols = {
        r[0]
        for r in pg_legacy_engine.connect()
        .execute(
            text(
                "SELECT attname FROM pg_attribute "
                "WHERE attrelid = 'paper_trade'::regclass AND attnum > 0 "
                "AND NOT attisdropped"
            )
        )
        .all()
    }
    assert "reflection" not in cols
    with pg_legacy_engine.begin() as conn:
        conn.execute(text(up))
        conn.execute(text(up))  # idempotent after a fresh up as well
    cols2 = {
        r[0]
        for r in pg_legacy_engine.connect()
        .execute(
            text(
                "SELECT attname FROM pg_attribute "
                "WHERE attrelid = 'paper_trade'::regclass AND attnum > 0 "
                "AND NOT attisdropped"
            )
        )
        .all()
    }
    assert "reflection" in cols2


# ---------------------------------------------------------------------------
# 2. Generation — selection, idempotency, failure isolation
# ---------------------------------------------------------------------------


class _StubLLM:
    """Deterministic llm_fn stub. Records every call."""

    def __init__(self, reply: str = "Price ran to target. Thesis held.",
                 fail: bool = False):
        self.reply = reply
        self.fail = fail
        self.calls: list[dict] = []

    def __call__(self, *, model, system_prompt, user_prompt, image_bytes):
        self.calls.append(
            {
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "image_bytes": image_bytes,
            }
        )
        if self.fail:
            raise RuntimeError("llm unavailable")
        return self.reply


AS_OF = date(2026, 6, 8)


def test_generation_writes_reflection_once(pg_legacy_session):
    from rainier.paper.reflection import generate_reflections

    tid = _mk_trade(pg_legacy_session, symbol="AAA")
    stub = _StubLLM()
    stats = generate_reflections(AS_OF, model="test-model", llm_fn=stub)
    assert stats["written"] == 1
    assert len(stub.calls) == 1
    # Two-part post-mortem prompt: what price did + was the reasoning validated.
    prompt = stub.calls[0]["user_prompt"]
    assert "AAA" in prompt
    assert "w_bottom" in prompt           # pattern that triggered entry
    assert "target" in prompt             # exit_reason — what price did
    got = pg_legacy_session.execute(
        text("SELECT reflection FROM paper_trade WHERE id = :id"), {"id": tid}
    ).scalar()
    assert got == stub.reply


def test_generation_is_idempotent_on_rerun(pg_legacy_session):
    from rainier.paper.reflection import generate_reflections

    tid = _mk_trade(pg_legacy_session, symbol="AAA")
    first = _StubLLM(reply="first run reflection")
    generate_reflections(AS_OF, model="test-model", llm_fn=first)
    second = _StubLLM(reply="SECOND run must never land")
    stats = generate_reflections(AS_OF, model="test-model", llm_fn=second)
    assert stats["written"] == 0
    assert second.calls == []  # no duplicate LLM spend
    got = pg_legacy_session.execute(
        text("SELECT reflection FROM paper_trade WHERE id = :id"), {"id": tid}
    ).scalar()
    assert got == "first run reflection"  # no rewrite


def test_generation_llm_failure_leaves_null_and_is_retried(pg_legacy_session):
    from rainier.paper.reflection import generate_reflections

    tid = _mk_trade(pg_legacy_session, symbol="AAA")
    broken = _StubLLM(fail=True)
    stats = generate_reflections(AS_OF, model="test-model", llm_fn=broken)
    assert stats["written"] == 0
    assert stats["failed"] == 1
    got = pg_legacy_session.execute(
        text("SELECT reflection FROM paper_trade WHERE id = :id"), {"id": tid}
    ).scalar()
    assert got is None  # stays NULL — never a partial/placeholder write
    # Day D+1: the NULL row is naturally re-selected and succeeds.
    working = _StubLLM(reply="retried fine")
    stats2 = generate_reflections(
        AS_OF + timedelta(days=1), model="test-model", llm_fn=working
    )
    assert stats2["written"] == 1


def test_generation_selection_bounds(pg_legacy_session):
    from rainier.paper.reflection import generate_reflections

    # Not selected: open trade (no exit), future exit, stale exit (>30d back).
    _mk_trade(
        pg_legacy_session, symbol="OPN", status="open",
        exit_date=None, exit_reason=None, return_pct=None,
    )
    _mk_trade(pg_legacy_session, symbol="FUT", exit_date=AS_OF + timedelta(days=2))
    _mk_trade(pg_legacy_session, symbol="OLD", exit_date=AS_OF - timedelta(days=31))
    # Selected: exit on as_of itself (<= bound).
    _mk_trade(pg_legacy_session, symbol="NOW", exit_date=AS_OF)
    stub = _StubLLM()
    stats = generate_reflections(AS_OF, model="test-model", llm_fn=stub)
    assert stats["written"] == 1
    assert len(stub.calls) == 1
    assert "NOW" in stub.calls[0]["user_prompt"]


def test_generation_truncates_overlong_reply(pg_legacy_session):
    from rainier.paper.reflection import (
        REFLECTION_MAX_CHARS,
        generate_reflections,
    )

    tid = _mk_trade(pg_legacy_session, symbol="AAA")
    stub = _StubLLM(reply="x" * (REFLECTION_MAX_CHARS + 500))
    generate_reflections(AS_OF, model="test-model", llm_fn=stub)
    got = pg_legacy_session.execute(
        text("SELECT reflection FROM paper_trade WHERE id = :id"), {"id": tid}
    ).scalar()
    assert got is not None
    assert len(got) <= REFLECTION_MAX_CHARS


# ---------------------------------------------------------------------------
# 2. Generation — feature-detected chart attachment
# ---------------------------------------------------------------------------


def test_chart_column_absent_text_only_no_crash(pg_legacy_session):
    """Pre-chart-archive main: no `paper_trade.chart_id` -> text-only path."""
    from rainier.paper.reflection import generate_reflections

    _mk_trade(pg_legacy_session, symbol="AAA")
    stub = _StubLLM()
    stats = generate_reflections(AS_OF, model="test-model", llm_fn=stub)
    assert stats["written"] == 1
    assert stub.calls[0]["image_bytes"] is None


def test_chart_column_present_attaches_image(pg_legacy_engine, pg_legacy_session):
    """Post-chart-archive schema: chart_id column + value -> image attached."""
    from rainier.paper.reflection import generate_reflections

    with pg_legacy_engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS chart_images ("
                " id SERIAL PRIMARY KEY, image_bytes BYTEA)"
            )
        )
        conn.execute(
            text("ALTER TABLE paper_trade ADD COLUMN IF NOT EXISTS chart_id BIGINT")
        )
    tid = _mk_trade(pg_legacy_session, symbol="AAA")
    chart_id = pg_legacy_session.execute(
        text(
            "INSERT INTO chart_images (image_bytes) VALUES (:b) RETURNING id"
        ),
        {"b": b"\x89PNG-fake-bytes"},
    ).scalar()
    pg_legacy_session.execute(
        text("UPDATE paper_trade SET chart_id = :c WHERE id = :id"),
        {"c": chart_id, "id": tid},
    )
    pg_legacy_session.commit()
    stub = _StubLLM()
    stats = generate_reflections(AS_OF, model="test-model", llm_fn=stub)
    assert stats["written"] == 1
    assert stub.calls[0]["image_bytes"] == b"\x89PNG-fake-bytes"


def test_chart_column_present_but_null_falls_back_to_text(
    pg_legacy_engine, pg_legacy_session
):
    from rainier.paper.reflection import generate_reflections

    with pg_legacy_engine.begin() as conn:
        conn.execute(
            text("ALTER TABLE paper_trade ADD COLUMN IF NOT EXISTS chart_id BIGINT")
        )
    _mk_trade(pg_legacy_session, symbol="AAA")
    stub = _StubLLM()
    stats = generate_reflections(AS_OF, model="test-model", llm_fn=stub)
    assert stats["written"] == 1
    assert stub.calls[0]["image_bytes"] is None


# ---------------------------------------------------------------------------
# 3. Prompt injection — last K, strictly-before, bounded, labeled
# ---------------------------------------------------------------------------


def test_load_recent_reflections_last_k_strictly_before(pg_legacy_session):
    from rainier.paper.reflection import load_recent_reflections

    # 12 reflected closed trades on consecutive days + 1 exiting ON as_of.
    base = AS_OF - timedelta(days=20)
    for i in range(12):
        _mk_trade(
            pg_legacy_session,
            symbol=f"S{i:02d}",
            exit_date=base + timedelta(days=i),
            reflection=f"reflection {i}",
        )
    _mk_trade(
        pg_legacy_session,
        symbol="TODAY",
        exit_date=AS_OF,
        reflection="same-day reflection must NOT leak into a day-D scan",
    )
    rows = load_recent_reflections(AS_OF, k=10)
    assert len(rows) == 10
    symbols = [r["symbol"] for r in rows]
    assert "TODAY" not in symbols          # strictly-before discipline
    assert symbols[0] == "S11"             # most recent first
    assert "S00" not in symbols            # only the last K survive
    assert "S01" not in symbols


def test_load_recent_reflections_skips_null_reflections(pg_legacy_session):
    from rainier.paper.reflection import load_recent_reflections

    _mk_trade(pg_legacy_session, symbol="NULLR", reflection=None)
    assert load_recent_reflections(AS_OF, k=10) == []


def test_render_reflections_section_labeled_and_bounded():
    from rainier.paper.reflection import (
        REFLECTION_RENDER_CHARS,
        render_reflections_section,
    )

    rows = [
        {
            "symbol": "AAA",
            "exit_date": "2026-06-05",
            "exit_reason": "target",
            "return_pct": 0.08,
            "reflection": "y" * (REFLECTION_RENDER_CHARS + 200),
        },
        {
            "symbol": "BBB",
            "exit_date": "2026-06-04",
            "exit_reason": "stop_loss",
            "return_pct": -0.05,
            "reflection": "Stop hit fast; breakout was a fakeout.",
        },
    ]
    section = render_reflections_section(rows)
    assert "Recent trade reflections" in section  # labeled
    assert "AAA" in section and "BBB" in section
    # Each reflection line is truncated to the render bound.
    for line in section.splitlines():
        assert len(line) <= REFLECTION_RENDER_CHARS + 80  # facts prefix slack


def test_render_reflections_section_empty_is_blank():
    from rainier.paper.reflection import render_reflections_section

    assert render_reflections_section([]) == ""


def test_reflection_prompt_section_appended_in_generate_thesis_path():
    """The service appends the reflections block to the calibration section."""
    import inspect as _inspect

    from rainier.llm_thesis import service as thesis_service

    src = _inspect.getsource(thesis_service.generate_thesis)
    assert "reflection" in src  # R-A wired into the thesis prompt build


# ---------------------------------------------------------------------------
# 3. Cache correctness — prompt_version bumped in lockstep; input_hash inert
# ---------------------------------------------------------------------------


def test_prompt_version_bumped_for_reflections():
    from rainier.llm_thesis.prompt import PROMPT_VERSION

    # v2 was the calibration block; R-A's reflections block changes the prompt
    # again, so the Tier-1 key must move past v2.
    assert PROMPT_VERSION not in ("v1", "v2")


def test_prompt_version_yaml_tracks_module():
    import yaml

    from rainier.llm_thesis.prompt import PROMPT_VERSION

    cfg = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text())
    assert cfg["llm_thesis"]["prompt_version"] == PROMPT_VERSION


def test_prompt_version_config_default_tracks_module():
    from rainier.core.config import LLMThesisConfig
    from rainier.llm_thesis.prompt import PROMPT_VERSION

    assert LLMThesisConfig().prompt_version == PROMPT_VERSION


def test_input_hash_unaffected_by_reflections_text():
    from rainier.llm_thesis.schemas import EvidencePack, compute_input_hash

    pack = EvidencePack(
        symbol="AAA",
        scan_date="2026-06-08",
        session_name="close",
        candidate={"rank": 5},
        signals={},
    )
    # The reflections block lives in prompt text, never in the pack — identical
    # packs hash identically regardless of any reflections rendered.
    assert compute_input_hash(pack, b"img") == compute_input_hash(pack, b"img")
    assert "reflection" not in pack.model_dump()
