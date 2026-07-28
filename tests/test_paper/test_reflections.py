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
    shadow: bool = False,
) -> int:
    thesis_id = _mk_thesis(session)
    tid = session.execute(
        text(
            "INSERT INTO paper_trade "
            "(thesis_id, symbol, scan_date, session_name, status, "
            " planned_entry_price, stop_loss, target_price, pattern_type, "
            " llm_confidence, verdict, entry_date, entry_price, shares, "
            " exit_date, exit_price, exit_reason, return_pct, pnl, reflection, "
            " shadow) "
            "VALUES "
            "(:thesis_id, :symbol, '2026-05-29', 'close', :status, "
            " 100.0, 95.0, 110.0, 'w_bottom', "
            " 7, 'setup_long', :entry_date, 100.0, 100, "
            " :exit_date, :exit_price, :exit_reason, :return_pct, :pnl, "
            " :reflection, :shadow) "
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
            "shadow": shadow,
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

    def __call__(self, *, model, system_prompt, user_prompt, image_bytes,
                 **_kwargs):
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


def test_generation_excludes_shadow_trades(pg_legacy_session):
    """WS A isolation — a closed SHADOW trade must never be a reflection
    candidate (no LLM spend on measurement rows; outcomes stay out of prompts)."""
    from rainier.paper.reflection import (
        generate_reflections,
        select_reflection_candidates,
    )

    _mk_trade(pg_legacy_session, symbol="SHD", exit_date=AS_OF, shadow=True)
    _mk_trade(pg_legacy_session, symbol="LIV", exit_date=AS_OF, shadow=False)

    cands = select_reflection_candidates(AS_OF)
    symbols = {c["symbol"] for c in cands}
    assert "SHD" not in symbols and "LIV" in symbols

    stub = _StubLLM()
    stats = generate_reflections(AS_OF, model="test-model", llm_fn=stub)
    assert stats["written"] == 1  # only the live trade
    assert all("SHD" not in c["user_prompt"] for c in stub.calls)


def test_load_recent_reflections_excludes_shadow(pg_legacy_session):
    """WS A isolation — even a shadow row that somehow carries reflection text
    must not be injected into the live thesis prompt."""
    from rainier.paper.reflection import load_recent_reflections

    _mk_trade(
        pg_legacy_session, symbol="SHD", exit_date=AS_OF - timedelta(days=1),
        reflection="shadow post-mortem", shadow=True,
    )
    _mk_trade(
        pg_legacy_session, symbol="LIV", exit_date=AS_OF - timedelta(days=1),
        reflection="live post-mortem", shadow=False,
    )
    rows = load_recent_reflections(AS_OF, k=10)
    symbols = {r["symbol"] for r in rows}
    assert "SHD" not in symbols and "LIV" in symbols


def test_generation_caps_llm_calls_per_run(pg_legacy_session):
    """Spend bound: a cold-start backlog drains max_per_run per night,
    oldest exit first; the deferred remainder is re-selected next run."""
    from rainier.paper.reflection import generate_reflections

    for i in range(3):
        _mk_trade(
            pg_legacy_session,
            symbol=f"C{i:02d}",
            exit_date=AS_OF - timedelta(days=5 - i),  # C00 oldest
        )
    stub = _StubLLM()
    stats = generate_reflections(
        AS_OF, model="test-model", llm_fn=stub, max_per_run=2
    )
    assert stats["written"] == 2
    assert stats["deferred"] == 1
    assert len(stub.calls) == 2  # the cap bounds LLM spend
    prompts = "\n".join(c["user_prompt"] for c in stub.calls)
    assert "C00" in prompts and "C01" in prompts  # oldest-first drain
    assert "C02" not in prompts
    # Next run picks up the deferred trade — nothing is lost.
    stub2 = _StubLLM()
    stats2 = generate_reflections(
        AS_OF, model="test-model", llm_fn=stub2, max_per_run=2
    )
    assert stats2["written"] == 1
    assert stats2["deferred"] == 0
    assert "C02" in stub2.calls[0]["user_prompt"]


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


def test_generation_continues_batch_after_one_failure(pg_legacy_session):
    """Per-trade isolation: one failing LLM call never blocks the batch."""
    from rainier.paper.reflection import generate_reflections

    _mk_trade(pg_legacy_session, symbol="BAD", exit_date=AS_OF - timedelta(days=2))
    ok_tid = _mk_trade(pg_legacy_session, symbol="GOOD", exit_date=AS_OF)

    class _FailFirst(_StubLLM):
        def __call__(self, **kw):
            super().__call__(**kw)
            if len(self.calls) == 1:
                raise RuntimeError("llm unavailable")
            return self.reply

    stub = _FailFirst(reply="second trade reflected fine")
    stats = generate_reflections(AS_OF, model="test-model", llm_fn=stub)
    assert stats == {"candidates": 2, "written": 1, "failed": 1, "deferred": 0}
    got = pg_legacy_session.execute(
        text("SELECT reflection FROM paper_trade WHERE id = :id"), {"id": ok_tid}
    ).scalar()
    assert got == "second trade reflected fine"


def test_generation_empty_reply_counts_failed(pg_legacy_session):
    """Whitespace-only LLM reply -> failed, reflection stays NULL (no
    placeholder write that would block the natural retry)."""
    from rainier.paper.reflection import generate_reflections

    tid = _mk_trade(pg_legacy_session, symbol="AAA")
    stub = _StubLLM(reply="   ")
    stats = generate_reflections(AS_OF, model="test-model", llm_fn=stub)
    assert stats["written"] == 0
    assert stats["failed"] == 1
    got = pg_legacy_session.execute(
        text("SELECT reflection FROM paper_trade WHERE id = :id"), {"id": tid}
    ).scalar()
    assert got is None


def test_generation_selects_at_lookback_boundary(pg_legacy_session):
    """exit_date == as_of - LOOKBACK days is inside the window (inclusive)."""
    from rainier.paper.reflection import (
        REFLECTION_LOOKBACK_DAYS,
        generate_reflections,
    )

    _mk_trade(
        pg_legacy_session,
        symbol="EDG",
        exit_date=AS_OF - timedelta(days=REFLECTION_LOOKBACK_DAYS),
    )
    stub = _StubLLM()
    stats = generate_reflections(AS_OF, model="test-model", llm_fn=stub)
    assert stats["written"] == 1
    assert "EDG" in stub.calls[0]["user_prompt"]


def test_generation_bumps_updated_at(pg_legacy_session):
    """Raw-SQL reflection write must advance updated_at (the ORM onupdate
    does not fire for raw UPDATEs; change-detection consumers read it)."""
    from rainier.paper.reflection import generate_reflections

    tid = _mk_trade(pg_legacy_session, symbol="AAA")
    pg_legacy_session.execute(
        text(
            "UPDATE paper_trade SET updated_at = NOW() - interval '1 day' "
            "WHERE id = :id"
        ),
        {"id": tid},
    )
    pg_legacy_session.commit()
    before = pg_legacy_session.execute(
        text("SELECT updated_at FROM paper_trade WHERE id = :id"), {"id": tid}
    ).scalar()
    generate_reflections(AS_OF, model="test-model", llm_fn=_StubLLM())
    after = pg_legacy_session.execute(
        text("SELECT updated_at FROM paper_trade WHERE id = :id"), {"id": tid}
    ).scalar()
    assert after > before


# ---------------------------------------------------------------------------
# 2. Generation — feature-detected chart attachment
# ---------------------------------------------------------------------------


def test_chart_probe_degrades_to_false_on_error():
    """The pg_attribute probe swallows any failure -> False (text-only path)."""
    from unittest.mock import Mock

    from rainier.paper.reflection import _chart_id_column_exists

    broken = Mock()
    broken.execute.side_effect = RuntimeError("no pg_attribute here")
    assert _chart_id_column_exists(broken) is False


def test_chart_fetch_degrades_to_none_on_error():
    """A failing chart fetch returns None (text-only), never raises."""
    from unittest.mock import Mock

    from rainier.paper.reflection import _load_chart_bytes

    broken = Mock()
    broken.execute.side_effect = RuntimeError("chart_images missing")
    assert _load_chart_bytes(broken, 1, has_chart_col=True) is None


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
        # chart_images + paper_trade.chart_id are created by the R-D archive
        # migration (0010), applied by the pg_legacy_engine fixture. Guard the
        # column add so this test still runs against a pre-0010 schema.
        conn.execute(
            text("ALTER TABLE paper_trade ADD COLUMN IF NOT EXISTS chart_id BIGINT")
        )
        conn.execute(
            text(
                "INSERT INTO stocks (symbol) VALUES ('AAA') "
                "ON CONFLICT (symbol) DO NOTHING"
            )
        )
    tid = _mk_trade(pg_legacy_session, symbol="AAA")
    # chart_images carries a NOT NULL symbol FK under the real (R-D) schema, so
    # supply it; the test only cares that image_bytes flows to the LLM.
    chart_id = pg_legacy_session.execute(
        text(
            "INSERT INTO chart_images (symbol, image_bytes) "
            "VALUES (:s, :b) RETURNING id"
        ),
        {"s": "AAA", "b": b"\x89PNG-fake-bytes"},
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


async def _run_thesis_capture(monkeypatch) -> tuple:
    """Drive generate_thesis with stubbed LLM/cache/persist; return
    (thesis, captured_user_prompt). Calibration/reflection loaders are
    monkeypatched by the caller BEFORE calling this."""
    import json
    from unittest.mock import patch

    from rainier.core.config import LLMThesisConfig, Settings
    from rainier.llm_thesis.schemas import EvidencePack
    from rainier.llm_thesis.service import generate_thesis

    valid = json.dumps(
        {
            "verdict": "setup_long",
            "setup_quality": 7,
            "llm_confidence": 7,
            "paragraph_radar": "r",
            "paragraph_evidence": "e",
            "paragraph_invalidation": "i",
            "risks": [],
            "watch_items": [],
            "evidence_used": ["pattern"],
            "signals_used": [],
            "patterns_in_chart_not_in_indicators": "none",
        }
    )
    pack = EvidencePack(
        symbol="NVDA",
        scan_date="2026-06-08",
        session_name="close",
        candidate={"rank": 5},
        signals={},
    )
    settings = Settings(database_url="postgresql://test:test@localhost/test")
    settings.llm_thesis = LLMThesisConfig(model="test-model", prompt_version="v3")

    captured: dict = {}

    def fake_call_llm(*, model, system_prompt, user_prompt, image_bytes,
                      **_kwargs):
        captured["user_prompt"] = user_prompt
        return valid, 10, 20

    with patch(
        "rainier.llm_thesis.service._tier1_lookup", return_value=None
    ), patch(
        "rainier.llm_thesis.service._call_llm", side_effect=fake_call_llm
    ), patch(
        "rainier.llm_thesis.service._persist_thesis", return_value=1
    ):
        thesis, _cost, _rid = await generate_thesis(
            symbol="NVDA",
            scan_date=date(2026, 6, 8),
            session_name="close",
            evidence_provider=lambda: (pack, [], None),
            settings=settings,
            max_usd_remaining=1.0,
        )
    return thesis, captured["user_prompt"]


_RFL_ROW = {
    "symbol": "RFL",
    "exit_date": "2026-06-05",
    "exit_reason": "target",
    "return_pct": 0.08,
    "reflection": "Breakout ran straight to target; thesis held.",
}


@pytest.mark.asyncio
async def test_reflections_block_lands_in_thesis_user_prompt(monkeypatch):
    """generate_thesis appends the reflections block to the LLM user prompt."""
    monkeypatch.setattr(
        "rainier.paper.calibration.load_latest_calibration",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "rainier.paper.reflection.load_recent_reflections",
        lambda *a, **k: [_RFL_ROW],
    )
    thesis, user_prompt = await _run_thesis_capture(monkeypatch)
    assert thesis is not None
    assert "Recent trade reflections" in user_prompt
    assert "RFL" in user_prompt
    assert "thesis held" in user_prompt


@pytest.mark.asyncio
async def test_thesis_survives_reflections_load_failure(monkeypatch):
    """Best-effort isolation: a raising reflections loader costs the block,
    never the thesis (the daily scan must not crash on a reflections bug)."""

    def _boom(*a, **k):
        raise RuntimeError("reflections DB unavailable")

    monkeypatch.setattr(
        "rainier.paper.calibration.load_latest_calibration",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "rainier.paper.reflection.load_recent_reflections", _boom
    )
    thesis, user_prompt = await _run_thesis_capture(monkeypatch)
    assert thesis is not None
    assert "Recent trade reflections" not in user_prompt


@pytest.mark.asyncio
async def test_reflections_appended_after_calibration_block(monkeypatch):
    """With both blocks present, calibration text survives and the
    reflections block is appended after it (the join branch)."""
    monkeypatch.setattr(
        "rainier.paper.calibration.load_latest_calibration",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(
        "rainier.paper.calibration.render_calibration_section",
        lambda *a, **k: "CALIB-BLOCK-SENTINEL",
    )
    monkeypatch.setattr(
        "rainier.paper.reflection.load_recent_reflections",
        lambda *a, **k: [_RFL_ROW],
    )
    thesis, user_prompt = await _run_thesis_capture(monkeypatch)
    assert thesis is not None
    assert "CALIB-BLOCK-SENTINEL" in user_prompt
    assert "Recent trade reflections" in user_prompt
    assert user_prompt.index("CALIB-BLOCK-SENTINEL") < user_prompt.index(
        "Recent trade reflections"
    )


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
    import inspect

    from rainier.llm_thesis.schemas import EvidencePack, compute_input_hash

    pack = EvidencePack(
        symbol="AAA",
        scan_date="2026-06-08",
        session_name="close",
        candidate={"rank": 5},
        signals={},
    )
    # Structural guarantee: the Tier-2 hash is computed from the EvidencePack +
    # image bytes ONLY — prompt text (calibration/reflections blocks) cannot
    # enter it because the function takes no prompt-text parameter and the pack
    # carries no reflection field. (PROMPT_VERSION busts Tier-1 instead.)
    params = list(inspect.signature(compute_input_hash).parameters)
    assert params == ["pack", "image_bytes"]
    assert "reflection" not in pack.model_dump()
    assert compute_input_hash(pack, b"img") == compute_input_hash(pack, b"img")
