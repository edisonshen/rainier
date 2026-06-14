"""WS A — shadow WATCH-buy: gates, long-shape guard, isolation, eval split.

Lanes:
  * pure-logic (no DB): watch-buy gate, long-shape guard, replay math, eval
    action-split;
  * postgres-lane (`pg_legacy_engine`, skips without PG): shadow creation,
    long-shape regression, live-read isolation (byte-identical), replay.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import select, text

from rainier.core.database import get_session
from rainier.core.models import PaperSkip, PaperTrade
from rainier.llm_thesis.eval import _hit
from rainier.paper.positions import (
    _is_long_shape,
    _is_watch_buy_signal,
    create_positions_for_theses,
    create_shadow_positions_for_theses,
)
from rainier.paper.replay import benchmark_return, shadow_book_return

# --------------------------------------------------------------------------
# Pure-logic lane
# --------------------------------------------------------------------------


def test_watch_buy_gate_accepts_watch_at_threshold():
    assert _is_watch_buy_signal("watch", 5, "close", 5) is True


def test_watch_buy_gate_rejects_below_threshold():
    assert _is_watch_buy_signal("watch", 4, "close", 5) is False


def test_watch_buy_gate_rejects_non_watch_verdict():
    assert _is_watch_buy_signal("no_setup", 6, "close", 5) is False
    assert _is_watch_buy_signal("setup_long", 6, "close", 5) is False


def test_watch_buy_gate_rejects_non_actionable_session():
    assert _is_watch_buy_signal("watch", 6, "morning", 5) is False


def test_long_shape_accepts_valid_bullish_setup():
    levels = {"entry_price": 100.0, "stop_loss": 95.0, "target_price": 110.0}
    assert _is_long_shape(levels, "w_bottom") is True


def test_long_shape_rejects_false_breakout_bearish_pattern():
    # Even with numerically long-looking levels, false_breakout is bearish.
    levels = {"entry_price": 100.0, "stop_loss": 95.0, "target_price": 110.0}
    assert _is_long_shape(levels, "false_breakout") is False


def test_long_shape_rejects_inverted_levels():
    levels = {"entry_price": 100.0, "stop_loss": 105.0, "target_price": 110.0}
    assert _is_long_shape(levels, "w_bottom") is False


def test_long_shape_rejects_missing_level():
    levels = {"entry_price": 100.0, "stop_loss": None, "target_price": 110.0}
    assert _is_long_shape(levels, "w_bottom") is False


def test_replay_book_return_equal_weight_mean():
    assert shadow_book_return([0.10, -0.04, 0.06]) == pytest.approx(0.04)


def test_replay_book_return_empty_is_cash():
    assert shadow_book_return([]) == 0.0


def test_benchmark_return_point_to_point():
    assert benchmark_return(100.0, 105.0) == pytest.approx(0.05)


def test_benchmark_return_missing_endpoint_is_zero():
    assert benchmark_return(None, 105.0) == 0.0
    assert benchmark_return(100.0, None) == 0.0


def test_eval_split_grades_acted_watch_by_trade_outcome():
    # A WATCH that opened a long: a positive move is a WIN (not the abstention
    # rule which would call return>0 a miss for watch).
    assert _hit("watch", 0.05, acted_long=True) is True
    assert _hit("watch", -0.03, acted_long=True) is False


def test_eval_abstention_grading_when_no_action():
    # A WATCH that did NOT act keeps the abstention grading.
    assert _hit("watch", -0.03, acted_long=False) is True
    assert _hit("watch", 0.05, acted_long=False) is False


def test_migration_0013_files_exist():
    from tests.test_paper.conftest import MIGRATION_0013_DOWN, MIGRATION_0013_UP

    assert MIGRATION_0013_UP.exists()
    assert MIGRATION_0013_DOWN.exists()


# --------------------------------------------------------------------------
# Postgres-lane — shadow creation + isolation. Skips without PG.
# --------------------------------------------------------------------------


def _seed_thesis(session, recommendation: str) -> int:
    res = session.execute(
        text(
            "INSERT INTO analysis_results (llm_model, prompt_template, "
            "recommendation) VALUES ('test', 'test', :rec) RETURNING id"
        ),
        {"rec": recommendation},
    )
    return int(res.scalar())


def _seed_screened(
    session,
    *,
    symbol: str,
    scan_date: date,
    thesis_id: int,
    pattern_type: str,
    entry: float,
    stop: float,
    target: float,
) -> None:
    session.execute(
        text(
            "INSERT INTO screened_stocks "
            "(scan_date, session_name, symbol, rule_rank, composite_score, "
            " pattern_type, thesis_id, entry_price, stop_loss, target_price) "
            "VALUES (:sd, 'close', :sym, 1, 0.5, :pt, :tid, :e, :s, :t)"
        ),
        {"sd": scan_date, "sym": symbol, "pt": pattern_type, "tid": thesis_id,
         "e": entry, "s": stop, "t": target},
    )


def _thesis_dict(thesis_id: int, symbol: str, verdict: str, conf: int) -> dict:
    return {
        "thesis_id": thesis_id,
        "symbol": symbol,
        "verdict": verdict,
        "llm_confidence": conf,
        "session_name": "close",
    }


def test_shadow_watch_buy_opens_isolated_position(pg_legacy_engine):
    scan_date = date(2026, 6, 2)
    with get_session() as session:
        tid = _seed_thesis(session, "watch")
        _seed_screened(
            session, symbol="AAA", scan_date=scan_date, thesis_id=tid,
            pattern_type="w_bottom", entry=100.0, stop=95.0, target=110.0,
        )
    res = create_shadow_positions_for_theses(
        [_thesis_dict(tid, "AAA", "watch", 5)],
        scan_date=scan_date, min_confidence=5,
    )
    assert res["created"] == 1
    with get_session() as session:
        rows = session.execute(select(PaperTrade)).scalars().all()
    assert len(rows) == 1
    assert rows[0].shadow is True
    assert rows[0].verdict == "watch"


def test_watch_on_false_breakout_does_not_open(pg_legacy_engine):
    """WS A acceptance 3: a watch on false_breakout (bearish) never opens."""
    scan_date = date(2026, 6, 2)
    with get_session() as session:
        tid = _seed_thesis(session, "watch")
        _seed_screened(
            session, symbol="BBB", scan_date=scan_date, thesis_id=tid,
            pattern_type="false_breakout", entry=100.0, stop=95.0, target=110.0,
        )
    res = create_shadow_positions_for_theses(
        [_thesis_dict(tid, "BBB", "watch", 6)],
        scan_date=scan_date, min_confidence=5,
    )
    assert res["created"] == 0
    with get_session() as session:
        rows = session.execute(select(PaperTrade)).scalars().all()
    assert rows == []


def test_watch_below_threshold_does_not_open(pg_legacy_engine):
    scan_date = date(2026, 6, 2)
    with get_session() as session:
        tid = _seed_thesis(session, "watch")
        _seed_screened(
            session, symbol="CCC", scan_date=scan_date, thesis_id=tid,
            pattern_type="w_bottom", entry=100.0, stop=95.0, target=110.0,
        )
    res = create_shadow_positions_for_theses(
        [_thesis_dict(tid, "CCC", "watch", 4)],
        scan_date=scan_date, min_confidence=5,
    )
    assert res["created"] == 0


def test_shadow_never_writes_live_skip_ledger(pg_legacy_engine):
    """A watch with no screened row would `missing_screened_record`-skip in the
    live path; shadow must NOT write the live skip ledger."""
    scan_date = date(2026, 6, 2)
    with get_session() as session:
        tid = _seed_thesis(session, "watch")  # no screened row seeded
    create_shadow_positions_for_theses(
        [_thesis_dict(tid, "DDD", "watch", 6)],
        scan_date=scan_date, min_confidence=5,
    )
    with get_session() as session:
        skips = session.execute(select(PaperSkip)).scalars().all()
    assert skips == []


def test_shadow_does_not_consume_live_active_slot(pg_legacy_engine):
    """Isolation invariant: a shadow position on a symbol must not block a LIVE
    setup_long on the SAME symbol from opening."""
    scan_date = date(2026, 6, 2)
    with get_session() as session:
        shadow_tid = _seed_thesis(session, "watch")
        live_tid = _seed_thesis(session, "setup_long")
        _seed_screened(
            session, symbol="EEE", scan_date=scan_date, thesis_id=shadow_tid,
            pattern_type="w_bottom", entry=100.0, stop=95.0, target=110.0,
        )
        _seed_screened(
            session, symbol="EEE", scan_date=date(2026, 6, 3),
            thesis_id=live_tid, pattern_type="w_bottom",
            entry=100.0, stop=95.0, target=110.0,
        )
    # Shadow opens first on EEE.
    s = create_shadow_positions_for_theses(
        [_thesis_dict(shadow_tid, "EEE", "watch", 6)],
        scan_date=scan_date, min_confidence=5,
    )
    assert s["created"] == 1
    # Live setup_long on EEE must STILL open (shadow didn't consume the slot).
    live = create_positions_for_theses(
        [_thesis_dict(live_tid, "EEE", "setup_long", 6)],
        scan_date=date(2026, 6, 3),
    )
    assert live["created"] == 1
    with get_session() as session:
        live_rows = session.execute(
            select(PaperTrade).where(PaperTrade.shadow.is_(False))
        ).scalars().all()
        shadow_rows = session.execute(
            select(PaperTrade).where(PaperTrade.shadow.is_(True))
        ).scalars().all()
    assert len(live_rows) == 1
    assert len(shadow_rows) == 1


def test_live_book_byte_identical_with_shadow_present(pg_legacy_engine):
    """Isolation regression: the live daily-report read returns the same rows
    whether or not a shadow position exists for a DIFFERENT symbol."""
    from rainier.paper.report import compute_daily_payload

    scan_date = date(2026, 6, 2)
    with get_session() as session:
        live_tid = _seed_thesis(session, "setup_long")
        _seed_screened(
            session, symbol="FFF", scan_date=scan_date, thesis_id=live_tid,
            pattern_type="w_bottom", entry=100.0, stop=95.0, target=110.0,
        )
    create_positions_for_theses(
        [_thesis_dict(live_tid, "FFF", "setup_long", 6)], scan_date=scan_date
    )
    payload_before = compute_daily_payload(date(2026, 6, 2))

    # Now add a shadow position on a different symbol.
    with get_session() as session:
        shadow_tid = _seed_thesis(session, "watch")
        _seed_screened(
            session, symbol="GGG", scan_date=scan_date, thesis_id=shadow_tid,
            pattern_type="w_bottom", entry=50.0, stop=45.0, target=60.0,
        )
    create_shadow_positions_for_theses(
        [_thesis_dict(shadow_tid, "GGG", "watch", 6)],
        scan_date=scan_date, min_confidence=5,
    )
    payload_after = compute_daily_payload(date(2026, 6, 2))

    # The live report must be byte-identical despite the shadow row's existence.
    assert payload_before == payload_after
