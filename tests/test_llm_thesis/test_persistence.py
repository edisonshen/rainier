"""Tests for persistence module — uses an in-memory SQLite shim where possible."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from rainier.core.types import StockCandidate
from rainier.llm_thesis import persistence as persist_mod


def _candidate(
    symbol: str = "NVDA",
    strength: float = 0.85,
    money_flow: float | None = 0.42,
) -> StockCandidate:
    return StockCandidate(
        symbol=symbol, rank=5, rank_change=0, long_short="Long in",
        capital_flow_direction="+", sector="Technology",
        signal_strength=strength,
        money_flow_score=money_flow,
        pattern_type="w_bottom", pattern_confidence=0.8,
    )


def _patched_session(rowcount: int = 1):
    fake_session = MagicMock()
    fake_session.execute.return_value = MagicMock(rowcount=rowcount)
    cm = MagicMock()
    cm.__enter__.return_value = fake_session
    cm.__exit__.return_value = False
    return cm, fake_session


def test_persist_screened_stocks_bulk_insert():
    cm, fake_session = _patched_session(rowcount=2)
    cands = [_candidate("NVDA"), _candidate("TSLA")]
    with patch.object(persist_mod, "get_session", return_value=cm):
        n = persist_mod.persist_screened_stocks(
            cands, scan_date=date(2026, 5, 7), session_name="afternoon",
        )
    assert n == 2
    fake_session.execute.assert_called_once()


def test_candidate_row_separates_composite_from_money_flow():
    """Codex P1 regression: composite_score must hold the post-boost composite
    (signal_strength) while money_flow_score holds the raw Layer-1 value;
    the two columns must NOT be a duplicate of each other."""
    cand = _candidate(symbol="NVDA", strength=0.85, money_flow=0.42)
    row = persist_mod._candidate_row(cand, rule_rank=1, scan_date=date(2026, 5, 7),
                                     session_name="afternoon")
    assert row["composite_score"] == 0.85
    assert row["money_flow_score"] == 0.42
    assert row["composite_score"] != row["money_flow_score"]


def test_candidate_row_money_flow_none_when_missing():
    """Legacy candidates without a money_flow_score map to NULL — a known
    absence is honest. We must NOT fall back to signal_strength."""
    cand = _candidate(symbol="NVDA", strength=0.85, money_flow=None)
    row = persist_mod._candidate_row(cand, rule_rank=1, scan_date=date(2026, 5, 7),
                                     session_name="afternoon")
    assert row["composite_score"] == 0.85
    assert row["money_flow_score"] is None


def test_persist_screened_stocks_dedups_duplicate_symbol():
    """P0 regression: an in-batch duplicate (scan_date, session_name, symbol)
    must be deduped BEFORE the bulk upsert, or psycopg2 raises
    CardinalityViolation ("ON CONFLICT DO UPDATE cannot affect row a second
    time"). Capture the rows handed to pg_insert().values() and assert keys are
    unique."""
    captured: dict[str, list[dict]] = {}

    def _capture(rows):
        captured["rows"] = rows
        return MagicMock()

    cm, _ = _patched_session(rowcount=1)
    cands = [_candidate("NVDA"), _candidate("TSLA"), _candidate("NVDA")]
    with patch.object(persist_mod, "get_session", return_value=cm), patch.object(
        persist_mod, "pg_insert"
    ) as pg:
        pg.return_value.values.side_effect = _capture
        n = persist_mod.persist_screened_stocks(
            cands, scan_date=date(2026, 6, 4), session_name="afternoon",
        )

    rows = captured["rows"]
    keys = [(r["scan_date"], r["session_name"], r["symbol"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate (scan_date,session,symbol) reached upsert"
    assert sorted(r["symbol"] for r in rows) == ["NVDA", "TSLA"]
    assert n == len(rows) == 2


def test_persist_screened_stocks_dedup_keeps_best_ranked():
    """When a symbol appears twice, keep the first (best-ranked) occurrence —
    candidates arrive sorted by composite score descending."""
    captured: dict[str, list[dict]] = {}

    def _capture(rows):
        captured["rows"] = rows
        return MagicMock()

    cm, _ = _patched_session(rowcount=1)
    # Two NVDA candidates: first has the stronger composite score.
    strong = _candidate("NVDA", strength=0.9)
    weak = _candidate("NVDA", strength=0.1)
    with patch.object(persist_mod, "get_session", return_value=cm), patch.object(
        persist_mod, "pg_insert"
    ) as pg:
        pg.return_value.values.side_effect = _capture
        persist_mod.persist_screened_stocks(
            [strong, weak], scan_date=date(2026, 6, 4), session_name="afternoon",
        )

    rows = captured["rows"]
    assert len(rows) == 1
    assert rows[0]["composite_score"] == 0.9


def test_persist_screened_stocks_empty_returns_zero():
    n = persist_mod.persist_screened_stocks(
        [], scan_date=date(2026, 5, 7), session_name="afternoon"
    )
    assert n == 0


def test_persist_screened_stocks_db_failure_propagates():
    cm = MagicMock()
    cm.__enter__.side_effect = RuntimeError("db down")
    with patch.object(persist_mod, "get_session", return_value=cm):
        with pytest.raises(RuntimeError):
            persist_mod.persist_screened_stocks(
                [_candidate()], scan_date=date(2026, 5, 7), session_name="afternoon"
            )


def test_update_with_thesis_returns_true_on_match():
    cm, _ = _patched_session(rowcount=1)
    with patch.object(persist_mod, "get_session", return_value=cm):
        ok = persist_mod.update_with_thesis(
            symbol="NVDA", scan_date=date(2026, 5, 7), session_name="afternoon",
            llm_confidence=7, shadow_combined_score=0.62, would_be_combined_rank=1,
            thesis_id=42, patterns_count=1,
        )
    assert ok is True


def test_update_with_thesis_zero_rows_logs_and_returns_false(caplog):
    cm, _ = _patched_session(rowcount=0)
    with patch.object(persist_mod, "get_session", return_value=cm):
        with caplog.at_level("WARNING"):
            ok = persist_mod.update_with_thesis(
                symbol="ZZZ", scan_date=date(2026, 5, 7), session_name="afternoon",
                llm_confidence=5, shadow_combined_score=0.4, would_be_combined_rank=2,
                thesis_id=10, patterns_count=0,
            )
    assert ok is False
    assert any("update_with_thesis_no_row_affected" in m for m in caplog.messages)
