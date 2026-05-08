"""Tests for persistence module — uses an in-memory SQLite shim where possible."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from rainier.core.types import StockCandidate
from rainier.llm_thesis import persistence as persist_mod


def _candidate(symbol: str = "NVDA", strength: float = 0.85) -> StockCandidate:
    return StockCandidate(
        symbol=symbol, rank=5, rank_change=0, long_short="Long in",
        capital_flow_direction="+", sector="Technology",
        signal_strength=strength,
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
