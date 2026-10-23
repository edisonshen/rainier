"""Spec: a persisted thesis keeps the numeric signal values its prompt used, in the
shape the dashboard's Tab 2 scatter reads (structured_output["signals"][name])."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from rainier.core.config import Settings
from rainier.dashboard import data as ddata
from rainier.llm_thesis import service
from rainier.llm_thesis.schemas import TradeThesis


def _thesis() -> TradeThesis:
    return TradeThesis(
        verdict="watch", setup_quality=4, llm_confidence=5,
        paragraph_radar="r", paragraph_evidence="e", paragraph_invalidation="i",
        risks=[], watch_items=[], evidence_used=["chart"],
        signals_used=["rank_trajectory"], patterns_in_chart_not_in_indicators="none",
    )


def test_persisted_thesis_carries_signal_values_for_the_dashboard():
    added = []
    fake = MagicMock()
    fake.add.side_effect = added.append

    @contextmanager
    def _session():
        yield fake

    signals = {"rank_trajectory": {"delta_10d": 18, "trend": "rising"},
               "sector_momentum": {"delta": -0.03},
               "fundamentals": {"pe_trailing": float("inf"), "growth": [1.0, float("nan")]}}
    with patch.object(service, "get_session", _session):
        service._persist_thesis(
            symbol="CRWV", thesis=_thesis(), settings=Settings(), input_hash="h",
            prompt_tokens=1, completion_tokens=1, cost_usd=0.0,
            session_name="afternoon", signal_values=signals,
        )

    blob = added[0].structured_output["signals"]
    assert blob["rank_trajectory"] == signals["rank_trajectory"]
    # yfinance can hand back inf/NaN, which Postgres JSONB rejects; they persist as null.
    assert blob["fundamentals"] == {"pe_trailing": None, "growth": [1.0, None]}
    assert ddata.SIGNAL_VALUE_EXTRACTORS["rank_trajectory"](blob["rank_trajectory"]) == 18.0
    assert TradeThesis.model_validate(added[0].structured_output).verdict == "watch"
