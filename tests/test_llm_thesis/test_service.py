"""Tests for generate_thesis: Tier 1 cache, Tier 2 LLM, retries, kill switch, persist."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from rainier.core.config import LLMThesisConfig, Settings
from rainier.llm_thesis.schemas import EvidencePack, TradeThesis
from rainier.llm_thesis.service import generate_thesis


def _settings(model: str = "test-model", prompt_v: str = "v1") -> Settings:
    s = Settings(database_url="postgresql://test:test@localhost/test")
    s.llm_thesis = LLMThesisConfig(
        model=model, prompt_version=prompt_v, max_usd_per_scan=1.0,
    )
    return s


def _valid_thesis_dict():
    return {
        "verdict": "setup_long",
        "setup_quality": 7,
        "llm_confidence": 7,
        "paragraph_radar": "Why on radar.",
        "paragraph_evidence": "Evidence text.",
        "paragraph_invalidation": "Invalidation rules.",
        "risks": ["earnings"],
        "watch_items": ["volume"],
        "evidence_used": ["pattern"],
        "signals_used": ["rank_trajectory"],
        "patterns_in_chart_not_in_indicators": ["narrowing volume"],
    }


def _provider(symbol: str = "NVDA"):
    pack = EvidencePack(
        symbol=symbol, scan_date=date(2026, 5, 7).isoformat(),
        session_name="afternoon", candidate={"rank": 5},
        signals={"rank_trajectory": {"delta_10d": 25}},
    )

    def _p():
        return pack, ["rank trajectory: rising"], b"FAKE-IMAGE"
    return _p


@pytest.mark.asyncio
async def test_tier1_cache_hit_skips_llm_and_provider():
    cached_record = (42, _valid_thesis_dict())
    provider_called = MagicMock(side_effect=_provider())
    with patch(
        "rainier.llm_thesis.service._tier1_lookup", return_value=cached_record
    ), patch(
        "rainier.llm_thesis.service._call_llm"
    ) as mock_llm:
        thesis, cost, rec_id = await generate_thesis(
            symbol="NVDA", scan_date=date(2026, 5, 7), session_name="afternoon",
            evidence_provider=provider_called, settings=_settings(),
            max_usd_remaining=1.0,
        )
    assert isinstance(thesis, TradeThesis)
    assert cost == 0.0
    assert rec_id == 42
    mock_llm.assert_not_called()
    provider_called.assert_not_called()


@pytest.mark.asyncio
async def test_tier2_miss_calls_llm_persists_returns_thesis():
    valid_payload = json.dumps(_valid_thesis_dict())
    with patch(
        "rainier.llm_thesis.service._tier1_lookup", return_value=None
    ), patch(
        "rainier.llm_thesis.service._call_llm",
        return_value=(valid_payload, 100, 200),
    ), patch(
        "rainier.llm_thesis.service._persist_thesis", return_value=99,
    ):
        thesis, cost, rec_id = await generate_thesis(
            symbol="NVDA", scan_date=date(2026, 5, 7), session_name="afternoon",
            evidence_provider=_provider(), settings=_settings(),
            max_usd_remaining=1.0,
        )
    assert thesis is not None
    assert thesis.verdict == "setup_long"
    assert rec_id == 99
    assert cost > 0


@pytest.mark.asyncio
async def test_three_validation_failures_returns_none():
    bad = '{"verdict": "setup_long"}'  # missing required fields
    with patch(
        "rainier.llm_thesis.service._tier1_lookup", return_value=None
    ), patch(
        "rainier.llm_thesis.service._call_llm",
        return_value=(bad, 50, 50),
    ) as mock_llm, patch(
        "rainier.llm_thesis.service._persist_thesis", return_value=None
    ):
        thesis, cost, rec_id = await generate_thesis(
            symbol="NVDA", scan_date=date(2026, 5, 7), session_name="afternoon",
            evidence_provider=_provider(), settings=_settings(),
            max_usd_remaining=1.0,
        )
    assert thesis is None
    assert rec_id is None
    assert mock_llm.call_count == 3


@pytest.mark.asyncio
async def test_max_usd_kill_switch_aborts_before_llm():
    with patch(
        "rainier.llm_thesis.service._tier1_lookup", return_value=None
    ), patch(
        "rainier.llm_thesis.service._call_llm"
    ) as mock_llm:
        thesis, cost, rec_id = await generate_thesis(
            symbol="NVDA", scan_date=date(2026, 5, 7), session_name="afternoon",
            evidence_provider=_provider(), settings=_settings(),
            max_usd_remaining=0.0,
        )
    assert thesis is None
    assert cost == 0.0
    assert rec_id is None
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_cost_overrun_after_call_aborts_with_charge():
    valid_payload = json.dumps(_valid_thesis_dict())
    # Tokens that compute to a cost above the remaining budget.
    huge_tokens = (10_000_000, 10_000_000)
    with patch(
        "rainier.llm_thesis.service._tier1_lookup", return_value=None
    ), patch(
        "rainier.llm_thesis.service._call_llm",
        return_value=(valid_payload, *huge_tokens),
    ), patch(
        "rainier.llm_thesis.service._persist_thesis", return_value=None
    ):
        thesis, cost, rec_id = await generate_thesis(
            symbol="NVDA", scan_date=date(2026, 5, 7), session_name="afternoon",
            evidence_provider=_provider(), settings=_settings(),
            max_usd_remaining=0.10,
        )
    assert thesis is None
    assert cost > 0.10
    assert rec_id is None


@pytest.mark.asyncio
async def test_retry_recovers_on_second_attempt():
    bad = '{"oops": true}'
    good = json.dumps(_valid_thesis_dict())
    with patch(
        "rainier.llm_thesis.service._tier1_lookup", return_value=None
    ), patch(
        "rainier.llm_thesis.service._call_llm",
        side_effect=[(bad, 50, 50), (good, 50, 50)],
    ) as mock_llm, patch(
        "rainier.llm_thesis.service._persist_thesis", return_value=77
    ):
        thesis, cost, rec_id = await generate_thesis(
            symbol="NVDA", scan_date=date(2026, 5, 7), session_name="afternoon",
            evidence_provider=_provider(), settings=_settings(),
            max_usd_remaining=1.0,
        )
    assert thesis is not None
    assert rec_id == 77
    assert mock_llm.call_count == 2
