"""Tests for generate_thesis: Tier 1 cache, Tier 2 LLM, retries, kill switch, persist."""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from rainier.core.config import LLMThesisConfig, Settings
from rainier.llm_thesis.schemas import EvidencePack, TradeThesis
from rainier.llm_thesis.service import _tier1_lookup, generate_thesis


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


# ---------------------------------------------------------------------------
# Regression: Tier-1 cache key narrows on (session_name, llm_model)
# Codex P1 — close-of-day rerun must NOT reuse the afternoon thesis.
# ---------------------------------------------------------------------------


def _stub_query_returning(rows: list[tuple[int, dict]]):
    """Build a fake SQLAlchemy query whose .filter().order_by().first() pops rows.

    Each call to .first() returns the first remaining row (or None when empty).
    We capture the .filter args on each call so the test can assert what was
    actually pushed into the WHERE clause.
    """
    captured_filters: list[tuple] = []

    class _Query:
        def query(self, *args, **kwargs):
            return self

        def filter(self, *args, **kwargs):
            captured_filters.append(args)
            return self

        def order_by(self, *args, **kwargs):
            return self

        def first(self):
            return rows.pop(0) if rows else None

    return _Query(), captured_filters


def _patched_get_session(query_obj):
    fake_session = MagicMock()
    fake_session.query = query_obj.query
    cm = MagicMock()
    cm.__enter__.return_value = fake_session
    cm.__exit__.return_value = False
    return cm


def test_tier1_lookup_returns_hit_for_same_session_and_model():
    payload = {**_valid_thesis_dict(), "_session_name": "afternoon"}
    q, _ = _stub_query_returning([(123, payload)])
    cm = _patched_get_session(q)
    with patch("rainier.llm_thesis.service.get_session", return_value=cm):
        result = _tier1_lookup(
            "NVDA",
            date(2026, 5, 7),
            "v1",
            session_name="afternoon",
            llm_model="test-model",
        )
    assert result is not None
    rec_id, output = result
    assert rec_id == 123
    assert output["_session_name"] == "afternoon"


def test_tier1_lookup_misses_when_session_name_differs():
    """Regression: close scan must not reuse afternoon thesis."""
    payload = {**_valid_thesis_dict(), "_session_name": "afternoon"}
    q, _ = _stub_query_returning([(123, payload)])
    cm = _patched_get_session(q)
    with patch("rainier.llm_thesis.service.get_session", return_value=cm):
        result = _tier1_lookup(
            "NVDA",
            date(2026, 5, 7),
            "v1",
            session_name="close",  # different session
            llm_model="test-model",
        )
    # Even though SQL returned a row for this date+symbol+model, the
    # _session_name guard inside the lookup must reject the cross-session reuse.
    assert result is None


def test_tier1_lookup_filter_includes_llm_model():
    """The SELECT must include the llm_model column in its WHERE so a v2
    multi-model A/B never crosses model boundaries at the SQL layer."""
    q, captured_filters = _stub_query_returning([])
    cm = _patched_get_session(q)
    from rainier.core.models import LLMAnalysisRecord

    with patch("rainier.llm_thesis.service.get_session", return_value=cm):
        _tier1_lookup(
            "NVDA",
            date(2026, 5, 7),
            "v1",
            session_name="afternoon",
            llm_model="claude-sonnet-4-6",
        )
    # captured_filters[0] is the tuple of expressions passed to .filter(...)
    assert captured_filters, "filter() was never called"
    expressions = captured_filters[0]
    # Stringified bound expressions include the column names and bound values.
    flat = " ".join(str(e) for e in expressions)
    assert "llm_model" in flat
    # And the model name is bound in the comparator (verified by checking that
    # an llm_model BinaryExpression exists on the LLMAnalysisRecord column).
    has_model_filter = any(
        getattr(getattr(e, "left", None), "key", None) == "llm_model"
        for e in expressions
    )
    assert has_model_filter, (
        f"Expected an llm_model filter expression; got: {expressions!r}"
    )
    # Sanity: ensure the column reference is from the LLMAnalysisRecord table.
    assert LLMAnalysisRecord.llm_model is not None


def test_tier1_lookup_legacy_row_without_session_marker_is_reusable():
    """Backwards compat: rows persisted before the _session_name stamp existed
    have no _session_name key. These should remain reusable rather than
    forcing a costly LLM re-call after the migration."""
    payload = dict(_valid_thesis_dict())  # no _session_name key
    q, _ = _stub_query_returning([(7, payload)])
    cm = _patched_get_session(q)
    with patch("rainier.llm_thesis.service.get_session", return_value=cm):
        result = _tier1_lookup(
            "NVDA",
            date(2026, 5, 7),
            "v1",
            session_name="afternoon",
            llm_model="test-model",
        )
    assert result is not None
    rec_id, _ = result
    assert rec_id == 7


# ---------------------------------------------------------------------------
# generate_thesis end-to-end: Tier-1 hit vs miss across sessions.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_thesis_close_session_misses_afternoon_cache():
    """End-to-end: close scan must call the LLM even though an afternoon row
    exists for the same date+symbol+model+prompt — the session narrows the key.

    We stub _tier1_lookup at the call boundary and assert generate_thesis
    forwards session_name + llm_model so a real DB-backed lookup would receive
    the narrowing key. Then we simulate the lookup returning None for the
    close session and verify the LLM is in fact invoked.
    """
    captured_kwargs: dict = {}

    def _fake_lookup(symbol, scan_date, prompt_version, *, session_name, llm_model):
        captured_kwargs["session_name"] = session_name
        captured_kwargs["llm_model"] = llm_model
        # Afternoon hit, close miss.
        return None if session_name == "close" else (42, _valid_thesis_dict())

    valid_payload = json.dumps(_valid_thesis_dict())
    with patch(
        "rainier.llm_thesis.service._tier1_lookup", side_effect=_fake_lookup
    ), patch(
        "rainier.llm_thesis.service._call_llm",
        return_value=(valid_payload, 100, 200),
    ) as mock_llm, patch(
        "rainier.llm_thesis.service._persist_thesis", return_value=99,
    ) as mock_persist:
        # Close scan: lookup returns None → LLM called → persist called with session.
        thesis, cost, rec_id = await generate_thesis(
            symbol="NVDA",
            scan_date=date(2026, 5, 7),
            session_name="close",
            evidence_provider=_provider(),
            settings=_settings(model="claude-sonnet-4-6", prompt_v="v1"),
            max_usd_remaining=1.0,
        )

    assert thesis is not None
    assert rec_id == 99
    assert cost > 0
    # Lookup received the narrowing key.
    assert captured_kwargs["session_name"] == "close"
    assert captured_kwargs["llm_model"] == "claude-sonnet-4-6"
    # LLM was actually invoked (no afternoon-row reuse).
    mock_llm.assert_called_once()
    # Persist was called with session_name=close so the row is stamped properly.
    persist_kwargs = mock_persist.call_args.kwargs
    assert persist_kwargs["session_name"] == "close"


@pytest.mark.asyncio
async def test_generate_thesis_same_session_hits_cache_no_llm_call():
    """Same date + symbol + prompt + session + model → Tier-1 hit, no LLM call."""

    def _fake_lookup(symbol, scan_date, prompt_version, *, session_name, llm_model):
        # Hit only when session+model match.
        if session_name == "afternoon" and llm_model == "claude-sonnet-4-6":
            return 42, _valid_thesis_dict()
        return None

    with patch(
        "rainier.llm_thesis.service._tier1_lookup", side_effect=_fake_lookup
    ), patch(
        "rainier.llm_thesis.service._call_llm"
    ) as mock_llm, patch(
        "rainier.llm_thesis.service._persist_thesis"
    ) as mock_persist:
        thesis, cost, rec_id = await generate_thesis(
            symbol="NVDA",
            scan_date=date(2026, 5, 7),
            session_name="afternoon",
            evidence_provider=_provider(),
            settings=_settings(model="claude-sonnet-4-6", prompt_v="v1"),
            max_usd_remaining=1.0,
        )

    assert thesis is not None
    assert rec_id == 42
    assert cost == 0.0
    mock_llm.assert_not_called()
    mock_persist.assert_not_called()


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
