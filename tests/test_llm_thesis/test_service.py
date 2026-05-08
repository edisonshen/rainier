"""Tests for generate_thesis: Tier 1 cache, Tier 2 LLM, retries, kill switch, persist."""

from __future__ import annotations

import asyncio
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


def _stub_query_returning(rows: list[tuple]):
    """Build a fake SQLAlchemy query whose .filter().order_by().all() returns rows.

    Each row may be a 2-tuple (id, output) — legacy form — or a 3-tuple
    (id, output, session_name) matching the post-PR2 SELECT shape. The stub
    normalizes 2-tuples by appending ``None`` for session_name so legacy
    fixtures keep working.

    We capture the .filter args on each call so the test can assert what was
    actually pushed into the WHERE clause.
    """
    captured_filters: list[tuple] = []
    normalized: list[tuple] = []
    for r in rows:
        if len(r) == 2:
            normalized.append((r[0], r[1], None))
        else:
            normalized.append(tuple(r))

    class _Query:
        def query(self, *args, **kwargs):
            return self

        def filter(self, *args, **kwargs):
            captured_filters.append(args)
            return self

        def order_by(self, *args, **kwargs):
            return self

        def first(self):
            return normalized.pop(0) if normalized else None

        def all(self):
            out = list(normalized)
            normalized.clear()
            return out

    return _Query(), captured_filters


def _patched_get_session(query_obj):
    fake_session = MagicMock()
    fake_session.query = query_obj.query
    cm = MagicMock()
    cm.__enter__.return_value = fake_session
    cm.__exit__.return_value = False
    return cm


def test_candidate_to_pattern_signal_round_trips_levels():
    """PR2 carry-over P2 #4: chart_export receives a real PatternSignal so
    the chart sent to the LLM gets the entry / SL / target overlays. We
    reconstruct it from the flat fields on StockCandidate.
    """
    from rainier.core.types import PatternSignal, StockCandidate
    from rainier.llm_thesis.service import _candidate_to_pattern_signal

    cand = StockCandidate(
        symbol="NVDA", rank=5, rank_change=1, long_short="Long in",
        capital_flow_direction="+", sector="Technology", signal_strength=0.8,
        pattern_type="w_bottom", pattern_direction="bullish",
        pattern_status="forming", pattern_confidence=0.72,
        entry_price=120.5, stop_loss=115.0, target_price=132.0,
        rr_ratio=2.1, volume_confirmed=True,
    )
    p = _candidate_to_pattern_signal(cand)
    assert isinstance(p, PatternSignal)
    assert p.pattern_type == "w_bottom"
    assert p.entry_price == 120.5
    assert p.stop_loss == 115.0
    assert p.target_wave1 == 132.0
    assert p.confidence == 0.72


def test_candidate_to_pattern_signal_returns_none_when_no_pattern():
    """No pattern fields → no overlay. Chart still renders cleanly."""
    from rainier.core.types import StockCandidate
    from rainier.llm_thesis.service import _candidate_to_pattern_signal

    cand = StockCandidate(
        symbol="NVDA", rank=5, rank_change=0, long_short="Long in",
        capital_flow_direction="+", sector="Technology", signal_strength=0.5,
    )
    assert _candidate_to_pattern_signal(cand) is None


@pytest.mark.asyncio
async def test_compute_theses_async_passes_pattern_signal_to_chart():
    """Cache miss must call render_chart_png with a reconstructed PatternSignal,
    not None — fix #4. Captures the kwarg value passed."""
    from datetime import date as _date

    from rainier.core.types import StockCandidate
    from rainier.llm_thesis.service import _compute_theses_async

    cand = StockCandidate(
        symbol="NVDA", rank=5, rank_change=0, long_short="Long in",
        capital_flow_direction="+", sector="Technology", signal_strength=0.8,
        pattern_type="bull_flag", pattern_direction="bullish",
        pattern_status="forming", pattern_confidence=0.7,
        entry_price=200.0, stop_loss=190.0, target_price=215.0,
        rr_ratio=1.5,
    )

    captured: dict = {}

    def _capture_chart(symbol, df, pattern=None, **kwargs):
        captured["pattern"] = pattern
        return b"png-bytes", "deadbeef"

    async def _fake_assemble(*args, **kwargs):
        from rainier.llm_thesis.schemas import EvidencePack
        pack = EvidencePack(
            symbol="NVDA", scan_date=_date(2026, 5, 7).isoformat(),
            session_name="afternoon", candidate={"rank": 5}, signals={},
        )
        return pack, []

    async def _fake_generate(**kwargs):
        # Force the cache-miss path by invoking the provider, just like the
        # real generate_thesis does (via asyncio.to_thread). The provider
        # internally calls asyncio.run, which only works off-loop, so we
        # match production by offloading to a worker thread here too.
        provider = kwargs["evidence_provider"]
        await asyncio.to_thread(provider)
        return TradeThesis.model_validate(_valid_thesis_dict()), 0.05, 99

    with (
        patch(
            "rainier.llm_thesis.service.generate_thesis", side_effect=_fake_generate,
        ),
        patch(
            "rainier.llm_thesis.chart_export.render_chart_png",
            side_effect=_capture_chart,
        ),
        patch(
            "rainier.llm_thesis.service.assemble_evidence",
            side_effect=_fake_assemble,
        ),
        patch("rainier.llm_thesis.service.update_with_thesis"),
    ):
        await _compute_theses_async(
            [cand], {"NVDA": MagicMock()},
            scan_date=_date(2026, 5, 7),
            session_name="afternoon",
            settings=_settings(),
        )

    assert captured.get("pattern") is not None, (
        "render_chart_png was called with pattern=None — fix #4 regression"
    )
    p = captured["pattern"]
    assert p.pattern_type == "bull_flag"
    assert p.entry_price == 200.0


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


def test_tier1_lookup_invalidates_when_signals_used_differs():
    """PR2 carry-over P2 #2: toggling a signal off in YAML must invalidate
    same-day cached theses. The cached row was produced with a stale signal
    set; reusing it would feed the LLM verdict drift into the next scan.
    """
    payload = {
        **_valid_thesis_dict(),
        "_session_name": "afternoon",
        "signals_used": ["rank_trajectory", "fundamentals"],  # cached with two
    }
    q, _ = _stub_query_returning([(123, payload, "afternoon")])
    cm = _patched_get_session(q)
    with patch("rainier.llm_thesis.service.get_session", return_value=cm):
        result = _tier1_lookup(
            "NVDA",
            date(2026, 5, 7),
            "v1",
            session_name="afternoon",
            llm_model="test-model",
            # Current enabled set differs — fundamentals was just toggled off.
            enabled_signals=["rank_trajectory"],
        )
    assert result is None, (
        "Expected cache miss because the cached signals_used differs from "
        "the current enabled-signals set."
    )


def test_tier1_lookup_hits_when_signals_used_matches():
    """Same set, just permuted — cache should still hit."""
    payload = {
        **_valid_thesis_dict(),
        "_session_name": "afternoon",
        "signals_used": ["fundamentals", "rank_trajectory"],
    }
    q, _ = _stub_query_returning([(123, payload, "afternoon")])
    cm = _patched_get_session(q)
    with patch("rainier.llm_thesis.service.get_session", return_value=cm):
        result = _tier1_lookup(
            "NVDA",
            date(2026, 5, 7),
            "v1",
            session_name="afternoon",
            llm_model="test-model",
            enabled_signals=["rank_trajectory", "fundamentals"],
        )
    assert result is not None
    rec_id, _ = result
    assert rec_id == 123


def test_tier1_lookup_session_column_takes_precedence_over_jsonb_legacy():
    """[P3] fix #6: with the new session_name column populated, the lookup
    must filter on the column directly. A row whose JSONB stamp says one
    session but whose session_name column says another should match the
    column (the JSONB stamp is informational only on PR2+ rows).
    """
    payload = {
        **_valid_thesis_dict(),
        "_session_name": "ignored-jsonb",  # legacy field still present
    }
    # Column-row says "afternoon"; the lookup is for "afternoon" too.
    q, _ = _stub_query_returning([(7, payload, "afternoon")])
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


def test_tier1_lookup_skips_cross_session_row_continues_search():
    """[P3] fix #6 regression: a cross-session row earlier in the partition
    must NOT block a same-session row from being found. The old code did
    `.first()` + post-filter, returning None whenever the first row was the
    wrong session. The new code iterates all rows and returns the first
    matching one.
    """
    rows = [
        # Most-recent row is the WRONG session (close).
        (200, {**_valid_thesis_dict(), "_session_name": "close"}, "close"),
        # Earlier same-session row should win.
        (100, {**_valid_thesis_dict(), "_session_name": "afternoon"}, "afternoon"),
    ]
    q, _ = _stub_query_returning(rows)
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
    assert rec_id == 100


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

    def _fake_lookup(
        symbol, scan_date, prompt_version, *, session_name, llm_model,
        enabled_signals=None,
    ):
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

    def _fake_lookup(
        symbol, scan_date, prompt_version, *, session_name, llm_model,
        enabled_signals=None,
    ):
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


# ---------------------------------------------------------------------------
# PR2 carry-over P2 #1: _compute_theses_async must defer evidence assembly
# until generate_thesis confirms a Tier-1 miss. A cache hit must skip the
# chart export + signal compute calls entirely.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compute_theses_async_cache_hit_skips_chart_and_assembly():
    """Cache-hit path must not invoke render_chart_png or assemble_evidence.

    Regression: a previous version of _compute_theses_async pre-resolved the
    evidence pack BEFORE handing it to generate_thesis, so even a Tier-1 hit
    paid for chart export + every signal compute. PR2 carry-over P2 #1 fixes
    this by deferring the evidence assembly behind a closure that only runs
    when generate_thesis decides the cache missed.
    """
    from datetime import date as _date
    from unittest.mock import AsyncMock as _AsyncMock

    from rainier.core.types import StockCandidate
    from rainier.llm_thesis.service import _compute_theses_async

    cand = StockCandidate(
        symbol="NVDA", rank=5, rank_change=0, long_short="Long in",
        capital_flow_direction="+", sector="Technology", signal_strength=0.8,
    )

    cached_thesis_dict = _valid_thesis_dict()
    chart_mock = MagicMock()
    assemble_mock = _AsyncMock()

    async def _fake_generate(**kwargs):
        # Simulate the Tier-1 hit — generate_thesis returns the cached thesis
        # without ever calling its evidence_provider.
        return TradeThesis.model_validate(cached_thesis_dict), 0.0, 42

    with (
        patch(
            "rainier.llm_thesis.service.generate_thesis",
            side_effect=_fake_generate,
        ),
        patch(
            "rainier.llm_thesis.chart_export.render_chart_png", chart_mock
        ),
        patch(
            "rainier.llm_thesis.service.assemble_evidence", assemble_mock
        ),
        patch("rainier.llm_thesis.service.update_with_thesis"),
    ):
        result = await _compute_theses_async(
            [cand], {"NVDA": MagicMock()},
            scan_date=_date(2026, 5, 7),
            session_name="afternoon",
            settings=_settings(),
        )

    assert "NVDA" in result
    # The cache-hit path must NOT have asked for the chart or run any signal.
    chart_mock.assert_not_called()
    assemble_mock.assert_not_called()


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
