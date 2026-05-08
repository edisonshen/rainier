"""Tests for TradeThesis + EvidencePack + compute_input_hash."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rainier.llm_thesis.schemas import EvidencePack, TradeThesis, compute_input_hash


def _valid_thesis_kwargs(**overrides):
    defaults = dict(
        verdict="setup_long",
        setup_quality=7,
        llm_confidence=7,
        paragraph_radar="Why on radar.",
        paragraph_evidence="Evidence text.",
        paragraph_invalidation="Invalidation rules.",
        risks=["earnings risk"],
        watch_items=["volume on neckline"],
        evidence_used=["pattern", "rank_trajectory"],
        signals_used=["rank_trajectory", "fundamentals"],
        patterns_in_chart_not_in_indicators=["narrowing volume on right shoulder"],
    )
    defaults.update(overrides)
    return defaults


def test_round_trip_keeps_fields():
    t = TradeThesis(**_valid_thesis_kwargs())
    payload = t.model_dump_json()
    parsed = TradeThesis.model_validate_json(payload)
    assert parsed.verdict == "setup_long"
    assert parsed.llm_confidence == 7


def test_setup_short_is_rejected():
    with pytest.raises(ValidationError):
        TradeThesis(**_valid_thesis_kwargs(verdict="setup_short"))


def test_paragraph_max_length_enforced():
    too_long = "x" * 401
    with pytest.raises(ValidationError):
        TradeThesis(**_valid_thesis_kwargs(paragraph_radar=too_long))


def test_patterns_field_accepts_list_of_strings():
    t = TradeThesis(
        **_valid_thesis_kwargs(
            patterns_in_chart_not_in_indicators=["a", "b"]
        )
    )
    assert t.patterns_in_chart_not_in_indicators == ["a", "b"]


def test_patterns_field_accepts_literal_none():
    t = TradeThesis(
        **_valid_thesis_kwargs(patterns_in_chart_not_in_indicators="none")
    )
    assert t.patterns_in_chart_not_in_indicators == "none"


def test_patterns_field_rejects_empty_list():
    with pytest.raises(ValidationError):
        TradeThesis(
            **_valid_thesis_kwargs(patterns_in_chart_not_in_indicators=[])
        )


def test_patterns_field_rejects_other_string():
    with pytest.raises(ValidationError):
        TradeThesis(
            **_valid_thesis_kwargs(patterns_in_chart_not_in_indicators="nope")
        )


def test_patterns_field_max_5():
    with pytest.raises(ValidationError):
        TradeThesis(
            **_valid_thesis_kwargs(
                patterns_in_chart_not_in_indicators=["a", "b", "c", "d", "e", "f"]
            )
        )


def test_compute_input_hash_deterministic():
    pack = EvidencePack(
        symbol="NVDA", scan_date="2026-05-07", session_name="afternoon",
        candidate={"rank": 5}, signals={"rank_trajectory": {"delta_10d": 25}},
    )
    h1 = compute_input_hash(pack, b"image-bytes")
    h2 = compute_input_hash(pack, b"image-bytes")
    assert h1 == h2
    assert len(h1) == 64


def test_compute_input_hash_changes_with_image():
    pack = EvidencePack(
        symbol="NVDA", scan_date="2026-05-07", session_name="afternoon",
        candidate={"rank": 5},
    )
    assert compute_input_hash(pack, b"a") != compute_input_hash(pack, b"b")


def test_compute_input_hash_changes_with_signals():
    base = EvidencePack(
        symbol="NVDA", scan_date="2026-05-07", session_name="afternoon",
        candidate={"rank": 5}, signals={"rank_trajectory": {"delta_10d": 25}},
    )
    other = EvidencePack(
        symbol="NVDA", scan_date="2026-05-07", session_name="afternoon",
        candidate={"rank": 5}, signals={"rank_trajectory": {"delta_10d": 26}},
    )
    assert compute_input_hash(base, b"img") != compute_input_hash(other, b"img")


def test_compute_input_hash_handles_no_image():
    pack = EvidencePack(
        symbol="NVDA", scan_date="2026-05-07", session_name="afternoon",
        candidate={"rank": 5},
    )
    h = compute_input_hash(pack, None)
    assert len(h) == 64
