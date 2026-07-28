"""Tests for the xhigh extended-thinking thesis LLM call.

Covers the deterministic thinking config passed to litellm.completion
(temperature==1.0, max_tokens > budget_tokens), response parsing when reasoning
is surfaced separately from the final answer, and cost accounting that bills
thinking tokens as output.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from rainier.llm_thesis.schemas import TradeThesis
from rainier.llm_thesis.service import (
    _FINAL_ANSWER_HEADROOM_TOKENS,
    _call_llm,
    _estimate_cost_usd,
    _parse_thesis,
)


def _valid_thesis_json() -> str:
    return json.dumps(
        {
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
    )


def _mock_resp(content: str, *, completion_tokens: int = 12000, reasoning: str = ""):
    """A minimal litellm-shaped response dict (supports [] access and .get)."""
    message = {"content": content}
    if reasoning:
        # litellm surfaces extended-thinking text on a separate key.
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 2800, "completion_tokens": completion_tokens},
    }


def test_committed_model_actually_engages_thinking_path():
    """Un-mocked guard against a silent no-op: the model in the committed config
    must resolve to the anthropic provider AND support reasoning in litellm's
    registry, or _call_llm falls through to the plain temp=0.2 call and every
    thesis silently degrades to non-thinking. Reads the model from config so a
    rename can't disable the whole feature undetected. No network — litellm's
    provider/registry lookups are local."""
    import litellm

    from rainier.core.config import LLMThesisConfig

    model = LLMThesisConfig().model
    assert litellm.get_llm_provider(model)[1] == "anthropic"
    assert litellm.supports_reasoning(model=model) is True


def test_call_llm_enables_thinking_with_budget_and_temp_one():
    budget = 24000
    with patch("litellm.supports_reasoning", return_value=True), \
            patch("litellm.completion", return_value=_mock_resp("{}")) as mock_comp:
        _call_llm(
            model="claude-sonnet-4-6",
            system_prompt="sys",
            user_prompt="user",
            image_bytes=None,
            thinking_budget_tokens=budget,
        )

    assert mock_comp.call_count == 1
    kwargs = mock_comp.call_args.kwargs
    # Extended thinking enabled at the exact configured budget.
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": budget}
    # Anthropic requires temperature == 1.0 with thinking on.
    assert kwargs["temperature"] == 1.0
    # Anthropic requires max_tokens > budget_tokens (headroom for the answer).
    assert kwargs["max_tokens"] == budget + _FINAL_ANSWER_HEADROOM_TOKENS
    assert kwargs["max_tokens"] > budget


def test_call_llm_budget_scales_max_tokens():
    with patch("litellm.supports_reasoning", return_value=True), \
            patch("litellm.completion", return_value=_mock_resp("{}")) as mock_comp:
        _call_llm(
            model="claude-sonnet-4-6",
            system_prompt="sys",
            user_prompt="user",
            image_bytes=None,
            thinking_budget_tokens=8000,
        )
    kwargs = mock_comp.call_args.kwargs
    assert kwargs["thinking"]["budget_tokens"] == 8000
    assert kwargs["max_tokens"] == 8000 + _FINAL_ANSWER_HEADROOM_TOKENS


def test_call_llm_returns_content_and_token_counts():
    resp = _mock_resp(_valid_thesis_json(), completion_tokens=13500)
    with patch("litellm.supports_reasoning", return_value=True), \
            patch("litellm.completion", return_value=resp):
        text, p_tok, c_tok = _call_llm(
            model="claude-sonnet-4-6",
            system_prompt="sys",
            user_prompt="user",
            image_bytes=None,
            thinking_budget_tokens=24000,
        )
    assert p_tok == 2800
    # completion_tokens is the output total (already includes thinking spend).
    assert c_tok == 13500
    assert json.loads(text)["verdict"] == "setup_long"


def test_thinking_text_does_not_leak_into_parsed_thesis():
    """content carries the final JSON; reasoning is on a separate key. The parsed
    thesis must come only from content, with no thinking text bleeding in."""
    reasoning = "SECRET CHAIN OF THOUGHT — must not appear in the thesis."
    resp = _mock_resp(_valid_thesis_json(), reasoning=reasoning)
    with patch("litellm.supports_reasoning", return_value=True), \
            patch("litellm.completion", return_value=resp):
        text, _, _ = _call_llm(
            model="claude-sonnet-4-6",
            system_prompt="sys",
            user_prompt="user",
            image_bytes=None,
            thinking_budget_tokens=24000,
        )
    assert "SECRET" not in text
    thesis = _parse_thesis(text)
    assert isinstance(thesis, TradeThesis)
    assert thesis.verdict == "setup_long"
    assert "SECRET" not in thesis.paragraph_evidence


def test_call_llm_falls_back_for_non_reasoning_model():
    """A non-reasoning provider must NOT receive thinking/max_tokens (LiteLLM
    would raise UnsupportedParamsError); fall back to the legacy temp=0.2 call."""
    with patch("litellm.supports_reasoning", return_value=False), \
            patch("litellm.completion", return_value=_mock_resp("{}")) as mock_comp:
        _call_llm(
            model="gpt-some-non-reasoning",
            system_prompt="sys",
            user_prompt="user",
            image_bytes=None,
            thinking_budget_tokens=24000,
        )
    kwargs = mock_comp.call_args.kwargs
    assert "thinking" not in kwargs
    assert "max_tokens" not in kwargs
    assert kwargs["temperature"] == 0.2


def test_call_llm_does_not_send_thinking_to_non_anthropic_reasoning_model():
    """The anthropic thinking payload is provider-specific: an OpenAI reasoning
    model (supports_reasoning True, provider != anthropic) must fall back rather
    than receive thinking={...}."""
    with patch("litellm.supports_reasoning", return_value=True), \
            patch("litellm.get_llm_provider", return_value=("o3", "openai", None, None)), \
            patch("litellm.completion", return_value=_mock_resp("{}")) as mock_comp:
        _call_llm(
            model="o3",
            system_prompt="sys",
            user_prompt="user",
            image_bytes=None,
            thinking_budget_tokens=24000,
        )
    kwargs = mock_comp.call_args.kwargs
    assert "thinking" not in kwargs
    assert kwargs["temperature"] == 0.2


def test_cost_estimate_bills_thinking_tokens_as_output():
    """A large completion-token count (thinking folded into output) must be
    billed at the $15/M output rate."""
    # 2800 input, 13500 output (incl. ~11k thinking).
    cost = _estimate_cost_usd(2800, 13500)
    expected = 2800 / 1_000_000 * 3.0 + 13500 / 1_000_000 * 15.0
    assert cost == expected
    # Sanity: a single xhigh ticker lands in the measured ~$0.15-0.40 range.
    assert 0.15 <= cost <= 0.40
