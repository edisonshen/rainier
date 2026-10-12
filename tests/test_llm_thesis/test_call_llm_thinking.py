"""Tests for the xhigh extended-thinking thesis LLM call.

Covers the deterministic thinking config passed to litellm.completion
(temperature==1.0, max_tokens > budget_tokens), response parsing when reasoning
is surfaced separately from the final answer, and cost accounting that bills
thinking tokens as output.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from rainier.llm_thesis.schemas import TradeThesis
from rainier.llm_thesis.service import (
    _FINAL_ANSWER_HEADROOM_TOKENS,
    _call_llm,
    _estimate_cost_usd,
    _parse_thesis,
)

# Pre-4.6 Claude: no adaptive thinking in litellm's catalog -> manual budget path.
_MANUAL_THINKING_MODEL = "claude-sonnet-4-5"


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


def test_call_llm_manual_thinking_model_uses_budget_and_temp_one():
    budget = 24000
    with patch("litellm.supports_reasoning", return_value=True), \
            patch("litellm.completion", return_value=_mock_resp("{}")) as mock_comp:
        _call_llm(
            model=_MANUAL_THINKING_MODEL,
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
            model=_MANUAL_THINKING_MODEL,
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
            model=_MANUAL_THINKING_MODEL,
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
            model=_MANUAL_THINKING_MODEL,
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


def test_call_llm_raises_for_non_reasoning_model():
    """A non-reasoning model can't enable thinking; the thesis pipeline requires
    it, so _call_llm must RAISE (never make a degraded no-thinking call that
    could be persisted + cached as a valid xhigh result)."""
    with patch("litellm.supports_reasoning", return_value=False), \
            patch("litellm.completion", return_value=_mock_resp("{}")) as mock_comp:
        with pytest.raises(RuntimeError, match="extended thinking unavailable"):
            _call_llm(
                model="gpt-some-non-reasoning",
                system_prompt="sys",
                user_prompt="user",
                image_bytes=None,
                thinking_budget_tokens=24000,
            )
    mock_comp.assert_not_called()  # no degraded thesis produced


def test_call_llm_raises_for_non_anthropic_reasoning_model():
    """The anthropic thinking payload is provider-specific: an OpenAI reasoning
    model (supports_reasoning True, provider != anthropic) must RAISE rather than
    silently produce a no-thinking thesis."""
    with patch("litellm.supports_reasoning", return_value=True), \
            patch("litellm.get_llm_provider", return_value=("o3", "openai", None, None)), \
            patch("litellm.completion", return_value=_mock_resp("{}")) as mock_comp:
        with pytest.raises(RuntimeError, match="extended thinking unavailable"):
            _call_llm(
                model="o3",
                system_prompt="sys",
                user_prompt="user",
                image_bytes=None,
                thinking_budget_tokens=24000,
            )
    mock_comp.assert_not_called()


def _capture_anthropic_request_body(**call_kwargs) -> dict:
    """Run _call_llm through litellm's real anthropic request builder and return
    the JSON body it would POST (HTTP layer intercepted; no network/API key)."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    captured: dict = {}

    def _fake_post(*_a, **kw):
        body = kw.get("data") or kw.get("json")
        captured["body"] = json.loads(body) if isinstance(body, (str, bytes)) else body
        raise RuntimeError("request captured")

    with patch.object(HTTPHandler, "post", side_effect=_fake_post), \
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with pytest.raises(Exception, match="request captured"):
            _call_llm(
                system_prompt="sys",
                user_prompt="user",
                image_bytes=b"png",
                **call_kwargs,
            )
    return captured["body"]


def test_committed_opus_config_sends_adaptive_thinking_request():
    """Regression for the Oct 2026 QU100-LLM outage: Opus 5.5 rejects manual
    thinking={"type": "enabled", "budget_tokens": N} and sampling params. The
    request litellm actually builds from the committed config must use adaptive
    thinking + effort, carry no temperature, and keep the max_tokens cap."""
    from rainier.core.config import LLMThesisConfig

    cfg = LLMThesisConfig()
    body = _capture_anthropic_request_body(
        model=cfg.model,
        thinking_budget_tokens=cfg.thinking_budget_tokens,
        thinking_effort=cfg.thinking_effort,
    )
    assert body["model"] == cfg.model
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": cfg.thinking_effort}
    assert "temperature" not in body
    assert "budget_tokens" not in body["thinking"]
    assert body["max_tokens"] == cfg.thinking_budget_tokens + _FINAL_ANSWER_HEADROOM_TOKENS


def test_settings_yaml_effort_is_accepted_by_litellm_for_thesis_model():
    """The deployed settings.yaml (not just the pydantic default) must yield a
    request litellm will build — it validates effort client-side."""
    from rainier.core.config import load_settings

    cfg = load_settings().llm_thesis
    body = _capture_anthropic_request_body(
        model=cfg.model,
        thinking_budget_tokens=cfg.thinking_budget_tokens,
        thinking_effort=cfg.thinking_effort,
    )
    assert body["output_config"] == {"effort": cfg.thinking_effort}


def test_manual_thinking_model_request_body_keeps_budget_form():
    body = _capture_anthropic_request_body(
        model=_MANUAL_THINKING_MODEL, thinking_budget_tokens=8000,
    )
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 8000}
    assert body["temperature"] == 1.0
    assert "output_config" not in body


def test_cost_estimate_bills_thinking_tokens_as_output():
    """A large completion-token count (thinking folded into output) must be
    billed at the model's output rate (Opus 5.5: $20/M)."""
    # 2800 input, 13500 output (incl. ~11k thinking).
    cost = _estimate_cost_usd(2800, 13500, "claude-opus-5-5")
    expected = 2800 / 1_000_000 * 4.0 + 13500 / 1_000_000 * 20.0
    assert cost == pytest.approx(expected)
    # Sanity: a single xhigh ticker lands in the ~$0.20-0.55 range.
    assert 0.20 <= cost <= 0.55


def test_cost_estimate_uses_catalog_rates_per_model():
    sonnet = _estimate_cost_usd(1_000_000, 1_000_000, "claude-sonnet-4-6")
    assert sonnet == pytest.approx(3.0 + 15.0)


def test_cost_estimate_unknown_model_falls_back_to_default_rates():
    cost = _estimate_cost_usd(1_000_000, 1_000_000, "not-a-real-model")
    assert cost == pytest.approx(4.0 + 20.0)
