"""Spec: Claude picks which screened names get a full thesis.

Scenario: the screener returns 50 "Long in" candidates; the scheduled
afternoon scan asks the LLM to pick 5 of them, theses exactly those picks
(even ones ranked outside the Discord top-20), and posts a thesis embed per
pick with the LLM's reason. A selector outage (e.g. Anthropic out of credit)
must fall back to the screener's composite top-5, never drop the thesis set.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from rainier.llm_thesis.service import ThesisBatch
from tests.test_pipeline.test_post_scrape import _candidates, _make_settings


def _llm_reply(picks: list[tuple[str, str]], *, fenced: bool = False) -> dict:
    body = json.dumps({"picks": [{"symbol": s, "reason": r} for s, r in picks]})
    if fenced:
        body = f"```json\n{body}\n```"
    return {
        "choices": [{"message": {"content": body}}],
        "usage": {"prompt_tokens": 4000, "completion_tokens": 400},
    }


def _thesis(symbol: str, reason: str | None) -> dict:
    thesis = {
        "verdict": "setup_long",
        "setup_quality": 7,
        "llm_confidence": 7,
        "paragraph_radar": f"{symbol} on radar.",
        "paragraph_evidence": "Evidence.",
        "paragraph_invalidation": "Invalidation.",
        "risks": [],
        "watch_items": [],
        "evidence_used": [],
        "signals_used": [],
        "patterns_in_chart_not_in_indicators": "none",
    }
    if reason:
        thesis["_selection_reason"] = reason
    return thesis


def _fake_compute(targets, _ohlcv, **kwargs):
    reasons = kwargs.get("selection_reasons") or {}
    return ThesisBatch({c.symbol: _thesis(c.symbol, reasons.get(c.symbol)) for c in targets}, {})


def _run_scan(llm_completion, *, selection_enabled: bool = True):
    """Run the real post-scrape pipeline + real Discord renderer; mock only the
    screener, DB writes, the thesis LLM batch, and the network edges."""
    settings = _make_settings(llm_selection_enabled=selection_enabled)
    posted_embeds: list[dict] = []

    with (
        patch(
            "rainier.pipeline.post_scrape.screen_stocks",
            return_value=(_candidates(50), {}),
        ),
        patch("rainier.pipeline.post_scrape.persist_screened_stocks"),
        patch(
            "rainier.pipeline.post_scrape.compute_theses_and_persist",
            side_effect=_fake_compute,
        ) as mock_compute,
        patch("litellm.completion", side_effect=llm_completion) as mock_llm,
        patch("rainier.alerts.discord.httpx.post"),
        patch(
            "rainier.alerts.discord._post_thesis_embed",
            side_effect=lambda **kw: posted_embeds.append(kw["embed"]),
        ),
        patch("rainier.pipeline.post_scrape.send_thesis_failure_alert"),
    ):
        from rainier.pipeline.post_scrape import run_post_scrape_pipeline

        run_post_scrape_pipeline(settings, "afternoon")

    return mock_compute, mock_llm, posted_embeds


def _embed_symbol(embed: dict) -> str:
    return embed["title"].split(" · ")[1]


def _picked_reason(embed: dict) -> str | None:
    for field in embed["fields"]:
        if field["name"] == "PICKED BY LLM":
            return field["value"]
    return None


def test_llm_picks_drive_thesis_and_discord_even_outside_top_20():
    reply = _llm_reply([
        ("S029", "fresh w_bottom breakout near entry"),
        ("s003", "rank momentum +12"),
        ("BOGUS", "not in the table"),
        ("S003", "duplicate"),
        ("S010", "capital inflow, tight stop"),
        ("S040", "sector diversifier"),
        ("S007", "clean flag"),
    ])
    mock_compute, mock_llm, embeds = _run_scan(lambda **_: reply)

    # The selector saw the full 50-name pool, not the Discord top-20.
    prompt = mock_llm.call_args.kwargs["messages"][1]["content"]
    assert "50. S049" in prompt

    targets = [c.symbol for c in mock_compute.call_args.args[0]]
    assert targets == ["S029", "S003", "S010", "S040", "S007"]
    kwargs = mock_compute.call_args.kwargs
    assert kwargs["already_spent_usd"] > 0  # selector spend counts toward the cap
    assert kwargs["selection_reasons"]["S029"] == "fresh w_bottom breakout near entry"

    assert [_embed_symbol(e) for e in embeds] == targets
    assert _picked_reason(embeds[0]) == "fresh w_bottom breakout near entry"


def test_short_llm_list_is_filled_in_screener_order():
    reply = _llm_reply([("S030", "only pick")], fenced=True)
    mock_compute, _, embeds = _run_scan(lambda **_: reply)

    targets = [c.symbol for c in mock_compute.call_args.args[0]]
    assert targets == ["S030", "S000", "S001", "S002", "S003"]
    assert _picked_reason(embeds[0]) == "only pick"
    assert _picked_reason(embeds[1]) is None  # screener fill has no LLM reason


@pytest.mark.parametrize(
    "llm_completion",
    [
        pytest.param(
            lambda **_: (_ for _ in ()).throw(
                RuntimeError("Your credit balance is too low to access the Anthropic API")
            ),
            id="provider-error",
        ),
        pytest.param(
            lambda **_: {"choices": [{"message": {"content": "I like S029"}}], "usage": {}},
            id="not-json",
        ),
        pytest.param(lambda **_: _llm_reply([("ZZZZ", "unknown")]), id="no-valid-symbols"),
    ],
)
def test_selector_failure_falls_back_to_screener_top_5(llm_completion):
    mock_compute, _, embeds = _run_scan(llm_completion)

    targets = [c.symbol for c in mock_compute.call_args.args[0]]
    assert targets == ["S000", "S001", "S002", "S003", "S004"]
    assert mock_compute.call_args.kwargs["selection_reasons"] == {}
    assert [_embed_symbol(e) for e in embeds] == targets


def test_selection_disabled_uses_composite_top_5_without_llm_call():
    mock_compute, mock_llm, _ = _run_scan(lambda **_: _llm_reply([]), selection_enabled=False)

    mock_llm.assert_not_called()
    targets = [c.symbol for c in mock_compute.call_args.args[0]]
    assert targets == ["S000", "S001", "S002", "S003", "S004"]
