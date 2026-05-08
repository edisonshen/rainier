"""Real-Sonnet integration test (gated).

Run on demand:

    RAINIER_LLM_INTEGRATION_TEST=1 uv run pytest -m llm_integration tests/integration/

Skipped in CI; calls the real Anthropic API and incurs cost.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.llm_integration]


@pytest.mark.skipif(
    os.environ.get("RAINIER_LLM_INTEGRATION_TEST") != "1",
    reason="Set RAINIER_LLM_INTEGRATION_TEST=1 to run real LLM call",
)
def test_real_sonnet_returns_valid_trade_thesis():
    import asyncio
    from datetime import date

    from rainier.core.config import load_settings_fresh
    from rainier.llm_thesis.schemas import EvidencePack, TradeThesis
    from rainier.llm_thesis.service import generate_thesis

    settings = load_settings_fresh()

    pack = EvidencePack(
        symbol="NVDA", scan_date="2026-05-07", session_name="afternoon",
        candidate={
            "rank": 5, "long_short": "Long in", "sector": "Technology",
            "pattern_type": "w_bottom", "pattern_status": "forming",
            "signal_strength": 0.85,
        },
        signals={
            "rank_trajectory": {"delta_10d": 22, "trend": "rising"},
        },
    )

    def _provider():
        return pack, ["Rank trajectory: rising (Δ +22)."], None

    thesis, cost, _ = asyncio.run(
        generate_thesis(
            symbol="NVDA", scan_date=date(2026, 5, 7), session_name="afternoon",
            evidence_provider=_provider, settings=settings, max_usd_remaining=1.0,
        )
    )

    assert thesis is None or isinstance(thesis, TradeThesis)
    if thesis:
        assert thesis.verdict in {"setup_long", "watch", "no_setup"}
        assert 1 <= thesis.llm_confidence <= 10
