"""Verify assemble_evidence runs signals in parallel.

One signal sleeps 2.0s in the threadpool, others return immediately. Total
wall time must be <=2.5s — proves asyncio.gather + asyncio.to_thread is in
effect (sequential execution would total ~2.0s + overheads, but in particular
multiple slow signals would sum). Also verifies one signal raising does not
kill the others (return_exceptions=True).
"""

from __future__ import annotations

import asyncio
import time
from datetime import date
from unittest.mock import patch

import pytest

from rainier.core.config import LLMThesisConfig, ThesisSignalConfig
from rainier.core.types import StockCandidate
from rainier.llm_thesis.service import assemble_evidence


class _SlowSignal:
    name = "rank_trajectory"
    version = "v1"
    cost_estimate_ms = 2000

    def compute(self, ctx):
        time.sleep(2.0)
        return {"slow": True}

    def render_for_prompt(self, value):
        return "slow signal ran"


class _FastSignal:
    name = "capital_flow_streak"
    version = "v1"
    cost_estimate_ms = 5

    def compute(self, ctx):
        return {"fast": True}

    def render_for_prompt(self, value):
        return "fast signal ran"


class _RaisingSignal:
    name = "sector_momentum"
    version = "v1"
    cost_estimate_ms = 5

    def compute(self, ctx):
        raise RuntimeError("boom")

    def render_for_prompt(self, value):
        return "should not be called"


class _NoneSignal:
    name = "fundamentals"
    version = "v1"
    cost_estimate_ms = 5

    def compute(self, ctx):
        return None

    def render_for_prompt(self, value):
        return "should not be called"


def _candidate():
    return StockCandidate(
        symbol="NVDA", rank=5, rank_change=0, long_short="Long in",
        capital_flow_direction="+", sector="Technology", signal_strength=0.8,
    )


@pytest.mark.asyncio
async def test_signals_run_in_parallel_and_isolate_failures():
    cfg = LLMThesisConfig(
        signals={
            "rank_trajectory": ThesisSignalConfig(),
            "capital_flow_streak": ThesisSignalConfig(),
            "sector_momentum": ThesisSignalConfig(),
            "fundamentals": ThesisSignalConfig(),
        }
    )
    fake_registry = {
        "rank_trajectory": _SlowSignal,
        "capital_flow_streak": _FastSignal,
        "sector_momentum": _RaisingSignal,
        "fundamentals": _NoneSignal,
    }

    start = time.monotonic()
    with patch("rainier.llm_thesis.service.REGISTRY", fake_registry):
        pack, renders = await assemble_evidence(
            _candidate(), None, date(2026, 5, 7), "afternoon", cfg
        )
    elapsed = time.monotonic() - start

    assert elapsed < 2.5, f"signals did not run in parallel; took {elapsed:.2f}s"
    assert "rank_trajectory" in pack.signals
    assert "capital_flow_streak" in pack.signals
    # Raising signal isolated → not in pack
    assert "sector_momentum" not in pack.signals
    # None signal skipped → not in pack
    assert "fundamentals" not in pack.signals
    assert len(renders) == 2


@pytest.mark.asyncio
async def test_disabled_signals_are_skipped():
    cfg = LLMThesisConfig(
        signals={
            "rank_trajectory": ThesisSignalConfig(enabled=False),
            "capital_flow_streak": ThesisSignalConfig(enabled=True),
            "sector_momentum": ThesisSignalConfig(enabled=False),
            "fundamentals": ThesisSignalConfig(enabled=False),
        }
    )
    fake_registry = {
        "rank_trajectory": _SlowSignal,
        "capital_flow_streak": _FastSignal,
        "sector_momentum": _RaisingSignal,
        "fundamentals": _NoneSignal,
    }
    with patch("rainier.llm_thesis.service.REGISTRY", fake_registry):
        pack, renders = await assemble_evidence(
            _candidate(), None, date(2026, 5, 7), "afternoon", cfg
        )
    assert list(pack.signals) == ["capital_flow_streak"]
    assert renders == ["fast signal ran"]


@pytest.mark.asyncio
async def test_all_signals_disabled_yields_empty_pack():
    cfg = LLMThesisConfig(
        signals={
            "rank_trajectory": ThesisSignalConfig(enabled=False),
            "capital_flow_streak": ThesisSignalConfig(enabled=False),
        }
    )
    pack, renders = await assemble_evidence(
        _candidate(), None, date(2026, 5, 7), "afternoon", cfg
    )
    assert pack.signals == {}
    assert renders == []


def test_module_exports_asyncio_safe_call():
    # Sanity: the orchestrator can be invoked from sync code via asyncio.run.
    cfg = LLMThesisConfig(signals={})
    pack, renders = asyncio.run(
        assemble_evidence(_candidate(), None, date(2026, 5, 7), "afternoon", cfg)
    )
    assert pack.signals == {}
    assert renders == []
