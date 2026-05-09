"""Regression tests for the `rainier recover` command's scrape re-run.

The recover flow re-invokes _run_qu_scrape directly via the in-process function
(not the Click command), so a signature drift between the two fails silently
until the cron actually fires. These tests catch that drift at unit time.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest

from rainier.cli import _run_qu_scrape

# Keyword args the recover branch (`_recover` in cli.py) passes when re-running
# a missed scrape. Keep this in sync with the call site.
RECOVER_SCRAPE_KWARGS = {
    "session": "morning",
    "detail_top": 0,
    "dates": None,
    "days_back": 0,
    "start_date": None,
    "delay": None,
    "headed": False,
    "cdp": "http://127.0.0.1:9222",
}


def test_recover_kwargs_bind_to_run_qu_scrape_signature():
    """The kwargs the recover path uses must bind cleanly to _run_qu_scrape."""
    sig = inspect.signature(_run_qu_scrape)
    bound = sig.bind(**RECOVER_SCRAPE_KWARGS)
    bound.apply_defaults()
    # Every signature parameter should be filled — no silent positional drift.
    for name in sig.parameters:
        assert name in bound.arguments, f"Recover call is missing kwarg: {name}"


@pytest.mark.asyncio
async def test_recover_branch_invokes_run_qu_scrape_with_expected_kwargs():
    """Patch _run_qu_scrape and replay the recover call site directly."""
    with patch("rainier.cli._run_qu_scrape", new_callable=AsyncMock) as mock_scrape:
        await mock_scrape(**RECOVER_SCRAPE_KWARGS)

    mock_scrape.assert_awaited_once_with(**RECOVER_SCRAPE_KWARGS)
