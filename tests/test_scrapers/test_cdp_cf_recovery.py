"""Unit tests for _cdp_ensure_auth Cloudflare-challenge wait/recover path.

Per TASK-PLAN-qu-verify-and-cf-recover-9f39 §3, the immediate "raise
RuntimeError on title == 'Just a moment...'" bail is replaced with a
25-second wait_for_function on the document title. CF JS challenges
typically resolve in 5-10s and Playwright's Chromium executes them
automatically; we just have to wait, not bail.

Test matrix:
  1. CF title detected, then clears within budget → no raise.
     wait_for_function called with timeout=25000.
  2. CF title detected, wait_for_function times out (challenge stuck) →
     raise RuntimeError with the new "stuck for >25s" message + QU URL.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from .conftest import make_mock_page as _make_mock_page
from .conftest import make_scraper


def _make_scraper():
    return make_scraper(is_cdp=True)


# ---------------------------------------------------------------------------


class TestCDPCFRecovery:
    """_cdp_ensure_auth waits up to 25s for CF challenge to clear."""

    @pytest.mark.asyncio
    async def test_cdp_cf_challenge_clears_in_time(self):
        """Title flips from 'Just a moment...' to 'QuantUnicorn' within 25s.

        wait_for_function is called with timeout=25000; no raise.
        """
        # First title call (CF detect) → "Just a moment..."; subsequent
        # calls (post-wait verification) → "QuantUnicorn".
        page, _ = _make_mock_page(titles=["Just a moment...", "QuantUnicorn"])

        scraper = _make_scraper()
        scraper._page = page

        with (
            patch(
                "rainier.scrapers.qu.scraper.get_session_path",
                return_value="/fake",
            ),
            patch(
                "rainier.scrapers.qu.scraper.goto_with_retry",
                new_callable=AsyncMock,
            ),
            patch(
                "rainier.scrapers.qu.scraper.login",
                new_callable=AsyncMock,
            ),
            patch("pathlib.Path.exists", return_value=False),
        ):
            # No raise expected.
            await scraper._cdp_ensure_auth()

        # wait_for_function must be called for the CF title.
        page.wait_for_function.assert_called_once()
        call_kwargs = page.wait_for_function.call_args.kwargs
        assert call_kwargs.get("timeout") == 25000
        # The JS predicate must check that the title no longer contains
        # "just a moment" (case-insensitive).
        js_arg = page.wait_for_function.call_args.args[0]
        assert "just a moment" in js_arg.lower()

    @pytest.mark.asyncio
    async def test_cdp_cf_challenge_stuck_raises_clearly(self):
        """wait_for_function times out → RuntimeError mentioning
        '>25s' and the QU URL so the operator knows how to recover.
        """
        page, _ = _make_mock_page(titles=["Just a moment..."])

        # The wait raises (Playwright TimeoutError surfaces as a
        # synchronous Exception subclass; we use a plain Exception here
        # since the fix's except clause catches broadly).
        page.wait_for_function = AsyncMock(
            side_effect=Exception("Timeout 25000ms exceeded")
        )

        scraper = _make_scraper()
        scraper._page = page

        with (
            patch(
                "rainier.scrapers.qu.scraper.get_session_path",
                return_value="/fake",
            ),
            patch(
                "rainier.scrapers.qu.scraper.goto_with_retry",
                new_callable=AsyncMock,
            ),
            patch(
                "rainier.scrapers.qu.scraper.login",
                new_callable=AsyncMock,
            ),
            patch("pathlib.Path.exists", return_value=False),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await scraper._cdp_ensure_auth()

        msg = str(exc_info.value)
        assert "stuck for >25s" in msg
        assert "quantunicorn.com/products" in msg
