"""Unit tests for QUScraper._verify_session (API-probe replacement).

After the fix in TASK-PLAN-qu-verify-and-cf-recover-9f39 §2,
_verify_session no longer waits for the .ant-table DOM element (which
only renders after click+Search on the QU SPA). Instead it probes the
same /api/v3/mf100 endpoint the scrape itself uses, via the in-page
fetch transport. _fetch_qu_api already encapsulates the 401/403
reauth-and-retry path (PR #86), so verification just delegates.

These tests pin the new behavior:
  1. API probe returns 200 → no raise, session is good.
  2. signin redirect on initial nav → force login + re-navigate.
  3. _fetch_qu_api propagates RuntimeError (e.g. persistent 401/403
     after reauth) → _verify_session re-raises so the caller sees it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from .conftest import make_mock_page as _make_mock_page
from .conftest import make_scraper as _make_scraper

# ---------------------------------------------------------------------------


class TestVerifySessionAPIProbe:
    """_verify_session probes /api/v3/mf100 via _fetch_qu_api."""

    @pytest.mark.asyncio
    async def test_verify_session_api_200_returns(self):
        """API probe succeeds (200, valid body) → no raise."""
        page, _ = _make_mock_page()
        scraper = _make_scraper()
        scraper._page = page

        # _fetch_qu_api returns the parsed JSON body (any non-raising value
        # means session OK).
        scraper._fetch_qu_api = AsyncMock(return_value={"data": []})

        with patch(
            "rainier.scrapers.qu.scraper.goto_with_retry",
            new_callable=AsyncMock,
        ) as mock_goto:
            await scraper._verify_session()

        # Probed the right endpoint. We don't pin the exact date (it's
        # market_date(now)), but the URL must hit /api/v3/mf100 with
        # top=true and frequency=daily.
        scraper._fetch_qu_api.assert_called_once()
        probe_url = scraper._fetch_qu_api.call_args.args[1]
        assert "/api/v3/mf100" in probe_url
        assert "top=true" in probe_url
        assert "frequency=daily" in probe_url
        # No DOM table wait should have happened.
        page.wait_for_selector.assert_not_called()
        # And we did navigate to the QU SPA at least once.
        mock_goto.assert_called()

    @pytest.mark.asyncio
    async def test_verify_session_signin_redirects_relogs(self):
        """Initial nav lands on /signin → call login(), re-navigate, return.

        Don't fall through to the API probe — the signin branch is the
        pre-auth pathway (saved storage_state was empty/stale at the file
        level, never even got cookies in the request).
        """
        page, page_url = _make_mock_page()

        goto_calls = {"n": 0}

        async def fake_goto(p, url):
            goto_calls["n"] += 1
            if goto_calls["n"] == 1:
                page_url["value"] = (
                    "https://www.quantunicorn.com/signin?next=/products"
                )
            else:
                page_url["value"] = url

        scraper = _make_scraper()
        scraper._page = page
        scraper._fetch_qu_api = AsyncMock(return_value={"data": []})

        with (
            patch(
                "rainier.scrapers.qu.scraper.goto_with_retry",
                new_callable=AsyncMock,
                side_effect=fake_goto,
            ),
            patch(
                "rainier.scrapers.qu.scraper.login",
                new_callable=AsyncMock,
            ) as mock_login,
        ):
            await scraper._verify_session()

        mock_login.assert_called_once_with(page)
        # After signin branch: do NOT fall through to API probe in this
        # call. The next scraper interaction (the actual scrape) will
        # exercise the API + 401/403 path if needed.
        scraper._fetch_qu_api.assert_not_called()
        # Goto was called twice: initial (→ signin) + post-login.
        assert goto_calls["n"] == 2

    @pytest.mark.asyncio
    async def test_verify_session_api_401_propagates_after_reauth_fails(self):
        """_fetch_qu_api already retries on 401/403 once; if it still
        raises, _verify_session must propagate.

        We don't simulate the reauth-and-retry path itself here —
        test_qu_api_retry.py owns that. We just assert that a raise
        from _fetch_qu_api bubbles out of _verify_session unchanged.
        """
        page, _ = _make_mock_page()
        scraper = _make_scraper()
        scraper._page = page

        scraper._fetch_qu_api = AsyncMock(
            side_effect=RuntimeError("qu api 401 after re-auth: ...")
        )

        with patch(
            "rainier.scrapers.qu.scraper.goto_with_retry",
            new_callable=AsyncMock,
        ):
            with pytest.raises(RuntimeError, match="qu api 401 after re-auth"):
                await scraper._verify_session()
