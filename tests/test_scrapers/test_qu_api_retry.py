"""Unit tests for QU in-page-fetch 401/403 retry-with-reauth helper.

Regression coverage for the PR #84 follow-up (TASK-PLAN-qu-401-403-recovery-9489):
the in-page-fetch path lost the `_recover_if_signin` mid-session auto-recovery
that the DOM path had. The helper under test is `QUScraper._fetch_qu_api`,
which inspects `page.evaluate(QU100_FETCH_JS, ...)` results and re-auths +
retries ONCE on 401/403. Bails on second auth failure or any other 4xx/5xx.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .conftest import make_scraper as _make_scraper

# ---------------------------------------------------------------------------
# Unit tests on QUScraper._fetch_qu_api
# ---------------------------------------------------------------------------


class TestFetchQuApi:
    """Direct unit tests of the retry-with-reauth wrapper."""

    async def test_200_returns_body(self):
        """Status 200 → returns the parsed body, no reauth, single fetch."""
        scraper = _make_scraper()
        body = {"data": [{"ticker": "TSLA"}]}
        page = MagicMock()
        page.evaluate = AsyncMock(
            return_value={"status": 200, "body": body, "text_excerpt": ""}
        )

        # Spy on _reauth so we can assert it was NOT called.
        scraper._reauth = AsyncMock()  # type: ignore[assignment]

        out = await scraper._fetch_qu_api(page, "https://x/api")

        assert out == body
        assert page.evaluate.await_count == 1
        scraper._reauth.assert_not_awaited()

    async def test_401_reauths_and_retries_once_then_succeeds(self):
        """First call 401 → reauth → second call 200 → returns body."""
        scraper = _make_scraper()
        body = {"data": [{"ticker": "NVDA"}]}
        page = MagicMock()
        page.evaluate = AsyncMock(
            side_effect=[
                {"status": 401, "body": None, "text_excerpt": "unauthorized"},
                {"status": 200, "body": body, "text_excerpt": ""},
            ]
        )
        scraper._reauth = AsyncMock()  # type: ignore[assignment]

        out = await scraper._fetch_qu_api(page, "https://x/api")

        assert out == body
        assert page.evaluate.await_count == 2
        scraper._reauth.assert_awaited_once()

    async def test_403_reauths_and_retries_once_then_succeeds(self):
        """First call 403 → reauth → second call 200 → returns body."""
        scraper = _make_scraper()
        body = {"data": [{"ticker": "AAPL"}]}
        page = MagicMock()
        page.evaluate = AsyncMock(
            side_effect=[
                {"status": 403, "body": None, "text_excerpt": "forbidden"},
                {"status": 200, "body": body, "text_excerpt": ""},
            ]
        )
        scraper._reauth = AsyncMock()  # type: ignore[assignment]

        out = await scraper._fetch_qu_api(page, "https://x/api")

        assert out == body
        assert page.evaluate.await_count == 2
        scraper._reauth.assert_awaited_once()

    async def test_401_then_401_raises(self):
        """Persistent 401 across reauth → raises RuntimeError mentioning the
        status and 'after re-auth'. _reauth fires exactly once."""
        scraper = _make_scraper()
        page = MagicMock()
        page.evaluate = AsyncMock(
            side_effect=[
                {"status": 401, "body": None, "text_excerpt": "unauthorized A"},
                {"status": 401, "body": None, "text_excerpt": "unauthorized B"},
            ]
        )
        scraper._reauth = AsyncMock()  # type: ignore[assignment]

        with pytest.raises(RuntimeError) as excinfo:
            await scraper._fetch_qu_api(page, "https://x/api")

        msg = str(excinfo.value)
        assert "401" in msg
        assert "after re-auth" in msg
        assert page.evaluate.await_count == 2
        scraper._reauth.assert_awaited_once()

    async def test_500_raises_without_reauth(self):
        """5xx is not auth-recoverable → raises immediately, no reauth."""
        scraper = _make_scraper()
        page = MagicMock()
        page.evaluate = AsyncMock(
            return_value={"status": 500, "body": None, "text_excerpt": "boom"}
        )
        scraper._reauth = AsyncMock()  # type: ignore[assignment]

        with pytest.raises(RuntimeError) as excinfo:
            await scraper._fetch_qu_api(page, "https://x/api")

        assert "500" in str(excinfo.value)
        assert page.evaluate.await_count == 1
        scraper._reauth.assert_not_awaited()

    async def test_200_non_json_raises(self):
        """Status 200 but body is None (JSON.parse failed) → raises."""
        scraper = _make_scraper()
        page = MagicMock()
        page.evaluate = AsyncMock(
            return_value={"status": 200, "body": None, "text_excerpt": "<html>..."}
        )
        scraper._reauth = AsyncMock()  # type: ignore[assignment]

        with pytest.raises(RuntimeError) as excinfo:
            await scraper._fetch_qu_api(page, "https://x/api")

        msg = str(excinfo.value)
        assert "non-JSON" in msg or "non-json" in msg.lower()
        assert page.evaluate.await_count == 1
        scraper._reauth.assert_not_awaited()


# ---------------------------------------------------------------------------
# Unit tests on QUScraper._reauth — regression for codex P1
# ---------------------------------------------------------------------------
#
# The first cut of _reauth delegated to ensure_authenticated (launch mode)
# or _cdp_ensure_auth (CDP mode). Both have local-state early-returns:
# ensure_authenticated bails out if the session file is "fresh" (TTL), and
# _cdp_ensure_auth bails out if the QU100 table is still on the page. By
# construction _reauth is only called AFTER the server rejected the current
# cookies with 401/403 — so both early-returns produce a no-op reauth and
# the retry replays the same rejected session. The fix: bypass both TTL and
# DOM heuristics, call `login` directly to force fresh cookies, then
# navigate back to the QU SPA so the in-page fetch retry has the right
# origin.


class TestReauthBypassesLocalHeuristics:
    """_reauth must FORCE a real login, ignoring TTL/DOM early-returns.

    Regression for codex iter-1 [P1]: when the server invalidates the
    session mid-scrape but the local session file is still within TTL
    (or the QU100 table is still rendered from cache in CDP mode), the
    old _reauth no-op'd and the retry failed against the same cookies.
    """

    async def test_reauth_calls_login_and_navigates_back(self):
        """_reauth calls login() then goto_with_retry() in that order.
        Does NOT consult ensure_authenticated / _cdp_ensure_auth — both
        have local early-returns that would skip the actual cookie
        refresh."""
        scraper = _make_scraper(is_cdp=False)
        scraper._page = MagicMock()

        with (
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock) as mock_login,
            patch(
                "rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock
            ) as mock_goto,
            patch(
                "rainier.scrapers.qu.scraper.ensure_authenticated",
                new_callable=AsyncMock,
            ) as mock_ensure,
        ):
            scraper._cdp_ensure_auth = AsyncMock()  # type: ignore[assignment]
            await scraper._reauth()

            # login() is the load-bearing call — it bypasses TTL.
            mock_login.assert_awaited_once_with(scraper._page)
            # goto_with_retry takes us back to the QU SPA after login.
            mock_goto.assert_awaited_once()
            assert mock_goto.await_args.args[0] is scraper._page
            assert "quantunicorn" in mock_goto.await_args.args[1]
            # Critical: TTL-gated helpers are NOT called.
            mock_ensure.assert_not_awaited()
            scraper._cdp_ensure_auth.assert_not_awaited()

    async def test_reauth_in_cdp_mode_still_forces_login(self):
        """CDP mode also force-logs-in. The old code routed CDP to
        _cdp_ensure_auth, which bails out if the QU100 table is still on
        the page (e.g., cached DOM after server-side session revocation).
        Now the same login path runs for both modes."""
        scraper = _make_scraper(is_cdp=True)
        scraper._page = MagicMock()

        with (
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock) as mock_login,
            patch(
                "rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock
            ) as mock_goto,
        ):
            scraper._cdp_ensure_auth = AsyncMock()  # type: ignore[assignment]
            await scraper._reauth()

            mock_login.assert_awaited_once_with(scraper._page)
            mock_goto.assert_awaited_once()
            # The DOM-heuristic CDP helper is NOT called from _reauth.
            scraper._cdp_ensure_auth.assert_not_awaited()
