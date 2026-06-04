"""Unit tests for QUScraper CDP authentication flow.

Tests the branching logic in _cdp_ensure_auth() and the post-login
paths in setup() and _scrape_qu100(). All Playwright Page interactions
are mocked — no real browser needed.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rainier.scrapers.qu import selectors as sel

from .conftest import make_mock_page, make_scraper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_page(url="https://www.quantunicorn.com/products#qu100", title="QU100"):
    # This flow needs query_selector to default to None ("no element found").
    return make_mock_page(url=url, title=title, query_selector_returns_element=False)


def _make_scraper(mock_browser=None):
    # CDP mode by default for this file's auth-flow tests.
    return make_scraper(mock_browser=mock_browser, is_cdp=True)


# ---------------------------------------------------------------------------
# setup() tests
# ---------------------------------------------------------------------------


class TestSetupCDP:
    """Tests for setup() in CDP mode."""

    @pytest.mark.asyncio
    async def test_cdp_setup_happy_path(self):
        """CDP mode (DESIGN D-10): fresh_tab_in_existing_context works,
        _cdp_ensure_auth is called."""
        page, _ = _make_mock_page()
        # Table already visible → early return from _cdp_ensure_auth
        page.query_selector = AsyncMock(return_value=AsyncMock())

        mock_browser = MagicMock()
        mock_browser._is_cdp = True
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=page)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_browser.fresh_tab_in_existing_context = MagicMock(return_value=mock_cm)

        scraper = _make_scraper(mock_browser)

        with (
            patch("rainier.scrapers.qu.scraper.get_session_path", return_value="/fake"),
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock),
            patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock),
            patch("pathlib.Path.exists", return_value=False),
        ):
            await scraper.setup()

        assert scraper._page is page
        mock_browser.fresh_tab_in_existing_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_cdp_setup_chrome_not_open(self):
        """CDP mode: fresh_tab_in_existing_context() raises when Chrome isn't
        reachable."""
        mock_browser = MagicMock()
        mock_browser._is_cdp = True
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(
            side_effect=ConnectionError("Cannot connect to Chrome on CDP")
        )
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_browser.fresh_tab_in_existing_context = MagicMock(return_value=mock_cm)

        scraper = _make_scraper(mock_browser)

        with pytest.raises(ConnectionError, match="Cannot connect to Chrome"):
            await scraper.setup()


# ---------------------------------------------------------------------------
# _cdp_ensure_auth() tests
# ---------------------------------------------------------------------------


class TestCDPEnsureAuth:
    """Tests for _cdp_ensure_auth() branching logic."""

    @pytest.mark.asyncio
    async def test_already_on_qu100_table_visible(self):
        """Already on QU100 page with table visible → no navigation, early return."""
        page, _ = _make_mock_page(url="https://www.quantunicorn.com/products#qu100")
        # query_selector for table returns a truthy element
        page.query_selector = AsyncMock(return_value=AsyncMock())

        scraper = _make_scraper()
        scraper._page = page

        with (
            patch("rainier.scrapers.qu.scraper.get_session_path", return_value="/fake"),
            patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock) as mock_goto,
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock) as mock_login,
            patch("pathlib.Path.exists", return_value=False),
        ):
            await scraper._cdp_ensure_auth()

        # Should NOT navigate (already on products page)
        mock_goto.assert_not_called()
        # Should NOT login
        mock_login.assert_not_called()

    @pytest.mark.asyncio
    async def test_cookies_work_navigate_and_search(self):
        """Page is elsewhere, cookies valid → navigate, click Search, table loads."""
        page, page_url = _make_mock_page(url="https://www.google.com")

        # After goto_with_retry navigates to products, URL updates
        async def fake_goto(p, url):
            page_url["value"] = url

        # First query_selector call (table) → None
        # Second query_selector call (search btn) → element
        search_btn = AsyncMock()
        table_mock = AsyncMock()
        page.query_selector = AsyncMock(
            side_effect=[None, search_btn]
        )
        # wait_for_selector for table succeeds (early return at "cdp_auth_ok")
        page.wait_for_selector = AsyncMock(return_value=table_mock)

        scraper = _make_scraper()
        scraper._page = page

        with (
            patch("rainier.scrapers.qu.scraper.get_session_path", return_value="/fake"),
            patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock,
                  side_effect=fake_goto),
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock) as mock_login,
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", create=True) as mock_open,
        ):
            mock_open.return_value.__enter__ = MagicMock(
                return_value=MagicMock(
                    read=MagicMock(return_value='{"cookies": [{"name": "sid", "value": "abc"}]}')
                )
            )
            with patch("json.load", return_value={"cookies": [{"name": "sid", "value": "abc"}]}):
                await scraper._cdp_ensure_auth()

        # Should navigate (was on google.com)
        mock_login.assert_not_called()
        search_btn.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_cookies_work_table_already_loaded(self):
        """Navigate to products, table already visible → no Search click needed."""
        page, page_url = _make_mock_page(url="https://other.com")

        async def fake_goto(p, url):
            page_url["value"] = url

        # query_selector for table returns element (table already there)
        page.query_selector = AsyncMock(return_value=AsyncMock())

        scraper = _make_scraper()
        scraper._page = page

        with (
            patch("rainier.scrapers.qu.scraper.get_session_path", return_value="/fake"),
            patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock,
                  side_effect=fake_goto),
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock) as mock_login,
            patch("pathlib.Path.exists", return_value=False),
        ):
            await scraper._cdp_ensure_auth()

        mock_login.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_session_file(self):
        """No session file on disk → skips cookie loading, continues normally."""
        page, page_url = _make_mock_page(url="https://www.quantunicorn.com/products#qu100")
        # Table already visible
        page.query_selector = AsyncMock(return_value=AsyncMock())

        scraper = _make_scraper()
        scraper._page = page

        with (
            patch("rainier.scrapers.qu.scraper.get_session_path", return_value="/nonexistent"),
            patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock),
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock),
            patch("pathlib.Path.exists", return_value=False),
        ):
            # Should not crash
            await scraper._cdp_ensure_auth()

        # Cookies were never loaded (no add_cookies call)
        page.context.add_cookies.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_file_corrupted(self):
        """Session file has invalid JSON → logs warning, continues."""
        page, _ = _make_mock_page(url="https://www.quantunicorn.com/products#qu100")
        page.query_selector = AsyncMock(return_value=AsyncMock())

        scraper = _make_scraper()
        scraper._page = page

        with (
            patch("rainier.scrapers.qu.scraper.get_session_path", return_value="/fake"),
            patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock),
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", side_effect=json.JSONDecodeError("bad", "", 0)),
        ):
            # Should not crash — logs warning and continues
            await scraper._cdp_ensure_auth()

    @pytest.mark.asyncio
    async def test_signin_redirect_triggers_login(self):
        """Cookies expired → redirected to signin → login → navigate back."""
        page, page_url = _make_mock_page(url="https://other.com")

        call_count = {"goto": 0}

        async def fake_goto(p, url):
            call_count["goto"] += 1
            if call_count["goto"] == 1:
                # First navigation → redirected to signin
                page_url["value"] = "https://www.quantunicorn.com/signin?next=/products"
            else:
                # After login, navigation succeeds
                page_url["value"] = url

        # After second navigation: table query returns element
        query_calls = {"n": 0}

        async def fake_query(selector):
            query_calls["n"] += 1
            if selector == sel.QU100_TABLE:
                # Table appears after login + navigation
                return AsyncMock()
            return None

        page.query_selector = AsyncMock(side_effect=fake_query)

        scraper = _make_scraper()
        scraper._page = page

        with (
            patch("rainier.scrapers.qu.scraper.get_session_path", return_value="/fake"),
            patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock,
                  side_effect=fake_goto),
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock) as mock_login,
            patch("pathlib.Path.exists", return_value=False),
        ):
            await scraper._cdp_ensure_auth()

        mock_login.assert_called_once()

    @pytest.mark.asyncio
    async def test_cloudflare_challenge_stuck_raises(self):
        """Cloudflare 'Just a moment...' challenge that doesn't clear within
        25s → RuntimeError pointing operator at the CDP browser.

        Post-fix (TASK-PLAN-qu-verify-and-cf-recover-9f39 §3) the
        immediate bail is replaced with a wait_for_function(25s). The
        stuck path still raises, but with new wording ("stuck for >25s")
        and a CDP-browser recovery hint, not the old "--headed" message.
        """
        page, _ = _make_mock_page(
            url="https://www.quantunicorn.com/products#qu100",
            title="Just a moment..."
        )
        # Force the wait to fail so we exercise the raise path.
        page.wait_for_function = AsyncMock(
            side_effect=Exception("Timeout 25000ms exceeded")
        )

        scraper = _make_scraper()
        scraper._page = page

        with (
            patch("rainier.scrapers.qu.scraper.get_session_path", return_value="/fake"),
            patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock),
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock),
            patch("pathlib.Path.exists", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="stuck for >25s"):
                await scraper._cdp_ensure_auth()

    @pytest.mark.asyncio
    async def test_login_button_fallback(self):
        """No table, no signin URL, but 注册/登录 button visible → click it → login."""
        page, page_url = _make_mock_page(url="https://www.quantunicorn.com/products#qu100")

        login_btn = AsyncMock()
        search_btn = AsyncMock()

        query_calls = []

        async def fake_query(selector):
            query_calls.append(selector)
            if selector == sel.QU100_TABLE:
                return None  # table never visible before login
            if selector == "text=注册/登录":
                return login_btn
            if selector == sel.SEARCH_BUTTON:
                return search_btn
            return None

        page.query_selector = AsyncMock(side_effect=fake_query)

        # wait_for_selector for table (first attempt) times out, then succeeds after login
        table_wait_calls = {"n": 0}

        async def fake_wait_for_selector(selector, timeout=None):
            if selector == sel.QU100_TABLE:
                table_wait_calls["n"] += 1
                if table_wait_calls["n"] == 1:
                    raise TimeoutError("table not found")
                return AsyncMock()
            if selector == sel.SEARCH_BUTTON:
                return search_btn
            return AsyncMock()

        page.wait_for_selector = AsyncMock(side_effect=fake_wait_for_selector)

        # Clicking login button changes URL to signin
        async def click_login():
            page_url["value"] = "https://www.quantunicorn.com/signin"

        login_btn.click = AsyncMock(side_effect=click_login)

        scraper = _make_scraper()
        scraper._page = page

        async def fake_goto(p, url):
            page_url["value"] = url

        with (
            patch("rainier.scrapers.qu.scraper.get_session_path", return_value="/fake"),
            patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock,
                  side_effect=fake_goto),
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock) as mock_login,
            patch("pathlib.Path.exists", return_value=False),
        ):
            await scraper._cdp_ensure_auth()

        login_btn.click.assert_called_once()
        mock_login.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_login_waits_for_search_button(self):
        """After login, uses wait_for_selector (not query_selector) for Search button.

        This is the race condition fix from PR #47.
        """
        page, page_url = _make_mock_page(url="https://www.quantunicorn.com/products#qu100")

        search_btn = AsyncMock()

        # Table never visible → falls through to login path
        # query_selector for table → None, for login button → None,
        # for QU100_TABLE (second check) → None
        page.query_selector = AsyncMock(return_value=None)

        wait_selectors = []

        async def fake_wait_for_selector(selector, timeout=None):
            wait_selectors.append(selector)
            if selector == sel.QU100_TABLE:
                raise TimeoutError("no table yet")
            if selector == sel.SEARCH_BUTTON:
                return search_btn
            return AsyncMock()

        page.wait_for_selector = AsyncMock(side_effect=fake_wait_for_selector)

        # URL stays on products (no signin redirect), so login button path
        # falls through to the "still no table" block at line 233
        # query_selector for table → None → enters the block
        # query_selector for login button → None → skips
        # "signin" not in URL → skips login call
        # Then: wait_for_selector(SEARCH_BUTTON) — THIS is what we're testing

        # Override: after the wait_for_selector calls, the final table wait succeeds
        call_idx = {"n": 0}

        async def fake_wait_v2(selector, timeout=None):
            call_idx["n"] += 1
            wait_selectors.append(selector)
            if selector == sel.QU100_TABLE and call_idx["n"] <= 1:
                # First table wait fails (in the "try to load" block)
                raise TimeoutError("no table")
            if selector == sel.SEARCH_BUTTON:
                return search_btn
            return AsyncMock()  # subsequent table waits succeed

        page.wait_for_selector = AsyncMock(side_effect=fake_wait_v2)

        scraper = _make_scraper()
        scraper._page = page

        with (
            patch("rainier.scrapers.qu.scraper.get_session_path", return_value="/fake"),
            patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock),
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock),
            patch("pathlib.Path.exists", return_value=False),
        ):
            await scraper._cdp_ensure_auth()

        # The critical assertion: SEARCH_BUTTON was passed to wait_for_selector
        assert sel.SEARCH_BUTTON in wait_selectors
        search_btn.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_cookies_expired_login_loops_to_signin(self):
        """All auth attempts fail (login succeeds but page still redirects to signin).

        Should eventually hit the final wait_for_selector timeout, not loop forever.
        """
        page, page_url = _make_mock_page(url="https://other.com")

        # Every navigation results in signin redirect
        async def fake_goto(p, url):
            page_url["value"] = "https://www.quantunicorn.com/signin?next=/products"

        # No elements ever found
        page.query_selector = AsyncMock(return_value=None)
        # Final wait_for_selector times out
        page.wait_for_selector = AsyncMock(
            side_effect=TimeoutError("Timeout 15000ms exceeded")
        )

        scraper = _make_scraper()
        scraper._page = page

        with (
            patch("rainier.scrapers.qu.scraper.get_session_path", return_value="/fake"),
            patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock,
                  side_effect=fake_goto),
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock) as mock_login,
            patch("pathlib.Path.exists", return_value=False),
        ):
            with pytest.raises(TimeoutError):
                await scraper._cdp_ensure_auth()

        # Login was called multiple times (first signin check + second signin check
        # + login button fallback path)
        assert mock_login.call_count >= 2


# ---------------------------------------------------------------------------
# _scrape_qu100() post-login tests
# ---------------------------------------------------------------------------


class TestScrapeQU100PostLogin:
    """Tests for the post-login safety net in _scrape_qu100()."""

    @pytest.mark.asyncio
    async def test_signin_redirect_during_scrape(self):
        """Navigate to products but server redirects to signin → login → retry."""
        page, page_url = _make_mock_page(
            url="https://www.quantunicorn.com/products#qu100"
        )

        search_btn = AsyncMock()
        page.wait_for_selector = AsyncMock(return_value=search_btn)
        page.get_attribute = AsyncMock(return_value="2026-04-09")
        page.evaluate = AsyncMock(return_value=[
            {"rank": "1", "symbol": "NVDA", "daily_change": "▲5",
             "sector": "Tech", "industry": "Semi", "long_short": "多"}
        ])

        goto_calls = {"n": 0}

        async def fake_goto(p, url):
            goto_calls["n"] += 1
            if goto_calls["n"] == 1:
                # First navigation: server redirects to signin (session expired)
                page_url["value"] = "https://www.quantunicorn.com/signin?next=/products"
            else:
                # After login: navigation succeeds
                page_url["value"] = url

        # Start on products page but it's stale; _scrape_qu100 navigates
        # because we force a redirect on first goto
        page_url["value"] = "https://other.com"

        scraper = _make_scraper()
        scraper._page = page

        from datetime import datetime, timezone

        from rainier.scrapers.base import ScrapeResult
        result = ScrapeResult(scraper_name="qu", started_at=datetime.now(timezone.utc))

        with (
            patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock,
                  side_effect=fake_goto),
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock) as mock_login,
            patch.object(scraper, "_persist_qu100", return_value=1),
        ):
            await scraper._scrape_qu100("afternoon", datetime.now(timezone.utc), result)

        mock_login.assert_called_once()

# NOTE: Tests for DOM-path internals have been removed as part of the
# in-page-fetch migration (DESIGN-qu-scraper-in-page-fetch / TASK-PLAN §3):
#
#   - test_signin_redirect_between_toggle_iterations_recovers
#   - test_search_button_uses_wait_for_selector
#   - class TestWaitForNoSpinner (3 tests)
#
# The behaviors they covered (mid-iteration toggle/Search-click recovery,
# spinner waits) no longer exist: the scrape is a single
# page.evaluate(fetch) per ranking and Cloudflare/auth failures surface as
# fetch errors caught by the per-iteration ``except`` in ``_scrape_qu100``.
#
# The pre-scrape signin-redirect path is still covered by
# ``test_signin_redirect_during_scrape`` above. The in-page-fetch happy
# path is covered by ``tests/test_scrapers/test_scraper_in_page_fetch.py``.
