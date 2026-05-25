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

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from rainier.scrapers.qu.scraper import QUScraper


def _make_mock_page(
    url: str = "https://www.quantunicorn.com/products#qu100",
    titles: list[str] | None = None,
):
    """Mock Page where successive .title() calls return entries from `titles`.

    On the (n+1)th call, the last value is repeated — so a list of
    ["Just a moment...", "QuantUnicorn"] gives the CF page first and
    every subsequent title check resolves cleanly.
    """
    page = AsyncMock()
    page_url = {"value": url}
    type(page).url = PropertyMock(side_effect=lambda: page_url["value"])

    titles = titles or ["QU100"]
    title_idx = {"n": 0}

    async def fake_title():
        idx = min(title_idx["n"], len(titles) - 1)
        title_idx["n"] += 1
        return titles[idx]

    page.title = AsyncMock(side_effect=fake_title)
    page.context = AsyncMock()
    page.context.add_cookies = AsyncMock()
    # Table query: return a truthy element so the post-CF path early-returns
    # at "table already loaded".
    page.query_selector = AsyncMock(return_value=AsyncMock())
    page.wait_for_selector = AsyncMock(return_value=AsyncMock())
    page.wait_for_load_state = AsyncMock()
    page.wait_for_function = AsyncMock()
    page.goto = AsyncMock(side_effect=lambda u, **kw: page_url.update(value=u))
    return page, page_url


def _make_scraper():
    mock_browser = MagicMock()
    mock_browser._is_cdp = True
    with patch("rainier.scrapers.qu.scraper.get_settings") as mock_settings:
        qu_config = MagicMock()
        qu_config.url = "https://www.quantunicorn.com/products#qu100"
        qu_config.login_url = "https://www.quantunicorn.com/signin"
        qu_config.session_file = "./data/auth/qu_session.json"
        qu_config.session_ttl_hours = 12
        qu_config.timeout_ms = 30000
        qu_config.backfill_delay_seconds = 2.0
        mock_settings.return_value.scraping.quantunicorn = qu_config
        return QUScraper(mock_browser)


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
