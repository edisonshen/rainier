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

from rainier.scrapers.qu.scraper import QUScraper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scraper(is_cdp: bool = False) -> QUScraper:
    mock_browser = MagicMock()
    mock_browser._is_cdp = is_cdp

    with patch("rainier.scrapers.qu.scraper.get_settings") as mock_settings:
        qu_config = MagicMock()
        qu_config.url = "https://www.quantunicorn.com/products#qu100"
        qu_config.login_url = "https://www.quantunicorn.com/signin"
        qu_config.session_file = "./data/auth/qu_session.json"
        qu_config.session_ttl_hours = 12
        qu_config.timeout_ms = 30000
        qu_config.backfill_delay_seconds = 2.0
        mock_settings.return_value.scraping.quantunicorn = qu_config
        scraper = QUScraper(mock_browser)
    return scraper


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
