"""Tests for the QU100 in-page-fetch migration (DESIGN-qu-scraper-in-page-fetch).

Covers:
- D-3 / D-4 / D-7 / D-9: _scrape_qu100 calls /api/v3/mf100 via
  page.evaluate(fetch(...)) and persists 200 rows (top + bottom) for one date.
- D-10: USCIS-style teardown closes the scrape page/tab on success, on
  run-time exception, and in CDP mode (where a fresh tab is opened in
  contexts[0] and closed on teardown without touching the operator's other
  tabs or the Chrome process).

All Playwright interactions are mocked — no real browser.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from rainier.scrapers.qu import selectors as sel
from rainier.scrapers.qu.scraper import QUScraper

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


def _make_mock_page(url: str = "https://www.quantunicorn.com/products#qu100") -> MagicMock:
    page = AsyncMock()
    page_url = {"value": url}
    type(page).url = PropertyMock(side_effect=lambda: page_url["value"])
    page.title = AsyncMock(return_value="QU100")
    page.context = AsyncMock()
    page.context.add_cookies = AsyncMock()
    page.query_selector = AsyncMock(return_value=AsyncMock())
    page.wait_for_selector = AsyncMock(return_value=AsyncMock())
    page.wait_for_load_state = AsyncMock()
    page.goto = AsyncMock(side_effect=lambda u, **kw: page_url.update(value=u))
    return page


def _make_scraper(mock_browser: MagicMock | None = None) -> QUScraper:
    if mock_browser is None:
        mock_browser = MagicMock()
        mock_browser._is_cdp = False

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
# D-3 / D-4 / D-7 / D-9: in-page fetch replay
# ---------------------------------------------------------------------------


class TestQu100FetchReplaysAgainstMock:
    """_scrape_qu100 makes two page.evaluate(fetch(...)) calls and persists
    100 top + 100 bottom rows for one date.
    """

    async def test_persists_200_rows_for_one_date(self):
        top_payload = _load_fixture("qu_api_mf100_top.json")
        bottom_payload = _load_fixture("qu_api_mf100_bottom.json")

        page = _make_mock_page()
        # page.evaluate returns the API payload from the fixture. The scraper
        # calls evaluate twice — once for top, once for bottom — in this order.
        page.evaluate = AsyncMock(side_effect=[top_payload, bottom_payload])

        scraper = _make_scraper()
        scraper._page = page

        # Capture persist calls instead of going to the DB.
        persist_calls: list[dict] = []

        def fake_persist(rows, ranking_type, session_name, captured_at, data_date=None):
            persist_calls.append({
                "rows": rows,
                "ranking_type": ranking_type,
                "session_name": session_name,
                "captured_at": captured_at,
                "data_date": data_date,
            })
            return len(rows)

        scraper._persist_qu100 = fake_persist  # type: ignore[assignment]

        from rainier.scrapers.base import ScrapeResult
        captured_at = datetime(2026, 5, 22, 13, 0, tzinfo=timezone.utc)
        result = ScrapeResult(scraper_name="qu", started_at=captured_at)

        with patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock):
            await scraper._scrape_qu100(
                "morning", captured_at, result, target_date="2026-05-22"
            )

        # Exactly two evaluate calls: top + bottom.
        assert page.evaluate.await_count == 2

        # Both calls hit the API URL constant (D-7) and carry the right
        # `top` flag and the requested date as query params (D-9).
        first_call = page.evaluate.await_args_list[0]
        second_call = page.evaluate.await_args_list[1]
        first_blob = json.dumps(list(first_call.args) + list(first_call.kwargs.values()))
        second_blob = json.dumps(list(second_call.args) + list(second_call.kwargs.values()))
        assert sel.QU100_API_URL in first_blob
        assert sel.QU100_API_URL in second_blob
        assert "2026-05-22" in first_blob
        assert "2026-05-22" in second_blob
        assert "top=true" in first_blob
        assert "top=false" in second_blob

        # 200 rows persisted: 100 top + 100 bottom.
        assert len(persist_calls) == 2
        assert persist_calls[0]["ranking_type"] == "top100"
        assert persist_calls[1]["ranking_type"] == "bottom100"
        assert len(persist_calls[0]["rows"]) == 100
        assert len(persist_calls[1]["rows"]) == 100
        # Field rename landed (ticker -> symbol via api_rows_to_qu100_rows).
        assert persist_calls[0]["rows"][0].symbol == top_payload["data"][0]["ticker"]
        # records_created accumulates both calls.
        assert result.records_created == 200
        # No errors recorded on the happy path.
        assert result.errors == []

    async def test_no_date_picker_click(self):
        """D-9: the API takes `date` as a query param. The scraper must NOT
        click the date picker."""
        top_payload = _load_fixture("qu_api_mf100_top.json")
        bottom_payload = _load_fixture("qu_api_mf100_bottom.json")

        page = _make_mock_page()
        page.evaluate = AsyncMock(side_effect=[top_payload, bottom_payload])

        scraper = _make_scraper()
        scraper._page = page
        scraper._persist_qu100 = MagicMock(return_value=100)  # type: ignore[assignment]

        from rainier.scrapers.base import ScrapeResult
        captured_at = datetime(2026, 5, 22, 13, 0, tzinfo=timezone.utc)
        result = ScrapeResult(scraper_name="qu", started_at=captured_at)

        with patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock):
            await scraper._scrape_qu100(
                "morning", captured_at, result, target_date="2026-05-22"
            )

        # No interactions with the date input. _set_date is gone; nothing
        # in the in-page-fetch path should locator()/fill() the date input.
        assert sel.DATE_INPUT not in str(page.locator.mock_calls)
        # And no Search-button clicks from _scrape_qu100: the DOM scrape
        # sequence is gone (the previous toggle/search/old-DOM-extract loop).
        click_targets = [c.args[0] for c in page.click.await_args_list if c.args]
        assert sel.SEARCH_BUTTON not in click_targets
        # The DOM-only selectors (TOP100_BUTTON, BOTTOM100_BUTTON) were
        # deleted from selectors.py — their absence is asserted by the
        # selector-deletion checks at module import time. Here we just
        # confirm no click() was made against the Chinese-text toggles or
        # any ant-table row selector either.
        assert not any(
            "select-button" in t for t in click_targets if isinstance(t, str)
        )


# ---------------------------------------------------------------------------
# D-10: USCIS-style teardown
# ---------------------------------------------------------------------------


class TestTeardownClosesPageOnSuccess:
    """Launch mode: after a successful execute(), the page context manager
    has been exited (page closed) and the scraper's bookkeeping is cleared."""

    async def test_page_cm_exited_after_execute(self):
        page = _make_mock_page()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=page)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_browser = MagicMock()
        mock_browser._is_cdp = False
        mock_browser.new_page = MagicMock(return_value=mock_cm)

        scraper = _make_scraper(mock_browser)

        # Stub run() so the test is independent of the scrape logic.
        scraper.run = AsyncMock(  # type: ignore[assignment]
            return_value=__import__(
                "rainier.scrapers.base", fromlist=["ScrapeResult"]
            ).ScrapeResult(
                scraper_name="qu", started_at=datetime.now(timezone.utc)
            )
        )

        with (
            patch("rainier.scrapers.qu.scraper.is_session_valid", return_value=False),
            patch("rainier.scrapers.qu.scraper.get_session_path", return_value="/fake"),
            patch("rainier.scrapers.qu.scraper.ensure_authenticated", new_callable=AsyncMock),
            patch.object(QUScraper, "_verify_session", new_callable=AsyncMock),
        ):
            await scraper.execute()

        mock_cm.__aexit__.assert_awaited_once()
        assert scraper._page is None
        assert scraper._page_cm is None


class TestTeardownClosesPageOnRunException:
    """If run() raises, teardown still closes the page (failure-path coverage)."""

    async def test_page_cm_exited_when_run_raises(self):
        page = _make_mock_page()

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=page)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_browser = MagicMock()
        mock_browser._is_cdp = False
        mock_browser.new_page = MagicMock(return_value=mock_cm)

        scraper = _make_scraper(mock_browser)
        scraper.run = AsyncMock(  # type: ignore[assignment]
            side_effect=RuntimeError("boom")
        )

        with (
            patch("rainier.scrapers.qu.scraper.is_session_valid", return_value=False),
            patch("rainier.scrapers.qu.scraper.get_session_path", return_value="/fake"),
            patch("rainier.scrapers.qu.scraper.ensure_authenticated", new_callable=AsyncMock),
            patch.object(QUScraper, "_verify_session", new_callable=AsyncMock),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await scraper.execute()

        mock_cm.__aexit__.assert_awaited_once()
        assert scraper._page is None
        assert scraper._page_cm is None


class TestTeardownCdpModeOpensAndClosesFreshTab:
    """CDP mode: setup() opens a fresh tab inside contexts[0] (via
    fresh_tab_in_existing_context, NOT existing_page or browser.new_context),
    teardown() closes that tab, the Chrome process stays alive, and the
    operator's other tabs are untouched."""

    async def test_setup_uses_fresh_tab_helper_and_teardown_closes_only_that_tab(self):
        page = _make_mock_page()

        # The fresh-tab helper is an async context manager. Track __aenter__
        # and __aexit__ separately so we can assert only the helper's page
        # got closed, never context.close() or browser.close().
        fresh_cm = AsyncMock()
        fresh_cm.__aenter__ = AsyncMock(return_value=page)
        fresh_cm.__aexit__ = AsyncMock(return_value=False)

        mock_browser = MagicMock()
        mock_browser._is_cdp = True
        mock_browser.fresh_tab_in_existing_context = MagicMock(return_value=fresh_cm)
        # If the scraper falls back to one of these, the test fails: those are
        # the wrong APIs in CDP mode per D-10.
        mock_browser.existing_page = MagicMock(
            side_effect=AssertionError("D-10: must NOT use existing_page in CDP mode")
        )
        mock_browser.new_page = MagicMock(
            side_effect=AssertionError("D-10: must NOT use new_page in CDP mode")
        )

        scraper = _make_scraper(mock_browser)
        scraper.run = AsyncMock(  # type: ignore[assignment]
            return_value=__import__(
                "rainier.scrapers.base", fromlist=["ScrapeResult"]
            ).ScrapeResult(
                scraper_name="qu", started_at=datetime.now(timezone.utc)
            )
        )

        with (
            patch("rainier.scrapers.qu.scraper.get_session_path", return_value="/fake"),
            patch("pathlib.Path.exists", return_value=False),
            patch("rainier.scrapers.qu.scraper.login", new_callable=AsyncMock),
            patch("rainier.scrapers.qu.scraper.goto_with_retry", new_callable=AsyncMock),
        ):
            await scraper.execute()

        # The fresh-tab helper was the entry point.
        mock_browser.fresh_tab_in_existing_context.assert_called_once()
        fresh_cm.__aenter__.assert_awaited_once()
        # And was exited exactly once on teardown — closing the scrape tab.
        fresh_cm.__aexit__.assert_awaited_once()
        # context.close / browser.close must NOT have been called (operator's
        # context + Chrome process must survive).
        assert not page.context.close.await_args_list  # type: ignore[attr-defined]
        assert scraper._page is None
        assert scraper._page_cm is None
