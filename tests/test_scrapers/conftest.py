"""Shared Playwright mock helpers for QUScraper unit tests.

Consolidates the copy-pasted ``_make_mock_page`` / ``_make_scraper`` helpers
that previously lived in each ``test_scrapers/*.py`` file (R5 of the slim
audit). These are module-level helper functions rather than pytest fixtures:
the call sites invoke them imperatively mid-test (often more than once, with
per-test URL/title overrides), which fits a plain factory better than a
fixture.

Behavior is the union of the prior per-file variants, gated by keyword args so
each call site keeps its exact previous semantics:

  - ``make_mock_page`` always returns ``(page, page_url)``; callers that only
    want the page unpack ``page, _``.
  - ``query_selector_returns_element`` toggles the default ``query_selector``
    result (element vs ``None``) — cdp_auth's flow needs ``None``.
  - ``titles`` supports the CF-recovery multi-call title sequence; otherwise a
    single ``title`` is returned on every call.
  - ``make_scraper`` takes ``is_cdp`` (and an optional pre-built browser) so the
    CDP-vs-launch distinction each file encoded as a default is explicit.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from rainier.scrapers.qu.scraper import QUScraper


def make_mock_page(
    url: str = "https://www.quantunicorn.com/products#qu100",
    title: str = "QU100",
    titles: list[str] | None = None,
    query_selector_returns_element: bool = True,
):
    """Create a mock Playwright Page; returns ``(page, page_url)``.

    ``page.url`` is a property backed by the mutable ``page_url`` dict so tests
    (and ``goto``) can change it mid-flow. When ``titles`` is given, successive
    ``.title()`` calls walk that list and then repeat the last entry (so e.g.
    ``["Just a moment...", "QuantUnicorn"]`` returns the CF page first and
    clears on every later check).
    """
    page = AsyncMock()
    page_url = {"value": url}
    type(page).url = PropertyMock(side_effect=lambda: page_url["value"])

    if titles is not None:
        title_idx = {"n": 0}

        async def fake_title():
            idx = min(title_idx["n"], len(titles) - 1)
            title_idx["n"] += 1
            return titles[idx]

        page.title = AsyncMock(side_effect=fake_title)
    else:
        page.title = AsyncMock(return_value=title)

    page.context = AsyncMock()
    page.context.add_cookies = AsyncMock()

    query_default = AsyncMock() if query_selector_returns_element else None
    page.query_selector = AsyncMock(return_value=query_default)
    page.wait_for_selector = AsyncMock(return_value=AsyncMock())
    page.wait_for_load_state = AsyncMock()
    # CF-challenge wait — harmless extra mock for flows that never await it.
    page.wait_for_function = AsyncMock()
    page.goto = AsyncMock(side_effect=lambda u, **kw: page_url.update(value=u))

    return page, page_url


def make_scraper(mock_browser: MagicMock | None = None, is_cdp: bool = False) -> QUScraper:
    """Create a QUScraper with a mocked browser and patched settings.

    When ``mock_browser`` is omitted, a ``MagicMock`` with ``_is_cdp=is_cdp`` is
    used. Settings are patched in-place so no real config file is read.
    """
    if mock_browser is None:
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
