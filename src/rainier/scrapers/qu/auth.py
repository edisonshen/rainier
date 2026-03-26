"""QuantUnicorn authentication — login flow + session management."""

from __future__ import annotations

import structlog
from playwright.async_api import Page

from rainier.core.config import get_settings
from rainier.scrapers.browser import BrowserManager
from rainier.scrapers.qu import selectors as sel

log = structlog.get_logger()


def is_session_valid() -> bool:
    """Check if the saved QU session is still fresh."""
    qu = get_settings().scraping.quantunicorn
    return BrowserManager.is_session_valid(qu.session_file, qu.session_ttl_hours)


def get_session_path() -> str:
    """Return the configured session file path."""
    return get_settings().scraping.quantunicorn.session_file


async def login(page: Page) -> None:
    """
    Perform QU login on the given page.

    Navigates to login URL, fills credentials, submits, waits for success,
    and saves the storage state to the session file.

    Raises:
        TimeoutError: if login form or post-login indicator doesn't appear.
        ValueError: if credentials are not configured.
    """
    settings = get_settings()
    qu = settings.scraping.quantunicorn

    if not settings.qu_username or not settings.qu_password:
        raise ValueError("QU_USERNAME and QU_PASSWORD must be set in .env")

    log.info("qu_login_starting", url=qu.login_url)

    await page.goto(qu.login_url)
    await page.wait_for_selector(sel.LOGIN_EMAIL_INPUT)

    await page.fill(sel.LOGIN_EMAIL_INPUT, settings.qu_username)
    await page.fill(sel.LOGIN_PASSWORD_INPUT, settings.qu_password)
    await page.click(sel.LOGIN_SUBMIT_BUTTON)

    # Wait for navigation away from signin page
    await page.wait_for_url(
        lambda url: "signin" not in url, timeout=qu.timeout_ms
    )
    await page.wait_for_load_state("networkidle", timeout=30000)

    log.info("qu_login_success", url=page.url)

    # Save session for reuse
    await BrowserManager.save_storage_state(page, qu.session_file)


async def ensure_authenticated(page: Page) -> None:
    """
    Ensure the page has a valid QU session.

    If session file is fresh, it was already loaded via storage_state in
    new_page(). If not, perform a fresh login.
    """
    if is_session_valid():
        log.info("qu_session_reused")
        return

    await login(page)
