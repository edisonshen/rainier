"""Playwright browser lifecycle manager."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import structlog
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from rainier.core.config import get_settings

log = structlog.get_logger()


class BrowserManager:
    """
    Manages a Playwright Chromium browser instance.

    Two modes:

    1. **Launch mode** (default) — starts a fresh Chromium::

        async with BrowserManager() as bm:
            async with bm.new_page() as page:
                await page.goto("https://example.com")

    2. **CDP mode** — connects to an already-running Chrome (bypasses bot detection)::

        async with BrowserManager(cdp_url="http://localhost:9222") as bm:
            async with bm.existing_page() as page:
                # uses the page already open in Chrome
                ...

    To start Chrome with CDP:
        /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222
    """

    def __init__(
        self,
        headless: bool | None = None,
        timeout_ms: int | None = None,
        cdp_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self._headless = (
            headless if headless is not None else settings.scraping.quantunicorn.headless
        )
        self._timeout_ms = timeout_ms or settings.scraping.quantunicorn.timeout_ms
        self._cdp_url = cdp_url
        self._playwright = None
        self._browser: Browser | None = None
        self._is_cdp = False

    async def start(self) -> None:
        """Launch or connect to the browser."""
        if self._browser is not None:
            return
        self._playwright = await async_playwright().start()

        if self._cdp_url:
            self._browser = await self._playwright.chromium.connect_over_cdp(self._cdp_url)
            self._is_cdp = True
            log.info("browser_connected_cdp", url=self._cdp_url)
        else:
            self._browser = await self._playwright.chromium.launch(
                headless=self._headless,
            )
            self._is_cdp = False
            log.info("browser_started", headless=self._headless)

    async def stop(self) -> None:
        """Close/disconnect the browser and Playwright."""
        if self._browser:
            if self._is_cdp:
                # Don't close the user's Chrome — just disconnect
                pass
            else:
                await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        log.info("browser_stopped")

    async def __aenter__(self) -> BrowserManager:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    @asynccontextmanager
    async def new_page(
        self,
        storage_state: str | Path | None = None,
    ) -> AsyncGenerator[Page, None]:
        """
        Create a new browser context + page. Closes both on exit.

        Args:
            storage_state: Path to JSON with cookies/localStorage to restore.
                          Pass None for a fresh session.
        """
        if self._browser is None:
            raise RuntimeError("Browser not started. Call start() or use 'async with'.")

        ctx_kwargs = {}
        if storage_state and Path(storage_state).exists():
            ctx_kwargs["storage_state"] = str(storage_state)
            log.info("session_restored", path=str(storage_state))

        context: BrowserContext = await self._browser.new_context(**ctx_kwargs)
        context.set_default_timeout(self._timeout_ms)
        page = await context.new_page()
        try:
            yield page
        finally:
            await context.close()

    @asynccontextmanager
    async def existing_page(self, tab_index: int = 0) -> AsyncGenerator[Page, None]:
        """
        Use an existing page/tab from a CDP-connected browser.

        Does NOT close the page on exit (it's the user's browser).
        Falls back to new_page() if not in CDP mode.
        """
        if not self._is_cdp:
            async with self.new_page() as page:
                yield page
            return

        if self._browser is None:
            raise RuntimeError("Browser not started. Call start() or use 'async with'.")

        contexts = self._browser.contexts
        if not contexts:
            log.warning("cdp_no_contexts", msg="No contexts in CDP browser, creating new page")
            context = await self._browser.new_context()
            context.set_default_timeout(self._timeout_ms)
            page = await context.new_page()
            yield page
            return

        pages = contexts[0].pages
        if not pages:
            log.warning("cdp_no_pages", msg="No open tabs in CDP browser, creating new page")
            page = await contexts[0].new_page()
            page.set_default_timeout(self._timeout_ms)
            yield page
            return

        # Filter out internal Chrome pages (chrome://, devtools://, etc.)
        real_pages = [p for p in pages if not (p.url or "").startswith("chrome")]
        if not real_pages:
            log.warning("cdp_no_real_pages", msg="Only internal Chrome tabs found, creating new page")
            page = await contexts[0].new_page()
            page.set_default_timeout(self._timeout_ms)
            yield page
            return

        if tab_index >= len(real_pages):
            log.warning("tab_index_out_of_range", requested=tab_index, available=len(real_pages))
            tab_index = 0

        page = real_pages[tab_index]
        page.set_default_timeout(self._timeout_ms)
        log.info("using_existing_page", url=page.url, tab=tab_index)
        yield page
        # Do NOT close — it's the user's tab

    @staticmethod
    async def save_storage_state(page: Page, path: str | Path) -> None:
        """Save cookies + localStorage from the current page's context."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = await page.context.storage_state()
        path.write_text(json.dumps(state, indent=2))
        log.info("session_saved", path=str(path))

    @staticmethod
    def is_session_valid(path: str | Path, ttl_hours: int) -> bool:
        """Check if a saved session file exists and is within TTL."""
        p = Path(path)
        if not p.exists():
            return False
        age_hours = (time.time() - p.stat().st_mtime) / 3600
        valid = age_hours < ttl_hours
        log.debug("session_check", path=str(p), age_hours=round(age_hours, 1), valid=valid)
        return valid
