"""Playwright browser lifecycle manager."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import structlog
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from rainier.core.config import get_settings

log = structlog.get_logger()

ENSURE_CHROME_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ensure-chrome.sh"

# CDP debug port. Must match scripts/ensure-chrome.sh's PORT — the kill
# path uses this to locate the Chrome process we spawned.
CDP_DEBUG_PORT = 9222


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
        # True only when this manager triggered ``_force_restart_chrome``
        # during ``start()`` — i.e., the Chrome listening on :9222 was
        # spawned by US, not by the operator. ``stop()`` checks this to
        # decide whether to kill the whole Chrome process (we own it) or
        # leave it running (D-10: it's the operator's browser).
        self._we_spawned_chrome = False

    async def start(self) -> None:
        """Launch or connect to the browser."""
        if self._browser is not None:
            return
        self._playwright = await async_playwright().start()

        if self._cdp_url:
            try:
                self._browser = await self._playwright.chromium.connect_over_cdp(self._cdp_url)
            except Exception as e:
                # Chrome can get into a bad CDP state after auto-updating in place
                # or sitting idle for days. Force-restart it once and retry.
                log.warning("cdp_connect_failed_restarting_chrome", error=str(e))
                await self._force_restart_chrome()
                # Record that WE own the Chrome process now. ``stop()`` will
                # kill it on teardown so we don't leak an empty window.
                self._we_spawned_chrome = True
                self._browser = await self._playwright.chromium.connect_over_cdp(self._cdp_url)
            self._is_cdp = True
            log.info(
                "browser_connected_cdp",
                url=self._cdp_url,
                we_spawned_chrome=self._we_spawned_chrome,
            )
        else:
            self._browser = await self._playwright.chromium.launch(
                headless=self._headless,
            )
            self._is_cdp = False
            log.info("browser_started", headless=self._headless)

    @staticmethod
    async def _force_restart_chrome() -> None:
        """Kill and relaunch the local debug Chrome via ensure-chrome.sh --force."""
        if not ENSURE_CHROME_SCRIPT.exists():
            raise RuntimeError(f"ensure-chrome.sh not found at {ENSURE_CHROME_SCRIPT}")
        proc = await asyncio.create_subprocess_exec(
            str(ENSURE_CHROME_SCRIPT),
            "--force",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode(errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(f"Chrome force-restart failed: {output}")
        log.info("chrome_force_restarted", output=output)

    async def stop(self) -> None:
        """Close/disconnect the browser and Playwright.

        CDP-mode teardown branches on ownership:

        - Attached to operator's pre-existing Chrome → just disconnect.
          The D-10 contract owns this case: the operator's window stays
          alive and the fresh scrape tab was already closed by the
          ``fresh_tab_in_existing_context`` exit path.
        - We force-restarted Chrome ourselves during ``start()`` → kill
          the whole Chrome process. Otherwise an empty Chrome window
          accumulates on :9222 after every cron run.

        Launch mode is unchanged: ``browser.close()`` tears down the
        Chromium we launched.
        """
        if self._browser:
            if self._is_cdp:
                if self._we_spawned_chrome:
                    # WE spawned this Chrome instance; nothing else owns it.
                    # Kill the process so the next ensure-chrome.sh call gets
                    # a clean start.
                    await self._kill_chrome_we_spawned()
                # else: D-10 — don't touch the operator's Chrome.
            else:
                await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        log.info("browser_stopped", we_spawned_chrome=self._we_spawned_chrome)

    @staticmethod
    async def _kill_chrome_we_spawned() -> None:
        """Terminate the Chrome process bound to :9222.

        Mirrors the pattern in ``scripts/ensure-chrome.sh kill_chrome``
        (``pkill -f remote-debugging-port=$PORT``) but in pure asyncio so
        we don't need a separate shell-script entrypoint. Best-effort:
        every failure is logged and swallowed because teardown must not
        break the calling code — the worst case is one leftover Chrome
        process the next ``ensure-chrome.sh`` invocation will recycle.

        Flow:
            lsof -ti :9222          → list pids holding the port
            for each pid: kill <pid>

        Returns silently when:
            - lsof finds no pids (Chrome already gone),
            - any subprocess call fails (logged as ``chrome_kill_failed``).
        """
        try:
            lsof = await asyncio.create_subprocess_exec(
                "lsof",
                "-ti",
                f":{CDP_DEBUG_PORT}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await lsof.communicate()
            pids = [p for p in stdout.decode(errors="replace").split() if p.strip()]
            if not pids:
                log.info("chrome_kill_skipped_no_pids", port=CDP_DEBUG_PORT)
                return
            for pid in pids:
                kill = await asyncio.create_subprocess_exec(
                    "kill",
                    pid,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await kill.communicate()
            log.info("chrome_we_spawned_killed", pids=pids, port=CDP_DEBUG_PORT)
        except Exception as exc:
            log.warning("chrome_kill_failed", error=str(exc))

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

    @asynccontextmanager
    async def fresh_tab_in_existing_context(
        self,
    ) -> AsyncGenerator[Page, None]:
        """Open a new page inside the CDP browser's existing context, close it on exit.

        Used by the QU scraper (DESIGN D-10) to satisfy the USCIS-style "always
        close the scrape window" guarantee in CDP mode without disturbing the
        operator's other tabs.

        We attach to ``contexts[0]`` — the operator's logged-in Chrome context
        carrying the live ``session`` + ``cf_clearance`` cookies — and call
        ``new_page()`` on it. **We do NOT call ``browser.new_context()``**:
        that would drop the operator's auth state. The new tab inherits the
        existing context's cookies, so the QU SPA + Cloudflare see us as
        logged in.

        On exit the helper closes only that one page. The context, the
        Chrome process, and every other tab the operator had open survive.

        Falls back to :meth:`new_page` if the manager is in launch mode (not
        CDP) so callers can stay mode-agnostic.
        """
        if not self._is_cdp:
            async with self.new_page() as page:
                yield page
            return

        if self._browser is None:
            raise RuntimeError("Browser not started. Call start() or use 'async with'.")

        contexts = self._browser.contexts
        if not contexts:
            # No context attached to the CDP browser. The fallback that used
            # to live here (``self._browser.new_context()``) materialized a
            # blank context with no operator cookies — every QU fetch from
            # that tab would hit Cloudflare's challenge wall and 401 within
            # one round-trip, so the workaround masked a misconfiguration
            # without recovering from it. Surface the state to the operator
            # instead so they know to open Chrome / restart the CDP debug
            # session before re-running.
            raise RuntimeError(
                "CDP browser has no contexts; cannot open fresh tab"
            )

        context = contexts[0]
        page = await context.new_page()
        page.set_default_timeout(self._timeout_ms)
        log.info("cdp_fresh_tab_opened", url=page.url)
        try:
            yield page
        finally:
            try:
                await page.close()
                log.info("cdp_fresh_tab_closed")
            except Exception as exc:
                # Closing a tab should never break the calling code. Log and
                # move on — the worst case is one leftover tab the operator
                # can close manually, which we treat as a non-fatal hygiene
                # bug rather than a scrape failure.
                log.warning("cdp_fresh_tab_close_failed", error=str(exc))

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
