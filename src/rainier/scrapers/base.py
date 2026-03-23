"""Base scraper interface — minimal abstraction for all site scrapers."""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from datetime import datetime

import structlog
from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeout

from rainier.scrapers.browser import BrowserManager

log = structlog.get_logger()


@dataclass
class ScrapeResult:
    """Summary returned by every scrape run."""

    scraper_name: str
    started_at: datetime
    finished_at: datetime | None = None
    records_created: int = 0
    records_updated: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.finished_at is not None and len(self.errors) == 0

    @property
    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


class BaseScraper(abc.ABC):
    """
    Abstract base for all Playwright-based scrapers.

    Subclasses must implement:
        name       -- unique string identifier (e.g., "qu")
        run()      -- the main scrape logic

    Optionally override:
        setup()    -- called before run() (e.g., login)
        teardown() -- called after run() (e.g., cleanup)
    """

    def __init__(self, browser: BrowserManager) -> None:
        self.browser = browser
        self.log = log.bind(scraper=self.name)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier for this scraper."""
        ...

    @abc.abstractmethod
    async def run(self, **kwargs) -> ScrapeResult:
        """Execute the scrape. kwargs allow CLI to pass scraper-specific options."""
        ...

    async def setup(self) -> None:
        """Pre-run hook (e.g., login). Default is no-op."""

    async def teardown(self) -> None:
        """Post-run hook. Default is no-op."""

    async def execute(self, **kwargs) -> ScrapeResult:
        """
        Full lifecycle: setup -> run -> teardown.
        This is what the CLI calls. Do not override.
        """
        self.log.info("scraper_starting")
        try:
            await self.setup()
            result = await self.run(**kwargs)
            self.log.info(
                "scraper_finished",
                records_created=result.records_created,
                duration=result.duration_seconds,
                errors=len(result.errors),
            )
            return result
        except Exception as exc:
            self.log.error("scraper_failed", error=str(exc))
            raise
        finally:
            await self.teardown()


async def goto_with_retry(
    page: Page, url: str, retries: int = 2, delay: float = 2.0
) -> None:
    """Navigate to URL with retries on timeout."""
    for attempt in range(retries + 1):
        try:
            await page.goto(url, wait_until="networkidle")
            return
        except PlaywrightTimeout:
            if attempt == retries:
                raise
            log.warning("navigation_retry", url=url, attempt=attempt + 1)
            await asyncio.sleep(delay)
