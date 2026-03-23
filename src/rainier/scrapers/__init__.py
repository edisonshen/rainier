"""Scraper framework — base classes and site implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rainier.scrapers.base import BaseScraper, ScrapeResult
from rainier.scrapers.browser import BrowserManager

if TYPE_CHECKING:
    pass

# Registry: add new scrapers here
SCRAPERS: dict[str, type[BaseScraper]] = {}


def _register_scrapers() -> None:
    """Lazy import to avoid circular deps and import cost at CLI startup."""
    from rainier.scrapers.qu.scraper import QUScraper

    SCRAPERS["qu"] = QUScraper
    # Future: SCRAPERS["tv"] = TVScraper


def get_scraper(name: str, browser: BrowserManager) -> BaseScraper:
    """Get a scraper instance by name."""
    if not SCRAPERS:
        _register_scrapers()
    if name not in SCRAPERS:
        available = ", ".join(sorted(SCRAPERS.keys()))
        raise ValueError(f"Unknown scraper '{name}'. Available: {available}")
    return SCRAPERS[name](browser=browser)


def list_scrapers() -> list[str]:
    """Return names of all registered scrapers."""
    if not SCRAPERS:
        _register_scrapers()
    return sorted(SCRAPERS.keys())


__all__ = [
    "BaseScraper",
    "BrowserManager",
    "ScrapeResult",
    "get_scraper",
    "list_scrapers",
]
