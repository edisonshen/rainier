"""QuantUnicorn scraper — orchestrates login, QU100 table, and detail pages."""

from __future__ import annotations

import asyncio
import random
from datetime import date as date_type
from datetime import datetime, timezone

import structlog
from sqlalchemy import func, select

from rainier.core.config import get_settings
from rainier.core.database import get_session
from rainier.core.models import MoneyFlowSnapshot, Stock, StockCapitalFlow
from rainier.scrapers.base import BaseScraper, ScrapeResult, goto_with_retry
from rainier.scrapers.browser import BrowserManager
from rainier.scrapers.qu import selectors as sel
from rainier.scrapers.qu.auth import ensure_authenticated, get_session_path, is_session_valid, login
from rainier.scrapers.qu.parsers import (
    QU100Row,
    parse_capital_flow_rows,
    parse_qu100_rows,
)

log = structlog.get_logger()


class QUScraper(BaseScraper):
    """Scrapes QuantUnicorn QU100 rankings and per-ticker detail pages."""

    @property
    def name(self) -> str:
        return "qu"

    def __init__(self, browser: BrowserManager) -> None:
        super().__init__(browser)
        self._qu_config = get_settings().scraping.quantunicorn
        self._page = None
        self._page_cm = None

    async def setup(self) -> None:
        """Open a page and ensure logged in.

        In CDP mode: uses the existing browser tab (user already logged in).
        In launch mode: opens a new page with saved session + auto-login.
        """
        if self.browser._is_cdp:
            # CDP mode: user's real Chrome — check if auth is needed
            self._page_cm = self.browser.existing_page()
            self._page = await self._page_cm.__aenter__()
            self.log.info("cdp_mode", url=self._page.url)

            # Auto-login if not authenticated
            await self._cdp_ensure_auth()
        else:
            # Launch mode: fresh Playwright browser, need auth
            storage = get_session_path() if is_session_valid() else None
            self._page_cm = self.browser.new_page(storage_state=storage)
            self._page = await self._page_cm.__aenter__()
            await ensure_authenticated(self._page)

            # Verify session works by navigating to QU100 page
            await self._verify_session()

    async def teardown(self) -> None:
        """Close the page/context."""
        if self._page_cm:
            await self._page_cm.__aexit__(None, None, None)
            self._page = None
            self._page_cm = None

    async def run(self, **kwargs) -> ScrapeResult:
        """
        Main scrape logic.

        Keyword args:
            session: str | None — capture session name ("morning", etc.)
                     If provided, scrapes QU100 table.
            dates: list[str] | None — specific dates to scrape (e.g., ["2026-03-10"])
                   If provided with session, scrapes each date.
            top_n: int — how many top stocks to scrape detail pages for (default 0)
            symbols: list[str] | None — specific symbols for detail scrape
        """
        session_name = kwargs.get("session")
        dates = kwargs.get("dates")
        top_n = kwargs.get("top_n", 0)
        symbols = kwargs.get("symbols")

        now = datetime.now(timezone.utc)
        result = ScrapeResult(scraper_name=self.name, started_at=now)

        # Phase A: QU100 table (Top100 + Bottom100)
        if session_name:
            if dates:
                # Multi-date scrape with per-date error handling
                delay = self._qu_config.backfill_delay_seconds
                for i, date_str in enumerate(dates):
                    try:
                        await self._scrape_qu100(
                            session_name, now, result, target_date=date_str,
                        )
                    except Exception as exc:
                        msg = f"Skipping {date_str}: {exc}"
                        self.log.warning("backfill_skip", date=date_str, error=str(exc))
                        result.errors.append(msg)
                    if i < len(dates) - 1:
                        wait = delay + random.uniform(0, delay * 0.25)
                        remaining = len(dates) - i - 1
                        self.log.info("backfill_delay", seconds=f"{wait:.1f}", remaining=remaining)
                        await asyncio.sleep(wait)
            else:
                await self._scrape_qu100(session_name, now, result)

        # Phase B: Detail pages (optional)
        if symbols:
            await self._scrape_details(symbols, now, result)
        elif top_n > 0:
            top_symbols = self._get_top_symbols(top_n)
            if top_symbols:
                await self._scrape_details(top_symbols, now, result)

        result.finished_at = datetime.now(timezone.utc)
        return result

    # ------------------------------------------------------------------
    # Session verification
    # ------------------------------------------------------------------

    async def _verify_session(self) -> None:
        """Navigate to QU100 and verify the session is live.

        If the saved session is stale (server-side), force a fresh login.
        """
        page = self._page
        await goto_with_retry(page, self._qu_config.url)

        # Check if we got redirected to signin
        if "signin" in (page.url or ""):
            self.log.warning("session_stale_redirected", url=page.url)
            await login(page)
            await goto_with_retry(page, self._qu_config.url)
            return

        # Check if QU100 table is visible (proves auth works)
        try:
            await page.wait_for_selector(sel.QU100_TABLE, timeout=10000)
            self.log.info("session_verified")
        except Exception:
            self.log.warning("session_stale_no_table", url=page.url)
            await login(page)
            await goto_with_retry(page, self._qu_config.url)
            await page.wait_for_selector(sel.QU100_TABLE, timeout=15000)
            self.log.info("session_verified_after_relogin")

    # ------------------------------------------------------------------
    # CDP auto-auth
    # ------------------------------------------------------------------

    async def _cdp_ensure_auth(self) -> None:
        """In CDP mode, navigate to QU100 and login if needed."""
        page = self._page
        url = page.url or ""

        # Navigate to QU100 page if not already there
        if "quantunicorn.com/products" not in url:
            await goto_with_retry(page, self._qu_config.url)
            await page.wait_for_load_state("networkidle", timeout=15000)

        # Check if redirected to signin — session is invalid regardless
        # of file age (server may have expired it)
        if "signin" in (page.url or ""):
            self.log.info("cdp_needs_login", url=page.url)
            await login(page)
            await goto_with_retry(page, self._qu_config.url)
            return

        # Check if on QU100 page but behind login wall (no table)
        table = await page.query_selector(sel.QU100_TABLE)
        if table is None:
            # Click the login/register button if present
            login_btn = await page.query_selector("text=注册/登录")
            if login_btn:
                self.log.info("cdp_clicking_login_button")
                await login_btn.click()
                await page.wait_for_load_state(
                    "networkidle", timeout=10000
                )
            if "signin" in (page.url or ""):
                self.log.info("cdp_needs_login", url=page.url)
                await login(page)
                await goto_with_retry(page, self._qu_config.url)

    # ------------------------------------------------------------------
    # Phase A: QU100 table
    # ------------------------------------------------------------------

    async def _scrape_qu100(
        self,
        session_name: str,
        captured_at: datetime,
        result: ScrapeResult,
        target_date: str | None = None,
    ) -> None:
        """Scrape Top100 and Bottom100 from the QU100 page."""
        page = self._page

        # Navigate to QU100 page if needed
        current_url = page.url or ""
        if "quantunicorn.com/products" not in current_url:
            if self.browser._is_cdp:
                self.log.warning("cdp_wrong_page", url=current_url,
                                 hint="Navigate to QU100 page in Chrome first")
            await goto_with_retry(page, self._qu_config.url)

        # Safety net: if redirected to signin after navigation, force login
        if "signin" in (page.url or ""):
            self.log.info("scrape_forced_login", url=page.url)
            await login(page)
            await goto_with_retry(page, self._qu_config.url)

        # Change date if requested (before initial search)
        if target_date:
            await self._set_date(target_date)

        # Wait for the table to appear (date change or initial load triggers it)
        await page.wait_for_selector(sel.QU100_TABLE, timeout=15000)

        # Read the data date from the date picker (e.g., "2026-03-13")
        data_date_str = await page.get_attribute(sel.DATE_INPUT, "value")
        try:
            data_date = date_type.fromisoformat(data_date_str) if data_date_str else captured_at.date()
        except ValueError:
            data_date = captured_at.date()
        self.log.info("qu100_data_date", date=str(data_date))

        for ranking_type, button_sel in [
            ("top100", sel.TOP100_BUTTON),
            ("bottom100", sel.BOTTOM100_BUTTON),
        ]:
            try:
                # Click the ranking toggle — table auto-refreshes
                await page.click(button_sel)
                await page.wait_for_selector(sel.QU100_TABLE_ROW)
                raw_rows = await page.evaluate(sel.QU100_EXTRACT_JS)
                parsed = parse_qu100_rows(raw_rows)
                count = self._persist_qu100(parsed, ranking_type, session_name, captured_at, data_date)
                result.records_created += count

                self.log.info("qu100_scraped", ranking_type=ranking_type, rows=count, date=str(data_date))

            except Exception as exc:
                error_msg = f"Failed to scrape {ranking_type}: {exc}"
                self.log.warning("qu100_failed", ranking_type=ranking_type, error=str(exc))
                result.errors.append(error_msg)

    async def _set_date(self, date_str: str) -> None:
        """Change the date picker to the given date and trigger search."""
        page = self._page
        date_input = page.locator(sel.DATE_INPUT)
        await date_input.click(click_count=3)
        await date_input.fill(date_str)
        await page.keyboard.press("Enter")
        self.log.info("date_changed", date=date_str)

    def _persist_qu100(
        self,
        rows: list[QU100Row],
        ranking_type: str,
        session_name: str,
        captured_at: datetime,
        data_date=None,
    ) -> int:
        """Save QU100 rows to the database using batch operations. Returns row count."""
        effective_date = data_date or captured_at.date()

        with get_session() as db:
            # Check if this (date, ranking_type) already exists — skip if so
            existing_count = db.execute(
                select(func.count()).where(
                    MoneyFlowSnapshot.data_date == effective_date,
                    MoneyFlowSnapshot.ranking_type == ranking_type,
                )
            ).scalar()
            if existing_count and existing_count >= len(rows):
                self.log.info("persist_skipped", date=str(effective_date),
                              ranking_type=ranking_type, reason="already_exists")
                return len(rows)

            # Ensure stocks exist
            symbols = {row.symbol for row in rows}
            existing_stocks = {
                s.symbol
                for s in db.execute(
                    select(Stock.symbol).where(Stock.symbol.in_(symbols))
                ).all()
            }
            new_stocks = [
                Stock(symbol=r.symbol, sector=r.sector, industry=r.industry)
                for r in rows
                if r.symbol not in existing_stocks
            ]
            if new_stocks:
                db.add_all(new_stocks)
                db.flush()

            # Bulk insert all snapshots
            db.add_all([
                MoneyFlowSnapshot(
                    captured_at=captured_at,
                    capture_session=session_name,
                    data_date=effective_date,
                    ranking_type=ranking_type,
                    symbol=row.symbol,
                    rank=row.rank,
                    daily_change=row.daily_change,
                    sector=row.sector,
                    industry=row.industry,
                    long_short=row.long_short,
                    raw_data=row.raw,
                )
                for row in rows
            ])

        return len(rows)

    # ------------------------------------------------------------------
    # Phase B: Detail pages
    # ------------------------------------------------------------------

    async def _scrape_details(
        self,
        symbols: list[str],
        captured_at: datetime,
        result: ScrapeResult,
    ) -> None:
        """Scrape capital flow detail pages for the given symbols."""
        page = self._page

        for symbol in symbols:
            try:
                self.log.info("detail_scraping", symbol=symbol)

                # Navigate to capital flow page and search for ticker
                await page.click(sel.CAPITAL_FLOW_NAV)
                await page.wait_for_selector(sel.TICKER_INPUT)
                await page.fill(sel.TICKER_INPUT, symbol)
                await page.click(sel.SEARCH_BUTTON)
                await page.wait_for_selector(sel.DAILY_RANK_TABLE)

                # Extract daily rank table
                daily_raw = await page.evaluate(
                    sel.DETAIL_TABLE_EXTRACT_JS, sel.DAILY_RANK_TABLE
                )
                daily_rows = parse_capital_flow_rows(daily_raw, "daily")

                # Extract weekly rank table
                weekly_raw = await page.evaluate(
                    sel.DETAIL_TABLE_EXTRACT_JS, sel.WEEKLY_RANK_TABLE
                )
                weekly_rows = parse_capital_flow_rows(weekly_raw, "weekly")

                count = self._persist_detail(
                    symbol, daily_rows + weekly_rows, captured_at
                )
                result.records_created += count

                self.log.info(
                    "detail_scraped",
                    symbol=symbol,
                    daily_rows=len(daily_rows),
                    weekly_rows=len(weekly_rows),
                )

                # Polite delay between tickers
                await asyncio.sleep(random.uniform(1.0, 3.0))

            except Exception as exc:
                error_msg = f"Failed to scrape detail for {symbol}: {exc}"
                self.log.warning("detail_failed", symbol=symbol, error=str(exc))
                result.errors.append(error_msg)

    def _persist_detail(
        self,
        symbol: str,
        rows: list,
        captured_at: datetime,
    ) -> int:
        """Save capital flow detail rows to the database."""
        count = 0
        with get_session() as db:
            stock = db.execute(
                select(Stock).where(Stock.symbol == symbol)
            ).scalar_one_or_none()

            if stock is None:
                self.log.warning("stock_not_found", symbol=symbol)
                return 0

            for row in rows:
                try:
                    flow_date = date_type.fromisoformat(row.flow_date)
                except ValueError:
                    self.log.warning(
                        "invalid_date", symbol=symbol, date=row.flow_date
                    )
                    continue

                flow = StockCapitalFlow(
                    symbol=symbol,
                    captured_at=captured_at,
                    flow_date=flow_date,
                    period_type=row.period_type,
                    week_start=(
                        date_type.fromisoformat(row.week_start)
                        if row.week_start
                        else None
                    ),
                    week_end=(
                        date_type.fromisoformat(row.week_end) if row.week_end else None
                    ),
                    capital_flow_direction=row.direction,
                    long_short=row.long_short,
                    rank=row.rank,
                    rank_total=row.rank_total,
                    raw_data=row.raw,
                )
                db.add(flow)
                count += 1

        return count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_top_symbols(self, n: int) -> list[str]:
        """Get top N symbols from the most recent QU100 scrape."""
        with get_session() as db:
            rows = (
                db.execute(
                    select(Stock.symbol)
                    .join(MoneyFlowSnapshot)
                    .where(MoneyFlowSnapshot.ranking_type == "top100")
                    .order_by(
                        MoneyFlowSnapshot.captured_at.desc(),
                        MoneyFlowSnapshot.rank,
                    )
                    .limit(n)
                )
                .scalars()
                .all()
            )
            return list(rows)
