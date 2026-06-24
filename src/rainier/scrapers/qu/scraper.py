"""QuantUnicorn scraper — orchestrates login, QU100 table, and detail pages."""

from __future__ import annotations

import asyncio
import random
import re
from datetime import date as date_type
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import structlog
from sqlalchemy import delete, select

from rainier.core.config import get_settings
from rainier.core.database import get_session
from rainier.core.models import MoneyFlowSnapshot, Stock, StockCapitalFlow
from rainier.scrapers.base import BaseScraper, ScrapeResult, goto_with_retry
from rainier.scrapers.browser import BrowserManager
from rainier.scrapers.qu import selectors as sel
from rainier.scrapers.qu.auth import ensure_authenticated, get_session_path, is_session_valid, login
from rainier.scrapers.qu.parsers import (
    QU100Row,
    _api_rows_to_qu100_rows,
    parse_capital_flow_rows,
)

# In-page fetch JS (DESIGN D-3/D-7). Defined at module scope so it's easy
# to find, easy to unit-test by reference, and not allocated per call.
#
# Cloudflare cf_clearance is bound to the page's TLS handshake — the only
# request shape that gets past the challenge is one that originates from
# the rendered page itself. We use ``credentials: 'include'`` so the
# session + cf_clearance cookies on the page are sent automatically.
#
# Return shape: ``{status, body, text_excerpt}`` so the Python caller can
# distinguish 401/403 (recoverable via reauth) from other errors instead of
# throwing inside the page context. ``body`` is the parsed JSON or ``None``
# when the response isn't valid JSON; ``text_excerpt`` is the first 500
# chars of the RAW (un-sanitized) text for diagnostics so genuine non-JSON
# error bodies stay legible in logs.
#
# QU's backend serializes with ``json.dumps(allow_nan=True)``, which emits
# the bare tokens ``NaN`` / ``Infinity`` / ``-Infinity`` (not legal JSON) in
# value position when a field like ``industry`` is missing. A naive
# ``JSON.parse`` throws on those, ``body`` becomes ``None``, and the whole
# date's ranking is dropped.
#
# ``QU100_NAN_LITERAL_RE`` rewrites those bare literals to ``null`` before
# parsing — but it must NOT touch the same byte sequence when it appears
# *inside* a quoted string (e.g. a field whose value is the text
# ``"ratio:NaN, see note"``). A colon-anchored pattern alone can't tell the
# two apart, so the regex alternates two branches:
#
#   1. a complete JSON string token ``"(?:[^"\\]|\\.)*"`` (handles escaped
#      quotes), matched but left UNCHANGED by the replacer, and
#   2. the value-position literal ``:\s*(NaN|-?Infinity)`` followed by a
#      structural terminator (``,`` / ``}`` / ``]``), rewritten to ``:null``.
#
# Because alternation is greedy left-to-right and strings are listed first,
# any NaN/Infinity that lives inside quotes is consumed as part of branch 1
# and never reaches branch 2 — so legitimate string content is preserved.
#
# The pattern is a single source-of-truth Python ``re.Pattern`` so a unit
# test can exercise the exact same regex without a JS engine; the JS below
# interpolates ``.pattern`` verbatim (JS and Python share identical syntax
# for it) and applies the same branch-aware replacer. The ``g`` flag is
# JS-only and lives in the literal below.
QU100_NAN_LITERAL_RE = re.compile(
    r'"(?:[^"\\]|\\.)*"|:\s*(?:NaN|-?Infinity)(?=\s*[,}\]])'
)


def _sanitize_qu_nan(text: str) -> str:
    """Coerce bare value-position NaN/Infinity literals to ``null``.

    Quote-aware: a NaN/Infinity sequence inside a JSON string value is left
    untouched. Mirrors the in-page JS replacer so both sides transform
    identically (see ``QU100_NAN_LITERAL_RE``).
    """
    return QU100_NAN_LITERAL_RE.sub(
        lambda m: m.group(0) if m.group(0).startswith('"') else ":null", text
    )


QU100_FETCH_JS = (
    """
async ({url}) => {
    const resp = await fetch(url, {
        method: 'GET',
        credentials: 'include',
        headers: {'accept': 'application/json, text/plain, */*'},
    });
    const text = await resp.text();
    // QU emits bare NaN/Infinity in value position (json.dumps allow_nan=True);
    // coerce only those structural literals to null so JSON.parse succeeds.
    // The string branch of the regex is matched first and returned verbatim so
    // NaN-like text inside a quoted value is never rewritten. The raw text is
    // still returned as text_excerpt for diagnostics.
    const sanitized = text.replace(/"""
    + QU100_NAN_LITERAL_RE.pattern
    + """/g, (m) => m[0] === '"' ? m : ':null');
    let body = null;
    try { body = JSON.parse(sanitized); } catch (_) { /* genuine non-JSON 4xx/5xx */ }
    return {
        status: resp.status,
        body: body,
        text_excerpt: text.slice(0, 500),
    };
}
"""
)

log = structlog.get_logger()

# QU is a US-equity product; the API's ``date=`` query param keys off the
# NYSE/NASDAQ calendar day. Manual operator runs at 19:00 PT (= 02:00 UTC
# next day) would otherwise request "tomorrow's" non-existent data. We
# hard-code America/New_York because QU is single-market — promote to a
# config knob only if a non-US market ever materializes.
MARKET_TZ = ZoneInfo("America/New_York")


def market_date(timestamp: datetime) -> date_type:
    """Return the US-market calendar date for a (UTC-or-other-tz) timestamp.

    ``timestamp`` must be timezone-aware; ``astimezone`` raises on naive
    inputs in Python 3.12+. The scraper always passes
    ``datetime.now(timezone.utc)``, so naive inputs are not expected.
    """
    return timestamp.astimezone(MARKET_TZ).date()


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

        In CDP mode (DESIGN D-10): opens a FRESH TAB inside the operator's
        existing logged-in Chrome context via
        ``BrowserManager.fresh_tab_in_existing_context``. The context's
        cookies (``session`` + ``cf_clearance``) carry through, so the
        QU SPA + Cloudflare see us as logged in. **We do NOT use
        ``existing_page`` (which reuses the operator's tab and refuses to
        close it) or ``browser.new_context`` (which drops auth).**

        In launch mode: opens a new page with saved session + auto-login.
        """
        if self.browser._is_cdp:
            # CDP mode: fresh tab in the operator's context. Always-close on
            # teardown — leaves the operator's other tabs and Chrome process
            # untouched.
            self._page_cm = self.browser.fresh_tab_in_existing_context()
            self._page = await self._page_cm.__aenter__()
            self.log.info("cdp_mode_fresh_tab", url=self._page.url)

            # Auto-login if not authenticated (rare — the operator's context
            # typically already has live cookies).
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
        """Close the scrape page/tab (USCIS-style, DESIGN D-10).

        Runs unconditionally from ``BaseScraper.execute``'s ``finally``, so
        it fires on success AND on every failure path (including a raise
        from ``run()``). Idempotent w.r.t. the outer ``async with
        BrowserManager`` — exiting the page context here means the outer
        block's exit is a no-op for the page (it still tears down the
        browser/playwright in launch mode).
        """
        if self._page_cm is not None:
            page_cm = self._page_cm
            # Clear bookkeeping BEFORE awaiting __aexit__ so a re-entrant
            # call (or a __aexit__ that itself raises) doesn't double-close
            # or leave dangling state.
            self._page_cm = None
            self._page = None
            await page_cm.__aexit__(None, None, None)

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
        """Verify the saved session works against the QU API.

        Uses the same in-page-fetch transport the scrape itself uses
        (DESIGN D-3/D-7). A 200 with valid JSON means the session +
        cf_clearance are alive; a 401/403 triggers ``_fetch_qu_api``'s
        one-shot reauth + retry; anything else propagates as a real
        error.

        The previous implementation waited for ``.ant-table`` to render
        on ``/products#qu100``. Post-PR-#84 the scrape no longer clicks
        Search, so the table never renders on bare nav — verification
        timed out every cron run. Probing the API directly is the only
        check that agrees with what the scrape actually does.
        """
        page = self._page
        await goto_with_retry(page, self._qu_config.url)

        # Pre-auth state — saved storage was empty or cookies wiped at
        # the file level; force a login then return. The next caller
        # (the scrape itself) will exercise the API path; no need to
        # double-probe here.
        if "signin" in (page.url or ""):
            self.log.warning("session_stale_redirected", url=page.url)
            await login(page)
            await goto_with_retry(page, self._qu_config.url)
            return

        # API probe. ``_fetch_qu_api`` handles 401/403 → reauth + retry
        # (PR #86); we just propagate any RuntimeError it raises.
        probe_date = market_date(datetime.now(timezone.utc)).isoformat()
        probe_url = (
            f"{sel.QU100_API_URL}?date={probe_date}"
            "&top=true&frequency=daily"
        )
        try:
            await self._fetch_qu_api(page, probe_url)
            self.log.info("session_verified_via_api")
        except RuntimeError as exc:
            self.log.error("session_verify_failed", error=str(exc))
            raise

    # ------------------------------------------------------------------
    # CDP auto-auth
    # ------------------------------------------------------------------

    async def _cdp_ensure_auth(self) -> None:
        """In CDP mode, navigate to QU100 and login if needed.

        Strategy: always try loading the saved session cookies first (even if
        "stale" by file age — the cf_clearance cookie is valid for a year).
        Only attempt programmatic login as a last resort.
        """
        page = self._page
        url = page.url or ""

        # Load saved session cookies into the CDP browser if available
        session_path = get_session_path()
        import json
        from pathlib import Path

        session_file = Path(session_path)
        if session_file.exists():
            try:
                with open(session_file) as f:
                    state = json.load(f)
                cookies = state.get("cookies", [])
                if cookies:
                    context = page.context
                    await context.add_cookies(cookies)
                    self.log.info("cdp_cookies_loaded", count=len(cookies))
            except Exception as exc:
                self.log.warning("cdp_cookies_load_failed", error=str(exc))

        # Navigate to QU100 page
        if "quantunicorn.com/products" not in url:
            await goto_with_retry(page, self._qu_config.url)
            await page.wait_for_load_state(
                "domcontentloaded", timeout=15000
            )

        # Check if redirected to signin — cookies didn't work
        if "signin" in (page.url or ""):
            self.log.warning("cdp_session_expired", url=page.url)
            await login(page)
            await goto_with_retry(page, self._qu_config.url)

        # Detect Cloudflare JS challenge. Playwright (Chromium) executes
        # the challenge JS automatically; we just need to wait for it to
        # resolve and issue a fresh cf_clearance cookie. Most challenges
        # clear in 5-10s. The old code bailed immediately and asked the
        # operator for a --headed run, which broke every unattended cron.
        title = await page.title()
        if "just a moment" in title.lower():
            self.log.info("cf_challenge_waiting", initial_title=title)
            try:
                # Wait for document.title to flip away from "Just a
                # moment..." (case-insensitive). On success the SPA loads
                # with title = "QuantUnicorn" (or similar). 25s is
                # generous; Cloudflare's own SLA for the challenge is
                # ~10s end-to-end.
                await page.wait_for_function(
                    """() => !document.title.toLowerCase().includes("just a moment")""",
                    timeout=25000,
                )
                new_title = await page.title()
                self.log.info("cf_challenge_cleared", new_title=new_title)
            except Exception as exc:
                # Challenge stuck > 25s — operator intervention needed.
                # The operator already has Chrome on :9222 (CDP mode), so
                # the simplest recovery is just to open QU there manually.
                raise RuntimeError(
                    "Cloudflare challenge stuck for >25s — cf_clearance "
                    "cookie expired and JS challenge didn't auto-resolve. "
                    f"Open {self._qu_config.url} in the CDP browser "
                    "(currently on :9222) and manually clear any "
                    "challenge, then retry."
                ) from exc

        # Check if redirected to signin — cookies didn't work
        if "signin" in (page.url or ""):
            self.log.warning("cdp_session_expired", url=page.url)
            await login(page)
            await goto_with_retry(page, self._qu_config.url)

        # Try to load the table — click Search button first
        table = await page.query_selector(sel.QU100_TABLE)
        if table is None:
            search_btn = await page.query_selector(sel.SEARCH_BUTTON)
            if search_btn:
                self.log.info("cdp_clicking_search_button")
                await search_btn.click()
            try:
                await page.wait_for_selector(
                    sel.QU100_TABLE, timeout=10000
                )
                self.log.info("cdp_auth_ok")
                return
            except Exception:
                pass

        # Still no table — need to login
        if await page.query_selector(sel.QU100_TABLE) is None:
            login_btn = await page.query_selector("text=注册/登录")
            if login_btn:
                self.log.info("cdp_clicking_login_button")
                await login_btn.click()
                await page.wait_for_load_state(
                    "domcontentloaded", timeout=10000
                )
            if "signin" in (page.url or ""):
                self.log.info("cdp_needs_login", url=page.url)
                await login(page)
                await goto_with_retry(page, self._qu_config.url)

            # After login, wait for React to render, then click Search
            search_btn = await page.wait_for_selector(
                sel.SEARCH_BUTTON, timeout=15000
            )
            if search_btn:
                await search_btn.click()
            await page.wait_for_selector(
                sel.QU100_TABLE, timeout=15000
            )
            self.log.info("cdp_auth_ok_after_login")

    # ------------------------------------------------------------------
    # In-page fetch + auth recovery
    # ------------------------------------------------------------------
    #
    # PR #84 dropped the DOM-path `_recover_if_signin` helper when the
    # scrape moved to in-page fetch. _fetch_qu_api restores the equivalent:
    # on 401/403 (cookie expiry, server-side session invalidation), reauth
    # ONCE and retry the same URL. Bail on second auth failure or any
    # other 4xx/5xx. No exponential backoff, no jitter — cookie expiry is
    # a single-shot recoverable condition.
    #
    #   ┌──────────────┐  401/403  ┌─────────┐  ok   ┌────────┐
    #   │ page.evaluate│──────────▶│ _reauth │──────▶│ retry  │──▶ body
    #   └──────────────┘           └─────────┘       └────────┘
    #          │                                          │
    #          │ ok                                       │ 401/403
    #          ▼                                          ▼
    #         body                                     RuntimeError

    async def _fetch_qu_api(self, page, url: str) -> dict:
        """In-page fetch with one-shot re-auth retry on 401/403.

        Returns the parsed JSON body. Raises ``RuntimeError`` on:
          - persistent 401/403 (after one re-auth attempt)
          - any other HTTP error (4xx/5xx not in {401, 403})
          - status 200 with a non-JSON body
        """
        result = await page.evaluate(QU100_FETCH_JS, {"url": url})
        if result["status"] in (401, 403):
            self.log.warning(
                "qu_api_auth_failure",
                status=result["status"],
                url=url,
                attempting="reauth_and_retry",
            )
            await self._reauth()
            result = await page.evaluate(QU100_FETCH_JS, {"url": url})
            if result["status"] in (401, 403):
                raise RuntimeError(
                    f"qu api {result['status']} after re-auth: "
                    f"{result['text_excerpt']}"
                )
        if result["status"] >= 400:
            raise RuntimeError(
                f"qu api {result['status']}: {result['text_excerpt']}"
            )
        if result["body"] is None:
            raise RuntimeError(
                f"qu api {result['status']} non-JSON: {result['text_excerpt']}"
            )
        return result["body"]

    async def _reauth(self) -> None:
        """Force a fresh login to recover from server-side session rejection.

        Called from ``_fetch_qu_api`` after a 401/403 — by construction the
        server just rejected the current cookies. Local heuristics
        (``is_session_valid`` TTL check in ``ensure_authenticated``, or the
        "table is already on the page" early-return in ``_cdp_ensure_auth``)
        would no-op and the retry would replay the same rejected session.
        So we bypass both and call ``login`` directly, which navigates to
        the login form, submits credentials, and overwrites the storage
        state with fresh cookies. After login we navigate back to the
        QU100 SPA so the in-page fetch on retry has the right origin.

        Works for both launch and CDP modes — ``login`` operates on
        whatever ``Page`` it's given.
        """
        await login(self._page)
        await goto_with_retry(self._page, self._qu_config.url)

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
        """Scrape Top100 + Bottom100 via in-page fetch (DESIGN D-3/D-7/D-9).

        Two ``page.evaluate(fetch(...))`` calls per date — one for
        ``top=true`` and one for ``top=false`` — against ``/api/v3/mf100``.
        The fetch originates from the rendered page, so Cloudflare's
        cf_clearance (TLS-bound) accepts the request. No DOM clicks. No
        date-picker interaction. No spinner waits.
        """
        page = self._page

        # Make sure the page is on the QU SPA so cf_clearance + session
        # cookies are scoped correctly. The in-page fetch path needs the
        # page to be on the same origin as the API.
        current_url = page.url or ""
        if "quantunicorn.com/products" not in current_url:
            if self.browser._is_cdp:
                self.log.info("cdp_navigating_for_fetch", url=current_url)
            await goto_with_retry(page, self._qu_config.url)

        # Safety net: if redirected to signin (stale session), force login
        # and try again. Same logic as before; just preserved here for the
        # one place it still matters.
        if "signin" in (page.url or ""):
            self.log.info("scrape_forced_login", url=page.url)
            await login(page)
            await goto_with_retry(page, self._qu_config.url)

        # The API takes ``date`` as a query param (DESIGN D-9). We use the
        # caller-supplied target_date if any, otherwise derive the US-market
        # calendar day from captured_at. ``captured_at`` itself is UTC and
        # stays UTC (it's the DB snapshot timestamp); only the *date*
        # going into the API query is shifted to America/New_York so
        # manual late-PT runs hit the correct trading day. See
        # docs/TASK-PLAN-qu-captured-at-tz-drift-44e3.md.
        if target_date:
            try:
                data_date = date_type.fromisoformat(target_date)
            except ValueError:
                self.log.warning("invalid_target_date_fallback",
                                 target_date=target_date)
                data_date = market_date(captured_at)
        else:
            data_date = market_date(captured_at)
        self.log.info("qu100_data_date", date=str(data_date))

        # One fetch per ranking. ``top=true|false`` per the SPA contract.
        for ranking_type, top_flag in [("top100", "true"), ("bottom100", "false")]:
            try:
                url = (
                    f"{sel.QU100_API_URL}?date={data_date.isoformat()}"
                    f"&top={top_flag}&frequency=daily"
                )
                payload = await self._fetch_qu_api(page, url)
                # ``payload.get("data") or []`` (not ``.get("data", [])``):
                # the API has been observed to return ``{"data": null}`` on
                # some dates (key present, value null). ``.get("data", [])``
                # only kicks in the default when the key is *missing*, so a
                # null value passes through and ``_api_rows_to_qu100_rows``
                # then raises ``TypeError: 'NoneType' is not iterable``.
                # Coalescing collapses both shapes ({}, {"data": null}) into
                # an empty list.
                api_data = (payload.get("data") or []) if isinstance(payload, dict) else []
                parsed = _api_rows_to_qu100_rows(api_data)
                count = self._persist_qu100(
                    parsed, ranking_type, session_name, captured_at, data_date
                )
                result.records_created += count
                self.log.info(
                    "qu100_scraped",
                    ranking_type=ranking_type,
                    rows=count,
                    date=str(data_date),
                )
            except Exception as exc:
                error_msg = f"Failed to scrape {ranking_type}: {exc}"
                self.log.warning(
                    "qu100_failed", ranking_type=ranking_type, error=str(exc)
                )
                result.errors.append(error_msg)

    def _persist_qu100(
        self,
        rows: list[QU100Row],
        ranking_type: str,
        session_name: str,
        captured_at: datetime,
        data_date=None,
    ) -> int:
        """Rebuild the day's ``(data_date, ranking_type)`` snapshot from this scrape.

        Each later same-day scrape must OVERRIDE the symbols it returns and CARRY
        FORWARD the rest, stamping the whole day with the latest ``captured_at`` —
        one snapshot generation per ``(data_date, ranking_type)``. An empty scrape
        (0 rows) is a no-op so a blank pull never wipes a good snapshot.

        Mechanics (one transaction):

            read existing rows  ─►  carried = stored symbols NOT in this scrape
            DELETE the day's rows
            INSERT  fresh rows  (this captured_at + this capture_session)
                  + carried rows (this captured_at, ORIGINAL capture_session,
                                  data copied into NEW instances — fresh ids)

        We never UPDATE ``captured_at`` in place (it is the hypertable partition
        key) and never re-``add`` the just-deleted ORM objects (identity-map
        pitfall); carried field values are copied into fresh ``MoneyFlowSnapshot``
        instances. ``captured_at`` advancing is itself the freshness signal.
        Returns the count of rows that now make up the day's snapshot.

        NOTE: delete-then-rebuild assumes scrapes for the same
        ``(data_date, ranking_type)`` are serial (cron / single-slot APScheduler).
        Overlapping same-day scrapes could race the rebuild — accepted under the
        serial scheduler; concurrency hardening is a tracked P2 follow-up.
        """
        effective_date = data_date or captured_at.date()

        # Empty scrape -> no-op. Never overwrite a good snapshot with a blank one.
        if not rows:
            self.log.info("persist_skipped", date=str(effective_date),
                          ranking_type=ranking_type, reason="empty_scrape")
            return 0

        scraped_symbols = {row.symbol for row in rows}

        with get_session() as db:
            # 1) Read the existing day's rows BEFORE deleting so we can carry
            #    forward symbols this scrape did not return.
            existing = db.execute(
                select(MoneyFlowSnapshot).where(
                    MoneyFlowSnapshot.data_date == effective_date,
                    MoneyFlowSnapshot.ranking_type == ranking_type,
                )
            ).scalars().all()
            # Carry forward only symbols absent from this scrape. Copy the field
            # VALUES out now (the ORM objects get deleted next), keeping each
            # carried row's ORIGINAL capture_session truthful.
            carried = [
                {
                    "symbol": row.symbol,
                    "rank": row.rank,
                    "daily_change": row.daily_change,
                    "sector": row.sector,
                    "industry": row.industry,
                    "long_short": row.long_short,
                    "raw_data": row.raw_data,
                    "view_type": row.view_type,
                    "capture_session": row.capture_session,
                }
                for row in existing
                if row.symbol not in scraped_symbols
            ]

            # 2) Ensure stocks exist for the scraped symbols (carried symbols
            #    already have a stock row from a prior scrape).
            existing_stocks = {
                s.symbol
                for s in db.execute(
                    select(Stock.symbol).where(Stock.symbol.in_(scraped_symbols))
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

            # 3) Delete the day's existing rows, then rebuild. flush() so the
            #    DELETE lands before the INSERTs in the same transaction.
            db.execute(
                delete(MoneyFlowSnapshot).where(
                    MoneyFlowSnapshot.data_date == effective_date,
                    MoneyFlowSnapshot.ranking_type == ranking_type,
                )
            )
            db.flush()

            # 4) Insert fresh rows (this session) + carried rows (this captured_at,
            #    original capture_session). Carried rows are NEW instances (fresh
            #    ids) — never the deleted ORM objects.
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
            db.add_all([
                MoneyFlowSnapshot(
                    captured_at=captured_at,
                    capture_session=c["capture_session"],
                    data_date=effective_date,
                    ranking_type=ranking_type,
                    symbol=c["symbol"],
                    rank=c["rank"],
                    daily_change=c["daily_change"],
                    sector=c["sector"],
                    industry=c["industry"],
                    long_short=c["long_short"],
                    raw_data=c["raw_data"],
                    view_type=c["view_type"],
                )
                for c in carried
            ])

        return len(rows) + len(carried)

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
