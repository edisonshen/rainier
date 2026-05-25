"""CSS selectors for QuantUnicorn pages.

Single source of truth. Discovered from real site HTML on 2026-03-15.
Site uses Ant Design (antd) React components.
"""

# ---------------------------------------------------------------------------
# Login page (https://www.quantunicorn.com/signin)
# ---------------------------------------------------------------------------
LOGIN_EMAIL_INPUT = '#signIn_email'
LOGIN_PASSWORD_INPUT = '#signIn_password'
LOGIN_SUBMIT_BUTTON = 'button[type="submit"]'

# Post-login indicator — not used as wait_for_selector anymore.
# After login, we wait for navigation away from /signin instead.
LOGGED_IN_INDICATOR = None

# ---------------------------------------------------------------------------
# QU100 page (https://www.quantunicorn.com/products#qu100)
# ---------------------------------------------------------------------------

# Main container
QU100_CONTAINER = ".qu100"

# Ant Design table — still referenced by the CDP auth probe (the table
# appearing proves we are logged in and the SPA is alive).
QU100_TABLE = ".ant-table"
QU100_TABLE_ROW = ".ant-table-tbody tr.ant-table-row"

# Daily/Weekly toggle
DAILY_BUTTON = ".select-button span:text-is('日线')"
WEEKLY_BUTTON = ".select-button span:text-is('周线')"

# Date picker (Ant Design) — kept for CDP auth probe; the in-page-fetch
# path (DESIGN D-9) does NOT click this; it passes the date as a query
# param.
DATE_INPUT = ".ant-picker-input input"
DATE_DISPLAY = ".ant-picker-input input"  # value attribute has the date

# Search/Query button (查 询) — kept for CDP auth flow + detail page.
# The QU100 fetch path no longer needs it (DESIGN D-3).
SEARCH_BUTTON = "button.ant-btn-default.button"

# Last updated text
UPDATE_TIME = ".update-time"

# ---------------------------------------------------------------------------
# QU100 in-page-fetch endpoint (DESIGN D-7)
# ---------------------------------------------------------------------------
# The SPA fetches /api/v3/mf100 with three query params: date (YYYY-MM-DD),
# top (true|false), frequency (daily). Cloudflare cf_clearance is bound to
# the page's TLS handshake, so the request MUST originate from inside the
# rendered page (page.evaluate(fetch(...))). See SPIKE-qu-json-api.md.
QU100_API_URL = "/api/v3/mf100"

# ---------------------------------------------------------------------------
# Detail page (QU Stock Capital Flow) — TODO: needs separate HTML inspection
# ---------------------------------------------------------------------------
CAPITAL_FLOW_NAV = "text=QU Stock Capital Flow"  # TODO: sidebar link
TICKER_INPUT = 'input[placeholder*="Ticker"]'  # TODO: discover
DETAIL_SEARCH_BUTTON = ".ant-btn:has-text('Search')"  # TODO: verify
DAILY_RANK_TABLE = ".ant-table"  # TODO: may need more specific selector
WEEKLY_RANK_TABLE = ".ant-table"  # TODO: may need more specific selector

# JavaScript to extract rows from daily/weekly rank tables on detail page.
# TODO: update after detail page HTML inspection.
DETAIL_TABLE_EXTRACT_JS = """
(tableSelector) => {
    const rows = document.querySelectorAll(tableSelector + ' .ant-table-tbody tr.ant-table-row');
    return Array.from(rows).map(row => {
        const cells = row.querySelectorAll('td.ant-table-cell');
        return {
            date: cells[0]?.textContent?.trim() || '',
            direction: cells[1]?.textContent?.trim() || '',
            long_short: cells[2]?.textContent?.trim() || '',
            rank: cells[3]?.textContent?.trim() || '',
        };
    });
}
"""
