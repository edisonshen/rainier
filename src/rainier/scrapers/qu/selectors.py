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

# Ant Design table
QU100_TABLE = ".ant-table"
QU100_TABLE_ROW = ".ant-table-tbody tr.ant-table-row"

# Top100/Bottom100 toggle — use text selectors (stable across DOM changes)
TOP100_BUTTON = ".select-button span:text-is('前100')"
BOTTOM100_BUTTON = ".select-button span:text-is('后100')"

# Daily/Weekly toggle
DAILY_BUTTON = ".select-button span:text-is('日线')"
WEEKLY_BUTTON = ".select-button span:text-is('周线')"

# Date picker (Ant Design)
DATE_INPUT = ".ant-picker-input input"
DATE_DISPLAY = ".ant-picker-input input"  # value attribute has the date

# Search/Query button (查 询) — MUST click after any toggle or date change
SEARCH_BUTTON = "button.ant-btn-default.button"

# Global search input within the table
TABLE_SEARCH_INPUT = ".table-input input.ant-input"

# Last updated text
UPDATE_TIME = ".update-time"

# JavaScript to extract rows from the QU100 Ant Design table.
# The table has 6 columns: Rank, Ticker, Daily Change, Sector, Industry, Long/Short
# - Ticker is inside <div class="purple">SYMBOL</div>
# - Daily change: <span class="green">▲</span><span>92</span> or "0" or "New"
QU100_EXTRACT_JS = """
() => {
    const rows = document.querySelectorAll('.ant-table-tbody tr.ant-table-row');
    return Array.from(rows).map(row => {
        const cells = row.querySelectorAll('td.ant-table-cell');
        // Daily change: get the raw text (includes ▲/▼ and number)
        const changeCell = cells[2];
        let dailyChange = '0';
        if (changeCell) {
            const spans = changeCell.querySelectorAll('span');
            if (spans.length >= 2) {
                // Has arrow span + number span
                const arrow = spans[0]?.textContent?.trim() || '';
                const num = spans[1]?.textContent?.trim() || '0';
                dailyChange = arrow + num;
            } else {
                dailyChange = changeCell.textContent?.trim() || '0';
            }
        }
        return {
            rank: cells[0]?.textContent?.trim() || '',
            symbol: cells[1]?.textContent?.trim() || '',
            daily_change: dailyChange,
            sector: cells[3]?.textContent?.trim() || '',
            industry: cells[4]?.textContent?.trim() || '',
            long_short: cells[5]?.textContent?.trim() || '',
        };
    });
}
"""

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
