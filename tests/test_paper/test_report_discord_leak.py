"""Regression: the daily paper-report Discord error path must never leak the
webhook token into logs.

``httpx.HTTPStatusError.__str__`` embeds the request URL, and a Discord webhook
URL carries its secret token as the last path segment. ``send_daily_paper_report``
catches send failures and used to ``log.exception`` the full traceback — which
serializes ``str(exc)`` and thus the token — into ``data/qu-scrape.log``. This is
the same P1 leak class fixed in 96fbd13 (Discord client) and the weekly sweep.

This file is UNMARKED on purpose: it mocks ``send_daily_report`` (httpx) and
captures the stdlib ``rainier.paper.report`` logger via ``caplog``. No DB is
touched, so it must NOT sit behind ``requires_postgres`` — otherwise the
regression is silently skipped in non-Postgres lanes.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from rainier.paper.report import send_daily_paper_report

PAYLOAD = {
    "report_type": "daily",
    "as_of_date": "2026-06-17",
    "counts_by_status": {"pending": 0, "open": 0, "closed": 0, "expired": 0},
    "realized_pnl": 0.0,
    "mtm_including_open_pnl": 0.0,
    "open_mtm_pnl": 0.0,
    "todays_exits": [],
    "win_rate_closed": None,
    "closed_trades": 0,
    "same_bar_ambiguous_exits": 0,
    "total_residual_cash": 0.0,
    "stale_mtm_symbols": [],
    "disclosure": "",
}


def _status_error(url: str) -> httpx.HTTPStatusError:
    """Build the real leak vector: raise_for_status auto-generates a message
    that embeds the request URL (which carries the token)."""
    req = httpx.Request("POST", url)
    resp = httpx.Response(401, request=req)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return exc
    raise AssertionError("raise_for_status did not raise")  # pragma: no cover


def test_daily_report_failure_never_leaks_webhook_token(caplog):
    secret = "SUPERSECRETtoken9999"
    url = f"https://discord.com/api/webhooks/123456/{secret}"
    config = SimpleNamespace(enabled=True, webhook_url=url)

    err = _status_error(url)
    # Guard: prove the test exercises the REAL leak vector, not a no-op.
    assert secret in str(err)

    with caplog.at_level(logging.INFO, logger="rainier.paper.report"):
        with patch(
            "rainier.alerts.discord.send_daily_report", side_effect=err
        ):
            result = send_daily_paper_report(PAYLOAD, config)

    # Non-fatal contract: returns False, never raises.
    assert result is False

    # A failure record was emitted with the exception class diagnostic.
    failed = [
        r for r in caplog.records if "paper_report_discord_failed" in r.getMessage()
    ]
    assert failed, "expected a paper_report_discord_failed log record"
    assert "HTTPStatusError" in failed[0].getMessage()

    # The token (and the webhook path) must not appear anywhere in the logs —
    # serialize every record the way the discord.py regression test does.
    blob = json.dumps(
        [r.getMessage() for r in caplog.records]
        + [(r.exc_text or "") for r in caplog.records]
    )
    assert secret not in blob, "webhook token leaked into report logs"
    assert "webhooks/123456" not in blob
