"""Notification sender using Apprise — supports email, Slack, Telegram, etc."""

from __future__ import annotations

import apprise
import structlog

from rainier.core.config import get_settings
from rainier.scrapers.base import ScrapeResult

log = structlog.get_logger()


def _get_apprise() -> apprise.Apprise | None:
    """Build an Apprise instance from config. Returns None if disabled or not configured."""
    settings = get_settings()

    if not settings.notify.enabled:
        log.debug("notifications_disabled")
        return None

    urls = settings.notify_urls.strip()
    if not urls:
        log.debug("no_notify_urls_configured")
        return None

    ap = apprise.Apprise()
    for url in urls.split(","):
        url = url.strip()
        if url:
            ap.add(url)

    if len(ap) == 0:
        log.warning("no_valid_notify_urls")
        return None

    return ap


def notify_scrape_result(session_name: str, result: ScrapeResult) -> None:
    """Send a notification after a successful scrape."""
    ap = _get_apprise()
    if ap is None:
        return

    settings = get_settings()
    prefix = settings.notify.subject_prefix

    if result.errors:
        title = f"{prefix} Scrape partial: {session_name}"
        body = (
            f"Session: {session_name}\n"
            f"Records: {result.records_created}\n"
            f"Errors: {len(result.errors)}\n"
            f"Duration: {result.duration_seconds:.1f}s\n\n"
            "Errors:\n" + "\n".join(f"  - {e}" for e in result.errors)
        )
        notify_type = apprise.NotifyType.WARNING
    else:
        title = f"{prefix} Scrape OK: {session_name}"
        body = (
            f"Session: {session_name}\n"
            f"Records: {result.records_created}\n"
            f"Duration: {result.duration_seconds:.1f}s"
        )
        notify_type = apprise.NotifyType.SUCCESS

    _send(ap, title, body, notify_type)


def notify_scrape_failure(session_name: str, error: str) -> None:
    """Send a notification when a scrape fails completely."""
    ap = _get_apprise()
    if ap is None:
        return

    settings = get_settings()
    prefix = settings.notify.subject_prefix

    title = f"{prefix} Scrape FAILED: {session_name}"
    body = f"Session: {session_name}\nError: {error}"

    _send(ap, title, body, apprise.NotifyType.FAILURE)


def _send(ap: apprise.Apprise, title: str, body: str, notify_type: str) -> None:
    """Send notification, catching all errors to never crash the scheduler."""
    try:
        ok = ap.notify(title=title, body=body, notify_type=notify_type)
        if ok:
            log.info("notification_sent", title=title)
        else:
            log.warning("notification_failed", title=title)
    except Exception as exc:
        log.error("notification_error", error=str(exc))
