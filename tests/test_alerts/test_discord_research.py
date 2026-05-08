"""Tests for the Discord research-report renderer (PR3)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import patch

from rainier.alerts.discord import _format_research_message, send_research_report
from rainier.core.config import DiscordConfig
from rainier.core.models import ResearchInsight


def _ins(
    *,
    id: int,
    kind: str = "signal_underperform",
    subject: str = "rank_trajectory",
    severity: str = "warn",
    rationale: str = "Default rationale.",
    recurrence: int = 1,
    action: dict | None = None,
    status: str = "pending",
) -> ResearchInsight:
    return ResearchInsight(
        id=id,
        kind=kind,
        subject=subject,
        severity=severity,
        evidence={"sample": 1},
        action=action or {"kind": "disable_signal", "target": subject, "params": {}},
        rationale=rationale,
        recurrence_count=recurrence,
        status=status,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def test_format_research_message_renders_pending_only():
    """Accepted/rejected/stale rows must NOT appear — only pending."""
    insights = [
        _ins(id=1, severity="warn", status="pending"),
        _ins(id=2, severity="warn", status="accepted"),
        _ins(id=3, severity="critical", status="rejected"),
        _ins(id=4, severity="critical", status="pending"),
    ]
    msg = _format_research_message(
        eval_date=date(2026, 5, 7), insights=insights
    )
    assert "#1" in msg
    assert "#4" in msg
    assert "#2" not in msg
    assert "#3" not in msg


def test_format_research_message_filters_info_by_default():
    """info-severity insights are suggestions; the report shows warn+critical
    only by default."""
    insights = [
        _ins(id=1, severity="info", status="pending"),
        _ins(id=2, severity="warn", status="pending"),
    ]
    msg = _format_research_message(
        eval_date=date(2026, 5, 7), insights=insights
    )
    assert "#2" in msg
    assert "#1" not in msg


def test_format_research_message_orders_critical_first():
    insights = [
        _ins(id=1, severity="warn", status="pending"),
        _ins(id=2, severity="critical", status="pending"),
        _ins(id=3, severity="warn", status="pending"),
    ]
    msg = _format_research_message(
        eval_date=date(2026, 5, 7), insights=insights
    )
    crit_idx = msg.index("#2")
    warn1_idx = msg.index("#1")
    warn3_idx = msg.index("#3")
    assert crit_idx < warn1_idx
    assert crit_idx < warn3_idx


def test_format_research_message_no_pending_insights():
    msg = _format_research_message(
        eval_date=date(2026, 5, 7), insights=[]
    )
    assert "No new pending" in msg


def test_format_research_message_under_1800_chars():
    insights = [
        _ins(
            id=i, severity="warn",
            rationale="rationale " * 30,  # long rationale
        )
        for i in range(20)
    ]
    msg = _format_research_message(
        eval_date=date(2026, 5, 7), insights=insights
    )
    assert len(msg) <= 1800


def test_format_research_message_recurrence_label():
    insights = [
        _ins(id=1, severity="warn", recurrence=3),
        _ins(id=2, severity="warn", recurrence=1),
    ]
    msg = _format_research_message(
        eval_date=date(2026, 5, 7), insights=insights
    )
    assert "(x3)" in msg
    # Recurrence=1 doesn't get a label (would be noisy).
    one_line = next(line for line in msg.splitlines() if "#2" in line)
    assert "(x1)" not in one_line


def test_format_research_message_includes_action_kind():
    insights = [
        _ins(
            id=1, severity="critical", kind="verdict_drift",
            action={"kind": "bump_prompt_version", "target": "v1", "params": {}},
        ),
    ]
    msg = _format_research_message(
        eval_date=date(2026, 5, 7), insights=insights
    )
    assert "bump_prompt_version" in msg


def test_send_research_report_skips_when_disabled():
    cfg = DiscordConfig(enabled=False, webhook_url="https://x")
    with patch("rainier.alerts.discord.httpx.post") as mock_post:
        send_research_report(
            eval_date=date(2026, 5, 7), insights=[], config=cfg,
        )
    mock_post.assert_not_called()


def test_send_research_report_skips_when_no_webhook():
    cfg = DiscordConfig(enabled=True, webhook_url="")
    with patch("rainier.alerts.discord.httpx.post") as mock_post:
        send_research_report(
            eval_date=date(2026, 5, 7), insights=[], config=cfg,
        )
    mock_post.assert_not_called()


def test_format_research_message_scrubs_at_mentions_in_subject():
    """[P2 from review] Adversarial LLM-generated `subject` text containing
    `@everyone` or `@here` must NOT pass through to Discord — the rendered
    message must replace `@` with a non-mention variant.
    """
    insights = [
        _ins(
            id=1, severity="warn",
            subject="@everyone bad",
            rationale="@here please review",
        ),
    ]
    msg = _format_research_message(
        eval_date=date(2026, 5, 7), insights=insights
    )
    assert "@everyone" not in msg
    assert "@here" not in msg
    # The fullwidth-at variant survives so the operator can still read it.
    assert "＠" in msg


def test_format_research_message_scrubs_backticks():
    """Backticks in rationale could break the surrounding code-block block
    formatting; they must be replaced (or stripped)."""
    insights = [
        _ins(
            id=1, severity="warn",
            rationale="bad `code injection` attempt",
        ),
    ]
    msg = _format_research_message(
        eval_date=date(2026, 5, 7), insights=insights
    )
    assert "`code" not in msg


def test_send_research_report_posts_when_enabled():
    cfg = DiscordConfig(enabled=True, webhook_url="https://example/test")
    with patch("rainier.alerts.discord.httpx.post") as mock_post:
        mock_post.return_value.raise_for_status = lambda: None
        send_research_report(
            eval_date=date(2026, 5, 7),
            insights=[_ins(id=1, severity="warn", status="pending")],
            config=cfg,
        )
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs.get("json", {}).get("content")
