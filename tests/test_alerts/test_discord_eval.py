"""Tests for Discord eval-report renderer (PR2)."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from rainier.alerts.discord import _format_eval_message, send_eval_report
from rainier.core.config import DiscordConfig
from rainier.llm_thesis.eval import HitRate, SignalContribution


def _hit_rates(setup_n=4, setup_wr=0.75, watch_n=2, watch_wr=1.0):
    return {
        "setup_long": [
            HitRate("setup_long", "1d", setup_n, setup_wr, 0.01),
            HitRate("setup_long", "5d", setup_n, setup_wr, 0.02),
            HitRate("setup_long", "10d", setup_n, setup_wr, 0.03),
        ],
        "watch": [
            HitRate("watch", "1d", watch_n, watch_wr, 0.0),
            HitRate("watch", "5d", watch_n, watch_wr, 0.0),
            HitRate("watch", "10d", watch_n, watch_wr, 0.0),
        ],
        "no_setup": [
            HitRate("no_setup", "1d", 0, 0.0, 0.0),
            HitRate("no_setup", "5d", 0, 0.0, 0.0),
            HitRate("no_setup", "10d", 0, 0.0, 0.0),
        ],
    }


def test_format_eval_message_renders_all_sections():
    rows = [
        {
            "scan_date": date(2026, 5, 6),
            "session_name": "afternoon",
            "symbol": "NVDA",
            "verdict": "setup_long",
            "llm_confidence": 8,
            "return_pct": 0.023,
            "hit": True,
        },
        {
            "scan_date": date(2026, 5, 6),
            "session_name": "afternoon",
            "symbol": "AAPL",
            "verdict": "watch",
            "llm_confidence": 5,
            "return_pct": -0.012,
            "hit": False,
        },
    ]
    contribs = [
        SignalContribution(
            "rank_trajectory", n_used=78, n_absent=80,
            mean_used=0.012, mean_absent=0.006, lift=0.006, p_value=0.012,
        ),
        SignalContribution(
            "fundamentals", n_used=78, n_absent=80,
            mean_used=0.001, mean_absent=0.0, lift=0.001, p_value=0.41,
        ),
    ]

    msg = _format_eval_message(
        eval_date=date(2026, 5, 7),
        yesterday_rows=rows,
        base_rates=_hit_rates(),
        signal_contribs=contribs,
    )

    assert "Eval report" in msg
    assert "2026-05-07" in msg
    # Yesterday's picks rendered with HIT/miss markers.
    assert "[HIT]" in msg and "NVDA" in msg
    assert "[miss]" in msg and "AAPL" in msg
    # Base rates section present.
    assert "30-day base rates" in msg
    assert "setup_long" in msg
    assert "watch" in msg
    # Signal contribution: rank_trajectory passes p<0.05; fundamentals doesn't.
    assert "rank_trajectory" in msg
    # The fundamentals row p>0.05 must be filtered out of the "significant" list.
    assert "fundamentals" not in msg or msg.count("fundamentals") == 0


def test_format_eval_message_under_1800_chars():
    rows = [
        {
            "scan_date": date(2026, 5, 6),
            "session_name": "afternoon",
            "symbol": "NVDA",
            "verdict": "setup_long",
            "llm_confidence": 8,
            "return_pct": 0.023,
            "hit": True,
        }
    ] * 10
    contribs = [
        SignalContribution(
            f"sig_{i}", n_used=20, n_absent=20,
            mean_used=0.01, mean_absent=0.0, lift=0.01, p_value=0.04,
        )
        for i in range(20)
    ]
    msg = _format_eval_message(
        eval_date=date(2026, 5, 7),
        yesterday_rows=rows,
        base_rates=_hit_rates(),
        signal_contribs=contribs,
    )
    assert len(msg) <= 1800


def test_format_eval_message_no_picks():
    """No graded picks yet (e.g. first 5 days of operation) — no crash."""
    msg = _format_eval_message(
        eval_date=date(2026, 5, 7),
        yesterday_rows=[],
        base_rates=_hit_rates(setup_n=0, setup_wr=0.0, watch_n=0, watch_wr=0.0),
        signal_contribs=[],
    )
    assert "no graded picks" in msg.lower()
    assert "insufficient data" in msg.lower()


def test_format_eval_message_no_significant_signals():
    """When no signal passes the p threshold, render the explicit "(no signals
    with p<X)" message instead of an empty bullet list."""
    contribs = [
        SignalContribution(
            "noise", n_used=20, n_absent=20,
            mean_used=0.0, mean_absent=0.0, lift=0.0, p_value=0.40,
        ),
    ]
    msg = _format_eval_message(
        eval_date=date(2026, 5, 7),
        yesterday_rows=[],
        base_rates=_hit_rates(),
        signal_contribs=contribs,
    )
    # "no signals with p<0.05" line must appear so the user understands why
    # the contribution block is empty.
    assert "no signals" in msg.lower()


def test_send_eval_report_skips_when_disabled():
    cfg = DiscordConfig(enabled=False, webhook_url="https://x")
    with patch("rainier.alerts.discord.httpx.post") as mock_post:
        send_eval_report(
            eval_date=date(2026, 5, 7),
            yesterday_rows=[], base_rates=_hit_rates(), signal_contribs=[],
            config=cfg,
        )
    mock_post.assert_not_called()


def test_send_eval_report_skips_when_no_webhook():
    cfg = DiscordConfig(enabled=True, webhook_url="")
    with patch("rainier.alerts.discord.httpx.post") as mock_post:
        send_eval_report(
            eval_date=date(2026, 5, 7),
            yesterday_rows=[], base_rates=_hit_rates(), signal_contribs=[],
            config=cfg,
        )
    mock_post.assert_not_called()


def test_send_eval_report_posts_when_enabled():
    cfg = DiscordConfig(enabled=True, webhook_url="https://example/test")
    with patch("rainier.alerts.discord.httpx.post") as mock_post:
        mock_post.return_value.raise_for_status = lambda: None
        send_eval_report(
            eval_date=date(2026, 5, 7),
            yesterday_rows=[
                {
                    "scan_date": date(2026, 5, 6),
                    "session_name": "afternoon",
                    "symbol": "NVDA",
                    "verdict": "setup_long",
                    "llm_confidence": 8,
                    "return_pct": 0.02,
                    "hit": True,
                }
            ],
            base_rates=_hit_rates(),
            signal_contribs=[],
            config=cfg,
        )
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs.get("json", {}).get("content")
