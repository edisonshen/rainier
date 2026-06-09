"""Tests for Discord stock candidate alerts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from rainier.alerts.discord import (
    DiscordSendResult,
    _build_payloads,
    _format_candidate_embed,
    _format_summary_embed,
    format_stock_candidates_json,
    send_stock_candidates,
)
from rainier.core.config import DiscordConfig
from rainier.core.types import StockCandidate


def _candidate(
    symbol: str = "NVDA",
    rank: int = 1,
    rank_change: int = 3,
    pattern_type: str | None = "w_bottom",
    pattern_direction: str | None = "bullish",
    pattern_status: str | None = "confirmed",
    pattern_confidence: float | None = 0.85,
    entry_price: float | None = 142.50,
    stop_loss: float | None = 135.00,
    target_price: float | None = 165.00,
    rr_ratio: float | None = 3.0,
    volume_confirmed: bool = True,
) -> StockCandidate:
    return StockCandidate(
        symbol=symbol,
        rank=rank,
        rank_change=rank_change,
        long_short="Long in",
        capital_flow_direction="+",
        sector="Technology",
        signal_strength=0.9,
        pattern_type=pattern_type,
        pattern_direction=pattern_direction,
        pattern_status=pattern_status,
        pattern_confidence=pattern_confidence,
        entry_price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        rr_ratio=rr_ratio,
        volume_confirmed=volume_confirmed,
    )


class TestFormatSummaryEmbed:

    def test_title_contains_count(self):
        candidates = [_candidate(symbol="NVDA"), _candidate(symbol="TSLA", rank=5)]
        embed = _format_summary_embed(candidates)
        assert "2" in embed["title"]

    def test_description_has_code_block(self):
        candidates = [_candidate()]
        embed = _format_summary_embed(candidates)
        assert "```" in embed["description"]

    def test_shows_all_symbols(self):
        candidates = [
            _candidate(symbol="NVDA"),
            _candidate(symbol="TSLA", rank=5),
            _candidate(symbol="AAPL", rank=8),
        ]
        embed = _format_summary_embed(candidates)
        for sym in ["NVDA", "TSLA", "AAPL"]:
            assert sym in embed["description"]

    def test_shows_pattern_label(self):
        candidates = [_candidate(pattern_type="w_bottom")]
        embed = _format_summary_embed(candidates)
        assert "W Bottom" in embed["description"]

    def test_no_pattern_shows_dash(self):
        candidates = [_candidate(pattern_type=None, pattern_confidence=None)]
        embed = _format_summary_embed(candidates)
        desc = embed["description"]
        # Should have dashes for pattern and confidence
        assert "-" in desc


class TestFormatCandidateEmbed:

    def test_bullish_is_green(self):
        embed = _format_candidate_embed(_candidate(pattern_direction="bullish"))
        assert embed["color"] == 0x00E676

    def test_bearish_is_red(self):
        embed = _format_candidate_embed(_candidate(pattern_direction="bearish"))
        assert embed["color"] == 0xFF1744

    def test_has_entry_sl_target_fields(self):
        embed = _format_candidate_embed(_candidate())
        field_names = {f["name"] for f in embed["fields"]}
        assert "Entry" in field_names
        assert "Stop Loss" in field_names
        assert "Target" in field_names
        assert "R:R" in field_names

    def test_has_pattern_label(self):
        embed = _format_candidate_embed(_candidate(pattern_type="bull_flag"))
        pattern_field = next(f for f in embed["fields"] if f["name"] == "Pattern")
        assert pattern_field["value"] == "Bull Flag"

    def test_volume_confirmed_shows_checkmark(self):
        embed = _format_candidate_embed(_candidate(volume_confirmed=True))
        vol_field = next(f for f in embed["fields"] if f["name"] == "Volume")
        assert "\u2705" in vol_field["value"]

    def test_volume_not_confirmed_shows_x(self):
        embed = _format_candidate_embed(_candidate(volume_confirmed=False))
        vol_field = next(f for f in embed["fields"] if f["name"] == "Volume")
        assert "\u274c" in vol_field["value"]

    def test_omits_none_price_fields(self):
        embed = _format_candidate_embed(
            _candidate(entry_price=None, stop_loss=None, target_price=None, rr_ratio=None)
        )
        field_names = {f["name"] for f in embed["fields"]}
        assert "Entry" not in field_names
        assert "Stop Loss" not in field_names
        assert "Target" not in field_names


class TestBuildPayloads:

    def test_single_payload_for_small_set(self):
        candidates = [_candidate(symbol=f"SYM{i}") for i in range(5)]
        payloads = _build_payloads(candidates)
        assert len(payloads) == 1
        # 1 summary + 5 detail embeds = 6 total
        assert len(payloads[0]["embeds"]) == 6

    def test_splits_beyond_10_embeds(self):
        # 12 candidates with patterns → 1 summary + 12 details = 13 → 2 payloads
        candidates = [_candidate(symbol=f"S{i}", rank=i) for i in range(12)]
        payloads = _build_payloads(candidates)
        assert len(payloads) == 2
        assert len(payloads[0]["embeds"]) == 10
        assert len(payloads[1]["embeds"]) == 3

    def test_no_detail_for_candidates_without_pattern(self):
        candidates = [_candidate(pattern_type=None)]
        payloads = _build_payloads(candidates)
        assert len(payloads) == 1
        assert len(payloads[0]["embeds"]) == 1  # summary only


class TestSendStockCandidates:

    def test_skips_when_empty(self):
        config = DiscordConfig(enabled=True, webhook_url="https://example.com/webhook")
        with patch("rainier.alerts.discord.httpx.post") as mock_post:
            send_stock_candidates([], config)
            mock_post.assert_not_called()

    def test_skips_when_disabled(self):
        config = DiscordConfig(enabled=False, webhook_url="https://example.com/webhook")
        with patch("rainier.alerts.discord.httpx.post") as mock_post:
            send_stock_candidates([_candidate()], config)
            mock_post.assert_not_called()

    def test_skips_when_no_webhook(self):
        config = DiscordConfig(enabled=True, webhook_url="", stock_webhook_url="")
        with patch("rainier.alerts.discord.httpx.post") as mock_post:
            send_stock_candidates([_candidate()], config)
            mock_post.assert_not_called()

    def test_uses_stock_webhook_when_set(self):
        config = DiscordConfig(
            enabled=True,
            webhook_url="https://main.com/hook",
            stock_webhook_url="https://stock.com/hook",
        )
        with patch("rainier.alerts.discord.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            send_stock_candidates([_candidate()], config)
            mock_post.assert_called()
            call_url = mock_post.call_args[0][0]
            assert call_url == "https://stock.com/hook"

    def test_falls_back_to_main_webhook(self):
        config = DiscordConfig(
            enabled=True,
            webhook_url="https://main.com/hook",
            stock_webhook_url="",
        )
        with patch("rainier.alerts.discord.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            send_stock_candidates([_candidate()], config)
            call_url = mock_post.call_args[0][0]
            assert call_url == "https://main.com/hook"

    def test_sends_valid_payload(self):
        config = DiscordConfig(enabled=True, webhook_url="https://example.com/hook")
        with patch("rainier.alerts.discord.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            send_stock_candidates([_candidate()], config)

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs["json"]
            assert "embeds" in payload
            assert len(payload["embeds"]) == 2  # 1 summary + 1 detail

    def test_handles_http_error_gracefully(self):
        config = DiscordConfig(enabled=True, webhook_url="https://example.com/hook")
        with patch("rainier.alerts.discord.httpx.post") as mock_post:
            mock_post.side_effect = Exception("Connection refused")
            # Should not raise
            send_stock_candidates([_candidate()], config)


class TestSendResultAccounting:
    """The return value must reflect what actually LANDED, so callers (and the
    pipeline's discord_sent log) stop reporting false success when the webhook
    silently fails. Regression for the 2026-06-09 'channel empty but logs say
    sent 20' incident."""

    def test_success_reports_truthful_counts(self):
        config = DiscordConfig(enabled=True, webhook_url="https://example.com/hook")
        with patch("rainier.alerts.discord.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            result = send_stock_candidates([_candidate()], config)
        assert isinstance(result, DiscordSendResult)
        assert result.candidates_reported == 1
        assert result.candidate_payloads_ok == 1
        assert result.candidate_payloads_failed == 0
        assert result.fully_ok is True
        assert result.summary_ok is True

    def test_partial_delivery_summary_landed(self):
        """If the summary payload (idx 0, the full candidate table) lands but a
        later detail payload fails, summary_ok stays True (operator saw the
        report) while fully_ok is False and the failure is counted. Pipeline
        logs discord_sent=len(candidates), not 0. (codex [P2] 2026-06-09.)"""
        config = DiscordConfig(enabled=True, webhook_url="https://example.com/hook")
        # 10 patterned candidates → 11 embeds → 2 payloads (summary+9, then 1).
        candidates = [_candidate(symbol=f"S{i}") for i in range(10)]
        assert len(_build_payloads(candidates)) == 2  # guards the fixture intent
        calls = {"n": 0}

        def _side_effect(url, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return MagicMock(status_code=204, raise_for_status=lambda: None)
            raise httpx.HTTPError("simulated detail-payload 429 rate-limit")

        with patch("rainier.alerts.discord.httpx.post", side_effect=_side_effect):
            result = send_stock_candidates(candidates, config)

        assert result.summary_ok is True       # table landed
        assert result.candidate_payloads_ok == 1
        assert result.candidate_payloads_failed == 1
        assert result.fully_ok is False        # but not every payload

    def test_failed_post_is_not_fully_ok(self):
        config = DiscordConfig(enabled=True, webhook_url="https://example.com/hook")
        with patch("rainier.alerts.discord.httpx.post") as mock_post:
            mock_post.side_effect = httpx.HTTPError("simulated 404 deleted webhook")
            result = send_stock_candidates([_candidate()], config)
        assert result.candidate_payloads_failed == 1
        assert result.candidate_payloads_ok == 0
        assert result.fully_ok is False
        assert result.summary_ok is False      # the table itself never landed

    def test_disabled_or_no_webhook_is_not_fully_ok(self):
        # enabled=False → nothing sent → fully_ok False (no false success).
        disabled = DiscordConfig(enabled=False, webhook_url="https://x/hook")
        r1 = send_stock_candidates([_candidate()], disabled)
        assert r1.candidate_payloads_ok == 0 and r1.fully_ok is False
        # no webhook URL → same.
        nowebhook = DiscordConfig(enabled=True, webhook_url="", stock_webhook_url="")
        r2 = send_stock_candidates([_candidate()], nowebhook)
        assert r2.candidate_payloads_ok == 0 and r2.fully_ok is False

    def test_empty_candidates_returns_zero_result(self):
        config = DiscordConfig(enabled=True, webhook_url="https://example.com/hook")
        result = send_stock_candidates([], config)
        assert result.candidates_reported == 0
        assert result.fully_ok is False

    def test_per_operation_logs_emitted(self):
        """Each send operation logs a structured event so the channel-delivery
        path is observable in data/qu-scrape.log."""
        from structlog.testing import capture_logs

        config = DiscordConfig(enabled=True, webhook_url="https://example.com/hook")
        with capture_logs() as caplogs:
            with patch("rainier.alerts.discord.httpx.post") as mock_post:
                mock_post.return_value = MagicMock(status_code=204)
                send_stock_candidates([_candidate()], config)

        events = [e.get("event", "") for e in caplogs]
        assert "discord_candidates_send_starting" in events
        assert "discord_payload_ok" in events
        assert "discord_candidates_send_done" in events

    def test_failed_payload_logs_status_and_error(self):
        """A failed POST logs discord_payload_failed with the HTTP status so
        the operator can tell 404 (deleted webhook) from 429 (rate-limit)."""
        from structlog.testing import capture_logs

        config = DiscordConfig(enabled=True, webhook_url="https://example.com/hook")
        resp = MagicMock(status_code=404)
        err = httpx.HTTPStatusError("404", request=MagicMock(), response=resp)
        with capture_logs() as caplogs:
            with patch("rainier.alerts.discord.httpx.post") as mock_post:
                mock_post.return_value = MagicMock(
                    raise_for_status=MagicMock(side_effect=err)
                )
                send_stock_candidates([_candidate()], config)

        failed = [e for e in caplogs if e.get("event") == "discord_payload_failed"]
        assert failed, "expected a discord_payload_failed event"
        assert failed[0].get("status") == 404


class TestFormatJson:

    def test_empty_returns_empty_array(self):
        assert format_stock_candidates_json([]) == "[]"

    def test_returns_valid_json(self):
        import json
        result = format_stock_candidates_json([_candidate()])
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert "embeds" in parsed[0]


class TestSendStockCandidatesWithTheses:
    """PR1 regression: theses=None preserves existing behavior; theses dict adds messages."""

    def _thesis(self, **overrides):
        defaults = dict(
            verdict="setup_long",
            setup_quality=7,
            llm_confidence=7,
            paragraph_radar="Why on radar.",
            paragraph_evidence="Evidence text.",
            paragraph_invalidation="Invalidation rules.",
            risks=["earnings"],
            watch_items=["volume"],
            evidence_used=["pattern"],
            signals_used=["rank_trajectory"],
            patterns_in_chart_not_in_indicators=["narrowing volume"],
        )
        defaults.update(overrides)
        return defaults

    def test_theses_none_preserves_existing_call_count(self):
        """Without theses, only the embeds payload(s) get posted — no per-ticker text messages."""
        config = DiscordConfig(enabled=True, webhook_url="https://example.com/hook")
        with patch("rainier.alerts.discord.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            send_stock_candidates([_candidate()], config)
            # 1 embed payload only.
            assert mock_post.call_count == 1

    def test_theses_dict_emits_summary_plus_thesis_messages(self):
        config = DiscordConfig(enabled=True, webhook_url="https://example.com/hook")
        candidates = [_candidate(symbol=f"S{i}") for i in range(5)]
        theses = {f"S{i}": self._thesis() for i in range(5)}
        with patch("rainier.alerts.discord.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            send_stock_candidates(candidates, config, theses=theses)
            # 1 embed payload + 5 thesis text messages = 6 calls minimum.
            assert mock_post.call_count == 6

    def test_thesis_message_under_1800_chars(self):
        from rainier.alerts.discord import _format_thesis_message
        long_thesis = self._thesis(
            paragraph_evidence="x" * 600, paragraph_radar="y" * 400,
            paragraph_invalidation="z" * 400,
        )
        msg = _format_thesis_message(_candidate(symbol="NVDA"), long_thesis)
        assert len(msg) <= 1800

    def test_thesis_only_renders_top_5_in_candidate_order(self):
        config = DiscordConfig(enabled=True, webhook_url="https://example.com/hook")
        candidates = [_candidate(symbol=f"S{i}") for i in range(8)]
        theses = {c.symbol: self._thesis() for c in candidates}
        with patch("rainier.alerts.discord.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            send_stock_candidates(candidates, config, theses=theses)
        # 1 embed payload + 5 thesis messages (top-5 only) = 6 total.
        assert mock_post.call_count == 6
