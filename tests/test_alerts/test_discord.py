"""Tests for Discord stock candidate alerts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rainier.alerts.discord import (
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
