"""Tests for the PR5 Discord webhook multipart attachment path.

These cover:
  * ``send_stock_candidates(theses=...)`` POSTs multipart/form-data when
    chart bytes load successfully — verifies ``payload_json`` + ``files[0]``
    structure.
  * Chart bytes are loaded from ``ChartImage`` (via
    ``dashboard.data.load_thesis_chart``), NOT re-rendered by kaleido.
  * Missing chart bytes (eg. cache hit returning None) → embed posts
    without the ``image`` field; no crash.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rainier.alerts.discord import send_stock_candidates
from rainier.core.config import DiscordConfig
from rainier.core.types import StockCandidate


def _candidate(symbol: str = "NVDA") -> StockCandidate:
    return StockCandidate(
        symbol=symbol,
        rank=5,
        rank_change=2,
        long_short="Long in",
        capital_flow_direction="+",
        sector="Technology",
        signal_strength=0.8,
        money_flow_score=0.6,
        pattern_type="w_bottom",
        pattern_direction="bullish",
        pattern_status="confirmed",
        pattern_confidence=0.9,
        entry_price=100.0,
        stop_loss=95.0,
        target_price=120.0,
        rr_ratio=4.0,
        volume_confirmed=True,
        current_price=105.0,
    )


def _thesis(**overrides) -> dict:
    base = dict(
        verdict="setup_long",
        setup_quality=7,
        llm_confidence=8,
        paragraph_radar="Strong setup.",
        paragraph_evidence="Volume + sector.",
        paragraph_invalidation="Reject below 95.",
        risks=["earnings"],
        watch_items=["pullback to 100"],
        evidence_used=["pattern"],
        signals_used=["rank_trajectory"],
        patterns_in_chart_not_in_indicators="none",
        _thesis_id=42,
    )
    base.update(overrides)
    return base


class TestMultipartAttachment:

    def test_thesis_with_chart_bytes_posts_multipart(self):
        config = DiscordConfig(enabled=True, webhook_url="https://x/hook")
        png_bytes = b"\x89PNG\r\n\x1a\nFAKE"
        with patch(
            "rainier.alerts.discord.httpx.post"
        ) as mock_post, patch(
            "rainier.alerts.discord._load_chart_bytes_for_thesis",
            return_value=png_bytes,
        ):
            mock_post.return_value = MagicMock(status_code=204)
            send_stock_candidates(
                [_candidate("NVDA")],
                config,
                theses={"NVDA": _thesis()},
            )

        # Top-20 summary call (json=) + thesis call (multipart with files=).
        calls = mock_post.call_args_list
        # Find the multipart call.
        multipart_calls = [c for c in calls if "files" in c.kwargs]
        assert len(multipart_calls) == 1
        call = multipart_calls[0]
        files = call.kwargs["files"]
        assert "files[0]" in files
        filename, body, mime = files["files[0]"]
        assert filename == "chart_NVDA.png"
        assert body == png_bytes
        assert mime == "image/png"
        # payload_json present in the data field.
        assert "data" in call.kwargs
        assert "payload_json" in call.kwargs["data"]

    def test_chart_bytes_loaded_from_db_not_rerendered(self):
        """The Discord renderer never calls kaleido directly — bytes must
        come from the ChartImage row via ``load_thesis_chart``."""
        config = DiscordConfig(enabled=True, webhook_url="https://x/hook")
        with patch(
            "rainier.alerts.discord.httpx.post"
        ) as mock_post, patch(
            "rainier.dashboard.data.load_thesis_chart"
        ) as mock_load, patch(
            "rainier.llm_thesis.chart_export.render_chart_png"
        ) as mock_render:
            mock_post.return_value = MagicMock(status_code=204)
            mock_load.return_value = b"FROM-DB"
            send_stock_candidates(
                [_candidate("NVDA")],
                config,
                theses={"NVDA": _thesis()},
            )

        mock_load.assert_called_once_with(42)
        mock_render.assert_not_called()

    def test_missing_chart_bytes_posts_without_image(self):
        """Cache-hit path returns None bytes — embed posts without image,
        no crash, no multipart."""
        config = DiscordConfig(enabled=True, webhook_url="https://x/hook")
        with patch(
            "rainier.alerts.discord.httpx.post"
        ) as mock_post, patch(
            "rainier.alerts.discord._load_chart_bytes_for_thesis",
            return_value=None,
        ):
            mock_post.return_value = MagicMock(status_code=204)
            send_stock_candidates(
                [_candidate("NVDA")],
                config,
                theses={"NVDA": _thesis()},
            )

        # Find the thesis call — it shouldn't be multipart.
        calls = mock_post.call_args_list
        # The summary embed call uses json=. The thesis call should also
        # use json= (no files) when bytes are missing.
        multipart_calls = [c for c in calls if "files" in c.kwargs]
        assert multipart_calls == []
        # And the thesis embed dict in the json payload should not have
        # an image field.
        json_calls = [c for c in calls if "json" in c.kwargs]
        # Find the thesis call (it has a single embed without table-style
        # description).
        thesis_calls = [
            c
            for c in json_calls
            if isinstance(c.kwargs["json"], dict)
            and isinstance(c.kwargs["json"].get("embeds"), list)
            and len(c.kwargs["json"]["embeds"]) == 1
            and c.kwargs["json"]["embeds"][0].get("title", "").startswith(
                "setup_long"
            )
        ]
        assert len(thesis_calls) == 1
        embed = thesis_calls[0].kwargs["json"]["embeds"][0]
        assert "image" not in embed

    def test_dashboard_base_url_is_passed_through(self):
        config = DiscordConfig(enabled=True, webhook_url="https://x/hook")
        with patch(
            "rainier.alerts.discord.httpx.post"
        ) as mock_post, patch(
            "rainier.alerts.discord._load_chart_bytes_for_thesis",
            return_value=None,
        ):
            mock_post.return_value = MagicMock(status_code=204)
            send_stock_candidates(
                [_candidate("NVDA")],
                config,
                theses={"NVDA": _thesis(_thesis_id=999)},
                dashboard_base_url="http://dashboard.local",
            )
        # Find the thesis call.
        json_calls = [
            c for c in mock_post.call_args_list if "json" in c.kwargs
        ]
        thesis_calls = [
            c
            for c in json_calls
            if isinstance(c.kwargs["json"], dict)
            and len(c.kwargs["json"].get("embeds") or []) == 1
            and c.kwargs["json"]["embeds"][0].get("title", "").startswith("setup_long")
        ]
        assert len(thesis_calls) == 1
        embed = thesis_calls[0].kwargs["json"]["embeds"][0]
        assert embed["url"] == "http://dashboard.local?thesis_id=999"
