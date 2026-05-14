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


# ---------------------------------------------------------------------------
# LLM-channel routing — DISCORD_LLM_WEBHOOK_URL splits LLM thesis embeds from
# the regular QU100 top-20 screener payload.
# ---------------------------------------------------------------------------


class TestLLMChannelRouting:
    """Routing per `_resolve_llm_webhook_url`:

    - Top-20 summary embed: always uses stock_webhook_url (or webhook_url
      fallback).
    - Per-ticker LLM thesis embed: uses llm_webhook_url when set; otherwise
      falls back to stock_webhook_url; otherwise webhook_url.
    """

    def _classify_calls(self, call_args_list):
        """Split httpx.post calls into (summary_calls, thesis_calls).

        Summary calls carry json={"embeds": [...]} where at least one embed
        contains a description with a ```code block``` (the QU100 table).
        Thesis calls either ride on multipart (files=, data=payload_json)
        OR carry a single embed whose title starts with the verdict label.
        """
        summary_calls = []
        thesis_calls = []
        for call in call_args_list:
            kwargs = call.kwargs
            if "files" in kwargs:
                thesis_calls.append(call)
                continue
            payload = kwargs.get("json")
            if not isinstance(payload, dict):
                continue
            embeds = payload.get("embeds") or []
            if not embeds:
                continue
            # Thesis embed: single embed, title starts with verdict label.
            if len(embeds) == 1 and embeds[0].get("title", "").startswith(
                ("setup_long", "watch", "no_setup")
            ):
                thesis_calls.append(call)
            else:
                summary_calls.append(call)
        return summary_calls, thesis_calls

    def test_thesis_embed_routes_to_llm_webhook_when_set(self):
        config = DiscordConfig(
            enabled=True,
            webhook_url="https://main/hook",
            stock_webhook_url="https://stock/hook",
            llm_webhook_url="https://llm/hook",
        )
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
        summary_calls, thesis_calls = self._classify_calls(
            mock_post.call_args_list
        )
        assert summary_calls, "expected at least one summary POST"
        assert thesis_calls, "expected at least one thesis POST"
        for call in summary_calls:
            assert call.args[0] == "https://stock/hook"
        for call in thesis_calls:
            assert call.args[0] == "https://llm/hook"

    def test_thesis_embed_falls_back_to_stock_webhook_when_llm_empty(self):
        config = DiscordConfig(
            enabled=True,
            webhook_url="https://main/hook",
            stock_webhook_url="https://stock/hook",
            llm_webhook_url="",
        )
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
        summary_calls, thesis_calls = self._classify_calls(
            mock_post.call_args_list
        )
        assert summary_calls and thesis_calls
        for call in summary_calls:
            assert call.args[0] == "https://stock/hook"
        for call in thesis_calls:
            assert call.args[0] == "https://stock/hook"

    def test_thesis_embed_falls_back_to_webhook_when_stock_and_llm_empty(self):
        config = DiscordConfig(
            enabled=True,
            webhook_url="https://main/hook",
            stock_webhook_url="",
            llm_webhook_url="",
        )
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
        summary_calls, thesis_calls = self._classify_calls(
            mock_post.call_args_list
        )
        assert summary_calls and thesis_calls
        for call in summary_calls:
            assert call.args[0] == "https://main/hook"
        for call in thesis_calls:
            assert call.args[0] == "https://main/hook"

    def test_regular_candidates_only_use_stock_webhook_regardless_of_llm(self):
        """No `theses=` arg — all calls should land on the stock channel
        even when llm_webhook_url is set."""
        config = DiscordConfig(
            enabled=True,
            webhook_url="https://main/hook",
            stock_webhook_url="https://stock/hook",
            llm_webhook_url="https://llm/hook",
        )
        with patch("rainier.alerts.discord.httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            send_stock_candidates([_candidate("NVDA")], config)
        # No theses → no thesis-embed POSTs; every call must go to stock URL.
        assert mock_post.call_count >= 1
        for call in mock_post.call_args_list:
            assert call.args[0] == "https://stock/hook"

    def test_multipart_thesis_post_uses_llm_webhook(self):
        """When a chart attachment rides on multipart, the destination still
        flips to llm_webhook_url."""
        config = DiscordConfig(
            enabled=True,
            webhook_url="https://main/hook",
            stock_webhook_url="https://stock/hook",
            llm_webhook_url="https://llm/hook",
        )
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
        multipart_calls = [
            c for c in mock_post.call_args_list if "files" in c.kwargs
        ]
        assert len(multipart_calls) == 1
        assert multipart_calls[0].args[0] == "https://llm/hook"


class TestResolveLLMWebhookUrl:
    """Direct coverage of the `_resolve_llm_webhook_url` helper."""

    def test_returns_llm_when_set(self):
        from rainier.alerts.discord import _resolve_llm_webhook_url

        cfg = DiscordConfig(
            webhook_url="https://main/hook",
            stock_webhook_url="https://stock/hook",
            llm_webhook_url="https://llm/hook",
        )
        assert _resolve_llm_webhook_url(cfg) == "https://llm/hook"

    def test_falls_back_to_stock_when_llm_empty(self):
        from rainier.alerts.discord import _resolve_llm_webhook_url

        cfg = DiscordConfig(
            webhook_url="https://main/hook",
            stock_webhook_url="https://stock/hook",
            llm_webhook_url="",
        )
        assert _resolve_llm_webhook_url(cfg) == "https://stock/hook"

    def test_falls_back_to_webhook_when_stock_and_llm_empty(self):
        from rainier.alerts.discord import _resolve_llm_webhook_url

        cfg = DiscordConfig(
            webhook_url="https://main/hook",
            stock_webhook_url="",
            llm_webhook_url="",
        )
        assert _resolve_llm_webhook_url(cfg) == "https://main/hook"

    def test_returns_none_when_all_empty(self):
        from rainier.alerts.discord import _resolve_llm_webhook_url

        cfg = DiscordConfig(
            webhook_url="",
            stock_webhook_url="",
            llm_webhook_url="",
        )
        assert _resolve_llm_webhook_url(cfg) is None


class TestSettingsPlumbing:
    """`DISCORD_LLM_WEBHOOK_URL` env var → `Settings.alerts.discord.llm_webhook_url`."""

    def test_env_var_populates_nested_field(self, monkeypatch, tmp_path):
        from rainier.core.config import load_settings

        yaml_path = tmp_path / "settings.yaml"
        yaml_path.write_text(
            "alerts:\n  discord:\n    enabled: true\n",
            encoding="utf-8",
        )
        # The loader also reads .env via load_dotenv; chdir to a clean tmp
        # path so the project's real .env doesn't leak into the test.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(
            "DISCORD_LLM_WEBHOOK_URL", "https://llm-channel.example/hook"
        )
        settings = load_settings(config_path=yaml_path)
        assert (
            settings.alerts.discord.llm_webhook_url
            == "https://llm-channel.example/hook"
        )

    def test_yaml_value_wins_when_env_empty(self, monkeypatch, tmp_path):
        from rainier.core.config import load_settings

        yaml_path = tmp_path / "settings.yaml"
        yaml_path.write_text(
            "alerts:\n"
            "  discord:\n"
            "    enabled: true\n"
            "    llm_webhook_url: https://yaml.example/hook\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DISCORD_LLM_WEBHOOK_URL", raising=False)
        settings = load_settings(config_path=yaml_path)
        assert (
            settings.alerts.discord.llm_webhook_url
            == "https://yaml.example/hook"
        )
