"""Tests for the PR5 Discord embed renderer.

These cover the pure-function ``format_thesis_embed`` and its helpers — no
HTTP, no DB. Multipart attachment + DB chart-load happens in
``test_discord_attachment.py``.
"""

from __future__ import annotations

from rainier.alerts.discord import (
    _VERDICT_COLORS,
    _build_chip_line,
    _levels_block,
    _llm_noticed,
    _risks_lines,
    _truncate_at_word,
    _watch_line,
    format_thesis_embed,
)
from rainier.core.types import StockCandidate


def _candidate(**overrides) -> StockCandidate:
    base = dict(
        symbol="CRWD",
        rank=14,
        rank_change=2,
        long_short="Long in",
        capital_flow_direction="+",
        sector="Technology",
        signal_strength=0.82,
        money_flow_score=0.7,
        pattern_type="w_bottom",
        pattern_direction="bullish",
        pattern_status="confirmed",
        pattern_confidence=0.95,
        entry_price=486.55,
        stop_loss=476.82,
        target_price=528.79,
        rr_ratio=4.3,
        volume_confirmed=True,
        current_price=505.72,
        distance_to_entry_pct=3.94,
    )
    base.update(overrides)
    return StockCandidate(**base)


def _thesis(**overrides) -> dict:
    base = dict(
        verdict="watch",
        setup_quality=6,
        llm_confidence=6,
        paragraph_radar=(
            "Rank improved sharply over four sessions. Volume confirms breakout."
        ),
        paragraph_evidence=(
            "Sector recovering from -0.17 to -0.06. Late entry — already +3.94%."
        ),
        paragraph_invalidation="Reject below 480 on volume.",
        risks=[
            "late entry, R/R degraded",
            "earnings 26d out",
            "cap flow neutral",
            "extra risk that should be cut",
        ],
        watch_items=["pullback 486-490 on low volume"],
        evidence_used=["pattern", "rank", "volume"],
        signals_used=["rank_trajectory", "sector_momentum"],
        patterns_in_chart_not_in_indicators=[
            "strong neckline breakout with extended upper wick",
        ],
        signals={
            "rank_trajectory": {
                "points": [["2026-05-04", 85], ["2026-05-08", 14]],
                "delta_10d": -71,
                "trend": "rising",
            },
            "sector_momentum": {
                "delta": 0.11,
                "sentiment_today": -0.06,
                "sentiment_5d_ago": -0.17,
            },
        },
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _truncate_at_word
# ---------------------------------------------------------------------------


class TestTruncateAtWord:

    def test_returns_input_when_under_cap(self):
        assert _truncate_at_word("hello world", 50) == "hello world"

    def test_appends_ellipsis_when_truncated(self):
        out = _truncate_at_word("hello world this is a long sentence", 16)
        assert out.endswith("…")
        assert len(out) <= 16

    def test_never_breaks_mid_word(self):
        # The naive cut would put us mid-word; the function must back up.
        text = "breaking thru extended upper wick area"
        out = _truncate_at_word(text, 20)
        # No word should be cut in half — the body before the ellipsis ends
        # at a real word boundary.
        assert out.endswith("…")
        body = out[:-1]
        assert not body.endswith("ext")  # would-be naive cut
        # Last char before ellipsis should be alphanumeric or end of word.
        assert body.rstrip()[-1].isalpha()

    def test_zero_or_negative_cap_returns_empty(self):
        assert _truncate_at_word("anything", 0) == ""
        assert _truncate_at_word("anything", -5) == ""

    def test_empty_input_returns_empty(self):
        assert _truncate_at_word("", 10) == ""
        assert _truncate_at_word(None, 10) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# format_thesis_embed — title + colors
# ---------------------------------------------------------------------------


class TestEmbedTitleAndColor:

    def test_setup_long_is_green(self):
        embed = format_thesis_embed(
            _thesis(verdict="setup_long", llm_confidence=8), _candidate()
        )
        assert embed["color"] == _VERDICT_COLORS["setup_long"] == 0x2ECC71

    def test_watch_is_yellow(self):
        embed = format_thesis_embed(_thesis(verdict="watch"), _candidate())
        assert embed["color"] == _VERDICT_COLORS["watch"] == 0xF1C40F

    def test_no_setup_is_gray(self):
        embed = format_thesis_embed(
            _thesis(verdict="no_setup", llm_confidence=2), _candidate()
        )
        assert embed["color"] == _VERDICT_COLORS["no_setup"] == 0x95A5A6

    def test_unknown_verdict_falls_back_to_no_setup_color(self):
        embed = format_thesis_embed(_thesis(verdict="???"), _candidate())
        assert embed["color"] == _VERDICT_COLORS["no_setup"]

    def test_title_format_is_verdict_symbol_score(self):
        embed = format_thesis_embed(
            _thesis(verdict="watch", llm_confidence=6),
            _candidate(symbol="CRWD"),
        )
        assert embed["title"] == "watch · CRWD · 6/10"


# ---------------------------------------------------------------------------
# Chip line
# ---------------------------------------------------------------------------


class TestChipLine:

    def test_chip_line_under_200_chars(self):
        line = _build_chip_line(_candidate(), _thesis(), "w_bottom")
        assert len(line) <= 200

    def test_chip_line_includes_pattern(self):
        line = _build_chip_line(_candidate(), _thesis(), "w_bottom")
        assert "w_bottom" in line

    def test_chip_line_includes_volume_check_when_confirmed(self):
        line = _build_chip_line(
            _candidate(volume_confirmed=True), _thesis(), "w_bottom"
        )
        assert "vol" in line

    def test_chip_line_omits_volume_when_not_confirmed(self):
        line = _build_chip_line(
            _candidate(volume_confirmed=False), _thesis(), "w_bottom"
        )
        assert "vol" not in line

    def test_chip_line_renders_rank_trajectory(self):
        line = _build_chip_line(_candidate(), _thesis(), "w_bottom")
        assert "rank #85" in line
        assert "#14" in line

    def test_chip_line_falls_back_to_rank_change_when_no_signal(self):
        line = _build_chip_line(
            _candidate(rank_change=3),
            {"signals": {}},
            "w_bottom",
        )
        # No signal payload → uses rank_change
        assert "rank +3" in line


# ---------------------------------------------------------------------------
# LEVELS field
# ---------------------------------------------------------------------------


class TestLevelsField:

    def test_levels_block_has_all_four_lines(self):
        block = _levels_block(_candidate())
        assert "Entry" in block
        assert "Stop" in block
        assert "Target" in block
        assert "Now" in block

    def test_levels_block_is_monospace(self):
        block = _levels_block(_candidate())
        assert block.startswith("```")
        assert block.endswith("```")

    def test_levels_renders_stop_pct(self):
        # entry=486.55, stop=476.82 -> ~-2.0%
        block = _levels_block(_candidate())
        assert "-2.0%" in block

    def test_levels_renders_target_pct_and_rr(self):
        # entry=486.55, target=528.79 -> ~+8.7%, R/R=4.3
        block = _levels_block(_candidate())
        assert "+8.7%" in block
        assert "4.3R" in block

    def test_levels_renders_now_pct(self):
        # entry=486.55, now=505.72 -> ~+3.94%
        block = _levels_block(_candidate())
        assert "+3.9%" in block or "+4.0%" in block  # rounding tolerance

    def test_levels_handles_missing_fields(self):
        c = _candidate(stop_loss=None, target_price=None, current_price=None)
        block = _levels_block(c)
        # Should still render Entry; missing rows show "-".
        assert "Entry" in block


# ---------------------------------------------------------------------------
# RISKS field
# ---------------------------------------------------------------------------


class TestRisksField:

    def test_risks_truncates_to_three(self):
        risks = _risks_lines(
            _thesis(risks=["a", "b", "c", "d", "e"])
        )
        assert len(risks) == 3

    def test_risks_under_80_chars_each(self):
        long = "x " * 200
        risks = _risks_lines(_thesis(risks=[long, long, long]))
        for r in risks:
            assert len(r) <= 80

    def test_risks_is_word_boundary_safe(self):
        long = "earnings risk extended position with major catalyst nearby"
        risks = _risks_lines(_thesis(risks=[long]))
        # Either the original (if under cap) or word-bounded with ellipsis.
        assert risks[0] == long or risks[0].endswith("…")

    def test_risks_skips_empty_strings(self):
        risks = _risks_lines(_thesis(risks=["", "  ", "real risk"]))
        assert risks == ["real risk"]


# ---------------------------------------------------------------------------
# WATCH field
# ---------------------------------------------------------------------------


class TestWatchField:

    def test_watch_renders_first_item_only(self):
        out = _watch_line(_thesis(watch_items=["item one", "item two", "item three"]))
        assert out == "item one"

    def test_watch_truncates_at_120_word_boundary(self):
        long = "watch for pullback to entry zone with low volume " * 5
        out = _watch_line(_thesis(watch_items=[long]))
        assert out is not None
        assert len(out) <= 120
        if len(long) > 120:
            assert out.endswith("…")

    def test_watch_returns_none_when_empty(self):
        assert _watch_line(_thesis(watch_items=[])) is None
        assert _watch_line(_thesis(watch_items=["", "  "])) is None


# ---------------------------------------------------------------------------
# LLM noticed field
# ---------------------------------------------------------------------------


class TestLLMNoticedField:

    def test_llm_noticed_omitted_when_none_string(self):
        out = _llm_noticed(_thesis(patterns_in_chart_not_in_indicators="none"))
        assert out is None

    def test_llm_noticed_omitted_when_empty_list_yields_falsy_field(self):
        # The Pydantic schema rejects empty lists, but the renderer
        # defends in case a degraded payload sneaks through.
        out = _llm_noticed({"patterns_in_chart_not_in_indicators": []})
        assert out is None

    def test_llm_noticed_renders_short_observation(self):
        out = _llm_noticed(
            _thesis(patterns_in_chart_not_in_indicators=["narrowing volume profile"])
        )
        assert out == "narrowing volume profile"

    def test_llm_noticed_truncates_at_200_word_boundary(self):
        long = "very long observation about an unusual chart pattern " * 10
        out = _llm_noticed(_thesis(patterns_in_chart_not_in_indicators=[long]))
        assert out is not None
        assert len(out) <= 200
        assert out.endswith("…")
        # Ellipsis must follow a complete word (no mid-word cut).
        body = out[:-1].rstrip()
        assert body[-1].isalpha()

    def test_llm_noticed_field_omitted_in_embed_when_none(self):
        embed = format_thesis_embed(
            _thesis(patterns_in_chart_not_in_indicators="none"), _candidate(),
        )
        names = [f["name"] for f in embed["fields"]]
        assert "LLM noticed" not in names

    def test_llm_noticed_field_present_in_embed_when_set(self):
        embed = format_thesis_embed(
            _thesis(patterns_in_chart_not_in_indicators=["strong neckline"]),
            _candidate(),
        )
        names = [f["name"] for f in embed["fields"]]
        assert "LLM noticed" in names


# ---------------------------------------------------------------------------
# Dashboard deep-link
# ---------------------------------------------------------------------------


class TestDashboardLink:

    def test_no_url_when_dashboard_base_url_is_none(self):
        embed = format_thesis_embed(_thesis(), _candidate(), dashboard_base_url=None)
        assert "url" not in embed

    def test_no_url_when_thesis_id_is_none(self):
        embed = format_thesis_embed(
            _thesis(),
            _candidate(),
            dashboard_base_url="http://localhost:8501",
            thesis_id=None,
        )
        assert "url" not in embed

    def test_url_built_when_both_set(self):
        embed = format_thesis_embed(
            _thesis(),
            _candidate(),
            dashboard_base_url="http://x",
            thesis_id=42,
        )
        assert embed["url"] == "http://x?thesis_id=42"

    def test_url_appends_with_amp_when_existing_query(self):
        embed = format_thesis_embed(
            _thesis(),
            _candidate(),
            dashboard_base_url="http://x?tab=3",
            thesis_id=42,
        )
        assert embed["url"] == "http://x?tab=3&thesis_id=42"


# ---------------------------------------------------------------------------
# Footer + image
# ---------------------------------------------------------------------------


class TestLLMTextScrubbing:
    """PR5 review iter-1 regression — LLM-generated text fields must NOT
    leak ``@everyone`` / ``@here`` / backticks into Discord because an
    adversarial completion could trigger a server-wide mention or break
    the surrounding embed formatting. Mirrors the existing scrub on the
    eval/research renderers (PR3 review iter-1 [P2]).
    """

    def test_risks_scrub_at_everyone(self):
        risks = _risks_lines(_thesis(risks=["@everyone bad risk"]))
        assert risks
        assert "@everyone" not in risks[0]
        # FULLWIDTH @ sign replacement — visible to the user, not a mention.
        assert "＠" in risks[0]

    def test_watch_scrub_at_here(self):
        out = _watch_line(_thesis(watch_items=["@here pullback to 100"]))
        assert out is not None
        assert "@here" not in out
        assert "＠" in out

    def test_llm_noticed_scrub_backticks(self):
        out = _llm_noticed(
            _thesis(
                patterns_in_chart_not_in_indicators=["pattern with ``code`` block"]
            )
        )
        assert out is not None
        # Backticks would break the surrounding embed code-block formatting.
        assert "`" not in out

    def test_why_bullets_scrub_paragraph_radar(self):
        from rainier.alerts.discord import _why_bullets
        bullets = _why_bullets(
            _thesis(
                paragraph_radar="@everyone strong setup. Volume confirms."
            )
        )
        for b in bullets:
            assert "@everyone" not in b


class TestFooterAndImage:

    def test_footer_renders_setup_quality_and_signals(self):
        embed = format_thesis_embed(
            _thesis(setup_quality=8, signals_used=["rank_trajectory", "sector_momentum"]),
            _candidate(),
        )
        footer = embed.get("footer", {}).get("text", "")
        assert "setup_quality 8/10" in footer
        assert "rank_trajectory" in footer

    def test_image_attachment_url_reflects_filename(self):
        embed = format_thesis_embed(
            _thesis(), _candidate(symbol="CRWD"),
            chart_filename="chart_CRWD.png",
        )
        assert embed["image"]["url"] == "attachment://chart_CRWD.png"

    def test_image_omitted_when_no_chart_filename(self):
        embed = format_thesis_embed(
            _thesis(), _candidate(symbol="CRWD"), chart_filename=None,
        )
        assert "image" not in embed
