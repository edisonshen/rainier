"""Tests for the PR5 Discord embed renderer.

Table-driven coverage of the pure-function ``format_thesis_embed`` and its
helpers — no HTTP, no DB. Multipart attachment + DB chart-load happens in
``test_discord_attachment.py``.

Organized around the renderer's core contracts (one parametrized test per
contract, new scenario = new row):

  1. ``_truncate_at_word``  — the shared word-boundary truncation primitive.
  2. embed color + title    — verdict → color bar, ``verdict · symbol · score``.
  3. chip line              — decision-critical chips (pattern/rank/vol), 200-cap.
  4. LEVELS block           — monospace Entry/Stop/Target/Now with %deltas + R/R.
  5. RISKS                  — top-3, ≤80 chars, word-safe, empties dropped.
  6. WATCH                  — first item only, ≤120 chars, None when empty.
  7. LLM-noticed            — optional observation, ≤200, present/omitted in embed.
  8. WHY bullets            — sentence bullets, ``_WHY_BULLET_MAX`` cap, field cap.
  9. LLM-text scrub         — @everyone/@here/backticks neutralized on every path.
 10. dashboard deep-link    — title URL only when base_url AND thesis_id set.
 11. footer + image         — setup_quality/signals footer, attachment:// image.
"""

from __future__ import annotations

import pytest

from rainier.alerts.discord import (
    _EMBED_FIELD_VALUE_MAX,
    _VERDICT_COLORS,
    _WHY_BULLET_MAX,
    _build_chip_line,
    _levels_block,
    _llm_noticed,
    _risks_lines,
    _truncate_at_word,
    _watch_line,
    _why_bullets,
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
# 1. _truncate_at_word — the shared word-boundary truncation primitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, cap, expected",
    [
        ("hello world", 50, "hello world"),   # under cap → unchanged
        ("", 10, ""),                          # empty input → empty
        (None, 10, ""),                        # None input → empty (defensive)
        ("anything", 0, ""),                   # zero cap → empty
        ("anything", -5, ""),                  # negative cap → empty
    ],
)
def test_truncate_at_word_exact_outputs(text, cap, expected):
    assert _truncate_at_word(text, cap) == expected  # type: ignore[arg-type]


def test_truncate_at_word_backs_off_to_word_boundary():
    # Over the cap → single trailing ellipsis, within the cap, and the cut
    # lands on a real word boundary (never mid-word like "ext...").
    text = "breaking thru extended upper wick area"
    out = _truncate_at_word(text, 20)
    assert out.endswith("…")
    assert len(out) <= 20
    body = out[:-1].rstrip()
    assert not body.endswith("ext")   # the would-be naive mid-word cut
    assert body[-1].isalpha()         # ends flush with a whole word


# ---------------------------------------------------------------------------
# 2. format_thesis_embed — verdict color bar + title
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict, expected_color",
    [
        ("setup_long", 0x2ECC71),                       # green
        ("watch", 0xF1C40F),                            # yellow
        ("no_setup", 0x95A5A6),                         # gray
        ("???", _VERDICT_COLORS["no_setup"]),           # unknown → gray fallback
    ],
)
def test_embed_verdict_color(verdict, expected_color):
    embed = format_thesis_embed(_thesis(verdict=verdict), _candidate())
    assert embed["color"] == expected_color


def test_embed_title_is_verdict_symbol_score():
    embed = format_thesis_embed(
        _thesis(verdict="watch", llm_confidence=6), _candidate(symbol="CRWD")
    )
    assert embed["title"] == "watch · CRWD · 6/10"


# ---------------------------------------------------------------------------
# 3. _build_chip_line — decision-critical chip line (pattern / rank / vol)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate_kwargs, thesis, present, absent",
    [
        # Full default: pattern + rank-trajectory (from signals payload) + vol✓.
        (
            {"volume_confirmed": True},
            None,
            ["w_bottom", "rank #85", "#14", "vol"],
            [],
        ),
        # Volume not confirmed → the vol✓ chip is dropped.
        (
            {"volume_confirmed": False},
            None,
            ["w_bottom"],
            ["vol"],
        ),
        # No rank_trajectory signal → falls back to candidate.rank_change.
        (
            {"rank_change": 3},
            {"signals": {}},
            ["rank +3"],
            [],
        ),
    ],
)
def test_chip_line_content_and_cap(candidate_kwargs, thesis, present, absent):
    line = _build_chip_line(
        _candidate(**candidate_kwargs),
        _thesis() if thesis is None else thesis,
        "w_bottom",
    )
    assert len(line) <= 200          # _CHIP_LINE_MAX — scannable in one glance
    for sub in present:
        assert sub in line
    for sub in absent:
        assert sub not in line


# ---------------------------------------------------------------------------
# 4. _levels_block — monospace Entry/Stop/Target/Now with %deltas + R/R
# ---------------------------------------------------------------------------


def test_levels_block_full_candidate():
    # entry=486.55, stop=476.82 (~-2.0%), target=528.79 (~+8.7%, 4.3R),
    # now=505.72 (~+3.9%). Rendered as a fenced monospace block.
    block = _levels_block(_candidate())
    assert block.startswith("```") and block.endswith("```")
    for label in ("Entry", "Stop", "Target", "Now"):
        assert label in block
    assert "-2.0%" in block
    assert "+8.7%" in block
    assert "4.3R" in block
    assert "+3.9%" in block or "+4.0%" in block  # rounding tolerance


def test_levels_block_tolerates_missing_fields():
    block = _levels_block(
        _candidate(stop_loss=None, target_price=None, current_price=None)
    )
    assert "Entry" in block   # still renders; absent rows collapse to "-"


# ---------------------------------------------------------------------------
# 5. _risks_lines — top-3, ≤80 chars each, word-safe, empties dropped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "risks, expected",
    [
        (["a", "b", "c", "d", "e"], ["a", "b", "c"]),           # capped at 3
        (["", "  ", "real risk"], ["real risk"]),               # empties dropped
    ],
)
def test_risks_selection(risks, expected):
    assert _risks_lines(_thesis(risks=risks)) == expected


def test_risks_each_truncated_word_safe_under_80():
    long = "x " * 200
    out = _risks_lines(_thesis(risks=[long, long, long]))
    assert len(out) == 3
    for r in out:
        assert len(r) <= 80
    # A single mid-length risk is either returned whole or word-bounded + "…".
    mid = "earnings risk extended position with major catalyst nearby"
    got = _risks_lines(_thesis(risks=[mid]))
    assert got[0] == mid or got[0].endswith("…")


# ---------------------------------------------------------------------------
# 6. _watch_line — first item only, ≤120 chars, None when empty
# ---------------------------------------------------------------------------


def test_watch_line_first_item_only():
    assert _watch_line(_thesis(watch_items=["item one", "item two"])) == "item one"


def test_watch_line_truncates_at_120_word_boundary():
    long = "watch for pullback to entry zone with low volume " * 5
    out = _watch_line(_thesis(watch_items=[long]))
    assert out is not None
    assert len(out) <= 120
    assert out.endswith("…")


@pytest.mark.parametrize("items", [[], ["", "  "]])
def test_watch_line_none_when_no_actionable_item(items):
    assert _watch_line(_thesis(watch_items=items)) is None


# ---------------------------------------------------------------------------
# 7. _llm_noticed — optional observation, ≤200, present/omitted in embed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("none", None),                                    # sentinel string
        ([], None),                                        # degraded empty list
        (["narrowing volume profile"], "narrowing volume profile"),  # short passthrough
    ],
)
def test_llm_noticed_value(raw, expected):
    assert _llm_noticed({"patterns_in_chart_not_in_indicators": raw}) == expected


def test_llm_noticed_truncates_at_200_word_boundary():
    long = "very long observation about an unusual chart pattern " * 10
    out = _llm_noticed(_thesis(patterns_in_chart_not_in_indicators=[long]))
    assert out is not None
    assert len(out) <= 200
    assert out.endswith("…")
    assert out[:-1].rstrip()[-1].isalpha()   # no mid-word cut


@pytest.mark.parametrize(
    "raw, present",
    [("none", False), (["strong neckline"], True)],
)
def test_llm_noticed_field_presence_in_embed(raw, present):
    embed = format_thesis_embed(
        _thesis(patterns_in_chart_not_in_indicators=raw), _candidate()
    )
    names = [f["name"] for f in embed["fields"]]
    assert ("LLM noticed" in names) is present


# ---------------------------------------------------------------------------
# 8. _why_bullets — sentence bullets, _WHY_BULLET_MAX cap, Discord field cap
# ---------------------------------------------------------------------------


def test_why_bullet_full_sentence_survives_whole():
    # Regression: real ~150-char thesis sentences used to get chopped mid-word
    # by an old 60-char cap. A sentence under _WHY_BULLET_MAX renders intact.
    sentence = (
        "Entry 670.60 sits 1.07 percent below the current price 677.79 "
        "and the stop rests just under the prior swing low which leaves "
        "a clean measured move higher"
    )
    assert len(sentence) < _WHY_BULLET_MAX  # guards the fixture premise
    bullets = _why_bullets(_thesis(paragraph_radar=sentence + "."))
    assert bullets[0] == sentence
    assert not bullets[0].endswith("…")


def test_why_bullet_over_cap_truncated_at_word_boundary():
    words = ["measured"] * 60  # 539 chars, well over the cap
    sentence = " ".join(words)
    assert len(sentence) > _WHY_BULLET_MAX
    bullets = _why_bullets(_thesis(paragraph_radar=sentence + "."))
    assert bullets[0].endswith("…")
    assert len(bullets[0]) <= _WHY_BULLET_MAX
    body = bullets[0][:-1]
    assert body.strip() == body               # no trailing partial-word/space
    assert all(tok == "measured" for tok in body.split())


def test_why_bullet_exact_boundary_transition():
    # Off-by-one: exactly _WHY_BULLET_MAX chars survives whole; one char over
    # gains the ellipsis. "aa " has period 3 so these slices never end on a
    # space (which _why_bullets would strip), keeping the lengths exact.
    filler = ("aa " * 100).strip()
    at_cap = filler[:_WHY_BULLET_MAX]
    over_cap = filler[: _WHY_BULLET_MAX + 1]
    assert len(at_cap) == _WHY_BULLET_MAX
    assert not at_cap.endswith(" ") and not over_cap.endswith(" ")

    at = _why_bullets(_thesis(paragraph_radar=at_cap, paragraph_evidence=""))
    assert at[0] == at_cap and not at[0].endswith("…")

    over = _why_bullets(_thesis(paragraph_radar=over_cap, paragraph_evidence=""))
    assert over[0].endswith("…") and len(over[0]) <= _WHY_BULLET_MAX


def test_why_field_value_within_discord_field_cap():
    # Field-cap safety: four long bullets (radar + evidence) still fit Discord's
    # 1024-char field-value limit — the call site backstops the joined WHY value.
    long_sentence = " ".join(["breakout"] * 40)  # ~319 chars, over the bullet cap
    embed = format_thesis_embed(
        _thesis(
            paragraph_radar=f"{long_sentence}. {long_sentence}.",
            paragraph_evidence=f"{long_sentence}. {long_sentence}.",
        ),
        _candidate(),
    )
    why = next(f for f in embed["fields"] if f["name"] == "WHY")
    assert len(why["value"]) <= _EMBED_FIELD_VALUE_MAX


# ---------------------------------------------------------------------------
# 9. LLM-text scrub — @everyone / @here / backticks neutralized on EVERY path
# ---------------------------------------------------------------------------
#
# LLM-generated prose flows into Discord text; an adversarial completion must
# not be able to trigger a mass-mention or break the embed's code-block
# formatting. Each renderer helper independently scrubs, so each path is a row.


@pytest.mark.parametrize(
    "render, forbidden, expect_fullwidth_at",
    [
        (
            lambda: _risks_lines(_thesis(risks=["@everyone bad risk"]))[0],
            "@everyone",
            True,
        ),
        (
            lambda: _watch_line(_thesis(watch_items=["@here pullback to 100"])),
            "@here",
            True,
        ),
        (
            lambda: _llm_noticed(
                _thesis(patterns_in_chart_not_in_indicators=["pattern ``code`` block"])
            ),
            "`",
            False,
        ),
        (
            lambda: " ".join(
                _why_bullets(_thesis(paragraph_radar="@everyone strong setup. Ok."))
            ),
            "@everyone",
            True,
        ),
    ],
)
def test_llm_text_is_scrubbed(render, forbidden, expect_fullwidth_at):
    out = render()
    assert out is not None
    assert forbidden not in out
    if expect_fullwidth_at:
        assert "＠" in out   # FULLWIDTH @ — visible, not a mention


# ---------------------------------------------------------------------------
# 10. Dashboard deep-link — title URL only when base_url AND thesis_id set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url, thesis_id, expected",
    [
        (None, 42, None),                                   # no base → no link
        ("http://localhost:8501", None, None),              # no id → no link
        ("http://x", 42, "http://x?thesis_id=42"),          # both → link
        ("http://x?tab=3", 42, "http://x?tab=3&thesis_id=42"),  # append with &
    ],
)
def test_dashboard_deeplink(base_url, thesis_id, expected):
    embed = format_thesis_embed(
        _thesis(), _candidate(), dashboard_base_url=base_url, thesis_id=thesis_id
    )
    if expected is None:
        assert "url" not in embed   # key omitted entirely, not set to None
    else:
        assert embed["url"] == expected


# ---------------------------------------------------------------------------
# 11. Footer + image attachment
# ---------------------------------------------------------------------------


def test_footer_renders_setup_quality_and_signals():
    embed = format_thesis_embed(
        _thesis(setup_quality=8, signals_used=["rank_trajectory", "sector_momentum"]),
        _candidate(),
    )
    footer = embed.get("footer", {}).get("text", "")
    assert "setup_quality 8/10" in footer
    assert "rank_trajectory" in footer


@pytest.mark.parametrize(
    "chart_filename, expected_url",
    [
        ("chart_CRWD.png", "attachment://chart_CRWD.png"),  # image references file
        (None, None),                                        # no file → no image
    ],
)
def test_image_attachment(chart_filename, expected_url):
    embed = format_thesis_embed(
        _thesis(), _candidate(symbol="CRWD"), chart_filename=chart_filename
    )
    if expected_url is None:
        assert "image" not in embed
    else:
        assert embed["image"]["url"] == expected_url
