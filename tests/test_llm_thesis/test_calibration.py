"""Phase 2 D7a — calibration block in the thesis prompt.

Covers:
  * the HEADLINE = unbiased fixed-horizon ThesisEvaluation stats (NOT realized-
    paper-only) with by-verdict + by-confidence-bucket breakdown;
  * the K=10 closed paper trades are rendered as LABELED supplementary context
    (realized-closed) alongside MTM-including-open — never as the headline;
  * the calibration text is injected into build_user_message;
  * bumping PROMPT_VERSION busts the Tier-1 cache key while compute_input_hash
    (Tier-2) is unaffected by the calibration text.
"""

from __future__ import annotations

from rainier.llm_thesis.prompt import PROMPT_VERSION, build_user_message
from rainier.llm_thesis.schemas import EvidencePack, compute_input_hash
from rainier.paper.calibration import (
    _headline_for_horizon,
    render_calibration_section,
)

# ---------------------------------------------------------------------------
# headline aggregation — unbiased fixed-horizon stats
# ---------------------------------------------------------------------------


def test_headline_breaks_down_by_verdict_and_confidence():
    # (verdict, llm_confidence, return_pct)
    rows = [
        ("setup_long", 9, 0.05),   # high-conf win
        ("setup_long", 8, -0.02),  # high-conf loss
        ("setup_long", 5, 0.01),   # mid-conf win
        ("watch", 3, -0.01),       # low-conf, correctly didn't buy → hit
    ]
    h = _headline_for_horizon(rows)
    assert h["overall"]["n"] == 4
    # setup_long: 2 of 3 are gains → win-rate 2/3.
    sl = h["by_verdict"]["setup_long"]
    assert sl["n"] == 3
    assert abs(sl["win_rate"] - 2 / 3) < 1e-9
    # watch with a loss is a HIT (right not to buy).
    assert h["by_verdict"]["watch"]["win_rate"] == 1.0
    # Confidence buckets present.
    assert "high (8-10)" in h["by_confidence_bucket"]
    assert h["by_confidence_bucket"]["high (8-10)"]["n"] == 2


# ---------------------------------------------------------------------------
# render — headline is primary, realized record is labeled supplementary
# ---------------------------------------------------------------------------


def _sample_payload():
    return {
        "as_of_date": "2026-02-02",
        "window_days": 60,
        "headline_fixed_horizon": {
            "5d": {
                "overall": {"n": 10, "win_rate": 0.6, "avg_return_pct": 0.012},
                "by_verdict": {
                    "setup_long": {"n": 8, "win_rate": 0.5, "avg_return_pct": 0.01}
                },
                "by_confidence_bucket": {
                    "high (8-10)": {"n": 4, "win_rate": 0.25, "avg_return_pct": -0.01}
                },
            }
        },
        "supplementary_realized": {
            "label": "realized-closed (survivorship-prone — NOT the headline)",
            "n_closed_sample": 2,
            "closed_win_rate": 1.0,
            "recent_closed": [
                {"symbol": "AAA", "outcome": "win", "exit_reason": "target",
                 "return_pct": 0.08},
                {"symbol": "BBB", "outcome": "win", "exit_reason": "time_stop",
                 "return_pct": 0.03},
            ],
            "realized_pnl_usd": 110.0,
            "mtm_including_open_pnl_usd": -25.0,
        },
    }


def test_render_headline_is_unbiased_and_supplementary_is_labeled():
    text = render_calibration_section(_sample_payload())
    # Headline framing present, primary.
    assert "HEADLINE" in text
    assert "unbiased fixed-horizon" in text
    assert "[5d] overall: win-rate 60%" in text
    assert "setup_long: win-rate 50%" in text
    assert "conf high (8-10): win-rate 25%" in text
    # The realized record is labeled supplementary / survivorship-prone — never
    # surfaced as the headline.
    assert "SUPPLEMENTARY" in text
    assert "survivorship-prone" in text
    assert "AAA win(target" in text
    # MTM-including-open carried.
    assert "MTM incl. open: $-25.00" in text


def test_render_empty_payload_is_blank():
    assert render_calibration_section(None) == ""
    assert render_calibration_section({}) == ""


# ---------------------------------------------------------------------------
# prompt injection
# ---------------------------------------------------------------------------


def test_build_user_message_injects_calibration():
    section = render_calibration_section(_sample_payload())
    msg = build_user_message(
        symbol="AAA",
        scan_date="2026-02-03",
        session_name="close",
        candidate_summary="{}",
        signal_renders=["rank up"],
        calibration_section=section,
    )
    assert "HEADLINE" in msg
    assert msg.strip().endswith("Produce the JSON object now.")


def test_build_user_message_without_calibration_is_unchanged_tail():
    msg = build_user_message(
        symbol="AAA",
        scan_date="2026-02-03",
        session_name="close",
        candidate_summary="{}",
        signal_renders=["rank up"],
    )
    assert "Calibration" not in msg
    assert "HEADLINE" not in msg


# ---------------------------------------------------------------------------
# cache correctness — PROMPT_VERSION busts Tier-1; input_hash unaffected
# ---------------------------------------------------------------------------


def test_prompt_version_bumped_for_calibration():
    # The Tier-1 key (date, symbol, prompt_version, session, model) includes
    # prompt_version. Shipping the calibration block bumped it past v1 so the
    # cache cannot serve a pre-calibration thesis.
    assert PROMPT_VERSION != "v1"


def test_prompt_version_config_tracks_module():
    # codex iter-1 [P2->P1]: the RUNTIME Tier-1 key is built from
    # settings.llm_thesis.prompt_version (service.generate_thesis), NOT the
    # prompt.PROMPT_VERSION module constant. If the config default lags the
    # module constant, the "bump busts Tier-1" mechanism silently no-ops —
    # calibrated prompts get served from / recorded as the stale version. Guard
    # the two in lockstep so the cache-bust actually fires in production.
    from rainier.core.config import LLMThesisConfig

    assert LLMThesisConfig().prompt_version == PROMPT_VERSION


def test_input_hash_unaffected_by_calibration_text():
    # compute_input_hash hashes only EvidencePack + image bytes; the calibration
    # section lives in the prompt text, NOT the pack — so it can never enter the
    # Tier-2 input_hash. Two identical packs hash identically regardless of any
    # calibration block rendered into the prompt.
    pack = EvidencePack(
        symbol="AAA",
        scan_date="2026-02-03",
        session_name="close",
        candidate={"rank": 5},
        signals={"rank_trajectory": {"value": 1}},
    )
    h1 = compute_input_hash(pack, b"img")
    h2 = compute_input_hash(pack, b"img")
    assert h1 == h2
    # The EvidencePack has no field that the calibration text could perturb.
    assert "calibration" not in pack.model_dump()
