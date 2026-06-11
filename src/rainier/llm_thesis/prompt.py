"""Static prompt prefix + JSON schema + 1 few-shot example for the LLM thesis.

PROMPT_VERSION is bumped manually whenever the prompt changes; old
LLMAnalysisRecord rows with different `prompt_template` stay queryable for
historical replay / A/B comparison.
"""

from __future__ import annotations

# v2 (Phase 2, D7a): adds the calibration section to the user message. The
# calibration text is INVISIBLE to compute_input_hash (it hashes only the
# EvidencePack + image), so the Tier-1 cache — keyed on prompt_version — would
# otherwise serve a pre-calibration thesis. Bumping the version busts Tier-1.
# v3 (R-A): the rolling last-K=10 post-exit trade reflections are appended to
# the calibration section — same Tier-1-bust rationale. Bump IN LOCKSTEP with
# core/config.py LLMThesisConfig.prompt_version and settings.yaml
# llm_thesis.prompt_version (the runtime cache key uses the config value).
PROMPT_VERSION = "v3"

OUTPUT_SCHEMA = """{
  "verdict": "setup_long" | "watch" | "no_setup",
  "setup_quality": int 1-10,
  "llm_confidence": int 1-10,
  "paragraph_radar": "string, max 400 chars",
  "paragraph_evidence": "string, max 600 chars",
  "paragraph_invalidation": "string, max 400 chars",
  "risks": ["string", ...],   // max 5
  "watch_items": ["string", ...],  // max 5
  "evidence_used": ["short list of evidence labels you actually used"],
  "signals_used": ["registry signal names that produced non-None values you used"],
  "patterns_in_chart_not_in_indicators":
      ["pattern name", ...]   // max 5; required to be falsifiable
      | "none"                 // explicit literal when nothing additional is in the chart
}"""

STYLE_GUIDE = """STYLE
- Speak in concrete trading language, not LLM filler. No "the data suggests", no hedging adverbs.
- Cite the chart image as a discovery channel BEYOND the engineered features. The
  `patterns_in_chart_not_in_indicators` field is REQUIRED and FALSIFIABLE — if you see no extra
  pattern, return the literal string "none". Empty list is rejected by validation.
- The screener is long-only ("Long in" candidates). Allowed verdicts: setup_long / watch / no_setup.
- Paragraphs:
    - paragraph_radar: ≤400 chars. Why this name is on the radar today.
    - paragraph_evidence: ≤600 chars. The pattern + flow + sector + fundamental case.
    - paragraph_invalidation: ≤400 chars. What kills the setup; specific levels/conditions.
- llm_confidence is integer 1–10. setup_quality is integer 1–10.
- Keep risks and watch_items to ≤5 each, terse bullets.
"""

# JSON example deliberately keeps long lines for fidelity to the prompt the LLM sees.
FEW_SHOT_EXAMPLE = (
    "EXAMPLE — fictional NVDA setup_long output:\n"
    "{\n"
    '  "verdict": "setup_long",\n'
    '  "setup_quality": 7,\n'
    '  "llm_confidence": 7,\n'
    '  "paragraph_radar": "NVDA QU100 rank improved from #28 → #6 over 8d, sector AI '
    'bullish, w_bottom forming with neckline at 142.50 and clean basing structure.",\n'
    '  "paragraph_evidence": "Pattern: w_bottom forming, neckline 142.50. Money flow: '
    '+5d streak, rank rising 22 places. Sector AI sentiment +0.61, +0.18 vs prior. '
    'Fundamentals: P/E 32, fwd 26, surprise +9.4% last quarter. Chart shows volume '
    'contraction inside the right shoulder — consistent with classical accumulation.",\n'
    '  "paragraph_invalidation": "Setup invalidated below 135.00 (recent swing low / '
    'pattern stop) or on a sector flip — rank back >40 or capital flow direction = \'-\' '
    'for 2+ days.",\n'
    '  "risks": ["Earnings in 9d", "Sector rotation if mega-cap leadership breaks"],\n'
    '  "watch_items": ["Volume on neckline break", "10d sector net_sentiment delta"],\n'
    '  "evidence_used": ["pattern", "rank_trajectory", "capital_flow_streak", '
    '"sector_momentum", "fundamentals"],\n'
    '  "signals_used": ["rank_trajectory", "capital_flow_streak", "sector_momentum", '
    '"fundamentals"],\n'
    '  "patterns_in_chart_not_in_indicators": ["narrowing volume on right shoulder"]\n'
    "}\n"
)

SYSTEM_PROMPT = f"""You are reading a daily QU100 stock candidate that already passed a 3-layer
rule-based screener (money flow + sector + 蔡森 chart pattern). Your job is to read the
chart image, the engineered evidence, the multi-day rank trajectory, capital flow streak,
sector momentum, and fundamentals — then produce a single JSON object matching this schema:

{OUTPUT_SCHEMA}

{STYLE_GUIDE}

{FEW_SHOT_EXAMPLE}

Return ONLY the JSON object — no surrounding text, no code fence.
"""


def build_user_message(
    symbol: str,
    scan_date: str,
    session_name: str,
    candidate_summary: str,
    signal_renders: list[str],
    calibration_section: str = "",
) -> str:
    """Stitch the per-call evidence into the user-side prompt.

    `signal_renders` is the list of `signal.render_for_prompt(value)` strings for
    every enabled signal that produced a non-None value. Order is REGISTRY order.

    `calibration_section` (D7a + R-A) is a bounded, pre-rendered block describing
    how prior theses graded out (unbiased fixed-horizon headline + labeled
    realized supplementary), followed by the last K=10 post-exit trade
    reflections (R-A). Empty string = nothing to inject (Phase-0 / fresh DB).
    NOTE: this text does NOT enter `compute_input_hash` (Tier-2), so
    PROMPT_VERSION is bumped to bust the Tier-1 cache whenever this block's
    shape changes.
    """
    body = (
        f"Symbol: {symbol}\n"
        f"Scan date: {scan_date}\n"
        f"Session: {session_name}\n\n"
        f"--- Screener candidate ---\n{candidate_summary}\n\n"
        f"--- Signals ---\n"
    )
    if signal_renders:
        body += "\n".join(f"- {s}" for s in signal_renders)
    else:
        body += "(no signals produced non-None values)"
    if calibration_section:
        body += f"\n\n{calibration_section}"
    body += "\n\nProduce the JSON object now."
    return body
