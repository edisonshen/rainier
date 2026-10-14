"""LLM pick step: Claude chooses which screened names get a full thesis.

The screener ranks QU100 by a rule-based composite score. Instead of blindly
thesis-ing the composite top-N, one cheap text-only call shows the LLM the
screener's top-``selection_pool_size`` candidates and asks it to pick the
``top_n`` most promising long setups. The expensive per-ticker thesis then runs
on those picks.

Any selector failure (provider error, bad JSON, unknown symbols) falls back to
the screener's own top-N, so a scan never loses its thesis set because of this
step. When the LLM returns fewer valid picks than requested, the remainder is
filled in screener order.
"""

from __future__ import annotations

import json
import logging
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field

from rainier.core.config import Settings
from rainier.core.types import StockCandidate

from .service import _estimate_cost_usd

log = logging.getLogger(__name__)

_SELECTION_MAX_TOKENS = 1500
_REASON_MAX_CHARS = 200

SELECTION_SYSTEM_PROMPT = """You are a swing-trading analyst picking which stocks deserve a deep \
chart review today.

You get a table of long-side ("Long in") candidates from a QU100 money-flow screener, already \
ranked by the screener's rule-based composite score. Pick the {top_n} names most likely to offer \
an actionable long setup over the next 5-10 trading days. Weigh money-flow strength and rank \
momentum, capital-flow direction, pattern quality and status, reward/risk, distance to entry \
(prefer names near a clean entry over extended ones), and sector diversification. You may pick \
names the composite ranks lower when the evidence justifies it.

Respond with JSON only, no prose, exactly this shape:
{{"picks": [{{"symbol": "TICKER", "reason": "one short sentence, max 200 chars"}}]}}
Return exactly {top_n} picks, best first, using only symbols from the table."""


class SelectionPick(BaseModel):
    symbol: str = Field(min_length=1)
    reason: str = ""


class SelectionOutput(BaseModel):
    picks: list[SelectionPick]


class ThesisSelection(NamedTuple):
    """Which candidates get a thesis, and why."""

    targets: list[StockCandidate]
    # {symbol: one-line LLM reason}; empty for screener-filled targets.
    reasons: dict[str, str]
    source: Literal["llm", "screener"]
    cost_usd: float
    error: str | None = None


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _candidate_row(position: int, c: StockCandidate) -> str:
    pattern = "none"
    if c.pattern_type:
        pattern = (
            f"{c.pattern_type} {c.pattern_status or '?'} conf {_fmt(c.pattern_confidence, '.2f')}"
            f" entry {_fmt(c.entry_price, '.2f')} SL {_fmt(c.stop_loss, '.2f')}"
            f" tgt {_fmt(c.target_price, '.2f')} R:R {_fmt(c.rr_ratio, '.1f')}"
            f" vol_confirmed {'yes' if c.volume_confirmed else 'no'}"
        )
    return (
        f"{position}. {c.symbol} | QU rank {c.rank} ({c.rank_change:+d}) | flow "
        f"{c.capital_flow_direction} | {c.sector} | composite {c.signal_strength:.3f} | "
        f"price {_fmt(c.current_price, '.2f')} | to-entry {_fmt(c.distance_to_entry_pct, '+.1f')}{'' if c.distance_to_entry_pct is None else '%'} "
        f"| pattern {pattern}"
    )


def build_selection_prompt(pool: list[StockCandidate], top_n: int) -> str:
    rows = "\n".join(_candidate_row(i, c) for i, c in enumerate(pool, start=1))
    return (
        f"Screener candidates ({len(pool)}), composite order:\n{rows}\n\n"
        f"Pick {top_n}."
    )


def _parse_selection(raw: str) -> SelectionOutput:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return SelectionOutput.model_validate(json.loads(text.strip()))


def _call_selector(model: str, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
    import litellm  # local import — tests mock this

    resp = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=_SELECTION_MAX_TOKENS,
    )
    usage = resp.get("usage") or {}
    return (
        resp["choices"][0]["message"]["content"],
        int(usage.get("prompt_tokens", 0)),
        int(usage.get("completion_tokens", 0)),
    )


def _resolve_picks(
    picks: list[SelectionPick], pool: list[StockCandidate], top_n: int
) -> tuple[list[StockCandidate], dict[str, str]]:
    """Map LLM picks onto pool candidates (unknown/duplicate symbols dropped),
    then fill any shortfall in screener order."""
    by_symbol = {c.symbol: c for c in pool}
    targets: list[StockCandidate] = []
    reasons: dict[str, str] = {}
    for pick in picks:
        symbol = pick.symbol.strip().upper()
        candidate = by_symbol.get(symbol)
        if candidate is None or symbol in reasons:
            continue
        targets.append(candidate)
        reasons[symbol] = pick.reason.strip()[:_REASON_MAX_CHARS]
        if len(targets) == top_n:
            break
    for candidate in pool:
        if len(targets) == top_n:
            break
        if candidate.symbol not in reasons:
            targets.append(candidate)
    return targets, reasons


def select_thesis_targets(
    candidates: list[StockCandidate], settings: Settings, *, top_n: int | None = None
) -> ThesisSelection:
    """Pick which screened candidates get a full LLM thesis.

    ``candidates`` is the screener output in composite order; the LLM sees the
    first ``llm_thesis.selection_pool_size`` of them. ``top_n`` defaults to
    ``llm_thesis.top_n``.
    """
    cfg = settings.llm_thesis
    n = cfg.top_n if top_n is None else top_n
    pool = candidates[: cfg.selection_pool_size]
    screener_pick = pool[:n]

    if not cfg.llm_selection_enabled or len(pool) <= n:
        return ThesisSelection(screener_pick, {}, "screener", 0.0)

    model = cfg.selection_model or cfg.model
    cost = 0.0
    try:
        raw, prompt_tokens, completion_tokens = _call_selector(
            model,
            SELECTION_SYSTEM_PROMPT.format(top_n=n),
            build_selection_prompt(pool, n),
        )
        cost = _estimate_cost_usd(prompt_tokens, completion_tokens, model)
        parsed = _parse_selection(raw)
    except Exception as exc:
        log.warning("thesis_selection_failed_using_screener_top_n error=%s", exc)
        return ThesisSelection(screener_pick, {}, "screener", cost, str(exc))

    targets, reasons = _resolve_picks(parsed.picks, pool, n)
    if not reasons:
        error = "LLM returned no symbols from the candidate pool"
        log.warning("thesis_selection_empty_using_screener_top_n raw=%s", raw[:500])
        return ThesisSelection(screener_pick, {}, "screener", cost, error)

    log.info(
        "thesis_selection_done picks=%s pool=%d cost_usd=%.4f",
        [c.symbol for c in targets], len(pool), cost,
    )
    return ThesisSelection(targets, reasons, "llm", cost)
