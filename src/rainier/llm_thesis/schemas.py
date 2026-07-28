"""Pydantic schemas for LLM thesis output + evidence pack + input hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TradeThesis(BaseModel):
    """LLM output schema — one per ticker, validated on every call.

    `verdict` is long-only/no_setup/watch — the screener never produces "Short in"
    candidates, so allowing setup_short would let the LLM contradict the upstream
    filter. This is enforced by the Literal below, not just by prompt language.
    """

    verdict: Literal["setup_long", "watch", "no_setup"]
    setup_quality: int = Field(ge=1, le=10)
    llm_confidence: int = Field(ge=1, le=10)
    paragraph_radar: str = Field(min_length=1, max_length=400)
    paragraph_evidence: str = Field(min_length=1, max_length=600)
    paragraph_invalidation: str = Field(min_length=1, max_length=400)
    risks: list[str] = Field(default_factory=list, max_length=5)
    watch_items: list[str] = Field(default_factory=list, max_length=5)
    evidence_used: list[str] = Field(default_factory=list)
    signals_used: list[str] = Field(default_factory=list)
    # Either a list (max 5) of named patterns OR the literal "none". Empty list
    # is rejected — the LLM must explicitly say "none" to mean nothing was found.
    patterns_in_chart_not_in_indicators: list[str] | Literal["none"]

    @field_validator("patterns_in_chart_not_in_indicators")
    @classmethod
    def _patterns_field_nonempty(
        cls, v: list[str] | str
    ) -> list[str] | Literal["none"]:
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError(
                    "patterns_in_chart_not_in_indicators must be a list of names "
                    "(max 5) or the literal string 'none' — empty list is rejected"
                )
            if len(v) > 5:
                raise ValueError("patterns_in_chart_not_in_indicators max 5 entries")
        elif v != "none":
            raise ValueError(
                "patterns_in_chart_not_in_indicators must be a list or 'none'"
            )
        return v


class EvidencePack(BaseModel):
    """Materialized evidence handed to the LLM (and hashed for the cache key).

    `signals` is the registry-keyed map of signal compute() outputs. The
    thesis prompt stitches each signal's render_for_prompt() into the
    final user message.
    """

    symbol: str
    scan_date: str  # ISO YYYY-MM-DD
    session_name: str
    candidate: dict[str, Any]
    signals: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Image bytes are hashed but never serialized into the JSON envelope.
    image_sha256: str | None = None


def compute_input_hash(
    pack: EvidencePack,
    image_bytes: bytes | None,
    thinking_budget_tokens: int | None = None,
) -> str:
    """Deterministic sha256 of the evidence pack + image bytes (+ thinking budget).

    Used as the idempotency key on `analysis_results.input_hash`. Two re-runs
    of the same scan with the same (symbol, signals, image, thinking budget)
    compute to the same value, so `INSERT ... ON CONFLICT DO NOTHING` guarantees
    zero duplicate LLM calls.

    ``thinking_budget_tokens`` is folded in because it changes the amount of
    reasoning the LLM does — a retuned budget must yield a DIFFERENT idempotency
    key so the regenerated thesis persists as its own row instead of colliding
    with the old-budget row on the (day, symbols, model, prompt, input_hash)
    unique index. It is NOT prompt text; the pack/image are still the only
    content inputs. ``None`` (the default) keeps legacy callers byte-identical.
    """
    pack_json = pack.model_dump(exclude={"image_sha256"})
    serialized = json.dumps(pack_json, sort_keys=True, default=str).encode("utf-8")
    image_hash = (
        hashlib.sha256(image_bytes).hexdigest()
        if image_bytes is not None
        else "no-image"
    )
    digest = serialized + image_hash.encode("ascii")
    if thinking_budget_tokens is not None:
        # Append only when provided so the legacy 2-arg form stays byte-identical
        # (other callers' idempotency keys are unaffected by this addition).
        digest += f"|budget:{int(thinking_budget_tokens)}".encode("ascii")
    return hashlib.sha256(digest).hexdigest()
