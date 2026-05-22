"""Research-side LLM output schemas (D-010).

``ResearchThesis`` is the v0/v1 thesis envelope — separate from production
``llm_thesis/schemas.py:TradeThesis`` so the research engine can iterate
without churning the production daily pipeline.

Slice 0 ships the skeleton (validators + serdes). Slice 1 wires it into
the L3 evaluator + per-call output store.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Verdict = Literal["setup_long", "no_setup", "watch"]


class ResearchThesis(BaseModel):
    """One per-(skill, ticker, day) LLM call.

    Invariants enforced at construction (see :func:`_validate_price_ordering`
    + :func:`_validate_now_price_not_hallucinated`):
      - ``stop < entry < target`` whenever ``verdict == "setup_long"``
      - ``|now_price - t_minus_1_close| / t_minus_1_close <= 0.02``

    Both invariants protect against LLM hallucination (D-010): wrong
    price ordering is a logic bug; ``now_price`` drift from the
    pinned T-1 close is the LLM inventing prices.
    """

    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(..., min_length=1)
    ticker: str = Field(..., min_length=1)
    day: date
    verdict: Verdict
    rationale: str = Field("", description="LLM's explanation; free-form.")

    now_price: float | None = None
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    holding_days_expected: int | None = None

    # T-1 close pinned at construction; the hallucination guard fires
    # when ``now_price`` diverges from this anchor by more than 2%.
    t_minus_1_close: float | None = None

    @model_validator(mode="after")
    def _validate_price_ordering(self) -> "ResearchThesis":
        if self.verdict == "setup_long":
            missing = [
                name
                for name, val in (
                    ("entry", self.entry),
                    ("stop", self.stop),
                    ("target", self.target),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"setup_long thesis missing required fields: {missing}"
                )
            if not (self.stop < self.entry < self.target):
                raise ValueError(
                    f"price ordering violated: stop={self.stop} "
                    f"entry={self.entry} target={self.target}"
                )
        return self

    @model_validator(mode="after")
    def _validate_now_price_not_hallucinated(self) -> "ResearchThesis":
        if self.now_price is None or self.t_minus_1_close is None:
            return self
        anchor = self.t_minus_1_close
        if anchor <= 0:
            return self
        drift = abs(self.now_price - anchor) / anchor
        if drift > 0.02:
            raise ValueError(
                f"now_price hallucination guard: {self.now_price} drifts "
                f"{drift:.2%} from t_minus_1_close={anchor} (cap 2%)"
            )
        return self
