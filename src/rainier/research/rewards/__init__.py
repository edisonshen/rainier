"""Input-typed reward registry for the A/B substrate.

Design: docs/DESIGN-qu100-ab-testing.md §4.2, §10.3.

A reward = a fact (the scored data) + an aggregation, declared as
config-as-data. Each registered reward carries:

  input_type  which candidate payload it can score —
              "basket" (screener, BasketOutcomes) or "trades" (LLM,
              TradeRecord series)
  role        "primary" (the objective being optimized),
              "guardrail" (blocks promotion on breach; carries threshold),
              "secondary" (reported, not gating)
  direction   "increase" (higher is better) or "decrease" (lower is better)
  threshold   guardrail-only breach bound

The registry REFUSES to score a candidate with a reward registered for a
different input type — a loud ValueError, never silent mis-scoring (§4.2).

Rewards are pure, deterministic, side-effect-free. The promote-time gate
that consumes ``guardrail_breached`` is ab-registry-promote-84a7's scope.

::

    candidate payload ──► score(candidate_type, reward_name, payload)
                              │
                              ├─ input_type mismatch ──► ValueError (loud)
                              └─ match ────────────────► spec.fn(payload)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

INPUT_TYPES = frozenset({"basket", "trades"})
ROLES = frozenset({"primary", "guardrail", "secondary"})
DIRECTIONS = frozenset({"increase", "decrease"})


@dataclass(frozen=True)
class RewardSpec:
    """Registered reward: the callable plus its declarative metadata."""

    name: str
    fn: Callable[..., float]
    input_type: str
    role: str
    direction: str
    threshold: float | None = None


#: name → RewardSpec. Populated ONLY via @register (see the ``basket``
#: import at the bottom of this module). Never insert directly — register()
#: is the sole write path; it enforces every metadata invariant.
REGISTRY: dict[str, RewardSpec] = {}


def register(
    name: str,
    *,
    input_type: str,
    role: str,
    direction: str,
    threshold: float | None = None,
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """Decorator factory: declare a reward's metadata and register it.

    Returns the function unwrapped, so the reward stays directly callable.
    Guardrails must carry a threshold; other roles must not.
    """
    if input_type not in INPUT_TYPES:
        raise ValueError(
            f"reward {name!r}: unknown input_type {input_type!r} (expected one of "
            f"{sorted(INPUT_TYPES)})"
        )
    if role not in ROLES:
        raise ValueError(
            f"reward {name!r}: unknown role {role!r} (expected one of {sorted(ROLES)})"
        )
    if direction not in DIRECTIONS:
        raise ValueError(
            f"reward {name!r}: unknown direction {direction!r} (expected one of "
            f"{sorted(DIRECTIONS)})"
        )
    if role == "guardrail" and threshold is None:
        raise ValueError(f"reward {name!r}: guardrail role requires a threshold")
    if role == "guardrail" and threshold is not None and not math.isfinite(threshold):
        # NaN threshold fails every >/< comparison -> guardrail silently
        # never breaches; inf is "unbounded" better expressed by not being
        # a guardrail at all. Same failure family as non-finite values.
        raise ValueError(
            f"reward {name!r}: guardrail threshold must be finite (got {threshold!r})"
        )
    if role != "guardrail" and threshold is not None:
        raise ValueError(f"reward {name!r}: threshold is guardrail-only (role={role!r})")
    if name in REGISTRY:
        raise ValueError(f"reward {name!r} already registered")

    def _decorator(fn: Callable[..., float]) -> Callable[..., float]:
        # Re-check at insertion time: two register(...) factories obtained
        # before either decorator is applied must not silently clobber.
        if name in REGISTRY:
            raise ValueError(f"reward {name!r} already registered")
        if role == "guardrail" and getattr(fn, "_is_ratio", False):
            # A ratio's ±inf zero-denominator sentinel collides with
            # guardrail_breached's non-finite rejection: the promote gate
            # would raise on the BEST candidate instead of passing it.
            raise ValueError(
                f"reward {name!r}: ratio rewards cannot be guardrails "
                f"(±inf sentinel vs non-finite rejection)"
            )
        fn_input_type = getattr(fn, "_input_type", None)
        if fn_input_type is not None and fn_input_type != input_type:
            # A make_ratio closure carries its constituents' input_type;
            # registering it under a different one would let score() route
            # the wrong payload into the aggregators (silent mis-scoring).
            raise ValueError(
                f"reward {name!r}: fn is bound to input_type {fn_input_type!r}, "
                f"cannot register as {input_type!r}"
            )
        REGISTRY[name] = RewardSpec(
            name=name,
            fn=fn,
            input_type=input_type,
            role=role,
            direction=direction,
            threshold=threshold,
        )
        return fn

    return _decorator


def get(name: str) -> RewardSpec:
    """Look up a reward by name; unknown names raise with the known set."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown reward {name!r}; registered: {names()}") from None


def names() -> list[str]:
    return sorted(REGISTRY.keys())


def score(candidate_type: str, reward_name: str, payload: Any, **kwargs: Any) -> float:
    """Score a candidate's payload with a named reward.

    Refuses an input_type mismatch (e.g. a ``trades`` reward applied to a
    screener ``basket``) with a loud ValueError — no silent mis-scoring.
    Extra kwargs (e.g. ``horizon=``) are forwarded to the reward.
    """
    if candidate_type not in INPUT_TYPES:
        raise ValueError(
            f"unknown candidate_type {candidate_type!r} (expected one of "
            f"{sorted(INPUT_TYPES)})"
        )
    spec = get(reward_name)
    if spec.input_type != candidate_type:
        raise ValueError(
            f"reward {reward_name!r} has input_type {spec.input_type!r}; "
            f"refusing to score a {candidate_type!r} candidate"
        )
    return spec.fn(payload, **kwargs)


def guardrail_breached(reward_name: str, value: float) -> bool:
    """Flag a guardrail breach: ``value`` vs the spec's threshold + direction.

    direction="decrease" (lower is better): breach when value > threshold.
    direction="increase" (higher is better): breach when value < threshold.
    At-threshold is clean. Non-guardrail rewards are rejected. A non-finite
    value (NaN/inf) is corrupt input and raises — NaN fails every ``>``/``<``
    comparison, so it would otherwise read as "clean" and silently disarm
    the guardrail (§4.2: loud, never silent).
    """
    spec = get(reward_name)
    if spec.role != "guardrail":
        raise ValueError(f"reward {reward_name!r} is not a guardrail (role={spec.role!r})")
    if spec.threshold is None:  # unreachable via register(); guards direct REGISTRY writes
        raise ValueError(f"guardrail {reward_name!r} has no threshold")
    if not math.isfinite(value):
        raise ValueError(
            f"guardrail {reward_name!r}: non-finite value {value!r} — refusing to "
            f"evaluate breach on corrupt input"
        )
    if spec.direction == "decrease":
        return value > spec.threshold
    return value < spec.threshold


def make_ratio(numerator: str, denominator: str) -> Callable[..., float]:
    """Compose two registered aggregations into a ratio reward (§4.2).

    Both must share an input_type. Zero denominator is sign-aware (never a
    ZeroDivisionError mid-evaluation): positive numerator → ``inf`` (a
    flawless candidate ranks first, not last), negative → ``-inf``, and
    0/0 → 0.0 (no data). Register the result under its own name to make
    it selectable.

    Ratios cannot be registered as guardrails — register() rejects the
    combination: the ``inf`` sentinel collides with ``guardrail_breached``'s
    non-finite rejection (the promote gate would raise on the best
    candidate instead of passing it).
    """
    num, den = get(numerator), get(denominator)
    if num.input_type != den.input_type:
        raise ValueError(
            f"ratio {numerator!r}/{denominator!r}: input_type mismatch "
            f"({num.input_type!r} vs {den.input_type!r})"
        )

    def _ratio(payload: Any, **kwargs: Any) -> float:
        d = den.fn(payload, **kwargs)
        if d == 0.0:
            n = num.fn(payload, **kwargs)
            if n == 0.0:
                return 0.0
            return math.inf if n > 0 else -math.inf
        return num.fn(payload, **kwargs) / d

    _ratio.__name__ = f"ratio_{numerator}_over_{denominator}"
    _ratio._is_ratio = True  # register() rejects guardrail role for ratios
    _ratio._input_type = num.input_type  # register() enforces matching input_type
    return _ratio


# Import for side effect: registers the screener basket reward set.
from rainier.research.rewards import basket as _basket  # noqa: E402,F401

__all__ = [
    "DIRECTIONS",
    "INPUT_TYPES",
    "REGISTRY",
    "ROLES",
    "RewardSpec",
    "get",
    "guardrail_breached",
    "make_ratio",
    "names",
    "register",
    "score",
]
