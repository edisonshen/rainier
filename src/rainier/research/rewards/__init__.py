"""Reward-function registry — concrete reward bodies land in Slice 1.

Slice 0 ships the registry surface (an empty dict) so other modules can
import it without breaking once Slice 1 lands the seven pre-registered
reward functions (D-007).
"""

from __future__ import annotations

from typing import Callable

#: name → callable. Each callable takes a TradeRecord and returns a float.
#: Slice 1 populates this with r_multiple_expectancy, all_call_R, etc.
REGISTRY: dict[str, Callable] = {}


def register(name: str, fn: Callable) -> Callable:
    """Decorator-style registrar — placeholder for Slice 1."""
    REGISTRY[name] = fn
    return fn


def names() -> list[str]:
    return sorted(REGISTRY.keys())
