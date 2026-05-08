"""Protocol conformance for every signal in the registry."""

from __future__ import annotations

from rainier.llm_thesis.signals import REGISTRY
from rainier.llm_thesis.signals.base import ThesisSignal


def test_registry_has_four_signals():
    assert set(REGISTRY) == {
        "rank_trajectory",
        "capital_flow_streak",
        "sector_momentum",
        "fundamentals",
    }


def test_every_signal_satisfies_protocol():
    for name, sig_cls in REGISTRY.items():
        instance = sig_cls()
        assert isinstance(instance, ThesisSignal), f"{name} fails ThesisSignal protocol"


def test_every_signal_has_required_class_attrs():
    for name, sig_cls in REGISTRY.items():
        instance = sig_cls()
        assert isinstance(instance.name, str) and instance.name == name
        assert isinstance(instance.version, str) and instance.version
        assert isinstance(instance.cost_estimate_ms, int)
