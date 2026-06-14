"""WS A/B — typed `paper` config block: defaults + env-override reload.

Acceptance A5: config typed + reload-tested; shadow default ON, live OFF.
"""

from __future__ import annotations

from rainier.core.config import PaperConfig, Settings


def test_paper_config_defaults():
    cfg = PaperConfig()
    # WS A: shadow measurement ON; the live buy-flip (A2) is NOT this config.
    assert cfg.watch_buy_shadow is True
    assert cfg.watch_buy_min_confidence == 5
    # WS B: auto-thesis OFF by default (detection + queue only).
    assert cfg.reclaim_auto_thesis is False
    assert cfg.reclaim_dedup_days == 5
    assert cfg.reclaim_window_days == 20
    assert cfg.reclaim_audit_min_events == 10
    assert cfg.reclaim_audit_horizon_days == 10


def test_settings_exposes_paper_block():
    s = Settings()
    assert isinstance(s.paper, PaperConfig)
    assert s.paper.watch_buy_shadow is True


def test_paper_config_env_override_reloads(monkeypatch):
    """A fresh Settings() picks up nested env overrides (reload contract)."""
    monkeypatch.setenv("PAPER__WATCH_BUY_MIN_CONFIDENCE", "6")
    monkeypatch.setenv("PAPER__RECLAIM_AUTO_THESIS", "true")
    s = Settings()
    assert s.paper.watch_buy_min_confidence == 6
    assert s.paper.reclaim_auto_thesis is True
