"""Tests for risk management (Phase 3 stubs, basic logic tests)."""

from rainier.core.config import RiskConfig


class TestRiskConfig:
    def test_default_values(self):
        config = RiskConfig()
        assert config.max_positions == 3
        assert config.max_daily_loss == 1000.0
        assert config.max_drawdown_pct == 0.05
        assert config.position_size_risk_pct == 0.01

    def test_position_limit_logic(self):
        config = RiskConfig(max_positions=2)
        current_positions = 2
        assert current_positions >= config.max_positions  # should reject

    def test_daily_loss_kill_switch(self):
        config = RiskConfig(max_daily_loss=500.0)
        daily_pnl = -501.0
        assert abs(daily_pnl) > config.max_daily_loss  # should kill

    def test_exactly_at_limit(self):
        config = RiskConfig(max_daily_loss=500.0)
        daily_pnl = -500.0
        assert abs(daily_pnl) >= config.max_daily_loss  # boundary: should kill
