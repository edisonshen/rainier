from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_strategy_signals_migration_and_model_contract():
    from rainier.core.models import HYPERTABLES, StrategySignal

    migration = (ROOT / "migrations" / "0015_strategy_signals.sql").read_text()
    downgrade = (
        ROOT / "migrations" / "0015_strategy_signals_downgrade.sql"
    ).read_text()
    sql = migration.lower()
    assert migration.lstrip().startswith("--")
    assert "begin;" in sql and "commit;" in sql
    assert "create table if not exists strategy_signals" in sql
    assert "unique (strategy, signal_date)" in sql
    assert "params      jsonb            not null" in sql
    assert "create_hypertable" not in sql
    assert "strategy_signals" not in HYPERTABLES
    assert "drop table if exists strategy_signals" in downgrade.lower()
    assert set(StrategySignal.__table__.columns.keys()) == {
        "id",
        "strategy",
        "signal_date",
        "fg",
        "qqq_close",
        "qqq_sma",
        "entry_price",
        "stop_price",
        "regime_sma",
        "holding",
        "armed",
        "bear",
        "locked",
        "action",
        "params",
        "created_at",
    }
    assert any(
        {column.name for column in constraint.columns} == {"strategy", "signal_date"}
        for constraint in StrategySignal.__table__.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    )


def test_fng_tqqq_cron_runs_after_fear_greed_capture():
    data = yaml.safe_load((ROOT / "config" / "cron.yaml").read_text())
    jobs = data["jobs"]
    names = [job["name"] for job in jobs]
    job = {job["name"]: job for job in jobs}["fng-tqqq-signal"]

    assert job["enabled"] is True
    assert job["schedule"] == "20 15 * * 1-5"
    assert job["command"] == "uv run rainier strategy fng-tqqq signal"
    assert job["log"] == "data/fng-tqqq.log"
    assert job["discord_on_failure"] is True
    assert names.index("fear-greed-daily") < names.index("fng-tqqq-signal")
    for name, schedule in (
        ("fng-tqqq-stop-check", "58 12 * * 1-5"),
        ("fng-tqqq-stop-check-halfday", "58 9 * * 1-5"),
    ):
        stop_job = {job["name"]: job for job in jobs}[name]
        assert stop_job["enabled"] is True
        assert stop_job["schedule"] == schedule
        assert stop_job["command"] == "uv run rainier strategy fng-tqqq stop-check"
        assert stop_job["log"] == "data/fng-tqqq.log"
        assert stop_job["discord_on_failure"] is True


def test_strategy_config_and_discord_webhook_environment_wiring(monkeypatch):
    from rainier.core.config import load_settings

    monkeypatch.setenv(
        "DISCORD_STRATEGY_WEBHOOK_URL", "https://discord.example/strategy"
    )
    settings = load_settings(ROOT / "config" / "settings.yaml")

    assert settings.strategies.fng_tqqq.buy_below == 30.0
    assert settings.strategies.fng_tqqq.sma_window == 38
    assert settings.strategies.fng_tqqq.regime_sma_window == 180
    assert settings.strategies.fng_tqqq.stop_loss_pct == 0.10
    assert settings.strategies.fng_tqqq.start.isoformat() == "2021-01-22"
    assert (
        settings.alerts.discord.strategy_webhook_url
        == "https://discord.example/strategy"
    )


def test_fng_tqqq_signal_and_pipeline_respect_module_boundaries():
    sources = [
        ROOT / "src" / "rainier" / "signals" / "fng_tqqq.py",
        ROOT / "src" / "rainier" / "pipeline" / "fng_tqqq_signal.py",
    ]
    for path in sources:
        tree = ast.parse(path.read_text())
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        imports.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            module.startswith("rainier.backtest") or module.startswith("rainier.trader")
            for module in imports
        )
