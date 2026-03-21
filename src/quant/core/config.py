"""Pydantic settings with nested config models. Each module gets only its own config."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class PivotConfig(BaseModel):
    lookback: int = 5  # bars each side for swing high/low


class SRHorizontalConfig(BaseModel):
    cluster_atr_mult: float = 0.3  # cluster within this * ATR(14)
    min_touches: int = 2
    # Strength weights (must sum to 1.0)
    weight_touches: float = 0.4
    weight_recency: float = 0.3
    weight_volume: float = 0.2
    weight_round_number: float = 0.1


class SRDiagonalConfig(BaseModel):
    min_swing_points: int = 2
    price_tolerance_atr_mult: float = 0.3  # how close price must be to line to count as "touch"
    slope_similarity_threshold: float = 0.05  # for deduplication
    intercept_similarity_atr_mult: float = 0.5
    min_touches: int = 2


class PinBarConfig(BaseModel):
    min_wick_body_ratio: float = 2.0
    max_body_pct: float = 0.30  # body < 30% of range
    wick_exceed_lookback: int = 5  # wick must exceed N prior candles
    sr_proximity_pct: float = 0.005  # within 0.5% of S/R level


class InsideBarConfig(BaseModel):
    max_consecutive: int = 5  # track up to N consecutive inside bars


class AnalysisConfig(BaseModel):
    pivot: PivotConfig = PivotConfig()
    sr_horizontal: SRHorizontalConfig = SRHorizontalConfig()
    sr_diagonal: SRDiagonalConfig = SRDiagonalConfig()
    pin_bar: PinBarConfig = PinBarConfig()
    inside_bar: InsideBarConfig = InsideBarConfig()


class ScorerConfig(BaseModel):
    min_confidence: float = 0.5  # minimum to generate signal
    # Sub-score weights
    weight_sr_strength: float = 0.30
    weight_wick_ratio: float = 0.20
    weight_volume_spike: float = 0.15
    weight_trend_alignment: float = 0.20
    weight_multi_tf_confluence: float = 0.15


class SignalConfig(BaseModel):
    scorer: ScorerConfig = ScorerConfig()
    default_rr_target: float = 2.0  # R:R when no next S/R for TP


class RiskConfig(BaseModel):
    max_positions: int = 3
    max_daily_loss: float = 1000.0  # dollars
    max_drawdown_pct: float = 0.05  # 5% of account
    position_size_risk_pct: float = 0.01  # risk 1% of account per trade


class DiscordConfig(BaseModel):
    webhook_url: str = ""
    enabled: bool = False


class AlertsConfig(BaseModel):
    discord: DiscordConfig = DiscordConfig()


class DatabaseConfig(BaseModel):
    url: str = "postgresql://postgres:quant@localhost:5432/quant"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    analysis: AnalysisConfig = AnalysisConfig()
    signal: SignalConfig = SignalConfig()
    risk: RiskConfig = RiskConfig()
    alerts: AlertsConfig = AlertsConfig()
    database: DatabaseConfig = DatabaseConfig()

    # Watchlist
    symbols: list[str] = ["NQ", "ES", "GC"]
    timeframes: list[str] = ["1D", "4H", "1H", "15m"]


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings, optionally merging from a YAML config file."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}
        return Settings(**yaml_data)
    return Settings()
