"""Unified configuration — .env (secrets) + YAML (app config)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Price action analysis config (futures)
# ---------------------------------------------------------------------------


class PivotConfig(BaseModel):
    lookback: int = 5  # bars each side for swing high/low


class SRHorizontalConfig(BaseModel):
    cluster_atr_mult: float = 0.5  # cluster within this * ATR(14)
    min_touches: int = 3  # need 3+ pin bar wick tips to form a level
    # Strength weights (must sum to 1.0)
    weight_touches: float = 0.4
    weight_recency: float = 0.3
    weight_volume: float = 0.2
    weight_round_number: float = 0.1


class SRDiagonalConfig(BaseModel):
    min_swing_points: int = 2
    price_tolerance_atr_mult: float = 0.3
    slope_similarity_threshold: float = 0.05
    intercept_similarity_atr_mult: float = 0.5
    min_touches: int = 2


class PinBarConfig(BaseModel):
    min_dominant_wick_ratio: float = 0.667
    max_secondary_wick_ratio: float = 0.333
    min_wick_body_ratio: float = 2.0
    min_amplitude_lookback: int = 20
    wick_exceed_lookback: int = 5
    sr_proximity_pct: float = 0.005


class InsideBarConfig(BaseModel):
    max_consecutive: int = 5


class AnalysisConfig(BaseModel):
    pivot: PivotConfig = PivotConfig()
    sr_horizontal: SRHorizontalConfig = SRHorizontalConfig()
    sr_diagonal: SRDiagonalConfig = SRDiagonalConfig()
    pin_bar: PinBarConfig = PinBarConfig()
    inside_bar: InsideBarConfig = InsideBarConfig()
    max_sr_levels: int = 10
    max_diagonal_levels: int = 3


class ScorerConfig(BaseModel):
    min_confidence: float = 0.60
    weight_sr_strength: float = 0.30
    weight_wick_ratio: float = 0.20
    weight_volume_spike: float = 0.15
    weight_trend_alignment: float = 0.20
    weight_multi_tf_confluence: float = 0.15


class SignalConfig(BaseModel):
    scorer: ScorerConfig = ScorerConfig()
    default_rr_target: float = 2.0
    min_rr_ratio: float = 1.5


class RiskConfig(BaseModel):
    max_positions: int = 3
    max_daily_loss: float = 1000.0
    max_drawdown_pct: float = 0.05
    position_size_risk_pct: float = 0.01


class DiscordConfig(BaseModel):
    webhook_url: str = ""
    enabled: bool = False


class AlertsConfig(BaseModel):
    discord: DiscordConfig = DiscordConfig()


# ---------------------------------------------------------------------------
# Scraping config (stocks / QuantUnicorn)
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    name: str = "rainier"
    data_dir: str = "./data"
    log_level: str = "INFO"
    timezone: str = "America/Los_Angeles"


class ScrapingSchedule(BaseModel):
    morning: str = "08:35"
    midday: str = "10:35"
    afternoon: str = "12:35"
    close: str = "14:35"


class QuantUnicornConfig(BaseModel):
    url: str = "https://www.quantunicorn.com/products#qu100"
    login_url: str = "https://www.quantunicorn.com/signin"
    session_file: str = "./data/auth/qu_session.json"
    session_ttl_hours: int = 12
    headless: bool = True
    timeout_ms: int = 30000


class TradingViewConfig(BaseModel):
    base_url: str = "https://www.tradingview.com/chart/"
    timeframe_days: int = 120
    chart_width: int = 1280
    chart_height: int = 720
    headless: bool = True
    output_dir: str = "./data/charts"


class ScrapingConfig(BaseModel):
    quantunicorn: QuantUnicornConfig = QuantUnicornConfig()
    tradingview: TradingViewConfig = TradingViewConfig()
    schedule: ScrapingSchedule = ScrapingSchedule()


# ---------------------------------------------------------------------------
# LLM analysis config
# ---------------------------------------------------------------------------


class LLMAnalysisConfig(BaseModel):
    default_model: str = "claude-cli"
    fallback_models: list[str] = Field(default_factory=list)
    local_model: str = "ollama/deepseek-r1:14b"
    default_prompt: str = "default_analysis"
    prompts_dir: str = "./config/prompts"
    max_retries: int = 3
    timeout_seconds: int = 120


class NotifyConfig(BaseModel):
    enabled: bool = True
    subject_prefix: str = "[Rainier]"


# ---------------------------------------------------------------------------
# Database config
# ---------------------------------------------------------------------------


class DatabaseConfig(BaseModel):
    echo: bool = False
    pool_size: int = 5


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Root settings — .env provides secrets, YAML provides app config."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    # Secrets from .env
    database_url: str = "postgresql://rainier:rainier_dev@localhost:5432/rainier"
    qu_username: str = ""
    qu_password: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    polygon_api_key: str = ""
    notify_urls: str = ""  # Apprise URL(s), comma-separated

    # App config from YAML
    app: AppConfig = AppConfig()
    database: DatabaseConfig = DatabaseConfig()

    # Price action analysis (futures)
    analysis: AnalysisConfig = AnalysisConfig()
    signal: SignalConfig = SignalConfig()
    risk: RiskConfig = RiskConfig()
    alerts: AlertsConfig = AlertsConfig()
    symbols: list[str] = ["NQ", "ES", "GC"]
    timeframes: list[str] = ["1D", "4H", "1H", "15m"]

    # Scraping (stocks)
    scraping: ScrapingConfig = ScrapingConfig()

    # LLM analysis
    llm: LLMAnalysisConfig = LLMAnalysisConfig()

    # Notifications
    notify: NotifyConfig = NotifyConfig()


class InstrumentConfig(BaseModel):
    symbol: str
    name: str = ""
    exchange: str = ""
    tick_size: float = 0.25
    point_value: float = 1.0
    min_touches: int = 3


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from .env (secrets) + YAML (app config)."""
    load_dotenv()

    if config_path is None:
        config_path = Path("config/settings.yaml")

    yaml_config: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f) or {}

    # Build nested config objects from YAML sections
    kwargs: dict[str, Any] = {}
    if "app" in yaml_config:
        kwargs["app"] = AppConfig(**yaml_config["app"])
    if "database" in yaml_config:
        kwargs["database"] = DatabaseConfig(**yaml_config["database"])
    if "analysis" in yaml_config:
        kwargs["analysis"] = AnalysisConfig(**yaml_config["analysis"])
    if "signal" in yaml_config:
        kwargs["signal"] = SignalConfig(**yaml_config["signal"])
    if "risk" in yaml_config:
        kwargs["risk"] = RiskConfig(**yaml_config["risk"])
    if "alerts" in yaml_config:
        kwargs["alerts"] = AlertsConfig(**yaml_config["alerts"])
    if "scraping" in yaml_config:
        raw = yaml_config["scraping"]
        kwargs["scraping"] = ScrapingConfig(
            quantunicorn=QuantUnicornConfig(**raw.get("quantunicorn", {})),
            tradingview=TradingViewConfig(**raw.get("tradingview", {})),
            schedule=ScrapingSchedule(**raw.get("schedule", {})),
        )
    if "llm" in yaml_config:
        kwargs["llm"] = LLMAnalysisConfig(**yaml_config["llm"])
    if "notify" in yaml_config:
        kwargs["notify"] = NotifyConfig(**yaml_config["notify"])

    return Settings(**kwargs)


# Singleton
_settings: Settings | None = None


def get_settings(config_path: str = "config/settings.yaml") -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = load_settings(Path(config_path))
    return _settings


def load_watchlist(watchlist_path: Path | None = None) -> dict[str, InstrumentConfig]:
    """Load watchlist, returning a symbol -> InstrumentConfig mapping."""
    if watchlist_path is None:
        watchlist_path = Path("config/watchlists/default.yaml")
    if not watchlist_path.exists():
        return {}
    with open(watchlist_path) as f:
        data = yaml.safe_load(f) or {}
    instruments = data.get("instruments", [])
    return {inst["symbol"]: InstrumentConfig(**inst) for inst in instruments}
