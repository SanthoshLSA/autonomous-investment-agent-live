"""
Configuration management for the Autonomous Investment Research Agent.

Loads settings from config.yaml and merges with environment variables from .env.
Provides type-safe access via Pydantic models with validation.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# ═══════════════════════════════════════════════════════════════════════════════
# Path Constants
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic Configuration Models
# ═══════════════════════════════════════════════════════════════════════════════


class WatchlistConfig(BaseModel):
    """Asset tickers to monitor daily."""

    us_stocks: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
    indian_stocks: list[str] = Field(
        default_factory=lambda: ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    )
    crypto: list[str] = Field(default_factory=lambda: ["BTC-USD", "ETH-USD"])
    indices: list[str] = Field(default_factory=lambda: ["^GSPC", "^NSEI"])

    @property
    def all_tickers(self) -> list[str]:
        """Return flattened list of all tickers across categories."""
        return self.us_stocks + self.indian_stocks + self.crypto + self.indices


class AnalysisConfig(BaseModel):
    """Technical analysis parameters."""

    lookback_days: int = 252
    sma_periods: list[int] = Field(default_factory=lambda: [20, 50, 200])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: int = 2


class RiskConfig(BaseModel):
    """Risk analysis parameters."""

    monte_carlo_simulations: int = 10000
    var_confidence: float = 0.95
    cvar_confidence: float = 0.95
    risk_free_rate: float = 0.05
    max_drawdown_threshold: float = 0.25


class SentimentConfig(BaseModel):
    """Sentiment analysis parameters."""

    news_lookback_days: int = 7
    max_articles_per_asset: int = 10
    decay_factor: float = 0.3


class ScoringConfig(BaseModel):
    """Composite scoring weights."""

    volatility_weight: float = 0.40
    sentiment_weight: float = 0.30
    market_beta_weight: float = 0.20
    drawdown_weight: float = 0.10

    @field_validator(
        "volatility_weight", "sentiment_weight", "market_beta_weight", "drawdown_weight"
    )
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Ensure each weight is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {v}")
        return v


class PortfolioConfig(BaseModel):
    """Portfolio optimization settings."""

    risk_tolerance: str = "conservative"
    optimization_method: str = "min_volatility"
    rebalance_threshold: float = 0.10
    rebalance_frequency: str = "quarterly"
    max_single_position: float = 0.25
    min_position_size: float = 0.02
    initial_capital: float = 1_000_000

    @field_validator("risk_tolerance")
    @classmethod
    def validate_risk_tolerance(cls, v: str) -> str:
        """Ensure risk tolerance is valid."""
        allowed = {"conservative", "moderate", "aggressive"}
        if v not in allowed:
            raise ValueError(f"risk_tolerance must be one of {allowed}, got '{v}'")
        return v


class BacktestConfig(BaseModel):
    """Backtesting parameters."""

    start_date: str = "2021-01-01"
    end_date: str | None = None
    benchmark: str = "^GSPC"
    transaction_cost: float = 0.001
    slippage: float = 0.0005


class LLMConfig(BaseModel):
    """Language model configuration."""

    provider: str = "ollama"
    model: str = "llama3.1:8b"
    fallback_model: str = "qwen2.5:7b"
    temperature: float = 0
    max_retries: int = 3
    timeout_seconds: int = 120

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Ensure provider is supported."""
        allowed = {"ollama", "openai", "groq"}
        if v not in allowed:
            raise ValueError(f"LLM provider must be one of {allowed}, got '{v}'")
        return v


class ScheduleConfig(BaseModel):
    """Scheduling parameters."""

    enabled: bool = True
    run_time: str = "09:00"
    timezone: str = "Asia/Kolkata"
    weekend_skip: bool = True


class NotificationsConfig(BaseModel):
    """Notification channel settings."""

    telegram_enabled: bool = False
    email_enabled: bool = False


class CacheConfig(BaseModel):
    """Data caching settings."""

    enabled: bool = True
    ttl_hours: int = 4
    database_path: str = "cache/market_data.db"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"
    file_path: str = "logs/agent.log"
    max_file_size_mb: int = 50
    backup_count: int = 5


class AppConfig(BaseModel):
    """Root configuration model aggregating all sections."""

    watchlist: WatchlistConfig = Field(default_factory=WatchlistConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ═══════════════════════════════════════════════════════════════════════════════
# API Keys (loaded from environment)
# ═══════════════════════════════════════════════════════════════════════════════


class APIKeys(BaseModel):
    """API keys loaded from environment variables."""

    newsapi_key: str | None = None
    finnhub_key: str | None = None
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    @classmethod
    def from_env(cls) -> APIKeys:
        """Load API keys from environment variables with Streamlit secrets fallback."""
        newsapi = os.getenv("NEWSAPI_KEY")
        finnhub = os.getenv("FINNHUB_KEY")
        openai = os.getenv("OPENAI_API_KEY")
        groq = os.getenv("GROQ_API_KEY")
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat = os.getenv("TELEGRAM_CHAT_ID")

        # Fallback to Streamlit secrets if running inside streamlit
        try:
            import streamlit as st

            # Check if secrets attribute exists and is populated
            if hasattr(st, "secrets") and st.secrets:
                newsapi = newsapi or st.secrets.get("NEWSAPI_KEY")
                finnhub = finnhub or st.secrets.get("FINNHUB_KEY")
                openai = openai or st.secrets.get("OPENAI_API_KEY")
                groq = groq or st.secrets.get("GROQ_API_KEY")
                telegram_token = telegram_token or st.secrets.get("TELEGRAM_BOT_TOKEN")
                telegram_chat = telegram_chat or st.secrets.get("TELEGRAM_CHAT_ID")
        except Exception:
            pass

        return cls(
            newsapi_key=newsapi,
            finnhub_key=finnhub,
            openai_api_key=openai,
            groq_api_key=groq,
            telegram_bot_token=telegram_token,
            telegram_chat_id=telegram_chat,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration Loading
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level singleton
_config: AppConfig | None = None
_api_keys: APIKeys | None = None


def load_config(config_path: Path | None = None) -> AppConfig:
    """
    Load application configuration from YAML file.

    Args:
        config_path: Path to config.yaml. Defaults to PROJECT_ROOT/config.yaml.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config validation fails.
    """
    global _config

    path = config_path or CONFIG_PATH
    if not path.exists():
        # Return defaults if no config file
        _config = AppConfig()
        return _config

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    _config = AppConfig(**raw)
    return _config


def load_api_keys() -> APIKeys:
    """
    Load API keys from .env file and environment.

    Returns:
        APIKeys instance with available keys.
    """
    global _api_keys

    # Load .env file if it exists
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)

    _api_keys = APIKeys.from_env()
    return _api_keys


def get_config() -> AppConfig:
    """
    Get the current configuration, loading if necessary.

    Returns:
        The application configuration singleton.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_api_keys() -> APIKeys:
    """
    Get API keys, loading if necessary.

    Returns:
        The API keys singleton.
    """
    global _api_keys
    if _api_keys is None:
        _api_keys = load_api_keys()
    return _api_keys
