"""Application settings loaded from .env and environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Singleton settings object. Read once at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Anthropic — required at runtime, but allow empty in tests/CI
    anthropic_api_key: str = Field(default="")
    anthropic_model: str = "claude-opus-4-7"

    # Agent runtime
    agent_max_iterations: int = 15
    agent_max_tokens_budget: int = 100_000

    # Logging
    log_level: str = "INFO"
    log_format: str = "console"  # 'console' or 'json'

    # Data
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".quant_agent_cache")


settings = Settings()
