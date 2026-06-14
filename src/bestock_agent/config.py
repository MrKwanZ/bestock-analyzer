"""Centralised settings loaded once from environment variables / .env file."""

from pathlib import Path

from pydantic import EmailStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from the project root so settings load correctly regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: str
    openai_model: str = "gpt-5.4-mini"

    # ── Financial data providers ───────────────────────────────────────────────
    alphavantage_api_key: str
    yfinance_enabled: bool = True

    # ── News providers ─────────────────────────────────────────────────────────
    brave_api_key: str = ""
    serpapi_api_key: str = ""

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "bestock-analyzer"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # ── Email ─────────────────────────────────────────────────────────────────
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str
    smtp_password: str
    default_email_recipient: EmailStr = "neilhkcheng@gmail.com"

    # ── Agent defaults ─────────────────────────────────────────────────────────
    default_lookback_days: int = 5
    advanced_analysis_enabled: bool = False
    rate_limit_backoff_seconds: int = 20

    @field_validator("default_lookback_days")
    @classmethod
    def _validate_lookback(cls, v: int) -> int:
        if not (3 <= v <= 30):
            raise ValueError("default_lookback_days must be between 3 and 30")
        return v

    @field_validator("rate_limit_backoff_seconds")
    @classmethod
    def _validate_rate_limit_backoff(cls, v: int) -> int:
        if v < 0:
            raise ValueError("rate_limit_backoff_seconds must be non-negative")
        return v

    @field_validator("smtp_port")
    @classmethod
    def _validate_smtp_port(cls, v: int) -> int:
        if v not in (25, 465, 587, 2525):
            raise ValueError(f"Unexpected SMTP port: {v}")
        return v


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return _settings


_settings = Settings()  # type: ignore[call-arg]
