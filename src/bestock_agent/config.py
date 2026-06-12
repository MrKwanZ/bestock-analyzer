"""Centralised settings loaded once from environment variables / .env file."""

from pydantic import EmailStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    # ── Financial data providers ───────────────────────────────────────────────
    alphavantage_api_key: str
    yfinance_enabled: bool = True

    # ── News providers ─────────────────────────────────────────────────────────
    brave_api_key: str = ""
    serpapi_api_key: str = ""

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "bestock-agent"

    # ── Email ─────────────────────────────────────────────────────────────────
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str
    smtp_password: str
    default_email_recipient: EmailStr = "neilhkcheng@gmail.com"

    # ── Agent defaults ─────────────────────────────────────────────────────────
    default_lookback_days: int = 5
    advanced_analysis_enabled: bool = False

    @field_validator("default_lookback_days")
    @classmethod
    def _validate_lookback(cls, v: int) -> int:
        if not (3 <= v <= 30):
            raise ValueError("default_lookback_days must be between 3 and 30")
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
