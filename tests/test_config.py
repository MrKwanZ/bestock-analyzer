"""Tests for Settings / environment loading."""

from bestock_agent.config import Settings, get_settings


def test_openai_model_loaded_from_env(monkeypatch):
    """OPENAI_MODEL in the environment should override the built-in default."""
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "av-test")
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret")

    settings = Settings()
    assert settings.openai_model == "gpt-5.4-mini"


def test_get_settings_uses_env_openai_model():
    """The cached settings instance should reflect OPENAI_MODEL from .env."""
    assert get_settings().openai_model == "gpt-5.4-mini"
