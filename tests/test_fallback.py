"""Tests for services/fallback.py — provider switching and retry logic."""

from datetime import datetime, timezone


from bestock_agent.schemas import AgentError, ErrorType
from bestock_agent.services.fallback import decide_fallback, failing_node, is_retryable


# ── Helpers ───────────────────────────────────────────────────────────────────


def _err(node: str, error_type: ErrorType = ErrorType.TOOL_ERROR, recoverable: bool = True) -> AgentError:
    return AgentError(
        error_type=error_type,
        message=f"Simulated {error_type.value} in {node}",
        node=node,
        timestamp=datetime.now(timezone.utc).isoformat(),
        recoverable=recoverable,
    )


def _state(**kwargs) -> dict:
    defaults = dict(
        errors=[],
        retry_count=0,
        rate_limit_backoff_used=False,
        active_financial_provider="alphavantage",
        active_news_provider="brave",
    )
    return {**defaults, **kwargs}


# ── is_retryable ──────────────────────────────────────────────────────────────


def test_retryable_when_budget_and_recoverable():
    state = _state(errors=[_err("fetch_top_gainer")], retry_count=0)
    assert is_retryable(state, max_retries=2) is True


def test_not_retryable_when_budget_exhausted():
    state = _state(errors=[_err("fetch_top_gainer")], retry_count=2)
    assert is_retryable(state, max_retries=2) is False


def test_not_retryable_when_unrecoverable():
    state = _state(errors=[_err("analyze_trend", recoverable=False)], retry_count=0)
    assert is_retryable(state, max_retries=2) is False


def test_not_retryable_for_validation_error():
    state = _state(
        errors=[_err("fetch_top_gainer", error_type=ErrorType.VALIDATION_ERROR)],
        retry_count=0,
    )
    assert is_retryable(state, max_retries=2) is False


def test_not_retryable_with_no_errors():
    state = _state(errors=[], retry_count=0)
    assert is_retryable(state, max_retries=2) is False


# ── decide_fallback ───────────────────────────────────────────────────────────


def test_alphavantage_falls_back_to_yfinance():
    state = _state(
        errors=[_err("fetch_top_gainer")],
        active_financial_provider="alphavantage",
    )
    updates = decide_fallback(state)
    assert updates.get("active_financial_provider") == "yfinance"


def test_yfinance_has_no_further_fallback():
    state = _state(
        errors=[_err("fetch_top_gainer")],
        active_financial_provider="yfinance",
    )
    updates = decide_fallback(state)
    assert "active_financial_provider" not in updates


def test_price_history_also_triggers_fallback():
    state = _state(
        errors=[_err("fetch_price_history")],
        active_financial_provider="alphavantage",
    )
    updates = decide_fallback(state)
    assert updates.get("active_financial_provider") == "yfinance"


def test_brave_falls_back_to_serpapi():
    state = _state(
        errors=[_err("fetch_news_and_sentiment")],
        active_news_provider="brave",
    )
    updates = decide_fallback(state)
    assert updates.get("active_news_provider") == "serpapi"


def test_first_rate_limit_defers_provider_switch_until_backoff():
    state = _state(
        errors=[_err("fetch_top_gainer", error_type=ErrorType.RATE_LIMIT)],
        active_financial_provider="alphavantage",
        rate_limit_backoff_used=False,
    )
    updates = decide_fallback(state)
    assert updates == {}


def test_rate_limit_after_backoff_switches_provider_and_resets_flag():
    state = _state(
        errors=[_err("fetch_top_gainer", error_type=ErrorType.RATE_LIMIT)],
        active_financial_provider="alphavantage",
        rate_limit_backoff_used=True,
    )
    updates = decide_fallback(state)
    assert updates.get("active_financial_provider") == "yfinance"
    assert updates.get("rate_limit_backoff_used") is False


def test_news_rate_limit_after_backoff_switches_provider_and_resets_flag():
    state = _state(
        errors=[_err("fetch_news_and_sentiment", error_type=ErrorType.RATE_LIMIT)],
        active_news_provider="brave",
        rate_limit_backoff_used=True,
    )
    updates = decide_fallback(state)
    assert updates.get("active_news_provider") == "serpapi"
    assert updates.get("rate_limit_backoff_used") is False


def test_validation_error_does_not_trigger_provider_switch():
    state = _state(
        errors=[_err("fetch_top_gainer", error_type=ErrorType.VALIDATION_ERROR)],
        active_financial_provider="alphavantage",
    )
    updates = decide_fallback(state)
    assert "active_financial_provider" not in updates


def test_fallback_injects_switch_note():
    state = _state(
        errors=[_err("fetch_top_gainer")],
        active_financial_provider="alphavantage",
    )
    updates = decide_fallback(state)
    switch_notes = updates.get("errors", [])
    assert len(switch_notes) == 1
    assert switch_notes[0].fallback_used is True
    assert "alphavantage" in switch_notes[0].message
    assert "yfinance" in switch_notes[0].message


# ── failing_node ──────────────────────────────────────────────────────────────


def test_failing_node_returns_originating_node():
    state = _state(errors=[_err("fetch_price_history")])
    assert failing_node(state) == "fetch_price_history"


def test_failing_node_skips_error_handler_entries():
    state = _state(errors=[
        _err("fetch_top_gainer"),
        _err("error_handler"),   # injected switch note — should be skipped
    ])
    assert failing_node(state) == "fetch_top_gainer"


def test_failing_node_none_when_no_errors():
    state = _state(errors=[])
    assert failing_node(state) is None
