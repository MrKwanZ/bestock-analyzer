"""Tests for graph routing functions and the error_handler node.

All tests use mocked providers so no real API calls are made.
"""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bestock_agent.nodes.error_handler import error_handler
from bestock_agent.schemas import AgentError, ErrorType, PriceBar, RunSummary, TopGainer
from bestock_agent.state import BestockState, initial_state


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _base_state(**overrides) -> BestockState:
    state = initial_state(
        recipient_email="test@example.com",
        target_date="2026-06-11",
        lookback_days=5,
    )
    state.update(overrides)  # type: ignore[arg-type]
    return state


def _gainer() -> TopGainer:
    return TopGainer(symbol="NVDA", name="NVIDIA", price=1200.0, change=50.0, change_pct=4.3)


def _bars() -> list[PriceBar]:
    return [
        PriceBar(date=date(2026, 6, 3), open=100, high=101, low=99, close=100, volume=1000),
        PriceBar(date=date(2026, 6, 4), open=101, high=102, low=100, close=102, volume=1000),
    ]


def _err(node: str, recoverable: bool = True, error_type: ErrorType = ErrorType.TOOL_ERROR) -> AgentError:
    return AgentError(
        error_type=error_type,
        message=f"Simulated error in {node}",
        node=node,
        timestamp=datetime.now(timezone.utc).isoformat(),
        recoverable=recoverable,
    )


# ── error_handler node ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_error_handler_retries_when_budget_available():
    state = _base_state(errors=[_err("fetch_top_gainer")], retry_count=0)
    result = await error_handler(state)
    # Should increment retry_count and NOT produce a terminal RunSummary
    assert result.get("retry_count", 0) == 1
    assert "run_summary" not in result or result.get("run_summary") is None


@pytest.mark.asyncio
async def test_error_handler_terminates_when_budget_exhausted():
    state = _base_state(errors=[_err("fetch_top_gainer")], retry_count=2)
    result = await error_handler(state)
    assert result.get("retry_count") == 2
    summary: RunSummary = result["run_summary"]
    assert summary.success is False
    assert "fetch_top_gainer" in summary.message


@pytest.mark.asyncio
async def test_error_handler_terminates_on_unrecoverable():
    state = _base_state(
        errors=[_err("analyze_trend", recoverable=False)],
        retry_count=0,
    )
    result = await error_handler(state)
    summary: RunSummary = result["run_summary"]
    assert summary.success is False


@pytest.mark.asyncio
async def test_error_handler_switches_to_yfinance_on_alphavantage_failure():
    state = _base_state(
        errors=[_err("fetch_top_gainer")],
        retry_count=0,
        active_financial_provider="alphavantage",
    )
    result = await error_handler(state)
    # Provider should switch
    assert result.get("active_financial_provider") == "yfinance"
    assert result.get("retry_count") == 1


@pytest.mark.asyncio
async def test_error_handler_message_contains_node_and_type():
    state = _base_state(errors=[_err("fetch_top_gainer")], retry_count=2)
    result = await error_handler(state)
    msg = result["run_summary"].message
    assert "fetch_top_gainer" in msg
    assert "tool_error" in msg


# ── fetch_top_gainer node routing ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_top_gainer_routes_to_error_on_provider_failure():
    """A provider exception must produce an AgentError and no top_gainer."""
    from bestock_agent.nodes.fetch_top_gainer import fetch_top_gainer

    state = _base_state()
    mock_provider = AsyncMock()
    mock_provider.get_top_nasdaq_gainer.side_effect = RuntimeError("API down")

    with patch("bestock_agent.nodes.fetch_top_gainer.get_financial_provider", return_value=mock_provider):
        result = await fetch_top_gainer(state)

    assert "top_gainer" not in result or result.get("top_gainer") is None
    assert len(result.get("errors", [])) == 1
    assert result["errors"][0].node == "fetch_top_gainer"


@pytest.mark.asyncio
async def test_fetch_top_gainer_validates_and_rejects_zero_price():
    from bestock_agent.nodes.fetch_top_gainer import fetch_top_gainer

    state = _base_state()
    bad_gainer = TopGainer(symbol="BAD", name="Bad Corp", price=0.0001, change=0, change_pct=0)

    # Force pydantic to accept price=0.0001 but then validation catches it via validate_top_gainer
    # (price > 0 passes pydantic but change_pct=-100 → implausible)
    absurd_gainer = TopGainer(symbol="BAD", name="Bad Corp", price=1.0, change=-10000, change_pct=-9999.0)
    mock_provider = AsyncMock()
    mock_provider.get_top_nasdaq_gainer.return_value = absurd_gainer

    with patch("bestock_agent.nodes.fetch_top_gainer.get_financial_provider", return_value=mock_provider):
        result = await fetch_top_gainer(state)

    errors = result.get("errors", [])
    assert len(errors) == 1
    assert errors[0].error_type == ErrorType.VALIDATION_ERROR


@pytest.mark.asyncio
async def test_fetch_price_history_validates_empty_bars():
    from bestock_agent.nodes.fetch_price_history import fetch_price_history

    state = _base_state(top_gainer=_gainer())
    mock_provider = AsyncMock()
    mock_provider.get_price_history.return_value = []

    with patch("bestock_agent.nodes.fetch_price_history.get_financial_provider", return_value=mock_provider):
        result = await fetch_price_history(state)

    errors = result.get("errors", [])
    assert len(errors) == 1
    assert errors[0].error_type == ErrorType.VALIDATION_ERROR
    assert "No price bars" in errors[0].message


@pytest.mark.asyncio
async def test_fetch_price_history_network_error_is_recoverable():
    from bestock_agent.nodes.fetch_price_history import fetch_price_history

    state = _base_state(top_gainer=_gainer())
    mock_provider = AsyncMock()
    mock_provider.get_price_history.side_effect = OSError("Connection reset")

    with patch("bestock_agent.nodes.fetch_price_history.get_financial_provider", return_value=mock_provider):
        result = await fetch_price_history(state)

    errors = result.get("errors", [])
    assert errors[0].error_type == ErrorType.NETWORK_ERROR
    assert errors[0].recoverable is True
