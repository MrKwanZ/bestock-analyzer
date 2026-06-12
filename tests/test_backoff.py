"""Tests for rate-limit backoff helpers."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from bestock_agent.schemas import AgentError, ErrorType
from bestock_agent.services.backoff import needs_same_provider_backoff, wait_for_rate_limit_backoff
from bestock_agent.state import initial_state


def _err(node: str, error_type: ErrorType = ErrorType.RATE_LIMIT) -> AgentError:
    return AgentError(
        error_type=error_type,
        message=f"Simulated {error_type.value} in {node}",
        node=node,
        timestamp=datetime.now(timezone.utc).isoformat(),
        recoverable=True,
    )


def test_needs_same_provider_backoff_for_first_rate_limit():
    state = initial_state(recipient_email="test@example.com", target_date="2026-06-11")
    state["errors"] = [_err("fetch_top_gainer")]
    state["rate_limit_backoff_used"] = False

    assert needs_same_provider_backoff(state) is True


def test_no_same_provider_backoff_after_backoff_used():
    state = initial_state(recipient_email="test@example.com", target_date="2026-06-11")
    state["errors"] = [_err("fetch_top_gainer")]
    state["rate_limit_backoff_used"] = True

    assert needs_same_provider_backoff(state) is False


def test_no_same_provider_backoff_for_tool_error():
    state = initial_state(recipient_email="test@example.com", target_date="2026-06-11")
    state["errors"] = [_err("fetch_top_gainer", ErrorType.TOOL_ERROR)]
    state["rate_limit_backoff_used"] = False

    assert needs_same_provider_backoff(state) is False


@pytest.mark.asyncio
async def test_wait_for_rate_limit_backoff_sleeps_for_configured_delay():
    with patch("bestock_agent.services.backoff.asyncio.sleep", new_callable=AsyncMock) as sleep:
        await wait_for_rate_limit_backoff(20, provider="alphavantage", node="fetch_top_gainer")

    sleep.assert_awaited_once_with(20)
