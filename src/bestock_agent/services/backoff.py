"""Rate-limit backoff helpers."""

from __future__ import annotations

import asyncio

from bestock_agent.logging import get_logger
from bestock_agent.schemas import AgentError, ErrorType
from bestock_agent.state import BestockState

log = get_logger("backoff")


def latest_pipeline_error(state: BestockState) -> AgentError | None:
    """Return the latest error raised by a graph node, ignoring handler notes."""
    errors = state.get("errors", [])
    return next((e for e in reversed(errors) if e.node != "error_handler"), None)


def needs_same_provider_backoff(state: BestockState) -> bool:
    """True when the latest rate-limit error has not used same-provider backoff."""
    latest = latest_pipeline_error(state)
    return (
        latest is not None
        and latest.error_type == ErrorType.RATE_LIMIT
        and not state.get("rate_limit_backoff_used", False)
    )


async def wait_for_rate_limit_backoff(seconds: int, *, provider: str, node: str) -> None:
    """Pause before retrying the same provider after a rate-limit response."""
    log.backoff_waiting(provider=provider, node=node, seconds=seconds)
    await asyncio.sleep(seconds)
