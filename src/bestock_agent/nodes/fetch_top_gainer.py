"""Node: fetch_top_gainer.

Queries the active financial provider for the NASDAQ stock with the highest
intraday percentage gain, validates the result, and stores it in state.
"""

from datetime import datetime, timezone

from bestock_agent.logging import get_logger, timer
from bestock_agent.providers import get_financial_provider
from bestock_agent.providers.financial_base import RateLimitError
from bestock_agent.schemas import AgentError, ErrorType
from bestock_agent.services.validation import ValidationError, validate_top_gainer
from bestock_agent.state import BestockState

log = get_logger("fetch_top_gainer")


async def fetch_top_gainer(state: BestockState) -> dict:
    """Fetch the top NASDAQ gainer, validate it, and write it to state."""
    provider_name = state["active_financial_provider"]
    provider = get_financial_provider(provider_name)
    try:
        with timer() as t:
            top_gainer = await provider.get_top_nasdaq_gainer()
        log.provider_call(provider_name, "get_top_nasdaq_gainer", latency_ms=t.elapsed_ms, symbol=top_gainer.symbol)
    except RateLimitError as exc:
        log.node_error("fetch_top_gainer", "RATE_LIMIT", str(exc))
        return {"errors": [AgentError(
            error_type=ErrorType.RATE_LIMIT,
            message=f"[{provider_name}] Rate limit hit: {exc}",
            node="fetch_top_gainer",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=True,
        )]}
    except OSError as exc:
        log.node_error("fetch_top_gainer", "NETWORK_ERROR", str(exc))
        return {"errors": [AgentError(
            error_type=ErrorType.NETWORK_ERROR,
            message=f"[{provider_name}] Network error: {exc}",
            node="fetch_top_gainer",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=True,
        )]}
    except Exception as exc:
        log.node_error("fetch_top_gainer", "TOOL_ERROR", str(exc))
        return {"errors": [AgentError(
            error_type=ErrorType.TOOL_ERROR,
            message=f"[{provider_name}] get_top_nasdaq_gainer failed: {exc}",
            node="fetch_top_gainer",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=True,
        )]}

    try:
        validate_top_gainer(top_gainer)
    except ValidationError as exc:
        log.validation_failure("top_gainer", exc.message)
        return {"errors": [AgentError(
            error_type=ErrorType.VALIDATION_ERROR,
            message=f"Top-gainer data invalid: {exc.message}",
            node="fetch_top_gainer",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=False,
        )]}

    return {"top_gainer": top_gainer}
