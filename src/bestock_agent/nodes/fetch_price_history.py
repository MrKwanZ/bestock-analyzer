"""Node: fetch_price_history.

Retrieves OHLCV bars for the selected stock over the configured lookback window
and validates the result.
"""

from datetime import datetime, timezone

from bestock_agent.providers import get_financial_provider
from bestock_agent.providers.financial_base import RateLimitError
from bestock_agent.schemas import AgentError, ErrorType
from bestock_agent.services.validation import ValidationError, validate_price_bars
from bestock_agent.state import BestockState


async def fetch_price_history(state: BestockState) -> dict:
    """Fetch price history for the top gainer, validate it, and write bars to state."""
    gainer = state["top_gainer"]
    if gainer is None:
        return {}

    provider_name = state["active_financial_provider"]
    provider = get_financial_provider(provider_name)
    lookback = state["lookback_days"]

    try:
        bars = await provider.get_price_history(gainer.symbol, lookback)
    except RateLimitError as exc:
        return {"errors": [AgentError(
            error_type=ErrorType.RATE_LIMIT,
            message=f"[{provider_name}] Rate limit hit fetching history for {gainer.symbol}: {exc}",
            node="fetch_price_history",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=True,
        )]}
    except OSError as exc:
        return {"errors": [AgentError(
            error_type=ErrorType.NETWORK_ERROR,
            message=f"[{provider_name}] Network error fetching history for {gainer.symbol}: {exc}",
            node="fetch_price_history",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=True,
        )]}
    except Exception as exc:
        return {"errors": [AgentError(
            error_type=ErrorType.TOOL_ERROR,
            message=f"[{provider_name}] get_price_history failed for {gainer.symbol}: {exc}",
            node="fetch_price_history",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=True,
        )]}

    try:
        validate_price_bars(bars, gainer.symbol, min_bars=2)
    except ValidationError as exc:
        return {"errors": [AgentError(
            error_type=ErrorType.VALIDATION_ERROR,
            message=f"Price-bar validation failed for {gainer.symbol}: {exc.message}",
            node="fetch_price_history",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=False,
        )]}

    return {"price_history": bars}
