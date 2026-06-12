"""Provider fallback routing and retry logic.

Keeps track of which provider is currently active and decides when to switch.
All state mutation happens through the returned dict so LangGraph's reducer
handles it cleanly.
"""

from __future__ import annotations

from bestock_agent.schemas import AgentError, ErrorType
from bestock_agent.state import BestockState

# Ordered fallback chains — try each in sequence
_FINANCIAL_CHAIN: list[str] = ["alphavantage", "yfinance"]
_NEWS_CHAIN: list[str] = ["brave", "serpapi"]

# Error types that are worth retrying (network / rate-limit / transient tool failures)
_RETRYABLE_ERROR_TYPES: frozenset[ErrorType] = frozenset(
    {ErrorType.NETWORK_ERROR, ErrorType.RATE_LIMIT, ErrorType.TOOL_ERROR}
)

# Nodes that touch the financial provider (switch when these fail)
_FINANCIAL_NODES: frozenset[str] = frozenset(
    {"fetch_top_gainer", "fetch_price_history"}
)
# Nodes that touch the news provider
_NEWS_NODES: frozenset[str] = frozenset({"fetch_news_and_sentiment"})


def _next_in_chain(current: str, chain: list[str]) -> str | None:
    """Return the provider that follows *current* in *chain*, or None."""
    try:
        idx = chain.index(current)
        return chain[idx + 1] if idx + 1 < len(chain) else None
    except ValueError:
        return None


def decide_fallback(state: BestockState) -> dict:
    """Inspect the latest error and decide whether to retry or switch provider.

    Returns a partial state dict with updated provider name(s), retry_count,
    and an informational ``AgentError`` if a provider switch is made.
    Updates are merged into graph state by the caller (error_handler node).
    """
    errors = state.get("errors", [])
    if not errors:
        return {}

    latest: AgentError = errors[-1]
    updates: dict = {}

    if latest.error_type == ErrorType.RATE_LIMIT and not state.get("rate_limit_backoff_used", False):
        return updates

    # Financial provider switch
    if latest.node in _FINANCIAL_NODES and latest.error_type in _RETRYABLE_ERROR_TYPES:
        current = state["active_financial_provider"]
        nxt = _next_in_chain(current, _FINANCIAL_CHAIN)
        if nxt:
            updates["active_financial_provider"] = nxt
            updates["rate_limit_backoff_used"] = False
            # Inject an informational error record so the CLI / logs show the switch
            switch_note = AgentError(
                error_type=ErrorType.TOOL_ERROR,
                message=(
                    f"Financial provider switched: {current} → {nxt} "
                    f"(reason: {latest.message[:80]})"
                ),
                node="error_handler",
                timestamp=latest.timestamp,
                recoverable=True,
                fallback_used=True,
            )
            updates["errors"] = [switch_note]

    # News provider switch (non-fatal — just swap and continue)
    elif latest.node in _NEWS_NODES and latest.error_type in _RETRYABLE_ERROR_TYPES:
        current = state["active_news_provider"]
        nxt = _next_in_chain(current, _NEWS_CHAIN)
        if nxt:
            updates["active_news_provider"] = nxt
            updates["rate_limit_backoff_used"] = False
            switch_note = AgentError(
                error_type=ErrorType.TOOL_ERROR,
                message=(
                    f"News provider switched: {current} → {nxt} "
                    f"(reason: {latest.message[:80]})"
                ),
                node="error_handler",
                timestamp=latest.timestamp,
                recoverable=True,
                fallback_used=True,
            )
            updates["errors"] = [switch_note]

    return updates


def is_retryable(state: BestockState, max_retries: int) -> bool:
    """Return True if the graph should attempt another pass through the failing node.

    Switch-notes injected by error_handler itself (node == "error_handler") are
    informational and must not count as a new retryable error — only errors from
    actual pipeline nodes trigger retries.
    """
    errors = state.get("errors", [])
    if not errors:
        return False
    # Find the latest error that originated from a real pipeline node
    pipeline_error = next(
        (e for e in reversed(errors) if e.node != "error_handler"),
        None,
    )
    if pipeline_error is None:
        return False
    return (
        pipeline_error.recoverable
        and pipeline_error.error_type in _RETRYABLE_ERROR_TYPES
        and state["retry_count"] < max_retries
    )


def failing_node(state: BestockState) -> str | None:
    """Return the node name from the most recent recoverable error, if any."""
    errors = state.get("errors", [])
    for err in reversed(errors):
        if err.recoverable and err.node not in ("error_handler",):
            return err.node
    return None
