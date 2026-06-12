"""Node: error_handler.

Decides whether to retry the failing node (with provider fallback if applicable)
or terminate the run with a failure RunSummary.

Routing outcome is determined by ``_route_error_handler`` in graph.py which
checks ``retry_count`` against ``_MAX_RETRIES``.
"""

from datetime import datetime, timezone

from bestock_agent.config import get_settings
from bestock_agent.logging import get_logger
from bestock_agent.schemas import RunSummary
from bestock_agent.services.backoff import needs_same_provider_backoff, wait_for_rate_limit_backoff
from bestock_agent.services.fallback import decide_fallback, is_retryable
from bestock_agent.state import BestockState

log = get_logger("error_handler")

_MAX_RETRIES = 2


def _provider_for_node(state: BestockState, node: str | None) -> str:
    """Return the active provider name associated with a failing node."""
    if node in {"fetch_top_gainer", "fetch_price_history"}:
        return state["active_financial_provider"]
    if node == "fetch_news_and_sentiment":
        return state["active_news_provider"]
    if node == "send_email":
        return "smtp"
    return "unknown"


async def error_handler(state: BestockState) -> dict:
    """Inspect the latest error, attempt provider fallback, and decide retry vs terminate."""
    errors = state.get("errors", [])
    latest = errors[-1] if errors else None
    gainer = state.get("top_gainer")
    retry_count = state.get("retry_count", 0)

    # ── Attempt fallback / retry ───────────────────────────────────────────────
    if is_retryable(state, _MAX_RETRIES):
        if needs_same_provider_backoff(state):
            settings = get_settings()
            await wait_for_rate_limit_backoff(
                settings.rate_limit_backoff_seconds,
                provider=_provider_for_node(state, latest.node if latest else None),
                node=latest.node if latest else "unknown",
            )
            log.info("retry_scheduled", retry=retry_count + 1, node=latest.node if latest else "unknown")
            return {"rate_limit_backoff_used": True}

        fallback_updates = decide_fallback(state)
        new_provider = fallback_updates.get("active_financial_provider")
        if new_provider and new_provider != state.get("active_financial_provider"):
            log.fallback_switch(
                state["active_financial_provider"],
                new_provider,
                reason=latest.message[:80] if latest else "",
            )
        log.info("retry_scheduled", retry=retry_count + 1, node=latest.node if latest else "unknown")
        return {
            "retry_count": retry_count + 1,
            **fallback_updates,
        }

    # ── No retry budget left — produce a terminal failure summary ─────────────
    if latest:
        message = (
            f"Agent failed after {retry_count} retry attempt(s).\n"
            f"  Node    : {latest.node}\n"
            f"  Error   : {latest.error_type.value}\n"
            f"  Detail  : {latest.message}"
        )
    else:
        message = "Agent failed with an unknown error (no error details in state)."

    run_summary = RunSummary(
        success=False,
        stock_symbol=gainer.symbol if gainer else None,
        stock_name=gainer.name if gainer else None,
        change_pct=gainer.change_pct if gainer else None,
        email_sent=False,
        email_recipient=state.get("recipient_email"),
        errors=errors,
        charts_generated=len(state.get("chart_artifacts", [])),
        message=message,
    )
    # Saturate retry_count so _route_error_handler routes to END
    return {"run_summary": run_summary, "retry_count": _MAX_RETRIES}
