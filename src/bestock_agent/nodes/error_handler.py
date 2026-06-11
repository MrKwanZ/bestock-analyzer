"""Node: error_handler.

Decides whether to retry the failing node (with provider fallback if applicable)
or terminate the run with a failure RunSummary.

Routing outcome is determined by ``_route_error_handler`` in graph.py which
checks ``retry_count`` against ``_MAX_RETRIES``.
"""

from datetime import datetime, timezone

from bestock_agent.schemas import RunSummary
from bestock_agent.services.fallback import decide_fallback, is_retryable
from bestock_agent.state import BestockState

_MAX_RETRIES = 2


async def error_handler(state: BestockState) -> dict:
    """Inspect the latest error, attempt provider fallback, and decide retry vs terminate."""
    errors = state.get("errors", [])
    latest = errors[-1] if errors else None
    gainer = state.get("top_gainer")
    retry_count = state.get("retry_count", 0)

    # ── Attempt fallback / retry ───────────────────────────────────────────────
    if is_retryable(state, _MAX_RETRIES):
        fallback_updates = decide_fallback(state)
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
