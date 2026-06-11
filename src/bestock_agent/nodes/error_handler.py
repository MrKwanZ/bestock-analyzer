"""Node: error_handler.

Inspects accumulated errors, decides whether to retry with a fallback provider,
and updates state accordingly.  Full retry / fallback logic added in Phase 4.
"""

from bestock_agent.state import BestockState


async def error_handler(state: BestockState) -> dict:
    """Log errors, switch providers if possible, and update retry_count."""
    raise NotImplementedError("Implemented in Phase 4")
