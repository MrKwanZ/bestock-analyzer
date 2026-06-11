"""Node: compare_with_indices.

Fetches S&P 500 (^GSPC) and Dow Jones (^DJI) performance over the same lookback
window and computes relative performance metrics.
Implemented in Phase 5.
"""

from bestock_agent.state import BestockState


async def compare_with_indices(state: BestockState) -> dict:
    """Fetch index bars and write IndexComparison to state."""
    raise NotImplementedError("Implemented in Phase 5")
