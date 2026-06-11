"""Node: fetch_top_gainer.

Queries the active financial provider for the NASDAQ stock with the highest
intraday percentage gain and stores it in state.
Implemented in Phase 3.
"""

from bestock_agent.state import BestockState


async def fetch_top_gainer(state: BestockState) -> dict:
    """Fetch the top NASDAQ gainer and write it to state."""
    raise NotImplementedError("Implemented in Phase 3")
