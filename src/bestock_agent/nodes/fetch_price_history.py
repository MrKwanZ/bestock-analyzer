"""Node: fetch_price_history.

Retrieves OHLCV bars for the selected stock over the configured lookback window.
Implemented in Phase 3.
"""

from bestock_agent.state import BestockState


async def fetch_price_history(state: BestockState) -> dict:
    """Fetch price history for the top gainer and write bars to state."""
    raise NotImplementedError("Implemented in Phase 3")
