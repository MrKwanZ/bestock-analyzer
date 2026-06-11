"""Node: build_charts.

Generates Plotly charts from analysis data and appends ChartArtifact entries
to state.  Basic price-trend chart added in Phase 3; advanced charts in Phase 5.
"""

from bestock_agent.state import BestockState


async def build_charts(state: BestockState) -> dict:
    """Generate Plotly charts and append ChartArtifact entries to state."""
    raise NotImplementedError("Implemented in Phase 3 / Phase 5")
