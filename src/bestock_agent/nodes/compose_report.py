"""Node: compose_report.

Runs the report LCEL chain to produce structured HTML and plain-text report
content and writes an EmailPayload to state.
Implemented in Phase 3.
"""

from bestock_agent.state import BestockState


async def compose_report(state: BestockState) -> dict:
    """Compose the analysis report and write EmailPayload to state."""
    raise NotImplementedError("Implemented in Phase 3")
