"""Node: analyze_trend.

Computes daily change percentages, average change, trend label, and volatility
from the price history stored in state.
"""

from datetime import datetime, timezone

from bestock_agent.schemas import AgentError, ErrorType
from bestock_agent.services.analysis import build_trend_analysis
from bestock_agent.state import BestockState


async def analyze_trend(state: BestockState) -> dict:
    """Run quantitative trend analysis and write TrendAnalysis to state."""
    bars = state["price_history"]
    gainer = state["top_gainer"]

    if not bars or gainer is None:
        error = AgentError(
            error_type=ErrorType.VALIDATION_ERROR,
            message="analyze_trend called with empty price_history or missing top_gainer",
            node="analyze_trend",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=False,
        )
        return {"errors": [error]}

    try:
        analysis = build_trend_analysis(gainer.symbol, bars)
        return {"trend_analysis": analysis}
    except Exception as exc:
        error = AgentError(
            error_type=ErrorType.TOOL_ERROR,
            message=f"build_trend_analysis failed: {exc}",
            node="analyze_trend",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=False,
        )
        return {"errors": [error]}
