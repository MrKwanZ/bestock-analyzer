"""Node: build_charts.

Phase 3: price-trend line chart + daily-change bar chart.
Phase 5 will extend with index-comparison and sentiment charts.
"""

import os
from datetime import datetime, timezone

from bestock_agent.schemas import AgentError, ErrorType
from bestock_agent.services.charts import generate_change_chart, generate_price_chart
from bestock_agent.state import BestockState

_OUTPUT_DIR = "outputs/charts"


async def build_charts(state: BestockState) -> dict:
    """Generate Plotly charts and append ChartArtifact entries to state."""
    analysis = state["trend_analysis"]
    if analysis is None:
        return {}

    artifacts = []
    errors = []

    try:
        price_chart = generate_price_chart(analysis.symbol, analysis.bars, _OUTPUT_DIR)
        artifacts.append(price_chart)
    except Exception as exc:
        errors.append(
            AgentError(
                error_type=ErrorType.TOOL_ERROR,
                message=f"Price chart generation failed: {exc}",
                node="build_charts",
                timestamp=datetime.now(timezone.utc).isoformat(),
                recoverable=False,
            )
        )

    try:
        dates = [str(b.date) for b in analysis.bars]
        change_chart = generate_change_chart(
            analysis.symbol, analysis.daily_changes, dates, _OUTPUT_DIR
        )
        artifacts.append(change_chart)
    except Exception as exc:
        errors.append(
            AgentError(
                error_type=ErrorType.TOOL_ERROR,
                message=f"Change chart generation failed: {exc}",
                node="build_charts",
                timestamp=datetime.now(timezone.utc).isoformat(),
                recoverable=False,
            )
        )

    result: dict = {"chart_artifacts": artifacts}
    if errors:
        result["errors"] = errors
    return result
