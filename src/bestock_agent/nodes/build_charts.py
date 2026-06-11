"""Node: build_charts.

Phase 3: price-trend line chart + daily-change bar chart.
Phase 5: adds index-comparison and sentiment charts when advanced_analysis_enabled.
"""

from datetime import datetime, timezone

from bestock_agent.schemas import AgentError, ErrorType
from bestock_agent.services.charts import (
    generate_change_chart,
    generate_index_comparison_chart,
    generate_price_chart,
    generate_sentiment_chart,
)
from bestock_agent.state import BestockState

_OUTPUT_DIR = "outputs/charts"


async def build_charts(state: BestockState) -> dict:
    """Generate Plotly charts and append ChartArtifact entries to state."""
    analysis = state["trend_analysis"]
    if analysis is None:
        return {}

    artifacts = []
    errors = []

    # ── Phase 3: core charts ──────────────────────────────────────────────────
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

    # ── Phase 5: advanced charts (only when advanced_analysis_enabled) ────────
    if state.get("advanced_analysis_enabled"):
        index_comparison = state.get("index_comparison")
        if index_comparison is not None:
            try:
                idx_chart = generate_index_comparison_chart(
                    analysis.symbol, analysis.bars, index_comparison, _OUTPUT_DIR
                )
                artifacts.append(idx_chart)
            except Exception as exc:
                errors.append(
                    AgentError(
                        error_type=ErrorType.TOOL_ERROR,
                        message=f"Index comparison chart failed: {exc}",
                        node="build_charts",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        recoverable=False,
                    )
                )

        sentiment = state.get("sentiment_result")
        if sentiment is not None:
            try:
                sent_chart = generate_sentiment_chart(sentiment, _OUTPUT_DIR)
                artifacts.append(sent_chart)
            except Exception as exc:
                errors.append(
                    AgentError(
                        error_type=ErrorType.TOOL_ERROR,
                        message=f"Sentiment chart failed: {exc}",
                        node="build_charts",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        recoverable=False,
                    )
                )

    result: dict = {"chart_artifacts": artifacts}
    if errors:
        result["errors"] = errors
    return result
