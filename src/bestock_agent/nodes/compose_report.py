"""Node: compose_report.

Runs the report chain to produce an EmailPayload stored in state.
Subject is always "Analysis Report on {SYMBOL}".
"""

from datetime import datetime, timezone

from bestock_agent.chains.report_chain import compose_report as run_report_chain
from bestock_agent.config import get_settings
from bestock_agent.schemas import AgentError, EmailPayload, ErrorType
from bestock_agent.state import BestockState


async def compose_report(state: BestockState) -> dict:
    """Compose the analysis report and write EmailPayload to state."""
    gainer = state["top_gainer"]
    analysis = state["trend_analysis"]

    if gainer is None or analysis is None:
        error = AgentError(
            error_type=ErrorType.VALIDATION_ERROR,
            message="compose_report called without top_gainer or trend_analysis in state",
            node="compose_report",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=False,
        )
        return {"errors": [error]}

    settings = get_settings()
    analysis_date = state["target_date"]
    chart_paths = [a.path for a in state["chart_artifacts"]]

    try:
        report = run_report_chain(
            gainer=gainer,
            analysis=analysis,
            analysis_date=analysis_date,
            openai_model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
        )
    except Exception as exc:
        error = AgentError(
            error_type=ErrorType.TOOL_ERROR,
            message=f"Report chain failed: {exc}",
            node="compose_report",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=False,
        )
        return {"errors": [error]}

    subject = f"Analysis Report on {gainer.symbol}"
    payload = EmailPayload(
        recipient=state["recipient_email"],
        subject=subject,
        body_html=report.html_body,
        body_text=report.text_body,
        chart_paths=chart_paths,
    )
    return {"email_payload": payload, "report_text": report.text_body}
