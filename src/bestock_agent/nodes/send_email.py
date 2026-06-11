"""Node: send_email.

Sends the composed report email via SMTP and updates run_summary in state.
"""

from datetime import datetime, timezone

from bestock_agent.config import get_settings
from bestock_agent.schemas import AgentError, ErrorType, RunSummary
from bestock_agent.services.email import send_report_email
from bestock_agent.state import BestockState


async def send_email(state: BestockState) -> dict:
    """Deliver the email payload via SMTP and mark email_sent in run_summary."""
    payload = state["email_payload"]
    gainer = state["top_gainer"]

    if payload is None:
        error = AgentError(
            error_type=ErrorType.VALIDATION_ERROR,
            message="send_email called without email_payload in state",
            node="send_email",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=False,
        )
        return {"errors": [error]}

    settings = get_settings()

    # Skip sending if SMTP credentials are placeholders
    creds_are_placeholders = (
        not settings.smtp_user
        or settings.smtp_user.startswith("<")
        or not settings.smtp_password
        or settings.smtp_password.startswith("<")
    )
    if creds_are_placeholders:
        run_summary = RunSummary(
            success=True,
            stock_symbol=gainer.symbol if gainer else None,
            stock_name=gainer.name if gainer else None,
            change_pct=gainer.change_pct if gainer else None,
            email_sent=False,
            email_recipient=payload.recipient,
            charts_generated=len(state["chart_artifacts"]),
            message="Email skipped — SMTP credentials not configured.",
        )
        return {"run_summary": run_summary}

    try:
        await send_report_email(payload, settings)
    except Exception as exc:
        error = AgentError(
            error_type=ErrorType.TOOL_ERROR,
            message=f"SMTP send failed: {exc}",
            node="send_email",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=True,
        )
        return {"errors": [error]}

    run_summary = RunSummary(
        success=True,
        stock_symbol=gainer.symbol if gainer else None,
        stock_name=gainer.name if gainer else None,
        change_pct=gainer.change_pct if gainer else None,
        email_sent=True,
        email_recipient=payload.recipient,
        charts_generated=len(state["chart_artifacts"]),
        message="Report delivered successfully.",
    )
    return {"run_summary": run_summary}
