"""Node: send_email.

Sends the composed report email via SMTP and updates run_summary in state.
Implemented in Phase 3.
"""

from bestock_agent.state import BestockState


async def send_email(state: BestockState) -> dict:
    """Deliver the email payload via SMTP and mark email_sent in run_summary."""
    raise NotImplementedError("Implemented in Phase 3")
