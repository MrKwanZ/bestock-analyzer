"""Email composition and SMTP delivery via aiosmtplib."""

from __future__ import annotations

import os
import ssl
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib
import certifi

from bestock_agent.schemas import EmailPayload


def _ssl_context() -> ssl.SSLContext:
    """Return an SSL context that uses certifi's CA bundle (fixes macOS cert issues)."""
    ctx = ssl.create_default_context(cafile=certifi.where())
    return ctx


async def send_report_email(payload: EmailPayload, settings) -> None:  # type: ignore[type-arg]
    """Send *payload* via SMTP using the credentials in *settings*.

    Chart PNGs listed in ``payload.chart_paths`` are attached as inline images.
    """
    # Build MIME structure: mixed > alternative (text + html) + image attachments
    root = MIMEMultipart("mixed")
    root["Subject"] = payload.subject
    root["From"] = settings.smtp_user
    root["To"] = payload.recipient

    alternative = MIMEMultipart("alternative")
    alternative.attach(MIMEText(payload.body_text, "plain", "utf-8"))
    alternative.attach(MIMEText(payload.body_html, "html", "utf-8"))
    root.attach(alternative)

    for chart_path in payload.chart_paths:
        if not os.path.isfile(chart_path):
            continue
        with open(chart_path, "rb") as f:
            img_data = f.read()
        img = MIMEImage(img_data, name=os.path.basename(chart_path))
        img.add_header(
            "Content-Disposition",
            "attachment",
            filename=os.path.basename(chart_path),
        )
        root.attach(img)

    ctx = _ssl_context()
    _common = dict(
        hostname=settings.smtp_host,
        username=settings.smtp_user,
        password=settings.smtp_password,
        tls_context=ctx,
        timeout=20,
    )
    try:
        # Port 587 — STARTTLS
        await aiosmtplib.send(root, port=settings.smtp_port, start_tls=True, **_common)
    except (aiosmtplib.SMTPConnectError, TimeoutError, OSError, aiosmtplib.SMTPException):
        if settings.smtp_port == 587:
            # Fallback: port 465 — implicit TLS (SSL)
            await aiosmtplib.send(root, port=465, use_tls=True, **_common)
        else:
            raise
