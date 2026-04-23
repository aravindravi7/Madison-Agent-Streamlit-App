"""Gmail SMTP sender for the generated brief.

Pure function — no Streamlit imports. Callers pass credentials as a
``(user, app_password)`` tuple; the calling layer is responsible for
reading secrets.

Unicode handling: UTF-8 via ``email.header.Header`` and the UTF-8 charset
on the MIMEText part. We do not pre-sanitize characters — that's the
charset's job.
"""

from __future__ import annotations

import smtplib
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

__all__ = ["send_brief_email"]

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 465


def send_brief_email(
    recipient: str,
    html_content: str,
    subject: str,
    credentials: tuple[str, str],
) -> tuple[bool, str]:
    """Send an HTML brief to ``recipient`` via Gmail SMTP over SSL.

    Returns ``(True, message)`` on success and ``(False, error)`` on failure.
    """
    user, password = credentials
    if not user or not password:
        return False, "Gmail credentials missing."

    recipient = (recipient or "").strip()
    if not recipient or "@" not in recipient:
        return False, "Enter a valid email address."

    msg = MIMEMultipart("alternative")
    msg["Subject"] = Header(subject or "Research Brief", "utf-8")
    msg["From"] = user
    msg["To"] = recipient
    msg.attach(MIMEText(html_content or "", "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL(_SMTP_HOST, _SMTP_PORT) as server:
            server.login(user, password)
            server.sendmail(user, recipient, msg.as_string())
    except Exception as e:  # smtplib / ssl errors surface here
        return False, str(e)

    return True, f"Brief sent to {recipient}."
