from __future__ import annotations

import os
import base64
import hashlib
import hmac
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class LineSendResult:
    ok: bool
    status_code: int | None
    error: str | None = None


class LineMessagingClient:
    endpoint = "https://api.line.me/v2/bot/message/push"

    def __init__(self, channel_access_token: str | None = None, user_id: str | None = None, timeout: int = 20) -> None:
        self.channel_access_token = channel_access_token or os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
        self.user_id = user_id or os.getenv("LINE_USER_ID")
        self.timeout = timeout

    def send_text(self, text: str) -> LineSendResult:
        if not self.channel_access_token or not self.user_id:
            return LineSendResult(ok=False, status_code=None, error="LINE credentials missing")
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.channel_access_token}", "Content-Type": "application/json"},
            json={"to": self.user_id, "messages": [{"type": "text", "text": text}]},
            timeout=self.timeout,
        )
        if response.ok:
            return LineSendResult(ok=True, status_code=response.status_code)
        return LineSendResult(ok=False, status_code=response.status_code, error=response.text)


def verify_line_signature(body: bytes, signature: str | None, channel_secret: str | None = None) -> bool:
    secret = channel_secret or os.getenv("LINE_CHANNEL_SECRET")
    if not secret or not signature:
        return False
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)
