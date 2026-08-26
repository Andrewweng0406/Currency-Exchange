from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_fixed


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderResult:
    name: str
    ok: bool
    rows: int = 0
    error: str | None = None


class HttpProvider:
    def __init__(self, timeout: int = 20, attempts: int = 3, wait_seconds: int = 2) -> None:
        self.timeout = timeout
        self.attempts = attempts
        self.wait_seconds = wait_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "twd-fx-monitor/0.1 (+personal risk monitor)"})

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        @retry(stop=stop_after_attempt(self.attempts), wait=wait_fixed(self.wait_seconds), reraise=True)
        def _get() -> requests.Response:
            response = self.session.get(url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response

        try:
            return _get()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"GET failed for {url}: {exc}") from exc
