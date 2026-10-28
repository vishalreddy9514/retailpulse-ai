"""Small HTTP client for the RetailPulse API, used by the dashboard."""

from __future__ import annotations

import os

import httpx

DEFAULT_URL = os.environ.get("RETAILPULSE_API_URL", "http://localhost:8000")


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class ApiClient:
    def __init__(self, base_url: str = DEFAULT_URL, http: httpx.Client | None = None):
        token = os.environ.get("RETAILPULSE_API_TOKEN")
        headers = {"X-API-Key": token} if token else {}
        self.http = http or httpx.Client(base_url=base_url, timeout=120.0, headers=headers)

    def _check(self, r: httpx.Response) -> dict:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(r.status_code, str(detail))
        return r.json()

    def get(self, path: str, **params) -> dict:
        return self._check(self.http.get(path, params=params))

    def post(self, path: str, body: dict) -> dict:
        return self._check(self.http.post(path, json=body))
