"""Dashboard tests: the real Streamlit app against the real API, in-process.

Streamlit's AppTest runs the script headless. Its API client is backed by FastAPI's
TestClient over synthetic artifacts, so no servers or network are needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from src.api.main import create_app
from src.dashboard.api_client import ApiClient
from tests.synthetic import build_synthetic_artifacts

APP = str(Path(__file__).resolve().parents[2] / "src" / "dashboard" / "app.py")


@pytest.fixture
def app_test(tmp_path, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "RETAILPULSE_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    api = create_app(build_synthetic_artifacts(), approvals_path=tmp_path / "approvals.sqlite")
    with TestClient(api) as http:
        at = AppTest.from_file(APP, default_timeout=60)
        at.session_state["api_client"] = ApiClient(http=http)
        at.run()
        yield at


def test_dashboard_renders_without_errors(app_test):
    at = app_test
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "RetailPulse AI"
    banner = at.info[0].value
    assert "Simulated real-time" in banner and "synthetic" in banner.lower()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Forecast origin (end of day)"] == "2011-12-08"
    assert metrics["Copilot modes"] == "offline"
    assert [t.label for t in at.tabs] == [
        "Overview",
        "Forecasts",
        "Unusual sales",
        "Stockout risk & approvals",
        "Copilot",
    ]


def test_approve_from_the_dashboard(app_test):
    at = app_test
    name = next(t for t in at.text_input if t.label.startswith("Your name"))
    name.input("Dashboard Tester")
    next(b for b in at.button if b.label == "Approve").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("approved by Dashboard Tester" in s.value for s in at.success)


def test_copilot_tab_answers(app_test):
    at = app_test
    at.text_area[0].input("Which products have unusual sales patterns?")
    next(b for b in at.button if b.label == "Ask").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Most unusual observed sales" in m.value for m in at.markdown)
    assert any("every number matches a tool result" in c.value for c in at.caption)
