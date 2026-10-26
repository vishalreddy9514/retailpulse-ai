"""REST API tests on small synthetic artifacts (no real data, no network, no API key)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.copilot.offline import OfflineCopilot
from tests.synthetic import build_synthetic_artifacts


@pytest.fixture
def api(tmp_path, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "RETAILPULSE_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    app = create_app(build_synthetic_artifacts(), approvals_path=tmp_path / "approvals.sqlite")
    with TestClient(app) as client:
        yield client


def pending_id(api) -> str:
    risk = api.get("/inventory/risk", params={"min_level": "low", "limit": 10}).json()
    return next(
        r["suggestion_id"] for r in risk["results"] if r["suggestion_status"] == "pending_approval"
    )


def test_health_states_data_origin_and_modes(api):
    h = api.get("/health").json()
    assert h["status"] == "ok" and h["forecast_origin"] == "2011-12-08"
    assert "simulated real-time" in h["data_note"] and "SYNTHETIC" in h["inventory_note"]
    assert h["copilot_modes"] == ["offline"]
    assert h["write_auth"].startswith("open")


def test_read_endpoints_return_provenance(api):
    for path, params in [
        ("/forecasts/top", {"horizon": 1, "limit": 2}),
        ("/sales/unusual", {}),
        ("/sales/daily-totals", {"days": 30}),
        ("/inventory/risk", {"min_level": "low"}),
        ("/products/22197", {}),
        ("/products/22197/demand-change", {}),
        ("/inventory/reorder/22410", {}),
        ("/products/search", {"q": "popcorn"}),
    ]:
        r = api.get(path, params=params)
        assert r.status_code == 200, (path, r.text)
        body = r.json()
        assert {"results", "value_types", "limitations"} <= set(body), path


def test_errors_map_to_http_status(api):
    assert api.get("/products/99999").status_code == 404
    assert api.get("/products/not-a-code").status_code == 422
    assert api.get("/forecasts/top", params={"horizon": 9}).status_code == 422
    assert api.get("/inventory/risk", params={"min_level": "extreme"}).status_code == 422


def test_approval_workflow(api):
    pending = api.get("/approvals").json()
    assert pending["count"] >= 1 and pending["returned"] == len(pending["suggestions"])
    sid = pending_id(api)
    ok = api.post(
        f"/approvals/{sid}/decision",
        json={"decision": "approve", "actor": "A. Buyer", "note": "ok"},
    )
    assert ok.status_code == 200 and ok.json()["status"] == "approved"
    assert [h["action"] for h in ok.json()["history"]] == ["registered", "approved"]
    again = api.post(f"/approvals/{sid}/decision", json={"decision": "reject", "actor": "B"})
    assert again.status_code == 409
    assert (
        api.post("/approvals/nope/decision", json={"decision": "approve", "actor": "A"}).status_code
        == 404
    )
    assert (
        api.post(
            f"/approvals/{sid}/decision", json={"decision": "approve", "actor": ""}
        ).status_code
        == 422
    )
    # The live status flows into the risk view and the reorder service.
    risk = api.get("/inventory/risk", params={"min_level": "low", "limit": 10}).json()
    assert (
        next(r for r in risk["results"] if r["suggestion_id"] == sid)["suggestion_status"]
        == "approved"
    )


def test_blank_actor_is_refused(api):
    sid = pending_id(api)
    r = api.post(f"/approvals/{sid}/decision", json={"decision": "approve", "actor": "   "})
    assert r.status_code == 422 and "named person" in r.json()["detail"]


def test_write_token_is_enforced_when_configured(api, monkeypatch):
    monkeypatch.setenv("RETAILPULSE_API_TOKEN", "s3cret")
    sid = pending_id(api)
    body = {"decision": "approve", "actor": "A. Buyer"}
    assert api.post(f"/approvals/{sid}/decision", json=body).status_code == 401
    assert (
        api.post(
            f"/approvals/{sid}/decision", json=body, headers={"X-API-Key": "wrong"}
        ).status_code
        == 401
    )
    assert (
        api.post(
            f"/approvals/{sid}/decision", json=body, headers={"X-API-Key": "s3cret"}
        ).status_code
        == 200
    )
    assert api.get("/health").json()["write_auth"] == "token"


def test_copilot_offline_and_claude_without_key(api):
    r = api.post("/copilot/ask", json={"question": "What reorder quantity is suggested for 22410?"})
    assert r.status_code == 200
    ans = r.json()
    assert ans["mode"] == "offline (no LLM)" and ans["verified"]
    assert "reorder_suggestion" in [t["tool"] for t in ans["tool_calls"]]
    r = api.post("/copilot/ask", json={"question": "hello there", "mode": "claude"})
    assert r.status_code == 503


def test_copilot_factory_is_used(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    seen = []

    def factory(svc, mode):
        seen.append(mode)
        return OfflineCopilot(svc)

    app = create_app(build_synthetic_artifacts(), tmp_path / "a.sqlite", copilot_factory=factory)
    with TestClient(app) as client:
        r = client.post("/copilot/ask", json={"question": "unusual sales?", "mode": "claude"})
    assert r.status_code == 200 and seen == ["claude"]
