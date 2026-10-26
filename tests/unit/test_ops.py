"""Observability and monitoring tests: JSON logs, metrics, request ids, model monitor."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import observability as obs
from src.api.main import create_app
from src.monitoring.model_monitor import forecast_performance, psi, run
from src.streaming.kafka_consumer import record_metrics
from src.streaming.stream_processor import BatchResult
from tests.synthetic import build_synthetic_artifacts

BASELINE = {
    "test": {"segments": {"model": {"methods": {"lightgbm": {"wape": 0.9}}}}},
    "test_cumulative_7day": {"model": {"lightgbm": {"wape": 0.6}}},
}


def sample(name: str, labels: dict | None = None) -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


# -- logging and metrics ---------------------------------------------------------------------


def test_json_formatter_merges_fields():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    record.fields = {"request_id": "abc", "status": 200}
    out = json.loads(obs.JsonFormatter().format(record))
    assert out["msg"] == "hello world" and out["request_id"] == "abc" and out["level"] == "INFO"


@pytest.fixture
def api(tmp_path, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "RETAILPULSE_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    with TestClient(create_app(build_synthetic_artifacts(), tmp_path / "a.sqlite")) as client:
        yield client


def test_request_id_is_echoed_or_generated(api):
    assert (
        api.get("/health", headers={"X-Request-ID": "trace-123"}).headers["X-Request-ID"]
        == "trace-123"
    )
    assert len(api.get("/health").headers["X-Request-ID"]) == 16


def test_metrics_use_route_templates_not_raw_paths(api):
    api.get("/products/22197")
    api.get("/products/99999")
    body = api.get("/metrics").text
    assert 'route="/products/{stock_code}"' in body
    assert "/products/22197" not in body
    assert 'status="404"' in body
    assert "retailpulse_pending_approvals" in body
    assert 'retailpulse_model_test_wape{window="7d"}' in body


def test_business_counters(api):
    before = sample(
        "retailpulse_copilot_answers_total",
        {"mode": "offline (no LLM)", "outcome": "answered", "verified": "true"},
    )
    api.post("/copilot/ask", json={"question": "Which products have unusual sales patterns?"})
    after = sample(
        "retailpulse_copilot_answers_total",
        {"mode": "offline (no LLM)", "outcome": "answered", "verified": "true"},
    )
    assert after == before + 1

    sid = api.get("/approvals").json()["suggestions"][0]["suggestion_id"]
    before = sample("retailpulse_approval_decisions_total", {"decision": "rejected"})
    api.post(f"/approvals/{sid}/decision", json={"decision": "reject", "actor": "A. Buyer"})
    assert sample("retailpulse_approval_decisions_total", {"decision": "rejected"}) == before + 1


def test_consumer_metrics_record_batches():
    before = sample("retailpulse_stream_messages_total", {"result": "applied"})
    count_before = sample("retailpulse_stream_end_to_end_latency_seconds_count")
    record_metrics(
        BatchResult(applied=3, duplicates=1, latencies_ms=[100.0, 200.0, 300.0]),
        0.05,
        "2011-12-08 17:00:00",
    )
    assert sample("retailpulse_stream_messages_total", {"result": "applied"}) == before + 3
    assert sample("retailpulse_stream_end_to_end_latency_seconds_count") == count_before + 3
    assert (
        sample("retailpulse_stream_event_time_watermark_seconds")
        == pd.Timestamp("2011-12-08 17:00", tz="UTC").timestamp()
    )


# -- model monitor ----------------------------------------------------------------------------


def test_psi_is_near_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(100, 10, 2000)
    assert psi(ref, rng.normal(100, 10, 2000)) < 0.05
    assert psi(ref, rng.normal(130, 10, 2000)) > 0.25


def monitor_inputs(forecast_scale: float, days: int = 160):
    dates = pd.bdate_range("2011-05-02", periods=days)
    rows = [(c, d, 10 + (i % 3), False) for c in ("A", "B") for i, d in enumerate(dates)]
    daily = pd.DataFrame(rows, columns=["stock_code", "date", "units_sold", "is_partial_day"])
    origin, targets = dates[-8], dates[-7:]
    fc = pd.DataFrame(
        [
            (c, origin, t, h, 11 * forecast_scale, "lightgbm")
            for c in ("A", "B")
            for h, t in enumerate(targets, 1)
        ],
        columns=[
            "stock_code",
            "forecast_origin",
            "target_date",
            "horizon_trading_days",
            "forecast_units",
            "method",
        ],
    )
    return fc, daily


def test_monitor_accuracy_ok_degraded_and_in_sample():
    fc, daily = monitor_inputs(1.0)
    ok = run(fc, daily, BASELINE, trained_through="2011-01-01")
    assert ok["checks"]["forecast_accuracy"]["status"] == "ok"
    assert ok["forecast_performance"]["target_days_with_actuals"] == 7
    assert set(ok["forecast_performance"]["wape"]) == {"daily", "7d"}

    bad_fc, _ = monitor_inputs(3.0)
    assert (
        run(bad_fc, daily, BASELINE, "2011-01-01")["checks"]["forecast_accuracy"]["status"]
        == "degraded"
    )

    in_sample = run(fc, daily, BASELINE, trained_through=str(fc["target_date"].max().date()))
    assert in_sample["checks"]["forecast_accuracy"]["status"] == "in_sample"


def test_monitor_excludes_partial_days_and_needs_complete_window_for_7d():
    fc, daily = monitor_inputs(1.0)
    last = fc["target_date"].max()
    daily.loc[daily["date"] == last, "is_partial_day"] = True
    perf = forecast_performance(fc, daily)
    assert perf["target_days_with_actuals"] == 6
    assert "7d" not in perf["wape"]
    report = run(fc, daily, BASELINE, "2011-01-01")
    assert report["checks"]["freshness"]["status"] == "insufficient_data"
