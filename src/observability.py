"""Structured logging and Prometheus metrics shared by the API and the stream consumer.

Logging: ``configure_logging()`` emits one JSON object per line (``RETAILPULSE_LOG_FORMAT=json``,
the default in containers) or plain text for local use. Extra fields passed with
``logger.info(..., extra={"fields": {...}})`` are merged into the JSON object.

Metrics: every metric RetailPulse exports is defined here, so names, labels and help
text live in one place. HTTP metrics are labelled by *route template* (``/products/{stock_code}``),
never by raw path, so one series is not created per stock code.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime

from prometheus_client import Counter, Gauge, Histogram


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    level = (level or os.environ.get("RETAILPULSE_LOG_LEVEL", "INFO")).upper()
    fmt = fmt or os.environ.get("RETAILPULSE_LOG_FORMAT", "text")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


# -- API ---------------------------------------------------------------------------------------
HTTP_REQUESTS = Counter(
    "retailpulse_http_requests_total",
    "HTTP requests by route template and status",
    ["method", "route", "status"],
)
HTTP_LATENCY = Histogram(
    "retailpulse_http_request_duration_seconds",
    "HTTP request duration",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
COPILOT_ANSWERS = Counter(
    "retailpulse_copilot_answers_total",
    "Copilot answers by mode, outcome and number-verification result",
    ["mode", "outcome", "verified"],
)
APPROVAL_DECISIONS = Counter(
    "retailpulse_approval_decisions_total", "Reorder suggestion decisions recorded", ["decision"]
)
PENDING_APPROVALS = Gauge(
    "retailpulse_pending_approvals", "Reorder suggestions awaiting a decision"
)
FORECAST_ORIGIN = Gauge(
    "retailpulse_forecast_origin_timestamp_seconds",
    "End-of-day forecast origin of the loaded artifacts (historical)",
)
ARTIFACTS_LOADED = Gauge(
    "retailpulse_artifacts_loaded_timestamp_seconds", "When the API loaded its artifacts"
)
RISK_PRODUCTS = Gauge(
    "retailpulse_stockout_risk_products",
    "Products by stockout risk level (synthetic stock)",
    ["level"],
)
MODEL_TEST_WAPE = Gauge(
    "retailpulse_model_test_wape", "Held-out test WAPE of the forecast model", ["window"]
)
MONITOR_WAPE = Gauge(
    "retailpulse_monitor_forecast_wape",
    "WAPE of a past forecast scored against later actuals",
    ["window"],
)
MONITOR_STATUS = Gauge(
    "retailpulse_monitor_degraded", "1 if the latest monitoring check flagged a problem", ["check"]
)
INPUT_DRIFT_PSI = Gauge(
    "retailpulse_input_drift_psi", "Population stability index of an input feature", ["feature"]
)

# -- stream consumer ---------------------------------------------------------------------------
STREAM_MESSAGES = Counter(
    "retailpulse_stream_messages_total", "Consumed messages by result", ["result"]
)
STREAM_E2E_LATENCY = Histogram(
    "retailpulse_stream_end_to_end_latency_seconds",
    "From producer ingestion_time to processed (applied events only)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 300),
)
STREAM_BATCH_SECONDS = Histogram(
    "retailpulse_stream_batch_seconds", "Time to process one poll batch"
)
STREAM_EVENT_WATERMARK = Gauge(
    "retailpulse_stream_event_time_watermark_seconds",
    "Latest event time processed (historical replay time)",
)
STREAM_LAST_PROCESSED = Gauge(
    "retailpulse_stream_last_processed_timestamp_seconds",
    "Wall-clock time of the last processed batch",
)
