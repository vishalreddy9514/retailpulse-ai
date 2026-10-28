"""RetailPulse REST API (FastAPI).

A thin HTTP layer over ``src.copilot.services``: the dashboard, the copilot and any other
client get the same numbers with the same provenance (``value_types``, ``forecast_origin``,
units, model and assumptions versions, limitations).

SIMULATED REAL-TIME: the data is historical UCI Online Retail II replayed in timestamp
order, and the inventory is SYNTHETIC. Both are stated in ``/health`` and in every response.

Run::

    uvicorn src.api.main:app --reload            # http://localhost:8000/docs

Write endpoints (approving or rejecting a reorder suggestion) record a named person's
decision and never place an order. If ``RETAILPULSE_API_TOKEN`` is set, they also require
the header ``X-API-Key: <token>``. Without it the API is in open local-development mode,
and ``/health`` reports that.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src import config
from src import observability as obs
from src.copilot.services import Artifacts, RetailPulseServices, ServiceError
from src.inventory.approvals import DB_PATH, ApprovalError, ApprovalStore

API_TOKEN_ENV = "RETAILPULSE_API_TOKEN"
MONITORING_REPORT = config.REPORTS_DIR / "monitoring_report.json"
logger = logging.getLogger("retailpulse.api")


class Decision(BaseModel):
    decision: Literal["approve", "reject"]
    actor: str = Field(min_length=1, max_length=100, description="name of the person deciding")
    note: str | None = Field(default=None, max_length=500)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["auto", "claude", "offline"] = "auto"


def has_anthropic_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def get_services(request: Request) -> RetailPulseServices:
    return request.app.state.services


# Module level so FastAPI can resolve the annotation (this module uses postponed annotations).
Svc = Annotated[RetailPulseServices, Depends(get_services)]


def require_token(x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = os.environ.get(API_TOKEN_ENV)
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


def create_app(
    artifacts: Artifacts | None = None,
    approvals_path: Path | str | None = None,
    copilot_factory=None,
) -> FastAPI:
    """Build the app. Tests pass small synthetic ``artifacts`` and a temporary approvals DB."""

    approvals_path = approvals_path or os.environ.get("RETAILPULSE_APPROVALS_DB", DB_PATH)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        arts = artifacts or Artifacts.load()
        store = ApprovalStore(approvals_path)
        try:
            store.register(arts.risk)  # idempotent: new suggestions start as pending_approval
        finally:
            store.close()
        arts.approvals_path = Path(approvals_path)
        app.state.services = RetailPulseServices(arts)
        app.state.approvals_path = approvals_path
        app.state.copilot_factory = copilot_factory
        _set_static_gauges(arts, app.state.services)
        logger.info("artifacts loaded", extra={"fields": app.state.services.status()})
        yield

    app = FastAPI(
        title="RetailPulse AI API",
        description="Demand forecasts, unusual sales, SYNTHETIC-inventory stockout risk, "
        "human-approved reorder suggestions and a copilot, over a SIMULATED real-time replay "
        "of historical UCI Online Retail II data.",
        version="0.9.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_and_measure(request: Request, call_next):
        """Request id, JSON access log and Prometheus metrics, keyed by route template."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            elapsed = time.perf_counter() - start
            route = getattr(request.scope.get("route"), "path", "unmatched")
            obs.HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
            obs.HTTP_LATENCY.labels(request.method, route).observe(elapsed)
            fields = {
                "request_id": request_id,
                "method": request.method,
                "route": route,
                "path": request.url.path,
                "status": status,
                "duration_ms": round(elapsed * 1000, 1),
            }
            logger.info("request", extra={"fields": fields})
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request) -> Response:
        store = ApprovalStore(request.app.state.approvals_path)
        try:
            obs.PENDING_APPROVALS.set(len(store.list("pending_approval")))
        finally:
            store.close()
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.exception_handler(ServiceError)
    async def service_error(_: Request, exc: ServiceError):
        status = 404 if "unknown" in str(exc) else 422
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    # -- status -----------------------------------------------------------------------------
    @app.get("/health")
    def health(svc: Svc) -> dict:
        return {
            "status": "ok",
            **svc.status(),
            "copilot_modes": ["offline", *(["claude"] if has_anthropic_credentials() else [])],
            "write_auth": "token" if os.environ.get(API_TOKEN_ENV) else "open (local development)",
        }

    # -- sales and forecasts ----------------------------------------------------------------
    @app.get("/sales/daily-totals")
    def daily_totals(svc: Svc, days: Annotated[int, Query(ge=7, le=700)] = 90) -> dict:
        return svc.daily_totals(days)

    @app.get("/sales/unusual")
    def unusual(
        svc: Svc,
        recent_days: Annotated[int, Query(ge=1, le=20)] = 5,
        limit: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> dict:
        return svc.unusual_sales(recent_days, limit)

    @app.get("/forecasts/top")
    def top_forecasts(
        svc: Svc,
        horizon: Annotated[int, Query(ge=1, le=7)] = 1,
        limit: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> dict:
        return svc.top_forecast_demand(horizon, limit)

    @app.get("/products/search")
    def search_products(
        svc: Svc,
        q: Annotated[str, Query(min_length=1, max_length=200)],
        limit: Annotated[int, Query(ge=1, le=20)] = 5,
    ) -> dict:
        return svc.product_info(q, limit)

    @app.get("/products/{stock_code}")
    def product(svc: Svc, stock_code: str, history_days: Annotated[int, Query(ge=7, le=365)] = 56):
        return svc.product_overview(stock_code, history_days)

    @app.get("/products/{stock_code}/demand-change")
    def demand_change(
        svc: Svc,
        stock_code: str,
        recent_days: Annotated[int, Query(ge=1, le=28)] = 7,
        baseline_days: Annotated[int, Query(ge=7, le=112)] = 28,
    ) -> dict:
        return svc.explain_demand_change(stock_code, recent_days, baseline_days)

    # -- inventory and approvals ------------------------------------------------------------
    @app.get("/inventory/risk")
    def risk(
        svc: Svc,
        min_level: Literal["low", "medium", "high"] = "high",
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> dict:
        return svc.stockout_risk(min_level, limit)

    @app.get("/inventory/reorder/{stock_code}")
    def reorder(svc: Svc, stock_code: str) -> dict:
        return svc.reorder_suggestion(stock_code)

    @app.get("/approvals")
    def list_approvals(
        request: Request,
        status: Literal["pending_approval", "approved", "rejected", "all"] = "pending_approval",
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> dict:
        store = ApprovalStore(request.app.state.approvals_path)
        try:
            rows = store.list(None if status == "all" else status)
        finally:
            store.close()
        return {
            "status": status,
            "count": int(len(rows)),
            "returned": int(min(limit, len(rows))),
            "suggestions": rows.head(limit).to_dict(orient="records"),
            "note": "Approval accepts a suggestion for manual ordering; nothing is ordered "
            "automatically. Quantities rest on SYNTHETIC inventory.",
        }

    @app.get("/approvals/{suggestion_id}")
    def get_approval(request: Request, suggestion_id: str) -> dict:
        store = ApprovalStore(request.app.state.approvals_path)
        try:
            return store.get(suggestion_id)
        except ApprovalError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        finally:
            store.close()

    @app.post("/approvals/{suggestion_id}/decision", dependencies=[Depends(require_token)])
    def decide(request: Request, suggestion_id: str, body: Decision) -> dict:
        store = ApprovalStore(request.app.state.approvals_path)
        decision = "approved" if body.decision == "approve" else "rejected"
        try:
            result = store.decide(suggestion_id, decision, body.actor, body.note)
            obs.APPROVAL_DECISIONS.labels(decision).inc()
            logger.info(
                "approval decision",
                extra={
                    "fields": {
                        "suggestion_id": suggestion_id,
                        "decision": decision,
                        "actor": body.actor,
                    }
                },
            )
            return result
        except ApprovalError as exc:
            msg = str(exc)
            code = 404 if "unknown" in msg else 409 if "already" in msg else 422
            raise HTTPException(status_code=code, detail=msg) from exc
        finally:
            store.close()

    # -- copilot ----------------------------------------------------------------------------
    @app.post("/copilot/ask")
    def ask(request: Request, body: Question, svc: Svc) -> dict:
        mode = body.mode
        if mode == "auto":
            mode = "claude" if has_anthropic_credentials() else "offline"
        if (
            mode == "claude"
            and not has_anthropic_credentials()
            and not request.app.state.copilot_factory
        ):
            raise HTTPException(
                status_code=503, detail="Claude mode needs ANTHROPIC_API_KEY; use mode=offline"
            )
        if request.app.state.copilot_factory:
            bot = request.app.state.copilot_factory(svc, mode)
        elif mode == "claude":
            from src.copilot.graph import Copilot

            bot = Copilot(svc)
        else:
            from src.copilot.offline import OfflineCopilot

            bot = OfflineCopilot(svc)
        ans = bot.ask(body.question)
        obs.COPILOT_ANSWERS.labels(ans.mode, ans.outcome, str(ans.verified).lower()).inc()
        logger.info(
            "copilot answer",
            extra={"fields": {"mode": ans.mode, "outcome": ans.outcome, "verified": ans.verified,
                              "tools": [t["tool"] for t in ans.tool_calls],
                              "unverified": ans.unverified_numbers}},
        )  # fmt: skip
        return ans.to_dict()

    return app


def _set_static_gauges(arts: Artifacts, svc: RetailPulseServices) -> None:
    obs.FORECAST_ORIGIN.set(svc.origin.timestamp())
    obs.ARTIFACTS_LOADED.set(time.time())
    for level, n in arts.risk["risk_level"].value_counts().items():
        obs.RISK_PRODUCTS.labels(level).set(int(n))
    ev = arts.forecast_eval
    obs.MODEL_TEST_WAPE.labels("daily").set(
        ev["test"]["segments"]["model"]["methods"]["lightgbm"]["wape"]
    )
    obs.MODEL_TEST_WAPE.labels("7d").set(ev["test_cumulative_7day"]["model"]["lightgbm"]["wape"])
    if MONITORING_REPORT.exists():
        report = json.loads(MONITORING_REPORT.read_text())
        for window, value in report["forecast_performance"]["wape"].items():
            obs.MONITOR_WAPE.labels(window).set(value)
        for check, result in report["checks"].items():
            obs.MONITOR_STATUS.labels(check).set(0 if result["status"] == "ok" else 1)
        for feature, psi in report["input_drift"]["psi"].items():
            obs.INPUT_DRIFT_PSI.labels(feature).set(psi)


obs.configure_logging()
app = create_app()
