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

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.copilot.services import Artifacts, RetailPulseServices, ServiceError
from src.inventory.approvals import DB_PATH, ApprovalError, ApprovalStore

API_TOKEN_ENV = "RETAILPULSE_API_TOKEN"


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
    approvals_path: Path | str = DB_PATH,
    copilot_factory=None,
) -> FastAPI:
    """Build the app. Tests pass small synthetic ``artifacts`` and a temporary approvals DB."""

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
        yield

    app = FastAPI(
        title="RetailPulse AI API",
        description="Demand forecasts, unusual sales, SYNTHETIC-inventory stockout risk, "
        "human-approved reorder suggestions and a copilot, over a SIMULATED real-time replay "
        "of historical UCI Online Retail II data.",
        version="0.9.0",
        lifespan=lifespan,
    )

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
            return store.decide(suggestion_id, decision, body.actor, body.note)
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
        return bot.ask(body.question).to_dict()

    return app


app = create_app()
