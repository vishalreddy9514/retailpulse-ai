"""Validated tools the copilot can call. The only way numbers reach an answer.

Each tool has a JSON schema sent to Claude with ``strict: true`` (Claude's arguments
always match the schema), and a pydantic model that validates the arguments again,
including ranges, before the service runs. Validation and service errors come back as
``is_error`` tool results, so the model can correct itself instead of guessing.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.copilot.services import RetailPulseServices, ServiceError


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TopForecastArgs(_Args):
    horizon_trading_days: int = Field(ge=1, le=7)
    limit: int = Field(ge=1, le=25)


class UnusualSalesArgs(_Args):
    recent_days: int = Field(ge=1, le=20)
    limit: int = Field(ge=1, le=25)


class StockoutRiskArgs(_Args):
    min_risk_level: Literal["low", "medium", "high"]
    limit: int = Field(ge=1, le=25)


class DemandChangeArgs(_Args):
    stock_code: str = Field(min_length=1, max_length=20)
    recent_days: int = Field(ge=1, le=28)
    baseline_days: int = Field(ge=7, le=112)


class ReorderArgs(_Args):
    stock_code: str = Field(min_length=1, max_length=20)


class ProductInfoArgs(_Args):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(ge=1, le=10)


class SearchDocsArgs(_Args):
    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(ge=1, le=6)


def _schema(props: dict) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


INT = {"type": "integer"}
STR = {"type": "string"}

TOOLS: dict[str, tuple[type[_Args], str, dict, str]] = {
    "top_forecast_demand": (
        TopForecastArgs,
        "Products with the highest FORECAST demand (model predictions) for a given number of "
        "trading days after the forecast origin. horizon_trading_days 1 = the next trading day "
        "('tomorrow'); 1-7. limit 1-25.",
        {"horizon_trading_days": INT, "limit": INT},
        "top_forecast_demand",
    ),
    "unusual_sales": (
        UnusualSalesArgs,
        "Products whose OBSERVED sales over the last recent_days trading days (1-20, typically 5) "
        "deviate most from their own 56-trading-day baseline, above or below. limit 1-25.",
        {"recent_days": INT, "limit": INT},
        "unusual_sales",
    ),
    "stockout_risk": (
        StockoutRiskArgs,
        "Products at risk of running out in the next 7 trading days, from forecasts and the "
        "SYNTHETIC inventory (not real stock). min_risk_level filters (high >= 50% probability, "
        "medium >= 20%). Includes suggested order quantities and approval status. limit 1-25.",
        {"min_risk_level": {"type": "string", "enum": ["low", "medium", "high"]}, "limit": INT},
        "stockout_risk",
    ),
    "explain_demand_change": (
        DemandChangeArgs,
        "Why a product's demand changed: OBSERVED recent window vs baseline window (units/day, "
        "orders/day, units per order, customers, price, returns, largest day) plus its forecast. "
        "Typical windows: recent_days 7, baseline_days 28.",
        {"stock_code": STR, "recent_days": INT, "baseline_days": INT},
        "explain_demand_change",
    ),
    "reorder_suggestion": (
        ReorderArgs,
        "Suggested reorder quantity for one product with the full calculation (lead time, review "
        "period, forecast demand over cover, safety stock, stock on hand) under the SYNTHETIC "
        "inventory assumptions, and its human-approval status.",
        {"stock_code": STR},
        "reorder_suggestion",
    ),
    "product_info": (
        ProductInfoArgs,
        "Find products by description keywords or stock code; returns stock codes, descriptions "
        "and OBSERVED lifetime sales. Use this to resolve a product name to a stock code.",
        {"query": STR, "limit": INT},
        "product_info",
    ),
    "search_docs": (
        SearchDocsArgs,
        "Search the project documentation (data cleaning and returns policy, forecasting method "
        "and evaluation, synthetic inventory assumptions, limitations). Returns cited passages.",
        {"query": STR, "limit": INT},
        "search_docs",
    ),
}


def tool_definitions() -> list[dict]:
    """Tool list for the Messages API, in a fixed order (keeps the prompt cache stable)."""
    return [
        {"name": name, "description": desc, "input_schema": _schema(props), "strict": True}
        for name, (_, desc, props, _) in TOOLS.items()
    ]


class ToolExecutor:
    def __init__(self, services: RetailPulseServices):
        self.services = services

    def run(self, name: str, raw_input: dict) -> tuple[str, bool, dict | None]:
        """Return (content for the tool_result, is_error, structured result or None)."""
        if name not in TOOLS:
            return f"Unknown tool '{name}'.", True, None
        model, _, _, method = TOOLS[name]
        try:
            args = model.model_validate(raw_input)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
            )
            return f"Invalid arguments for {name}: {problems}", True, None
        try:
            result = getattr(self.services, method)(**args.model_dump())
        except ServiceError as exc:
            return f"{name} could not answer: {exc}", True, None
        return json.dumps(result, default=str, allow_nan=False), False, result
