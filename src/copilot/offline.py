"""Deterministic copilot without a language model (no API key needed).

Routes a question to the same validated tools by keyword intent and renders a templated
answer from their results. Its answers are less flexible than Claude's but come from the
same services, so the numbers are identical. Used for tests, CI, demos without a key, and
as the API's fallback. Every answer is marked ``mode = "offline (no LLM)"``.
"""

from __future__ import annotations

import re

from src.copilot.graph import CopilotAnswer, evidence_from
from src.copilot.services import RetailPulseServices
from src.copilot.tools import ToolExecutor
from src.copilot.verify import verify_numbers

CODE_RE = re.compile(r"\b(\d{5}[A-Za-z]{0,3}|DCGS[0-9A-Za-z]+|SP\d{4})\b")

INTENTS = [
    (
        "reorder",
        ("reorder", "order quantity", "how much should we order", "how many should we order"),
    ),
    (
        "stockout",
        ("stockout", "stock-out", "stock out", "run out", "running out", "out of stock", "at risk"),
    ),
    ("unusual", ("unusual", "anomal", "spike", "abnormal", "strange", "unexpected")),
    ("change", ("why", "changed", "change", "drop", "increase", "decrease", "compared")),
    ("forecast", ("forecast", "tomorrow", "next day", "highest demand", "top demand", "predicted")),
    (
        "docs",
        (
            "how does",
            "how do",
            "how are",
            "how is",
            "handled",
            "calculated",
            "policy",
            "method",
            "assumption",
            "explain how",
            "what is wape",
        ),
    ),
]


def detect_intent(question: str) -> str:
    q = question.lower()
    for name, keys in INTENTS:
        if any(k in q for k in keys):
            return name
    return "product"


class OfflineCopilot:
    def __init__(self, services: RetailPulseServices):
        self.executor = ToolExecutor(services)

    def ask(self, question: str) -> CopilotAnswer:
        intent = detect_intent(question)
        code_match = CODE_RE.search(question)
        code = code_match.group(1).upper() if code_match else None
        log: list[dict] = []

        def call(tool: str, **args) -> dict | None:
            content, is_error, result = self.executor.run(tool, args)
            log.append({"tool": tool, "input": args, "is_error": is_error, "result": result,
                        "error": content if is_error else None})  # fmt: skip
            return result

        if intent in ("change", "reorder") and code is None:
            found = call("product_info", query=question, limit=1)
            code = found["results"][0]["stock_code"] if found and found["results"] else None

        if intent == "forecast":
            text = self._forecast(call("top_forecast_demand", horizon_trading_days=1, limit=5))
        elif intent == "unusual":
            text = self._unusual(call("unusual_sales", recent_days=5, limit=5))
        elif intent == "stockout":
            text = self._stockout(call("stockout_risk", min_risk_level="high", limit=5))
        elif intent == "change" and code:
            r = call("explain_demand_change", stock_code=code, recent_days=7, baseline_days=28)
            text = self._change(r) if r else log[-1]["error"]
        elif intent == "reorder" and code:
            r = call("reorder_suggestion", stock_code=code)
            text = self._reorder(r) if r else log[-1]["error"]
        elif intent == "docs":
            text = self._docs(call("search_docs", query=question, limit=2))
        else:
            text = self._products(call("product_info", query=question, limit=5))

        results = [t["result"] for t in log if t["result"]]
        check = verify_numbers(text, results, question)
        return CopilotAnswer(
            question=question,
            answer=text,
            outcome="answered",
            verified=check.passed,
            unverified_numbers=check.unverified,
            tool_calls=[{k: t[k] for k in ("tool", "input", "is_error")} for t in log],
            evidence=evidence_from(log),
            model=None,
            mode="offline (no LLM)",
        )

    # -- templates (every number is copied from a tool result) -----------------------------------
    @staticmethod
    def _limits(r: dict) -> str:
        return "Limitations: " + " ".join(r["limitations"][:2])

    def _forecast(self, r: dict) -> str:
        rows = r["results"]
        lines = [f"Highest forecast demand for {rows[0]['target_date']} (model predictions made at "
                 f"{r['forecast_origin']}, model {r['model_version']}):"]  # fmt: skip
        lines += [
            f"- {x['stock_code']} {x['product_description']}: {x['forecast_units']} units forecast "
            f"(observed 28-day mean {x['recent_mean_28_units_per_day']} units/day; {x['method']})"
            for x in rows
        ]
        return "\n".join([*lines, "", self._limits(r)])

    def _unusual(self, r: dict) -> str:
        lines = [
            f"Most unusual observed sales up to {r['data_as_of']} (vs each product's own baseline):"
        ]
        lines += [
            f"- {x['stock_code']} {x['product_description']}: {x['recent_units']} units, "
            f"{x['direction']} (expected {x['expected_units_from_baseline']}; z-score {x['z_score']})"
            for x in r["results"]
        ]
        return "\n".join([*lines, "", self._limits(r)])

    def _stockout(self, r: dict) -> str:
        lines = [f"Products at high stockout risk over the next 7 trading days after {r['forecast_origin']} "
                 "(SYNTHETIC inventory, model forecasts):"]  # fmt: skip
        lines += [
            f"- {x['stock_code']} {x['product_description']}: synthetic stock {x['current_stock']} units, "
            f"forecast {x['forecast_units_7d']} units, stockout probability {x['stockout_probability_7d']}, "
            f"suggested order {x['suggested_order_qty']} units ({x['suggestion_status']})"
            for x in r["results"]
        ]
        return "\n".join([*lines, "", self._limits(r)])

    def _change(self, r: dict) -> str:
        x = r["results"]
        rec, base, ch = x["recent"], x["baseline"], x["change"]
        lines = [
            f"{x['stock_code']} {x['product_description']}: observed {rec['units_per_day']} units/day "
            f"({rec['from']} to {rec['to']}) vs {base['units_per_day']} units/day "
            f"({base['from']} to {base['to']}), a change of {ch['units_per_day_pct']}%.",
            f"- orders/day {rec['orders_per_day']} vs {base['orders_per_day']} ({ch['orders_per_day_pct']}%)",
            f"- units/order {rec['units_per_order']} vs {base['units_per_order']} ({ch['units_per_order_pct']}%)",
            f"- average price GBP {rec['avg_unit_price_gbp']} vs {base['avg_unit_price_gbp']} "
            f"({ch['avg_unit_price_pct']}%)",
        ]
        lines += [f"- {d}" for d in x["drivers"]]
        f = x["forecast"]
        lines.append(
            f"Forecast for the next 7 trading days: {f['next_7_trading_days_units']} units ({f['method']})."
        )
        return "\n".join([*lines, "", self._limits(r)])

    def _reorder(self, r: dict) -> str:
        x = r["results"]
        c = x["calculation"]
        if not x["reorder_suggested"]:
            head = f"No reorder suggested for {x['stock_code']} {x['product_description']}: {x['reorder_reason']}."
        else:
            head = (
                f"Suggested reorder for {x['stock_code']} {x['product_description']}: "
                f"{x['suggested_order_qty']} units, status {x['approval_status']} (needs human approval)."
            )
        lines = [
            head,
            f"- synthetic stock on hand {c['stock_on_hand_units']} units, safety stock {c['safety_stock_units']}, "
            f"reorder point {c['reorder_point_units']}, lead time {c['lead_time_days']} trading days",
            f"- forecast demand over lead time + review period: {c['forecast_demand_over_cover_units']} units",
            f"- reason: {x['reorder_reason']}",
        ]
        return "\n".join([*lines, "", self._limits(r)])

    def _docs(self, r: dict) -> str:
        if not r["results"]:
            return "No matching documentation found."
        top = r["results"][0]
        return f"From {top['source']} ({top['section']}):\n{top['text']}"

    def _products(self, r: dict) -> str:
        if not r["results"]:
            return "No matching products found. Try a stock code or different description keywords."
        lines = ["Matching products (observed lifetime sales):"]
        lines += [
            f"- {x['stock_code']} {x['product_description']}: {x['total_units_sold']} units, "
            f"GBP {x['total_gross_revenue_gbp']}, sold {x['first_sale_date']} to {x['last_sale_date']}"
            for x in r["results"]
        ]
        return "\n".join(lines)
