"""Copilot tests: services, tools, number verification, LangGraph loop, offline mode.

The Anthropic client is replaced by a scripted stand-in, so the real graph, tools and
verifier run without network access or an API key. Data is small and synthetic.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.copilot.graph import FALLBACK_BETA, MODEL, Copilot
from src.copilot.knowledge import KnowledgeBase, chunk_markdown, stem
from src.copilot.offline import OfflineCopilot, detect_intent
from src.copilot.services import SIMULATION_NOTE, Artifacts, RetailPulseServices
from src.copilot.tools import ToolExecutor, tool_definitions
from src.copilot.verify import verify_numbers
from src.inventory.risk_engine import ErrorModel, assess
from src.inventory.synthetic_inventory import load_assumptions
from src.processing.trading_calendar import expected_trading_days

ORIGIN = pd.Timestamp("2011-12-08")
PRODUCTS = {
    "22197": ("SMALL POPCORN HOLDER", lambda i: 40 + (300 if i >= 53 else 0)),
    "85099B": ("JUMBO BAG RED RETROSPOT", lambda i: 30 + i % 5),
    "22410": ("MONEY BOX HOUSEKEEPING DESIGN", lambda i: 10 if i % 2 else 0),
}


# -- synthetic artifacts ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    days = [pd.Timestamp("2011-09-01")] + expected_trading_days("2011-09-01", 400)
    days = [d for d in days if d <= ORIGIN]
    rows = []
    for code, (_, f) in PRODUCTS.items():
        for i, d in enumerate(days):
            u = f(i)
            rows.append(
                (
                    code,
                    d,
                    u,
                    max(u // 10, 1 if u else 0),
                    1 if u else 0,
                    round(u * 1.5, 2),
                    0,
                    False,
                )
            )
    daily = pd.DataFrame(
        rows,
        columns=["stock_code", "date", "units_sold", "invoices", "customers", "gross_revenue",
                 "units_returned", "is_partial_day"],
    )  # fmt: skip
    summary = pd.DataFrame(
        {
            "stock_code": list(PRODUCTS),
            "product_description": [v[0] for v in PRODUCTS.values()],
            "first_sale_date": days[0],
            "last_sale_date": ORIGIN,
            "total_units": daily.groupby("stock_code")["units_sold"]
            .sum()
            .reindex(list(PRODUCTS))
            .to_numpy(),
            "total_gross_revenue": 1000.5,
            "selling_days": 60,
            "demand_class": "erratic",
        }
    )
    targets = expected_trading_days(ORIGIN, 7)
    fc = pd.DataFrame(
        [
            (code, ORIGIN, t, h, rate, "lightgbm", "test-model", 20.0, 10.0)
            for code, rate in {"22197": 300.5, "85099B": 31.25, "22410": 6.0}.items()
            for h, t in enumerate(targets, 1)
        ],
        columns=["stock_code", "forecast_origin", "target_date", "horizon_trading_days",
                 "forecast_units",
                 "method", "model_version", "recent_mean_28", "units_last_trading_day"],
    )  # fmt: skip
    inv = pd.DataFrame(
        [("22197", 50, 5, 30, 1530), ("85099B", 5000, 3, 10, 104), ("22410", 2, 7, 8, 50)],
        columns=["stock_code", "current_stock", "lead_time_days", "safety_stock", "reorder_point"],
    )
    cum = pd.DataFrame(
        [(f"P{i}", ORIGIN, h, a * h / 7, 70.0 * h / 7)
         for i, a in enumerate(np.linspace(35, 105, 100))
         for h in range(1, 8)],
        columns=["stock_code", "origin_date", "h", "cum_actual", "cum_forecast"],
    )  # fmt: skip
    assumptions = load_assumptions()
    risk = assess(fc, inv, ErrorModel.fit(cum), assumptions)
    forecast_eval = {
        "test": {"segments": {"model": {"methods": {"lightgbm": {"wape": 0.9887}}}}},
        "test_cumulative_7day": {"model": {"lightgbm": {"wape": 0.5926}}},
        "splits": {"test": {"targets": ["2011-10-01", "2011-12-08"]}},
    }
    return Artifacts(
        product_daily=daily,
        product_summary=summary,
        forecasts=fc,
        risk=risk,
        forecast_eval=forecast_eval,
        risk_eval={"mean_predicted_probability": 0.139, "observed_stockout_rate": 0.165},
        assumptions=assumptions,
        knowledge=KnowledgeBase.build(summary),
    )


@pytest.fixture(scope="module")
def services(artifacts):
    return RetailPulseServices(artifacts)


# -- services and tools ------------------------------------------------------------------------


def test_every_tool_result_carries_provenance_and_valid_json(services):
    ex = ToolExecutor(services)
    calls = {
        "top_forecast_demand": {"horizon_trading_days": 1, "limit": 2},
        "unusual_sales": {"recent_days": 5, "limit": 3},
        "stockout_risk": {"min_risk_level": "low", "limit": 3},
        "explain_demand_change": {"stock_code": "22197", "recent_days": 7, "baseline_days": 28},
        "reorder_suggestion": {"stock_code": "22410"},
        "product_info": {"query": "popcorn", "limit": 2},
        "search_docs": {"query": "returns policy", "limit": 2},
    }
    for name, args in calls.items():
        content, is_error, result = ex.run(name, args)
        assert not is_error, content
        json.loads(content)  # strict JSON (no NaN/Infinity)
        assert {"forecast_origin", "results", "value_types", "limitations"} <= set(result)
        assert SIMULATION_NOTE in result["limitations"]


def test_tool_inputs_are_validated(services):
    ex = ToolExecutor(services)
    msg, err, _ = ex.run("top_forecast_demand", {"horizon_trading_days": 9, "limit": 2})
    assert err and "horizon_trading_days" in msg
    msg, err, _ = ex.run("top_forecast_demand", {"horizon_trading_days": 1, "limit": 2, "x": 1})
    assert err and "Extra inputs" in msg
    msg, err, _ = ex.run("reorder_suggestion", {"stock_code": "NOT A CODE"})
    assert err and "not a valid product stock code" in msg
    msg, err, _ = ex.run("reorder_suggestion", {"stock_code": "99999"})
    assert err and "unknown product" in msg
    msg, err, _ = ex.run("drop_tables", {})
    assert err and "Unknown tool" in msg


def test_tool_definitions_are_strict(services):
    for t in tool_definitions():
        assert t["strict"] is True
        schema = t["input_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_stockout_values_are_labelled_synthetic(services):
    out = services.stockout_risk("low", 5)
    assert out["value_types"]["current_stock"] == "synthetic"
    assert out["value_types"]["forecast_units_7d"] == "model_prediction"
    assert any("SYNTHETIC" in lim for lim in out["limitations"])
    assert any("human approval" in lim for lim in out["limitations"])


def test_reorder_calculation_reproduces_the_suggested_quantity(services):
    r = services.reorder_suggestion("22410")["results"]
    c = r["calculation"]
    assert r["reorder_suggested"]
    expected = math.ceil(
        c["forecast_demand_over_cover_units"] + c["safety_stock_units"] - c["stock_on_hand_units"]
    )
    assert r["suggested_order_qty"] == expected


def test_demand_change_decomposition(services):
    r = services.explain_demand_change("22197", 7, 28)["results"]
    assert r["recent"]["units_per_day"] > r["baseline"]["units_per_day"]
    assert r["change"]["units_per_day_pct"] > 0
    assert r["forecast"]["next_7_trading_days_units"] == pytest.approx(300.5 * 7, abs=0.1)


def test_unusual_sales_flags_the_spike(services):
    out = services.unusual_sales(5, 3)
    top = out["results"][0]
    assert top["stock_code"] == "22197" and top["direction"] == "above baseline"


# -- number verification -------------------------------------------------------------------------


def test_verify_accepts_rounding_and_percentages():
    results = [{"forecast_units": 853.151, "p": 0.994, "date": "2011-12-09", "qty": 1530}]
    ok = verify_numbers(
        "Forecast 853.2 units (about 853), 99.4% risk, order 1,530 on 2011-12-09.", results
    )
    assert ok.passed, ok.unverified


def test_verify_flags_invented_numbers():
    check = verify_numbers("Forecast 853.2 units, rising to 1,000 next week.", [{"x": 853.151}])
    assert check.unverified == ["1,000"]


def test_verify_allows_numbers_from_the_question():
    assert verify_numbers("The top 15 are listed.", [], question="show the top 15").passed


# -- LangGraph copilot with a scripted client ---------------------------------------------------


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(name, args, id_="tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=args, id=id_)


def response(content, stop_reason):
    usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0)
    return SimpleNamespace(content=content, stop_reason=stop_reason, model=MODEL, usage=usage)


class ScriptedClient:
    """Stands in for anthropic.Anthropic(); each script step sees the request kwargs."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        step = self.script.pop(0)
        return step(kwargs) if callable(step) else step


def last_tool_result(kwargs) -> dict:
    return json.loads(kwargs["messages"][-1]["content"][0]["content"])


def test_tool_loop_answers_with_verified_numbers(services):
    def answer(kwargs):
        top = last_tool_result(kwargs)["results"][0]
        return response(
            [
                text_block(
                    f"{top['stock_code']} {top['product_description']}: "
                    f"{top['forecast_units']} units "
                    f"forecast for {top['target_date']} (model prediction)."
                )
            ],  # fmt: skip
            "end_turn",
        )

    client = ScriptedClient(
        [
            response(
                [tool_block("top_forecast_demand", {"horizon_trading_days": 1, "limit": 3})],
                "tool_use",
            ),
            answer,
        ]
    )
    ans = Copilot(services, client=client).ask(
        "Which products have the highest forecast demand tomorrow?"
    )

    assert ans.outcome == "answered" and ans.verified
    assert "22197" in ans.answer and "300.5" in ans.answer
    assert ans.tool_calls == [
        {
            "tool": "top_forecast_demand",
            "input": {"horizon_trading_days": 1, "limit": 3},
            "is_error": False,
        }
    ]
    assert ans.evidence["model_version"] == ["test-model"]
    assert ans.evidence["value_types"]["forecast_units"] == "model_prediction"


def test_request_shape_follows_the_model_requirements(services):
    client = ScriptedClient([response([text_block("No data needed.")], "end_turn")])
    Copilot(services, client=client).ask("hello")
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default"
    assert "tool_choice" not in kw  # forced tool choice is rejected on this model
    assert "thinking" not in kw and "temperature" not in kw
    assert kw["output_config"] == {"effort": "medium"}
    assert all(t["strict"] for t in kw["tools"])


def test_history_is_append_only(services):
    first = response(
        [tool_block("stockout_risk", {"min_risk_level": "high", "limit": 2})], "tool_use"
    )
    client = ScriptedClient([first, response([text_block("Done.")], "end_turn")])
    Copilot(services, client=client).ask("stockout risk?")
    second = client.calls[1]["messages"]
    assert second[0] == {"role": "user", "content": "stockout risk?"}
    assert second[1]["content"] is first.content  # assistant turn passed back unchanged
    assert second[2]["content"][0]["type"] == "tool_result"


def test_unverified_numbers_trigger_one_correction(services):
    def corrected(kwargs):
        assert "do not appear in any tool result: 9999" in kwargs["messages"][-1]["content"]
        return response([text_block("Corrected: forecast 300.5 units.")], "end_turn")

    client = ScriptedClient(
        [
            response(
                [tool_block("top_forecast_demand", {"horizon_trading_days": 1, "limit": 1})],
                "tool_use",
            ),
            response([text_block("Forecast 9999 units.")], "end_turn"),
            corrected,
        ]
    )
    ans = Copilot(services, client=client).ask("top forecast?")
    assert ans.verified and ans.answer.startswith("Corrected")
    assert len(client.calls) == 3


def test_persistent_invented_numbers_are_flagged_not_hidden(services):
    client = ScriptedClient(
        [
            response([text_block("Demand will be 4242 units.")], "end_turn"),
            response([text_block("Demand will be 4242 units.")], "end_turn"),
        ]  # fmt: skip
    )
    ans = Copilot(services, client=client).ask("forecast?")
    assert not ans.verified and ans.unverified_numbers == ["4242"]


def test_tool_errors_are_returned_to_the_model(services):
    def recover(kwargs):
        result = kwargs["messages"][-1]["content"][0]
        assert result["is_error"] and "unknown product" in result["content"]
        return response([text_block("That product code does not exist.")], "end_turn")

    client = ScriptedClient(
        [response([tool_block("reorder_suggestion", {"stock_code": "99999"})], "tool_use"), recover]
    )
    ans = Copilot(services, client=client).ask("reorder 99999?")
    assert ans.tool_calls[0]["is_error"] and ans.outcome == "answered"


def test_refusal_is_reported(services):
    client = ScriptedClient([response([], "refusal")])
    ans = Copilot(services, client=client).ask("something declined")
    assert ans.outcome == "refused" and not ans.verified


# -- offline mode and retrieval -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "tool"),
    [
        ("Which products have the highest forecast demand tomorrow?", "top_forecast_demand"),
        ("Which products have unusual sales patterns?", "unusual_sales"),
        ("Which products are at risk of stockout in the next seven days?", "stockout_risk"),
        (
            "Why has demand for 22197 changed compared with its recent baseline?",
            "explain_demand_change",
        ),
        ("What reorder quantity is suggested for 22410?", "reorder_suggestion"),
        ("How are returns handled?", "search_docs"),
    ],
)
def test_offline_answers_the_required_questions(services, question, tool):
    ans = OfflineCopilot(services).ask(question)
    assert tool in [t["tool"] for t in ans.tool_calls]
    assert ans.verified, ans.unverified_numbers
    assert ans.mode == "offline (no LLM)"


def test_offline_resolves_product_names():
    assert detect_intent("Why did popcorn holder sales change?") == "change"


def test_retrieval_stemming_and_chunking():
    assert stem("returns") == "return" and stem("handled") == stem("handling")
    chunks = chunk_markdown("doc.md", "# A\nalpha\n## B\nbeta gamma")
    assert [(c.section, c.text) for c in chunks] == [("A", "alpha"), ("B", "beta gamma")]
