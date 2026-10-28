"""RetailPulse copilot: Claude + validated backend tools, orchestrated with LangGraph.

Graph::

    START -> model --tool_use--> tools -> model ...
                   --end_turn--> verify --numbers ok--> END
                                        --unverified (first time)--> model (asked to fix)
                   --refusal / max_tokens / turn limit--> END

* **model**: one Messages API call to Claude Opus 5.5 through the official ``anthropic``
  SDK. Tools are strict JSON schemas. The system prompt and tools are stable, so they're
  cached. The response is appended to the history unchanged (thinking blocks included),
  which keeps the history append-only.
* **tools**: runs every ``tool_use`` block through ``ToolExecutor`` (pydantic-validated
  services) and returns all results in a single user message.
* **verify**: every number in the final answer must match a tool result (rounding allowed).
  The first failure sends the answer back with the offending numbers; a second failure
  returns the answer flagged ``verified = False``.

Server-side refusal fallback (``fallbacks: "default"``) is enabled, so a request that
Claude Opus 5.5's safety classifiers decline is retried on Anthropic's recommended model
for that category.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

import anthropic
from langgraph.graph import END, START, StateGraph

from src.copilot.services import RetailPulseServices
from src.copilot.tools import ToolExecutor, tool_definitions
from src.copilot.verify import verify_numbers

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 16000
MAX_MODEL_TURNS = 8
MAX_VERIFY_RETRIES = 1

SYSTEM_PROMPT = """\
You are RetailPulse Copilot, an analyst assistant for the demand-forecasting system of a \
UK online gift retailer. The data is the public UCI Online Retail II history (December 2009 \
to December 2011) replayed as a simulated real-time stream, not live data. "Now" is the end \
of the forecast origin day stated in the tool results; "tomorrow" is the next trading day.

Numbers: every number you state must come from a tool result in this conversation. You may \
round it or show a probability as a percentage, but don't compute new figures (sums, \
differences, ratios) yourself; if a needed figure isn't returned by any tool, say it isn't \
available. When a question needs data, call the tools first instead of answering from memory.

Say what kind of number each figure is, using the tool results' value_types: observed sales \
(history), model prediction (forecast), or synthetic inventory assumption. Inventory figures \
are synthetic and must never be described as the retailer's real stock. Reorder quantities \
are suggestions that require human approval; never imply an order was placed.

Answer directly first, then give the key figures with their units, product descriptions next \
to stock codes, and the forecast origin or date range they refer to. End with a short \
"Limitations" line drawn from the tool results' limitations. Keep it concise. Resolve product \
names to stock codes with product_info. For questions about methods or policies, use \
search_docs and cite the source file and section."""


class CopilotState(TypedDict, total=False):
    question: str
    messages: list
    tool_log: list
    turns: int
    verify_retries: int
    stop_reason: str | None
    served_model: str | None
    usage: dict
    answer: str | None
    verification: dict | None
    outcome: str


@dataclass
class CopilotAnswer:
    question: str
    answer: str
    outcome: str  # answered | refused | max_tokens | turn_limit
    verified: bool
    unverified_numbers: list[str]
    tool_calls: list[dict]
    evidence: dict
    model: str | None
    mode: str = "claude"
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _text(content) -> str:
    return "\n".join(b.text for b in content if getattr(b, "type", None) == "text").strip()


def evidence_from(tool_log: list[dict]) -> dict:
    """Provenance from the successful tool results: what the numbers refer to."""
    ok = [t["result"] for t in tool_log if t.get("result")]
    limitations: list[str] = []
    for r in ok:
        for item in r.get("limitations", []):
            if item not in limitations:
                limitations.append(item)
    pick = lambda key: sorted({str(r[key]) for r in ok if r.get(key)})  # noqa: E731
    value_types: dict[str, str] = {}
    for r in ok:
        value_types.update(r.get("value_types", {}))
    return {
        "forecast_origin": pick("forecast_origin"),
        "model_version": pick("model_version"),
        "assumptions_version": pick("assumptions_version"),
        "value_types": value_types,
        "limitations": limitations,
    }


class Copilot:
    def __init__(
        self,
        services: RetailPulseServices,
        client: Any | None = None,
        model: str = MODEL,
        effort: str = "medium",
    ):
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.executor = ToolExecutor(services)
        self.tools = tool_definitions()
        self.graph = self._build()

    # -- nodes -------------------------------------------------------------------------------
    def _model(self, state: CopilotState) -> CopilotState:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=self.tools,
            messages=state["messages"],
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        usage = dict(state.get("usage") or {})
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
            usage[k] = usage.get(k, 0) + (getattr(response.usage, k, 0) or 0)
        return {
            "messages": [*state["messages"], {"role": "assistant", "content": response.content}],
            "turns": state.get("turns", 0) + 1,
            "stop_reason": response.stop_reason,
            "served_model": response.model,
            "usage": usage,
        }

    def _tools(self, state: CopilotState) -> CopilotState:
        last = state["messages"][-1]["content"]
        results, log = [], list(state.get("tool_log", []))
        for block in last:
            if getattr(block, "type", None) != "tool_use":
                continue
            content, is_error, structured = self.executor.run(block.name, dict(block.input))
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                }
            )
            log.append(
                {
                    "tool": block.name,
                    "input": dict(block.input),
                    "is_error": is_error,
                    "result": structured,
                    "error": content if is_error else None,
                }
            )
        return {
            "messages": [*state["messages"], {"role": "user", "content": results}],
            "tool_log": log,
        }

    def _verify(self, state: CopilotState) -> CopilotState:
        answer = _text(state["messages"][-1]["content"])
        results = [t["result"] for t in state.get("tool_log", []) if t.get("result")]
        check = verify_numbers(answer, results, state["question"])
        update: CopilotState = {
            "answer": answer,
            "verification": {"checked": check.checked, "unverified": check.unverified},
        }
        if not check.passed and state.get("verify_retries", 0) < MAX_VERIFY_RETRIES:
            feedback = (
                "Automatic check: these numbers in your answer do not appear in any tool result: "
                f"{', '.join(check.unverified)}. Rewrite the answer using only figures returned by "
                "the tools (call tools again if needed), or remove those numbers."
            )
            update["messages"] = [*state["messages"], {"role": "user", "content": feedback}]
            update["verify_retries"] = state.get("verify_retries", 0) + 1
            update["outcome"] = "retry"
        else:
            update["outcome"] = "answered"
        return update

    def _finish(self, state: CopilotState) -> CopilotState:
        outcome = {"refusal": "refused", "max_tokens": "max_tokens"}.get(
            state.get("stop_reason"), "turn_limit"
        )
        text = (
            _text(state["messages"][-1]["content"])
            if state["messages"][-1]["role"] == "assistant"
            else ""
        )
        return {
            "outcome": outcome,
            "answer": text,
            "verification": {"checked": 0, "unverified": []},
        }

    # -- routing -------------------------------------------------------------------------------
    def _after_model(self, state: CopilotState) -> str:
        reason = state.get("stop_reason")
        if reason == "tool_use" and state["turns"] < MAX_MODEL_TURNS:
            return "tools"
        if reason == "end_turn":
            return "verify"
        return "finish"

    @staticmethod
    def _after_verify(state: CopilotState) -> str:
        return "model" if state["outcome"] == "retry" else END

    def _build(self):
        g = StateGraph(CopilotState)
        g.add_node("model", self._model)
        g.add_node("tools", self._tools)
        g.add_node("verify", self._verify)
        g.add_node("finish", self._finish)
        g.add_edge(START, "model")
        g.add_conditional_edges("model", self._after_model, ["tools", "verify", "finish"])
        g.add_edge("tools", "model")
        g.add_conditional_edges("verify", self._after_verify, ["model", END])
        g.add_edge("finish", END)
        return g.compile()

    # -- public ----------------------------------------------------------------------------------
    def ask(self, question: str) -> CopilotAnswer:
        state = self.graph.invoke(
            {
                "question": question,
                "messages": [{"role": "user", "content": question}],
                "tool_log": [],
                "turns": 0,
                "verify_retries": 0,
            },  # fmt: skip
            {"recursion_limit": 4 * MAX_MODEL_TURNS},
        )
        v = state.get("verification") or {"unverified": []}
        refused = state.get("outcome") == "refused"
        return CopilotAnswer(
            question=question,
            answer=state.get("answer") or ("The request was declined." if refused else ""),
            outcome=state["outcome"],
            verified=state["outcome"] == "answered" and not v["unverified"],
            unverified_numbers=v["unverified"],
            tool_calls=[
                {k: t[k] for k in ("tool", "input", "is_error")} for t in state["tool_log"]
            ],
            evidence=evidence_from(state["tool_log"]),
            model=state.get("served_model"),
            usage=state.get("usage", {}),
        )
