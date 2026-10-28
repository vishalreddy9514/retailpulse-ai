"""Evaluate the copilot on a fixed question set.

Checks per question (pass/fail, no LLM judge):

* ``tools``: every expected tool was called, and no call ended in an error;
* ``numbers``: every number in the answer matches a tool result;
* ``labels``: answers involving inventory call it synthetic; forecast answers mark predictions;
* ``limitations``: the answer states a limitation (Claude mode; the offline templates always do).

Usage::

    python -m src.copilot.evaluate --offline           # free, deterministic
    python -m src.copilot.evaluate                     # Claude: real API calls, costs credits
"""

from __future__ import annotations

import argparse
import json
import sys

from src import config
from src.copilot.services import Artifacts, RetailPulseServices

QUESTIONS = [
    {
        "q": "Which products have the highest forecast demand tomorrow?",
        "tools": ["top_forecast_demand"],
        "labels": ["forecast"],
    },
    {
        "q": "Which products have unusual sales patterns?",
        "tools": ["unusual_sales"],
        "labels": [],
    },
    {
        "q": "Which products are at risk of stockout in the next seven days?",
        "tools": ["stockout_risk"],
        "labels": ["synthetic"],
    },
    {
        "q": "Why has demand for SMALL POPCORN HOLDER changed compared with its recent baseline?",
        "tools": ["explain_demand_change"],
        "labels": [],
    },
    {
        "q": (
            "What reorder quantity is suggested for 22410 "
            "under the configured inventory assumptions?"
        ),
        "tools": ["reorder_suggestion"],
        "labels": ["synthetic", "approval"],
    },
    {
        "q": "How are returns and cancellations handled in the demand figures?",
        "tools": ["search_docs"],
        "labels": [],
    },
]


def score(item: dict, ans) -> dict:
    called = [t["tool"] for t in ans.tool_calls]
    text = ans.answer.lower()
    checks = {
        "tools": all(t in called for t in item["tools"])
        and not any(t["is_error"] for t in ans.tool_calls),
        "numbers": ans.verified,
        "labels": all(label in text for label in item["labels"]),
        "limitations": ans.mode != "claude" or "limitation" in text,
    }
    return {"question": item["q"], "passed": all(checks.values()), "checks": checks,
            "tools_called": called, "unverified": ans.unverified_numbers}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RetailPulse copilot")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    services = RetailPulseServices(Artifacts.load())
    if args.offline:
        from src.copilot.offline import OfflineCopilot

        bot, mode = OfflineCopilot(services), "offline"
    else:
        from src.copilot.graph import Copilot

        bot, mode = Copilot(services), "claude"
    results = [score(item, bot.ask(item["q"])) for item in QUESTIONS]
    passed = sum(r["passed"] for r in results)
    report = {"mode": mode, "passed": passed, "total": len(results), "results": results}
    out = config.REPORTS_DIR / f"copilot_evaluation_{mode}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for r in results:
        failed = [k for k, v in r["checks"].items() if not v]
        print(
            f"[{'PASS' if r['passed'] else 'FAIL'}] {r['question']}"
            + (f"  failed: {failed}" if failed else "")
        )
    print(f"{passed}/{len(results)} passed ({mode}); wrote {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
