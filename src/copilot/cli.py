"""Ask the RetailPulse copilot a question from the command line.

Usage::

    python -m src.copilot.cli "Which products are at risk of stockout in the next seven days?"
    python -m src.copilot.cli --offline "Which products have unusual sales patterns?"
    python -m src.copilot.cli --json "Why has demand for 22197 changed?"

Claude mode needs Anthropic credentials (``ANTHROPIC_API_KEY``). Without them the CLI says
so; pass ``--offline`` for the deterministic, no-LLM mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from src.copilot.services import Artifacts, RetailPulseServices


def has_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RetailPulse copilot")
    parser.add_argument("question")
    parser.add_argument("--offline", action="store_true", help="deterministic mode, no LLM")
    parser.add_argument(
        "--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"]
    )
    parser.add_argument("--json", action="store_true", help="print the full structured answer")
    args = parser.parse_args(argv)

    services = RetailPulseServices(Artifacts.load())
    if args.offline:
        from src.copilot.offline import OfflineCopilot

        bot = OfflineCopilot(services)
    else:
        if not has_credentials():
            print(
                "No Anthropic credentials: set ANTHROPIC_API_KEY, or use --offline.",
                file=sys.stderr,
            )
            return 2
        from src.copilot.graph import Copilot

        bot = Copilot(services, effort=args.effort)

    ans = bot.ask(args.question)
    if args.json:
        print(json.dumps(ans.to_dict(), indent=2, default=str))
        return 0
    print(ans.answer)
    print()
    model = f" | model: {ans.model}" if ans.model else ""
    print(f"[mode: {ans.mode}{model} | outcome: {ans.outcome}]")
    print(f"[tools: {', '.join(t['tool'] for t in ans.tool_calls) or 'none'}]")
    status = (
        "all numbers match tool results"
        if ans.verified
        else f"UNVERIFIED: {ans.unverified_numbers}"
    )
    print(f"[number check: {status}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
