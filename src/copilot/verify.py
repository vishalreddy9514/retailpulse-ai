"""Check that every number in an answer comes from a tool result.

The copilot may round a figure (853.151 -> 853.2 or 853) and may show a probability
as a percentage (0.994 -> 99.4%). It may not introduce a number that no tool returned
and the user did not mention. ISO dates are ignored; their parts are matched like any
other number. Small integers 1-7 are always allowed, because they name the forecast
horizon ("next 7 trading days").
"""

from __future__ import annotations

import json
import re
from contextlib import suppress
from dataclasses import dataclass, field

NUMBER_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?(?![\w])")
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
ALWAYS_ALLOWED = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}


@dataclass
class Verification:
    checked: int
    unverified: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.unverified


def numbers_in_text(text: str) -> list[tuple[str, float, int, bool]]:
    """(token, value, decimals, is_percent) for each number in free text, ignoring ISO dates."""
    out = []
    for m in NUMBER_RE.finditer(ISO_DATE_RE.sub(" ", text)):
        tok = m.group(0)
        pct = tok.endswith("%")
        core = tok.rstrip("%").replace(",", "")
        try:
            value = float(core)
        except ValueError:
            continue
        decimals = len(core.split(".")[1]) if "." in core else 0
        out.append((tok, value, decimals, pct))
    return out


def numbers_in_results(obj) -> set[float]:
    """Every number in tool results: numeric fields plus numbers inside strings."""
    found: set[float] = set()

    def walk(x):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, int | float):
            found.add(float(x))
        elif isinstance(x, str):
            for part in re.findall(r"-?\d[\d,]*(?:\.\d+)?", x):
                with suppress(ValueError):
                    found.add(float(part.replace(",", "")))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list | tuple):
            for v in x:
                walk(v)

    walk(obj)
    return found


def _matches(value: float, decimals: int, pct: bool, allowed: set[float]) -> bool:
    targets = {abs(value)}
    for y in allowed:
        for cand in {abs(y), abs(y) * 100} if pct or abs(y) <= 1 else {abs(y)}:
            if round(cand, decimals) in targets or (decimals == 0 and round(cand) in targets):
                return True
    return False


def verify_numbers(answer: str, tool_results: list, question: str = "") -> Verification:
    allowed = numbers_in_results(tool_results) | ALWAYS_ALLOWED
    allowed |= {v for _, v, _, _ in numbers_in_text(question)}
    allowed |= {-v for v in allowed}
    found = numbers_in_text(answer)
    unverified = [tok for tok, v, d, pct in found if not _matches(v, d, pct, allowed)]
    return Verification(checked=len(found), unverified=unverified)


def to_json(results: list) -> str:
    return json.dumps(results, default=str)
