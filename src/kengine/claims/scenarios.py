"""SCENARIOS: does a case the formal forms allow fit the statements as written?

When the givens' formal forms leave a conclusion open, z3 has a case where it fails and one
where it holds. Both satisfy the formal forms. If one makes a given false as written in
English, that given's formal form says less than the sentence, and the conclusion may not
be open after all. The model reads the cases; it can only cast doubt, never confirm."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from kengine.backend.types import Message
from kengine.prompts import load_prompt

type Case = Sequence[tuple[str, str]]


class Broken(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case: int = Field(ge=1)
    statement: int = Field(ge=1)
    why: str = Field(max_length=400)


class Scenarios(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    broken: list[Broken] = Field(max_length=60)


def render_case(case: Case) -> str:
    true = [a for a, v in case if v == "true"]
    false = [a for a, v in case if v == "false"]
    numbers = [f"{a} = {v}" for a, v in case if v not in ("true", "false")]
    lines = [f"- true: {', '.join(true) or '(none)'}", f"- false: {', '.join(false) or '(none)'}"]
    if numbers:
        lines.append(f"- numbers: {', '.join(numbers)}")
    return "\n".join(lines)


def scenario_messages(statements: Sequence[str], cases: Sequence[Case]) -> list[Message]:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))
    shown = "\n\n".join(f"Case {i}:\n{render_case(c)}" for i, c in enumerate(cases, 1))
    user = f"## Statements\n{numbered}\n\n## Cases\n{shown}"
    return [
        Message(role="system", content=load_prompt("scenarios")),
        Message(role="user", content=user),
    ]


def read_scenarios(parsed: Scenarios, n: int, cases: int) -> dict[int, str]:
    """Statement number -> why a case makes it false, for numbers and cases that exist."""
    out: dict[int, str] = {}
    for b in parsed.broken:
        if b.statement <= n and b.case <= cases:
            out.setdefault(b.statement, f"case {b.case}: {b.why}")
    return out
