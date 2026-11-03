"""FORMALIZE: one call that gives the task's givens formal forms (spec §5.3)."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from kengine.backend.types import Message
from kengine.claims.extraction import ExtractedFormal, to_formal
from kengine.claims.formal import FormalSyntax, JsonStatement
from kengine.claims.schema import Formal
from kengine.prompts import system_prompt


class FormalizedGiven(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    given: int = Field(ge=1)
    statements: list[str | JsonStatement] = Field(
        default_factory=list[str | JsonStatement], max_length=8
    )


class Formalization(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    givens: list[FormalizedGiven] = Field(max_length=60)


def formalize_messages(goal: str, statements: Sequence[str], syntax: FormalSyntax) -> list[Message]:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))
    user = f"## Task\n{goal}\n\n## Statements to translate\n{numbered}"
    return [
        Message(role="system", content=system_prompt("formalize", syntax)),
        Message(role="user", content=user),
    ]


def read_formalization(parsed: Formalization, n: int) -> tuple[dict[int, Formal], list[str]]:
    out: dict[int, Formal] = {}
    errors: list[str] = []
    for item in parsed.givens:
        if not 1 <= item.given <= n:
            errors.append(f"given {item.given} doesn't exist")
            continue
        if not item.statements or item.given in out:
            continue
        try:
            out[item.given] = to_formal(ExtractedFormal(statements=item.statements))
        except ValueError as e:
            errors.append(f"given {item.given}: {e}")
    return out, errors
