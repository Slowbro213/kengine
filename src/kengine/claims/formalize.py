"""FORMALIZE: one call that gives the task's givens formal forms (spec §5.3)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from kengine.backend.types import Message
from kengine.claims.extraction import ExtractedFormal, to_formal
from kengine.claims.formal import FormalSyntax, JsonStatement, parse_statements, variables_of
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


def says_everything(formal: Formal) -> str | None:
    """A statement with variables and no conditions, such as `four_sided(x)`: it says the
    property holds of every individual. A translation of "All X are Y" that lost its "if"
    part looks like this, and it makes every step that uses it pass for nothing."""
    for text in formal.statements:
        for s in parse_statements(text):
            if not s.body and variables_of(s):
                return f'`{s}` says it holds for every individual; did it lose an "if" part?'
    return None


def reformalize_messages(
    goal: str,
    statements: Sequence[str],
    current: Mapping[int, Formal],
    redo: Mapping[int, str],
    syntax: FormalSyntax,
) -> list[Message]:
    """Ask for the statements in `redo` (number -> why) to be translated again, showing every
    statement's current translation so the names stay the same."""
    lines: list[str] = []
    for i, s in enumerate(statements, 1):
        formal = current.get(i)
        now = "; ".join(f"`{t}`" for t in formal.statements) if formal else "(none)"
        lines.append(f"{i}. {s}\n   now: {now}")
    asks = "\n".join(f"- Statement {i}: {why}" for i, why in redo.items())
    user = (
        f"## Task\n{goal}\n\n## Statements and their current translations\n"
        + "\n".join(lines)
        + f"\n\n## Translate again\nThese translations look wrong:\n{asks}\n\n"
        "Translate only these statements again, from the sentence. Say everything it says, "
        'including any "if" part, and use the names the other translations use.'
    )
    return [
        Message(role="system", content=system_prompt("formalize", syntax)),
        Message(role="user", content=user),
    ]
