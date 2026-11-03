"""Builds each step's prompt from the belief store, within a token budget (spec §5.8).

The prompt is rebuilt every step; it's never a growing transcript.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from kengine.backend.types import Message
from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import TERMINAL, Status
from kengine.checkers.base import BeliefView
from kengine.claims.formal import FormalSyntax
from kengine.claims.schema import ClaimType
from kengine.loop.task import Task
from kengine.prompts import load_prompt, system_prompt

HISTORY_STEPS = 5

type TokenCounter = Callable[[str], int]


class StepSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    action_kind: str
    note: str


class StepContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: list[Message]
    included: list[str]
    dropped: list[str]
    prompt_tokens: int


class ContextOverflowError(ValueError):
    pass


def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / 3.5)


def belief_line(rec: BeliefRecord) -> str:
    claim = rec.claim
    line = f"{claim.id} [{rec.status}] ({claim.type}) {claim.text}"
    if claim.computation is not None:
        line += f" :: {claim.computation.expression} = {claim.computation.claimed_result}"
    if claim.formal is not None:
        label = "formal" if claim.formal.mode == "holds" else f"formal ({claim.formal.mode})"
        line += f" :: {label}: {'; '.join(claim.formal.statements)}"
    return line


def _section(title: str, lines: Sequence[str]) -> str:
    return f"## {title}\n" + "\n".join(lines)


def _join(sections: Sequence[str]) -> str:
    return "\n\n".join(s for s in sections if s)


class _Budget:
    """Running token count. Pieces are counted separately, which never undercounts for
    `approx_tokens` (ceil is subadditive) and is close enough for real tokenizers."""

    def __init__(self, budget: int, count: TokenCounter, mandatory: Sequence[str]) -> None:
        self.budget = budget
        self.count = count
        self.used = sum(count(text) for text in mandatory)
        if self.used > budget:
            raise ContextOverflowError(
                f"task, documents and feedback need {self.used} tokens; budget is {budget}"
            )

    def take(self, text: str) -> bool:
        cost = self.count(text)
        if self.used + cost > self.budget:
            return False
        self.used += cost
        return True


def _fill(
    groups: Sequence[tuple[str, list[BeliefRecord]]], budget: _Budget
) -> tuple[list[tuple[str, list[BeliefRecord]]], list[str]]:
    """Take belief lines group by group, newest first, until one doesn't fit."""
    taken: dict[str, list[BeliefRecord]] = {title: [] for title, _ in groups}
    dropped: list[str] = []
    full = False
    for title, recs in groups:
        for rec in reversed(recs):
            header = "" if taken[title] else f"\n\n## {title}\n"
            if not full and budget.take(header + belief_line(rec) + "\n"):
                taken[title].append(rec)
            else:
                full = True
                dropped.append(rec.claim.id)
    sections = [(title, sorted(taken[title], key=_seq)) for title, _ in groups]
    return sections, sorted(dropped, key=_id_seq)


def _seq(rec: BeliefRecord) -> int:
    return _id_seq(rec.claim.id)


def _id_seq(id: str) -> int:
    return int(id[1:])


def assemble_think_context(
    task: Task,
    view: BeliefView,
    feedback: Sequence[str],
    history: Sequence[StepSummary],
    budget_tokens: int,
    count_tokens: TokenCounter,
) -> StepContext:
    system = load_prompt("think")
    goal = task.goal
    if task.answer_format:
        goal += f"\n\nAnswer format: {task.answer_format}"
    mandatory = [
        _section("Task", [goal]),
        _section("Documents", [f"[{i}] {d}" for i, d in enumerate(task.documents, 1)])
        if task.documents
        else "",
        _section("Problems found last step", list(feedback)) if feedback else "",
    ]
    budget = _Budget(budget_tokens, count_tokens, [system, _join(mandatory)])

    live = [r for r in view.all() if r.status not in TERMINAL]
    established = [
        r for r in live if r.status == Status.VERIFIED or r.claim.type == ClaimType.GIVEN
    ]
    rest = [r for r in live if r not in established]
    groups = [
        ("Established", established),
        ("Assumptions in play", [r for r in rest if r.status == Status.ASSUMED]),
        ("Unchecked claims", [r for r in rest if r.status != Status.ASSUMED]),
    ]
    sections, dropped = _fill(groups, budget)
    belief_sections = [_section(t, [belief_line(r) for r in recs]) for t, recs in sections if recs]

    recent = history[-HISTORY_STEPS:]
    history_section = ""
    if recent:
        lines = [f"step {h.step}: {h.action_kind} — {h.note}" for h in recent]
        candidate = _section("Your previous steps", lines)
        if budget.take("\n\n" + candidate):
            history_section = candidate

    user = _join([*mandatory, *belief_sections, history_section])
    included = sorted((r.claim.id for _, recs in sections for r in recs), key=_id_seq)
    return StepContext(
        messages=[Message(role="system", content=system), Message(role="user", content=user)],
        included=included,
        dropped=dropped,
        prompt_tokens=count_tokens(system) + count_tokens(user),
    )


def assemble_extract_context(
    view: BeliefView,
    think_text: str,
    budget_tokens: int,
    count_tokens: TokenCounter,
    syntax: FormalSyntax = "text",
) -> StepContext:
    system = system_prompt("extract", syntax)
    reasoning = _section("Reasoning to convert", [think_text])
    budget = _Budget(budget_tokens, count_tokens, [system, reasoning])
    live = [r for r in view.all() if r.status not in TERMINAL]
    sections, dropped = _fill([("Existing claims you may reference", live)], budget)
    belief_sections = [_section(t, [belief_line(r) for r in recs]) for t, recs in sections if recs]
    user = _join([*belief_sections, reasoning])
    return StepContext(
        messages=[Message(role="system", content=system), Message(role="user", content=user)],
        included=[r.claim.id for _, recs in sections for r in recs],
        dropped=dropped,
        prompt_tokens=count_tokens(system) + count_tokens(user),
    )
