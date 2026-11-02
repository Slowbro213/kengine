"""The bare-model baseline: one thinking-mode call, same token budget, no checks (§6)."""

from __future__ import annotations

import time

from kengine.backend.types import Message, ModelBackend, RoleConfig
from kengine.eval.answers import extract_final_answer
from kengine.eval.types import EvalItem, SolveResult
from kengine.prompts import load_prompt

EXCERPT_CHARS = 2000


def render_user_message(item: EvalItem) -> str:
    parts: list[str] = []
    if item.documents:
        docs = "\n".join(f"[{i}] {d}" for i, d in enumerate(item.documents, start=1))
        parts.append(f"## Documents\n{docs}")
    parts.append(f"## Question\n{item.question}")
    parts.append(item.answer_format)
    return "\n\n".join(parts)


def transcript_excerpt(reasoning: str | None, content: str) -> str:
    return f"{(reasoning or '')[-EXCERPT_CHARS:]}\n---\n{content[-EXCERPT_CHARS:]}"


class BaselineSolver:
    name = "baseline"

    def __init__(self, backend: ModelBackend, role: RoleConfig, token_budget: int) -> None:
        self._backend = backend
        self._role = role.model_copy(update={"max_tokens": token_budget})
        self._system = load_prompt("baseline")

    def solve(self, item: EvalItem) -> SolveResult:
        start = time.monotonic()
        messages = [
            Message(role="system", content=self._system),
            Message(role="user", content=render_user_message(item)),
        ]
        completion = self._backend.complete(messages, self._role)
        answer = extract_final_answer(completion.content)
        if answer is not None:
            status = "answered"
        elif completion.finish_reason == "length":
            status = "partial"
        else:
            status = "abstained"
        return SolveResult(
            answer=answer,
            status=status,
            model_calls=1,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
            wall_s=time.monotonic() - start,
            transcript_excerpt=transcript_excerpt(completion.reasoning, completion.content),
        )
