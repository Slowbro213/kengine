"""What the agent is asked to do."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

_SENTENCE_END_RE = re.compile(r"(?<=[.?!])\s+|\n+")


class Task(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    goal: str = Field(min_length=1)
    documents: list[str] = Field(default_factory=list[str])
    answer_format: str | None = None
    # Statements the task supplies. Each becomes a `given` claim before the first step,
    # so the model can cite it instead of restating it as an unchecked fact.
    givens: list[str] = Field(default_factory=list[str])


def split_sentences(text: str) -> list[str]:
    """Sentences and lines of `text`, stripped. Questions are kept: their numbers are task
    data too ("How much do 12 pens cost?")."""
    return [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]
