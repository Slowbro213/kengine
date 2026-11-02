"""What the agent is asked to do."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Task(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    goal: str = Field(min_length=1)
    documents: list[str] = Field(default_factory=list[str])
    answer_format: str | None = None
