"""Types shared across the eval harness."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

SuiteName = Literal["gsm8k", "hotpotqa", "folio"]
Size = Literal["smoke", "small", "full"]
SIZE_COUNTS: dict[str, int] = {"smoke": 10, "small": 100}


class EvalItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    suite: SuiteName
    question: str
    documents: list[str] = []
    gold: str
    answer_format: str


SolveStatus = Literal["answered", "abstained", "partial", "blocked", "error"]


class SolveResult(BaseModel):
    answer: str | None
    status: SolveStatus
    model_calls: int
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    session_dir: str | None = None
    transcript_excerpt: str | None = None
    error: str | None = None


class ItemScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    correct: bool
    detail: str = ""


class Solver(Protocol):
    @property
    def name(self) -> str: ...

    def solve(self, item: EvalItem) -> SolveResult: ...


class RunMetrics(BaseModel):
    suite: str
    solver: str
    n: int
    headline_accuracy: float
    abstention_rate: float
    answered_accuracy: float | None
    status_counts: dict[str, int]
    mean_model_calls: float
    mean_prompt_tokens: float
    mean_completion_tokens: float
    mean_wall_s: float
