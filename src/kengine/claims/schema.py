"""Claims as stored in the belief graph (spec §5.3)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

PremiseRef = Annotated[str, StringConstraints(pattern=r"^[bc][0-9]+$")]
ClaimText = Annotated[str, StringConstraints(min_length=1, max_length=400)]


class ClaimType(StrEnum):
    OBSERVATION = "observation"
    GIVEN = "given"
    FACT = "fact"
    DEFINITION = "definition"
    ASSUMPTION = "assumption"
    COMPUTATION = "computation"
    INFERENCE = "inference"
    HYPOTHESIS = "hypothesis"
    PRECONDITION = "precondition"


class SourceKind(StrEnum):
    MODEL = "model"
    TOOL = "tool"
    SKILL = "skill"
    SUBAGENT = "subagent"
    USER = "user"
    TASK = "task"


class Source(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: SourceKind
    ref: str = ""


class Computation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    expression: str
    claimed_result: str


def check_computation(type_: str, computation: Computation | None) -> None:
    """A computation is attached if and only if the claim is a computation."""
    is_computation = type_ == ClaimType.COMPUTATION
    if is_computation and computation is None:
        raise ValueError("computation claims need a computation")
    if not is_computation and computation is not None:
        raise ValueError(f"{type_} claims can't have a computation")


class Claim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Annotated[str, StringConstraints(pattern=r"^b[0-9]+$")]
    type: ClaimType
    text: ClaimText
    premises: tuple[PremiseRef, ...] = ()
    computation: Computation | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: Source
    step: int = Field(ge=0)

    @model_validator(mode="after")
    def _computation_iff_type(self) -> Claim:
        check_computation(self.type, self.computation)
        return self
