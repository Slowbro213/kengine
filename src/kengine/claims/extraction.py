"""What the extract role returns: the schema sent as `response_format` (spec §5.3)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from kengine.claims.schema import ClaimText, Computation, PremiseRef, check_computation

# `observation`, `given` and `precondition` come from tools, the task or the gate, never
# from the model's own extraction.
ExtractableType = Literal[
    "fact", "definition", "assumption", "computation", "inference", "hypothesis"
]


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: Annotated[str, StringConstraints(pattern=r"^c[0-9]+$")]
    type: ExtractableType
    text: ClaimText
    premises: list[PremiseRef] = Field(default_factory=list[PremiseRef])
    computation: Computation | None = None

    @model_validator(mode="after")
    def _computation_iff_type(self) -> ExtractedClaim:
        check_computation(self.type, self.computation)
        return self


class ProposedAction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["final_answer", "continue"]
    answer: str | None = None
    preconditions: list[str] = Field(default_factory=list[str])


class Extraction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    claims: list[ExtractedClaim] = Field(max_length=12)
    action: ProposedAction
