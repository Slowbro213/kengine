"""What the extract role returns: the schema sent as `response_format` (spec §5.3)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from kengine.claims.formal import FormalSyntaxError, JsonStatement, canonical, json_statement_text
from kengine.claims.schema import (
    ClaimText,
    Computation,
    Formal,
    FormalMode,
    PremiseRef,
    check_computation,
)

# `observation`, `given` and `precondition` come from tools, the task or the gate, never
# from the model's own extraction.
ExtractableType = Literal[
    "fact", "definition", "assumption", "computation", "inference", "hypothesis"
]


class ExtractedFormal(BaseModel):
    """Raw formal form from the model. Parsed by `to_formal`, so a typo costs only this
    field, not the whole sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: FormalMode = "holds"
    statements: list[str | JsonStatement] = Field(min_length=1, max_length=8)


def to_formal(f: ExtractedFormal) -> Formal:
    texts = [s if isinstance(s, str) else json_statement_text(s) for s in f.statements]
    statements = [c for t in texts for c in canonical(t)]  # raises FormalSyntaxError
    if len(statements) > 12:
        raise FormalSyntaxError(f"too many statements ({len(statements)} > 12)")
    return Formal(mode=f.mode, statements=tuple(statements))


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: Annotated[str, StringConstraints(pattern=r"^c[0-9]+$")]
    type: ExtractableType
    text: ClaimText
    premises: list[PremiseRef] = Field(default_factory=list[PremiseRef])
    computation: Computation | None = None
    formal: ExtractedFormal | None = None

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
