"""The checker contract (spec §5.5)."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import Status
from kengine.claims.schema import Claim


class CheckerKind(StrEnum):
    DETERMINISTIC = "deterministic"
    EVIDENCE = "evidence"
    CRITIC = "critic"


class Outcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    checker_id: str
    checker_kind: CheckerKind
    outcome: Outcome
    # None: record the verdict but leave the belief's status alone.
    status: Status | None
    explanation: str
    evidence: list[str] = Field(default_factory=list[str])
    counterexample: str | None = None


class BeliefView(Protocol):
    def get(self, id: str) -> BeliefRecord: ...
    def exists(self, id: str) -> bool: ...
    def all(self) -> list[BeliefRecord]: ...
    def premise_closure(self, ids: Sequence[str]) -> list[str]: ...


class Checker(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def kind(self) -> CheckerKind: ...
    def applies_to(self, claim: Claim) -> bool: ...
    def check(self, claim: Claim, view: BeliefView) -> Verdict: ...
