"""A claim plus its current status. Kept apart from the store so checkers can import it."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from kengine.beliefs.status import Status
from kengine.claims.schema import Claim


class BeliefRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    claim: Claim
    status: Status
    status_reason: str
