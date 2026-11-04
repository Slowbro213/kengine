"""The final-answer gate (spec §5.6): an answer is only as good as what it rests on."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import TERMINAL, Status
from kengine.checkers.base import BeliefView
from kengine.claims.schema import ClaimType
from kengine.loop.profiles import GatePolicy

_STANDARD_OK = frozenset(
    {Status.VERIFIED, Status.SUPPORTED, Status.CRITIC_ACCEPTED, Status.ASSUMED}
)


class GateDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    blocking: list[str]
    acted_on_assumptions: list[str]
    reasons: list[str]


def evaluate_gate(
    preconditions: Sequence[str], view: BeliefView, policy: GatePolicy
) -> GateDecision:
    """Check the cited claims and everything they transitively rest on."""
    blocking: list[str] = []
    assumed: list[str] = []
    reasons: list[str] = []
    if not preconditions and policy != GatePolicy.PERMISSIVE:
        reasons.append("final answer cites no supporting claims")
    known = [p for p in dict.fromkeys(preconditions) if view.exists(p)]
    only_givens = all(view.get(p).claim.type == ClaimType.GIVEN for p in known)
    if known and only_givens and policy != GatePolicy.PERMISSIVE:
        reasons.append(
            "final answer rests only on the task's givens; rest it on the claim that states it"
        )
    for p in dict.fromkeys(preconditions):
        if not view.exists(p):
            blocking.append(p)
            reasons.append(f"{p}: unknown claim")
    for id in view.premise_closure(known):
        record = view.get(id)
        match _judge(record, policy):
            case "ok":
                pass
            case "assumed":
                assumed.append(id)
            case "block":
                blocking.append(id)
                reasons.append(f"{id} is {record.status}")
    return GateDecision(
        allowed=not reasons, blocking=blocking, acted_on_assumptions=assumed, reasons=reasons
    )


def _judge(record: BeliefRecord, policy: GatePolicy) -> Literal["ok", "assumed", "block"]:
    status = record.status
    if status in TERMINAL:
        return "block"
    if record.claim.type == ClaimType.GIVEN or status == Status.VERIFIED:
        return "ok"
    match policy:
        case GatePolicy.STRICT:
            return "block"
        case GatePolicy.STANDARD:
            if status == Status.ASSUMED:
                return "assumed"
            return "ok" if status in _STANDARD_OK else "block"
        case GatePolicy.PERMISSIVE:
            return "assumed" if status in (Status.ASSUMED, Status.UNCHECKED) else "ok"
