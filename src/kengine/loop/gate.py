"""The final-answer gate (spec §5.6): an answer is only as good as what it rests on.

When the task has a conclusion to judge and its formal form is known, a True/False/Uncertain
answer must also be shown by what it rests on (`check_answer`), not just rest on checked
claims."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import TERMINAL, Status
from kengine.checkers.base import BeliefView
from kengine.checkers.logic import check_answer
from kengine.claims.schema import ClaimType, Formal
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
    preconditions: Sequence[str],
    view: BeliefView,
    policy: GatePolicy,
    answer: str | None = None,
    conclusion: Formal | None = None,
    doubtful: Mapping[str, str] | None = None,
) -> GateDecision:
    """Check the cited claims and everything they transitively rest on, and that `answer`
    is right about `conclusion` when there is one. `doubtful` maps givens whose formal form
    may say less than the sentence to why; see `check_answer`."""
    blocking: list[str] = []
    assumed: list[str] = []
    reasons: list[str] = []
    if not preconditions and policy != GatePolicy.PERMISSIVE:
        reasons.append("final answer cites no supporting claims")
    known = [p for p in dict.fromkeys(preconditions) if view.exists(p)]
    judged = conclusion is not None and answer is not None and policy != GatePolicy.PERMISSIVE
    only_givens = all(view.get(p).claim.type == ClaimType.GIVEN for p in known)
    # With a conclusion to judge, the answer check proves it from the givens; without one,
    # an answer citing only givens has nothing checked.
    if known and only_givens and not judged and policy != GatePolicy.PERMISSIVE:
        reasons.append(
            "final answer rests only on the task's givens; rest it on the claim that states it"
        )
    for p in dict.fromkeys(preconditions):
        if not view.exists(p):
            blocking.append(p)
            reasons.append(f"{p}: unknown claim")
    closure = view.premise_closure(known)
    for id in closure:
        record = view.get(id)
        match _judge(record, policy):
            case "ok":
                pass
            case "assumed":
                assumed.append(id)
            case "block":
                blocking.append(id)
                reasons.append(f"{id} is {record.status}")
    if judged and known:
        assert conclusion is not None and answer is not None
        why = check_answer(answer, conclusion, closure, view, doubtful)
        if why is not None:
            reasons.append(why)
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
