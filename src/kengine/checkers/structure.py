"""Structural checks on the belief graph (spec §5.5 #1).

Missing premises and cycles can't happen: the store rejects unknown premises.
"""

from __future__ import annotations

from kengine.beliefs.status import TERMINAL, Status
from kengine.checkers.base import BeliefView, CheckerKind, Outcome, Verdict
from kengine.claims.schema import Claim, ClaimType


class StructureChecker:
    id = "structure"
    kind = CheckerKind.DETERMINISTIC

    def applies_to(self, claim: Claim) -> bool:
        return True

    def check(self, claim: Claim, view: BeliefView) -> Verdict:
        if claim.type == ClaimType.INFERENCE and not claim.premises:
            return self._verdict(Outcome.FAIL, None, "inference has no premises")
        for pid in claim.premises:
            status = view.get(pid).status
            if status in TERMINAL:
                return self._verdict(
                    Outcome.FAIL, Status.INVALIDATED, f"rests on {pid}, which is {status}"
                )
        # Structure alone never verifies anything.
        return self._verdict(Outcome.PASS, None, "well-formed")

    def _verdict(self, outcome: Outcome, status: Status | None, explanation: str) -> Verdict:
        return Verdict(
            checker_id=self.id,
            checker_kind=self.kind,
            outcome=outcome,
            status=status,
            explanation=explanation,
        )
