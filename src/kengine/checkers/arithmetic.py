"""Re-evaluates computation claims (spec §5.5 #2)."""

from __future__ import annotations

from kengine.beliefs.status import Status
from kengine.checkers.arith_eval import ArithError, format_number, results_match, safe_eval
from kengine.checkers.base import BeliefView, CheckerKind, Outcome, Verdict
from kengine.claims.schema import Claim, ClaimType


class ArithmeticChecker:
    id = "arithmetic"
    kind = CheckerKind.DETERMINISTIC

    def applies_to(self, claim: Claim) -> bool:
        return claim.type == ClaimType.COMPUTATION

    def check(self, claim: Claim, view: BeliefView) -> Verdict:
        assert claim.computation is not None  # guaranteed by Claim's validator
        expr = claim.computation.expression
        claimed = claim.computation.claimed_result
        try:
            actual = safe_eval(expr)
        except ArithError as e:
            return self._verdict(Outcome.UNKNOWN, None, f"could not evaluate: {e}")
        match results_match(actual, claimed):
            case None:
                return self._verdict(
                    Outcome.UNKNOWN, None, f"could not evaluate: {claimed!r} isn't a number"
                )
            case True:
                return self._verdict(
                    Outcome.PASS, Status.VERIFIED, f"`{expr}` = {format_number(actual)}"
                )
            case False:
                shown = format_number(actual)
                return self._verdict(
                    Outcome.FAIL,
                    Status.CONTRADICTED,
                    f"`{expr}` evaluates to {shown}, not {claimed}",
                    counterexample=shown,
                )

    def _verdict(
        self,
        outcome: Outcome,
        status: Status | None,
        explanation: str,
        counterexample: str | None = None,
    ) -> Verdict:
        return Verdict(
            checker_id=self.id,
            checker_kind=self.kind,
            outcome=outcome,
            status=status,
            explanation=explanation,
            counterexample=counterexample,
        )
