"""Facts that restate the task's numbers (spec §5.5 #5, with the task as the only corpus).

A `fact` such as "each pen costs $1.25 (from b1)", where b1 is a `given` from the task,
is supported when every number it states appears in what it cites: givens, or verified
computations. With no premises it may instead be backed by a single given that has all
its numbers and shares a word with it. It's a lexical check, so it reaches `supported`,
never `verified`. Facts without numbers are left to the fact checker (M3).
"""

from __future__ import annotations

import re
from fractions import Fraction

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import Status
from kengine.checkers.arith_eval import format_number
from kengine.checkers.base import BeliefView, CheckerKind, Outcome, Verdict
from kengine.claims.schema import Claim, ClaimType
from kengine.numbers import numbers_in

_WORD_RE = re.compile(r"[a-z]{3,}")
_STOPWORDS = frozenset(
    "the and for are was were has have had that this with from each per there their they "
    "she her his him its into than then what which who how much many does did is".split()
)


def _stems(text: str) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    return {w.removesuffix("s") for w in words if w not in _STOPWORDS}


def _numbers(rec: BeliefRecord) -> set[Fraction]:
    claim = rec.claim
    found = numbers_in(claim.text)
    if claim.computation is not None:
        found |= numbers_in(claim.computation.expression)
        found |= numbers_in(claim.computation.claimed_result)
    return found


def _citable(rec: BeliefRecord) -> bool:
    return rec.claim.type == ClaimType.GIVEN or (
        rec.claim.type == ClaimType.COMPUTATION and rec.status == Status.VERIFIED
    )


class RestatementChecker:
    id = "restatement"
    kind = CheckerKind.EVIDENCE

    def applies_to(self, claim: Claim) -> bool:
        return claim.type == ClaimType.FACT and bool(numbers_in(claim.text))

    def check(self, claim: Claim, view: BeliefView) -> Verdict:
        wanted = numbers_in(claim.text)
        if claim.premises:
            cited = [r for r in (view.get(p) for p in claim.premises) if _citable(r)]
            have: set[Fraction] = set()
            for rec in cited:
                have |= _numbers(rec)
            evidence = [r.claim.id for r in cited]
        else:
            givens = [r for r in view.all() if r.claim.type == ClaimType.GIVEN]
            words = _stems(claim.text)
            match = next(
                (g for g in givens if wanted <= _numbers(g) and words & _stems(g.claim.text)),
                None,
            )
            have = _numbers(match) if match is not None else set()
            evidence = [match.claim.id] if match is not None else []
        missing = sorted(wanted - have)
        if not missing and evidence:
            return self._verdict(
                Outcome.PASS,
                Status.SUPPORTED,
                f"restates {', '.join(evidence)}",
                evidence,
            )
        shown = ", ".join(format_number(n) for n in missing)
        return self._verdict(
            Outcome.UNKNOWN,
            None,
            f"{shown} isn't in the task's givens or verified calculations it cites. If a given "
            "states it, cite that given. If you worked it out, even by counting, write the "
            f"calculation as its own item (`<expression> = {format_number(missing[0])}`) and "
            "build on that item instead of this fact",
            evidence,
        )

    def _verdict(
        self, outcome: Outcome, status: Status | None, explanation: str, evidence: list[str]
    ) -> Verdict:
        return Verdict(
            checker_id=self.id,
            checker_kind=self.kind,
            outcome=outcome,
            status=status,
            explanation=explanation,
            evidence=evidence,
        )
