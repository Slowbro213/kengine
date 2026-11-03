"""The logic checker (spec §5.5 #3): does a formal claim follow from what it rests on?

An inference is checked against the premises it cites, so a step that silently needs
another premise fails with a counterexample. A fact is checked against everything trusted
(the givens and verified claims). An `undetermined` claim ("the premises settle X neither
way") is verified only if X is open, and every given has a formal form; otherwise a given
we couldn't translate might settle it.

The givens' formal forms are the model's translation (FORMALIZE). A `verified` here means
"follows in the fragment from those forms". The audit trail records them, so a
mistranslation can be found.
"""

from __future__ import annotations

from collections.abc import Sequence

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import TERMINAL, Status
from kengine.checkers.base import BeliefView, CheckerKind, Outcome, Verdict
from kengine.claims.formal import FormalSyntaxError, Statement, parse_statements
from kengine.claims.schema import Claim, ClaimType, Formal
from kengine.logic.entail import decide
from kengine.logic.render import render_world


def _ids(ids: Sequence[str]) -> str:
    return ", ".join(dict.fromkeys(ids))


def _has(ids: Sequence[str]) -> str:
    return "has" if len(set(ids)) == 1 else "have"


def _statements(formal: Formal) -> list[Statement]:
    return [s for text in formal.statements for s in parse_statements(text)]


def _holds(rec: BeliefRecord) -> bool:
    return rec.claim.formal is not None and rec.claim.formal.mode == "holds"


def _trusted(view: BeliefView, exclude: str) -> list[BeliefRecord]:
    return [
        r
        for r in view.all()
        if r.claim.id != exclude
        and r.status not in TERMINAL
        and _holds(r)
        and (r.claim.type == ClaimType.GIVEN or r.status == Status.VERIFIED)
    ]


class LogicChecker:
    id = "logic"
    kind = CheckerKind.DETERMINISTIC

    def applies_to(self, claim: Claim) -> bool:
        return claim.formal is not None and claim.type in (ClaimType.FACT, ClaimType.INFERENCE)

    def check(self, claim: Claim, view: BeliefView) -> Verdict:
        formal = claim.formal
        if formal is None:
            return self._verdict(Outcome.UNKNOWN, None, "no formal form")
        cited = [r for r in (view.get(p) for p in claim.premises) if r.status not in TERMINAL]
        step_check = formal.mode == "holds" and claim.type == ClaimType.INFERENCE
        if step_check:
            base = [r for r in cited if _holds(r)]
        else:
            base = _trusted(view, claim.id)
            base += [r for r in cited if _holds(r) and r not in base]
        try:
            premises: list[Statement] = []
            owners: list[str] = []
            for rec in base:
                assert rec.claim.formal is not None
                for s in _statements(rec.claim.formal):
                    premises.append(s)
                    owners.append(rec.claim.id)
            goal = _statements(formal)
        except FormalSyntaxError as e:  # stored forms are canonical; this is a safety net
            return self._verdict(Outcome.UNKNOWN, None, f"logic check gave up: {e}")
        d = decide(premises, goal)
        core = [owners[i] for i in d.core]

        if d.kind == "inconsistent":
            return self._verdict(
                Outcome.UNKNOWN,
                None,
                f"{_ids(core)} contradict each other, so this can't be checked",
                core,
            )
        if d.kind == "unknown":
            return self._verdict(Outcome.UNKNOWN, None, f"logic check gave up: {d.reason}")

        if formal.mode == "undetermined":
            if d.kind == "entailed":
                return self._verdict(
                    Outcome.FAIL,
                    Status.CONTRADICTED,
                    f"it isn't open: it follows from {_ids(core)}",
                    core,
                )
            if d.kind == "refuted":
                return self._verdict(
                    Outcome.FAIL,
                    Status.CONTRADICTED,
                    f"it isn't open: its negation follows from {_ids(core)}",
                    core,
                )
            informal = [
                r.claim.id
                for r in view.all()
                if r.claim.type == ClaimType.GIVEN
                and r.status not in TERMINAL
                and r.claim.formal is None
            ]
            if informal:
                return self._verdict(
                    Outcome.UNKNOWN,
                    None,
                    f"can't confirm it's open: {_ids(informal)} {_has(informal)} no formal form",
                )
            return self._verdict(
                Outcome.PASS,
                Status.VERIFIED,
                "the premises settle it neither way",
                [r.claim.id for r in base],
            )

        if d.kind == "entailed":
            text = f"follows from {_ids(core)}" if core else "is true in every case"
            return self._verdict(Outcome.PASS, Status.VERIFIED, text, core)
        if d.kind == "refuted":
            return self._verdict(
                Outcome.FAIL, Status.CONTRADICTED, f"contradicts {_ids(core)}", core
            )
        world = render_world(d.world)
        if not step_check:
            return self._verdict(
                Outcome.UNKNOWN, None, f"the givens don't settle it: they allow {world}"
            )
        informal = [r.claim.id for r in cited if r.claim.formal is None]
        formal_ids = [r.claim.id for r in base]
        if informal:
            alone = _ids(formal_ids) if formal_ids else "the rest"
            return self._verdict(
                Outcome.UNKNOWN,
                None,
                f"can't check: {_ids(informal)} {_has(informal)} no formal form, and {alone} "
                f"alone allow {world}",
            )
        cites = _ids(formal_ids) if formal_ids else "its premises"
        return self._verdict(
            Outcome.FAIL,
            None,
            f"doesn't follow from {cites}: they allow {world}; cite what else it needs or fix "
            "the step",
            formal_ids,
            counterexample=world,
        )

    def _verdict(
        self,
        outcome: Outcome,
        status: Status | None,
        explanation: str,
        evidence: Sequence[str] = (),
        counterexample: str | None = None,
    ) -> Verdict:
        return Verdict(
            checker_id=self.id,
            checker_kind=self.kind,
            outcome=outcome,
            status=status,
            explanation=explanation,
            evidence=list(dict.fromkeys(evidence)),
            counterexample=counterexample,
        )
