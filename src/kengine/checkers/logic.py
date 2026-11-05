"""The logic checker (spec §5.5 #3): does a formal claim follow from what it rests on?

An inference is checked against the premises it cites, so a step that silently needs
another premise fails with a counterexample. A fact is checked against everything trusted:
the givens, and verified claims that rest only on trusted ones. An `undetermined` claim ("the
premises settle X neither way") is verified only if X is open, and every given has a formal
form; otherwise a given we couldn't translate might settle it.

A step that names a new number (`total = 25`) that nothing it cites defines is left to
arithmetic: the sum lives in the model's working, and the logic checker would only find
that `total` could be anything.

A True/False/Uncertain answer about a conclusion is checked too (`check_answer`): True must
follow from the claims the answer rests on, False must contradict them, and Uncertain must
be left open by the givens. A valid chain that ends somewhere else doesn't count.

The givens' formal forms are the model's translation (FORMALIZE). A `verified` here means
"follows in the fragment from those forms". The audit trail records them, so a
mistranslation can be found.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import TERMINAL, Status
from kengine.checkers.base import BeliefView, CheckerKind, Outcome, Verdict
from kengine.claims.formal import (
    Compare,
    FormalSyntaxError,
    Statement,
    atoms_of,
    parse_statements,
)
from kengine.claims.schema import Claim, ClaimType, Formal
from kengine.logic.entail import decide
from kengine.logic.render import render_world


def _ids(ids: Sequence[str]) -> str:
    return ", ".join(dict.fromkeys(ids))


def _has(ids: Sequence[str]) -> str:
    return "has" if len(set(ids)) == 1 else "have"


def _statements(formal: Formal) -> list[Statement]:
    return [s for text in formal.statements for s in parse_statements(text)]


def _undefined_numbers(premises: Sequence[Statement], goal: Sequence[Statement]) -> list[str]:
    """The numeric names in `goal` that no premise mentions."""
    known = {a.pred for s in premises for lit in (*s.body, *s.head) for a in atoms_of(lit)}
    return list(
        dict.fromkeys(
            a.pred
            for s in goal
            for lit in (*s.body, *s.head)
            if isinstance(lit.atom, Compare)
            for a in atoms_of(lit)
            if a.pred not in known
        )
    )


def _holds(rec: BeliefRecord) -> bool:
    return rec.claim.formal is not None and rec.claim.formal.mode == "holds"


def _trusted(view: BeliefView, exclude: str) -> list[BeliefRecord]:
    """The givens and verified claims whose premises are all trusted too: a step verified
    from an unchecked claim only shows it follows, not that it holds."""
    sound = {
        r.claim.id
        for r in view.all()
        if r.status not in TERMINAL
        and (r.claim.type == ClaimType.GIVEN or r.status == Status.VERIFIED)
    }
    return [
        r
        for r in view.all()
        if r.claim.id != exclude
        and _holds(r)
        and all(id in sound for id in view.premise_closure([r.claim.id]))
    ]


def _informal_givens(view: BeliefView) -> list[str]:
    return [
        r.claim.id
        for r in view.all()
        if r.claim.type == ClaimType.GIVEN and r.status not in TERMINAL and r.claim.formal is None
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
            informal = _informal_givens(view)
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
        undefined = _undefined_numbers(premises, goal)
        if undefined:
            verb = "is" if len(undefined) == 1 else "are"
            return self._verdict(
                Outcome.UNKNOWN,
                None,
                f"can't check: nothing it cites says what {', '.join(undefined)} {verb}. If you "
                "worked it out, write the calculation as a computation; if a given defines it, "
                "cite that given",
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


ANSWER_LABELS = ("True", "False", "Uncertain")


def answer_label(answer: str) -> str | None:
    """`answer` as one of ANSWER_LABELS ("**true.**" is True), or None."""
    word = "".join(ch for ch in answer.lower() if ch.isalpha())
    aliases = {"unknown": "Uncertain"}
    return next((label for label in ANSWER_LABELS if label.lower() == word), aliases.get(word))


def check_answer(
    answer: str,
    conclusion: Formal,
    rests_on: Sequence[str],
    view: BeliefView,
    doubtful: Mapping[str, str] | None = None,
) -> str | None:
    """Why a True/False/Uncertain answer about `conclusion` doesn't stand, or None if it does.

    `rests_on` is everything the answer rests on, its citations and their premises.
    `doubtful` maps givens whose formal form may say less than the sentence to why. Then an
    open conclusion may only look open, so Uncertain doesn't stand."""
    label = answer_label(answer)
    if label is None:
        return f"answer with exactly one of: {', '.join(ANSWER_LABELS)}"
    goal = _statements(conclusion)
    if label == "Uncertain":
        base = _trusted(view, exclude="")
    else:
        records = [view.get(id) for id in rests_on]
        base = [r for r in records if r.status not in TERMINAL and _holds(r)]
    premises, owners = _premises(base)
    d = decide(premises, goal)
    core = _ids([owners[i] for i in d.core])
    if d.kind == "inconsistent":
        return f"can't check it: {core} contradict each other"
    if d.kind == "unknown":
        return f"the answer check gave up: {d.reason}"

    if label == "Uncertain":
        if d.kind == "entailed":
            return f"the conclusion isn't open: it follows from {core}"
        if d.kind == "refuted":
            return f"the conclusion isn't open: its negation follows from {core}"
        informal = _informal_givens(view)
        if informal:
            return (
                f"can't confirm the conclusion is open: {_ids(informal)} {_has(informal)} no "
                "formal form"
            )
        if doubtful:
            ids = list(doubtful)
            whys = "; ".join(f"{id}: {why}" for id, why in doubtful.items())
            say, them = ("says", "it") if len(ids) == 1 else ("say", "them")
            return (
                f"can't confirm the conclusion is open: its formal check allows a case that "
                f"makes {_ids(ids)} false as written ({whys}). Read what {_ids(ids)} {say}: "
                f"the conclusion may follow from {them}, or be ruled out"
            )
        return None

    want = "entailed" if label == "True" else "refuted"
    if d.kind == want:
        return None
    if d.kind == "refuted":
        return f"the conclusion is false given {core}, not true"
    if d.kind == "entailed":
        return f"the conclusion follows from {core}, so it isn't false"
    cited = _ids([r.claim.id for r in base]) if base else "nothing formal"
    informal = [id for id in rests_on if not _holds(view.get(id))]
    note = f" ({_ids(informal)} {_has(informal)} no formal form)" if informal else ""
    missing = _also_needed(goal, rests_on, view, want)
    them = "it" if len(missing) == 1 else "them"
    if label == "True":
        why = (
            f"what it rests on ({cited}) doesn't show the conclusion: it allows "
            f"{render_world(d.world)}{note}."
        )
        if missing:
            return f"{why} Adding {_ids(missing)} shows it: rest the answer on {them} too"
        return f"{why} Rest the answer on a claim that states the conclusion"
    why = f"what it rests on ({cited}) doesn't rule the conclusion out{note}."
    if missing:
        return f"{why} Adding {_ids(missing)} rules it out: rest the answer on {them} too"
    return f"{why} Rest the answer on a claim that states the conclusion's negation"


def _premises(records: Sequence[BeliefRecord]) -> tuple[list[Statement], list[str]]:
    """The statements of `records`' formal forms, and the id of the claim each came from."""
    premises: list[Statement] = []
    owners: list[str] = []
    for rec in records:
        assert rec.claim.formal is not None
        for s in _statements(rec.claim.formal):
            premises.append(s)
            owners.append(rec.claim.id)
    return premises, owners


def _also_needed(
    goal: Sequence[Statement], rests_on: Sequence[str], view: BeliefView, want: str
) -> list[str]:
    """The givens and verified claims, outside `rests_on`, that would let it decide `goal` as
    `want` says. Empty when even all of them don't."""
    premises, owners = _premises(_trusted(view, exclude=""))
    d = decide(premises, goal)
    if d.kind != want:
        return []
    return [id for id in dict.fromkeys(owners[i] for i in d.core) if id not in rests_on]
