"""The consistency checker (spec §5.5 #4): does a new fact or definition contradict a claim
already given or checked?

Embeddings pick the candidates and the entailment model scores contradiction. The model
also "contradicts" claims about different people, so a pair is compared only when the two
share a proper name and are close in meaning, and it counts only when both directions
score as a contradiction. Claims with formal content are the logic checker's."""

from __future__ import annotations

import re

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import Status
from kengine.checkers.base import BeliefView, CheckerKind, Outcome, Verdict
from kengine.claims.schema import Claim, ClaimType
from kengine.config import FactsConfig
from kengine.retrieval.models import Embedder, EntailmentModel

_NAME_RE = re.compile(r"\b[A-Z][A-Za-z'-]+\b")
_NOT_NAMES = frozenset(
    "The A An In On At It Its He She His Her They Their This That These Those There Then "
    "When While After Before So But And Or If As By For From Of To With Both Neither Either "
    "Yes No I We You".split()
)
_APPLIES = (ClaimType.FACT, ClaimType.DEFINITION)
_COMPARED = (ClaimType.FACT, ClaimType.DEFINITION, ClaimType.OBSERVATION)
_CHECKED = (Status.VERIFIED, Status.SUPPORTED)


def proper_names(text: str) -> set[str]:
    return {w for w in _NAME_RE.findall(text) if w not in _NOT_NAMES}


class ConsistencyChecker:
    id = "consistency"
    kind = CheckerKind.EVIDENCE

    def __init__(self, nli: EntailmentModel, embedder: Embedder, cfg: FactsConfig) -> None:
        self._nli, self._embedder, self._cfg = nli, embedder, cfg

    def applies_to(self, claim: Claim) -> bool:
        return claim.type in _APPLIES and claim.formal is None

    def check(self, claim: Claim, view: BeliefView) -> Verdict:
        names = proper_names(claim.text)
        others = [
            r
            for r in view.all()
            if r.claim.id != claim.id and _comparable(r) and names & proper_names(r.claim.text)
        ]
        if others:
            v = self._embedder.embed([claim.text, *(r.claim.text for r in others)])
            cos = v[1:] @ v[0]
            close = [float(c) >= self._cfg.consistency_cosine for c in cos]
            others = [r for r, ok in zip(others, close, strict=True) if ok]
        for r in others:
            a, b = self._nli.score([(r.claim.text, claim.text), (claim.text, r.claim.text)])
            if min(a.contradict, b.contradict) >= self._cfg.contradict:
                return self._verdict(
                    Outcome.FAIL,
                    Status.CONTRADICTED,
                    f"contradicts {r.claim.id}: {r.claim.text}",
                    [r.claim.id],
                )
        return self._verdict(
            Outcome.PASS, None, f"no conflict with {len(others)} close checked claim(s)", []
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


def _comparable(r: BeliefRecord) -> bool:
    if r.claim.type == ClaimType.GIVEN:
        return True
    return r.claim.type in _COMPARED and r.status in _CHECKED
