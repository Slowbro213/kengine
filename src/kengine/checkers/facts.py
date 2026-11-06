"""The fact checker (spec §5.5 #5): retrieval plus the entailment model.

The top passages for a fact are scored as premises for it. A passage that entails it above
`verify_entail` makes it `verified`; above `support_entail`, `supported`. Contradiction
needs a guard: the entailment model also "contradicts" a fact with passages about another
person or thing, so a contradiction counts only when the passage's document is about
something the fact names. Both backing and contradiction is `conflicting`, and neither is
`insufficient`: both stay unchecked."""

from __future__ import annotations

from kengine.beliefs.status import Status
from kengine.checkers.base import BeliefView, CheckerKind, Outcome, Verdict
from kengine.claims.schema import Claim, ClaimType
from kengine.config import FactsConfig
from kengine.retrieval.index import Corpus
from kengine.retrieval.models import EntailmentModel, NLIScores
from kengine.retrieval.passages import Passage, bare_title
from kengine.retrieval.web import WebSearch, web_passages

_SHOWN = 200


def cite(p: Passage) -> str:
    where = f"web {p.source}, untrusted" if p.source else f"doc {p.doc}"
    return f"{where}: {p.text[:_SHOWN]}"


class FactChecker:
    id = "facts"
    kind = CheckerKind.EVIDENCE

    def __init__(
        self,
        corpus: Corpus,
        nli: EntailmentModel,
        cfg: FactsConfig,
        web: WebSearch | None = None,
    ) -> None:
        self._corpus, self._nli, self._cfg, self._web = corpus, nli, cfg, web

    def applies_to(self, claim: Claim) -> bool:
        return claim.type == ClaimType.FACT

    def check(self, claim: Claim, view: BeliefView) -> Verdict:
        verdict = self._judge(claim)
        if verdict.outcome == Outcome.UNKNOWN and not verdict.evidence and self._web:
            self._corpus.add(web_passages(self._web.search(claim.text)))
            verdict = self._judge(claim)
        return verdict

    def _judge(self, claim: Claim) -> Verdict:
        passages = self._corpus.search(claim.text, self._cfg.top_k)
        scores = self._nli.score([(p.text, claim.text) for p in passages])
        scored = list(zip(passages, scores, strict=True))
        backing = max(scored, key=lambda ps: ps[1].entail, default=None)
        against = [
            p for p, s in scored if s.contradict >= self._cfg.contradict and _names(claim, p)
        ]
        if backing is not None and backing[1].entail >= self._cfg.support_entail:
            p, s = backing
            if against:
                return self._verdict(
                    Outcome.UNKNOWN,
                    None,
                    f"the documents both back and contradict it: {cite(p)} / "
                    f"{cite(against[0])}. Say which one you rely on and why",
                    [cite(p), cite(against[0])],
                )
            if s.entail >= self._cfg.verify_entail:
                return self._verdict(
                    Outcome.PASS, Status.VERIFIED, f"the documents state it: {cite(p)}", [cite(p)]
                )
            return self._verdict(
                Outcome.PASS,
                Status.SUPPORTED,
                f"the documents partly back it ({_pct(s)}): {cite(p)}",
                [cite(p)],
            )
        if against:
            return self._verdict(
                Outcome.FAIL,
                Status.CONTRADICTED,
                f"the documents say otherwise: {cite(against[0])}",
                [cite(against[0])],
            )
        return self._verdict(
            Outcome.UNKNOWN,
            None,
            "no document passage states this. Restate closely the sentence that says it and "
            "name its document, or drop the claim if no document says it",
            [],
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


def _names(claim: Claim, p: Passage) -> bool:
    title = bare_title(p.title).lower()
    return bool(title) and title in claim.text.lower()


def _pct(s: NLIScores) -> str:
    return f"{round(s.entail * 100)}% entailment"
