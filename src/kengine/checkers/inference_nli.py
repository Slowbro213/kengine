"""An entailment-model check for inferences without formal content (added in M3; the spec
gives these to the M5 critic). The cited claims' texts, joined, are the premise. A strong
entailment makes the inference `supported`; it's a model's judgement, so never `verified`.
Only built for tasks with documents, where inferences bridge facts the fact checker has
checked."""

from __future__ import annotations

from kengine.beliefs.status import Status
from kengine.checkers.base import BeliefView, CheckerKind, Outcome, Verdict
from kengine.claims.schema import Claim, ClaimType
from kengine.config import FactsConfig
from kengine.retrieval.models import EntailmentModel


class InferenceEntailmentChecker:
    id = "inference-nli"
    kind = CheckerKind.EVIDENCE

    def __init__(self, nli: EntailmentModel, cfg: FactsConfig) -> None:
        self._nli, self._cfg = nli, cfg

    def applies_to(self, claim: Claim) -> bool:
        return claim.type == ClaimType.INFERENCE and bool(claim.premises) and claim.formal is None

    def check(self, claim: Claim, view: BeliefView) -> Verdict:
        cited = [p for p in claim.premises if view.exists(p)]
        premise = " ".join(view.get(p).claim.text for p in cited)
        (s,) = self._nli.score([(premise, claim.text)])
        names = ", ".join(cited)
        if s.entail >= self._cfg.inference_entail:
            return self._verdict(
                Outcome.PASS,
                Status.SUPPORTED,
                f"the entailment model finds it follows from {names} ({round(s.entail * 100)}%)",
                cited,
            )
        return self._verdict(
            Outcome.UNKNOWN,
            None,
            f"the entailment model doesn't find that it follows from {names}. State the "
            "missing link as a fact from the documents and cite it too",
            cited,
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
