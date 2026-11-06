from collections.abc import Sequence

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import Status
from kengine.checkers.base import Outcome
from kengine.checkers.inference_nli import InferenceEntailmentChecker
from kengine.claims.schema import Claim, ClaimType, Formal, Source, SourceKind
from kengine.config import FactsConfig
from kengine.retrieval.fake import FakeEntailment
from kengine.retrieval.models import NLIScores

SURE = NLIScores(0.95, 0.04, 0.01)


def claim(id: str, text: str, type_: ClaimType, premises: tuple[str, ...] = ()) -> Claim:
    return Claim(
        id=id,
        type=type_,
        text=text,
        premises=premises,
        source=Source(kind=SourceKind.MODEL),
        step=1,
    )


class View:
    def __init__(self, claims: Sequence[Claim]) -> None:
        self._r = {
            c.id: BeliefRecord(claim=c, status=Status.VERIFIED, status_reason="") for c in claims
        }

    def get(self, id: str) -> BeliefRecord:
        return self._r[id]

    def exists(self, id: str) -> bool:
        return id in self._r

    def all(self) -> list[BeliefRecord]:
        return list(self._r.values())

    def premise_closure(self, ids: Sequence[str]) -> list[str]:
        return list(ids)


B1 = claim("b1", "Heat was directed by Michael Mann.", ClaimType.FACT)
B2 = claim("b2", "Michael Mann was born in Chicago.", ClaimType.FACT)
INF = claim("b3", "The director of Heat was born in Chicago.", ClaimType.INFERENCE, ("b1", "b2"))
JOINED = f"{B1.text} {B2.text}"


def test_an_inference_the_model_finds_entailed_is_supported_not_verified():
    nli = FakeEntailment([(JOINED, INF.text, SURE)])
    v = InferenceEntailmentChecker(nli, FactsConfig()).check(INF, View([B1, B2, INF]))
    assert (v.outcome, v.status) == (Outcome.PASS, Status.SUPPORTED)
    assert nli.calls == [(JOINED, INF.text)]
    assert v.evidence == ["b1", "b2"]


def test_otherwise_it_stays_unchecked_with_a_hint():
    v = InferenceEntailmentChecker(FakeEntailment(), FactsConfig()).check(INF, View([B1, B2, INF]))
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert "doesn't find that it follows from b1, b2" in v.explanation


def test_formal_or_premiseless_inferences_are_left_to_other_checkers():
    c = InferenceEntailmentChecker(FakeEntailment(), FactsConfig())
    assert c.applies_to(INF)
    assert not c.applies_to(INF.model_copy(update={"premises": ()}))
    formal = Formal(statements=("born(mann, chicago)",))
    assert not c.applies_to(INF.model_copy(update={"formal": formal}))
    assert not c.applies_to(B1)
