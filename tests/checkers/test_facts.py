from collections.abc import Sequence

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import Status
from kengine.checkers.base import Outcome
from kengine.checkers.facts import FactChecker
from kengine.claims.schema import Claim, ClaimType, Source, SourceKind
from kengine.config import FactsConfig
from kengine.retrieval.fake import FakeEmbedder, FakeEntailment
from kengine.retrieval.index import Corpus
from kengine.retrieval.models import NLIScores

DOCS = [
    "Ann Lee: Ann Lee is a painter. She was born in Paris in 1990.",
    "Bo Chan: Bo Chan was born in Oslo in 1960.",
]
SURE = NLIScores(0.97, 0.02, 0.01)
PARTLY = NLIScores(0.6, 0.3, 0.1)
AGAINST = NLIScores(0.01, 0.02, 0.97)


def fact(text: str) -> Claim:
    return Claim(
        id="b1", type=ClaimType.FACT, text=text, source=Source(kind=SourceKind.MODEL), step=1
    )


class NoBeliefs:
    def get(self, id: str) -> BeliefRecord:
        raise KeyError(id)

    def exists(self, id: str) -> bool:
        return False

    def all(self) -> list[BeliefRecord]:
        return []

    def premise_closure(self, ids: Sequence[str]) -> list[str]:
        return list(ids)


def checker(*rules: tuple[str, str, NLIScores]) -> FactChecker:
    return FactChecker(
        Corpus.from_documents(DOCS, FakeEmbedder()), FakeEntailment(rules), FactsConfig()
    )


def test_a_fact_a_passage_entails_is_verified_with_the_passage_cited():
    text = "Ann Lee was born in Paris."
    v = checker(("born in Paris", text, SURE)).check(fact(text), NoBeliefs())
    assert (v.outcome, v.status) == (Outcome.PASS, Status.VERIFIED)
    assert v.evidence[0].startswith("doc 1: Ann Lee: ")
    assert "documents state it" in v.explanation


def test_weaker_entailment_is_supported():
    text = "Ann Lee is a French painter."
    v = checker(("is a painter", text, PARTLY)).check(fact(text), NoBeliefs())
    assert (v.outcome, v.status) == (Outcome.PASS, Status.SUPPORTED)


def test_a_contradiction_from_the_named_entitys_document_contradicts():
    text = "Bo Chan was born in 1970."
    v = checker(("Oslo in 1960", text, AGAINST)).check(fact(text), NoBeliefs())
    assert (v.outcome, v.status) == (Outcome.FAIL, Status.CONTRADICTED)
    assert v.evidence == ["doc 2: Bo Chan: Bo Chan was born in Oslo in 1960."]


def test_a_contradiction_from_another_entitys_document_is_ignored():
    text = "Ann Lee was born in 1960."
    v = checker(("Oslo in 1960", text, AGAINST)).check(fact(text), NoBeliefs())
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert "no document passage states this" in v.explanation


def test_backing_and_contradiction_together_are_conflicting():
    text = "Ann Lee was born in Paris."
    v = checker(("born in Paris", text, SURE), ("Ann Lee is a painter.", text, AGAINST)).check(
        fact(text), NoBeliefs()
    )
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert "both back and contradict" in v.explanation


def test_it_applies_to_facts_only():
    c = checker()
    assert c.applies_to(fact("x"))
    assert not c.applies_to(fact("x").model_copy(update={"type": ClaimType.DEFINITION}))
