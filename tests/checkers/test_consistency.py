from collections.abc import Sequence

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import Status
from kengine.checkers.base import Outcome
from kengine.checkers.consistency import ConsistencyChecker, proper_names
from kengine.claims.schema import Claim, ClaimType, Source, SourceKind
from kengine.config import FactsConfig
from kengine.retrieval.fake import FakeEmbedder, FakeEntailment
from kengine.retrieval.models import NLIScores

AGAINST = NLIScores(0.01, 0.02, 0.97)


def rec(id: str, text: str, type_: ClaimType, status: Status) -> BeliefRecord:
    c = Claim(id=id, type=type_, text=text, source=Source(kind=SourceKind.MODEL), step=1)
    return BeliefRecord(claim=c, status=status, status_reason="")


class View:
    def __init__(self, records: Sequence[BeliefRecord]) -> None:
        self._r = {r.claim.id: r for r in records}

    def get(self, id: str) -> BeliefRecord:
        return self._r[id]

    def exists(self, id: str) -> bool:
        return id in self._r

    def all(self) -> list[BeliefRecord]:
        return list(self._r.values())

    def premise_closure(self, ids: Sequence[str]) -> list[str]:
        return list(ids)


OLD = rec("b1", "Ann Lee was born in Paris in 1990.", ClaimType.FACT, Status.VERIFIED)
NEW = rec("b2", "Ann Lee was born in Rome in 1990.", ClaimType.FACT, Status.UNCHECKED)


def both_ways(a: str, b: str) -> list[tuple[str, str, NLIScores]]:
    return [(a, b, AGAINST), (b, a, AGAINST)]


def test_proper_names_skip_sentence_starters():
    assert proper_names("The film was made by Ann Lee in Paris.") == {"Ann", "Lee", "Paris"}


def test_a_claim_contradicting_a_verified_one_both_ways_is_contradicted():
    nli = FakeEntailment(both_ways(OLD.claim.text, NEW.claim.text))
    v = ConsistencyChecker(nli, FakeEmbedder(), FactsConfig()).check(NEW.claim, View([OLD, NEW]))
    assert (v.outcome, v.status, v.evidence) == (Outcome.FAIL, Status.CONTRADICTED, ["b1"])
    assert v.explanation.startswith("contradicts b1: Ann Lee was born in Paris")


def test_one_direction_is_not_enough():
    nli = FakeEntailment([(OLD.claim.text, NEW.claim.text, AGAINST)])
    v = ConsistencyChecker(nli, FakeEmbedder(), FactsConfig()).check(NEW.claim, View([OLD, NEW]))
    assert (v.outcome, v.status) == (Outcome.PASS, None)


def test_claims_without_a_shared_name_are_not_compared():
    other = rec("b1", "Bo Chan was born in Paris in 1990.", ClaimType.FACT, Status.VERIFIED)
    nli = FakeEntailment(both_ways(other.claim.text, NEW.claim.text))
    v = ConsistencyChecker(nli, FakeEmbedder(), FactsConfig()).check(NEW.claim, View([other, NEW]))
    assert v.outcome == Outcome.PASS and nli.calls == []


def test_unchecked_claims_are_not_compared():
    old = OLD.model_copy(update={"status": Status.UNCHECKED})
    nli = FakeEntailment(both_ways(OLD.claim.text, NEW.claim.text))
    v = ConsistencyChecker(nli, FakeEmbedder(), FactsConfig()).check(NEW.claim, View([old, NEW]))
    assert v.outcome == Outcome.PASS and nli.calls == []


def test_dissimilar_claims_are_not_compared():
    cfg = FactsConfig(consistency_cosine=0.99)
    nli = FakeEntailment(both_ways(OLD.claim.text, NEW.claim.text))
    ConsistencyChecker(nli, FakeEmbedder(), cfg).check(NEW.claim, View([OLD, NEW]))
    assert nli.calls == []


def test_it_applies_to_informal_facts_and_definitions():
    c = ConsistencyChecker(FakeEntailment(), FakeEmbedder(), FactsConfig())
    assert c.applies_to(NEW.claim)
    assert not c.applies_to(NEW.claim.model_copy(update={"type": ClaimType.INFERENCE}))


def test_a_document_reference_is_not_a_shared_name():
    assert proper_names("Bo Chan was born in Oslo (Document 1)") == {"Bo", "Chan", "Oslo"}
    a = rec("b1", "Bo Chan was born in Oslo (Document 1)", ClaimType.FACT, Status.VERIFIED)
    b = rec("b2", "The tower is tall (Document 1)", ClaimType.FACT, Status.UNCHECKED)
    nli = FakeEntailment(both_ways(a.claim.text, b.claim.text))
    v = ConsistencyChecker(nli, FakeEmbedder(), FactsConfig()).check(b.claim, View([a, b]))
    assert v.outcome == Outcome.PASS and nli.calls == []


def test_supported_claims_are_not_compared():
    old = OLD.model_copy(update={"status": Status.SUPPORTED})
    nli = FakeEntailment(both_ways(OLD.claim.text, NEW.claim.text))
    v = ConsistencyChecker(nli, FakeEmbedder(), FactsConfig()).check(NEW.claim, View([old, NEW]))
    assert v.outcome == Outcome.PASS and nli.calls == []
