from kengine.beliefs.status import Status
from kengine.beliefs.store import BeliefStore
from kengine.checkers.base import CheckerKind, Outcome
from kengine.checkers.structure import StructureChecker
from kengine.claims.schema import Claim, ClaimType, Source, SourceKind

SRC = Source(kind=SourceKind.MODEL)


def test_identity_and_scope():
    checker = StructureChecker()
    assert (checker.id, checker.kind) == ("structure", CheckerKind.DETERMINISTIC)
    assert checker.applies_to(Claim(id="b1", type=ClaimType.GIVEN, text="x", source=SRC, step=0))


def test_inference_without_premises_fails_without_status():
    store = BeliefStore.open(":memory:")
    claim = store.add(type=ClaimType.INFERENCE, text="so", premises=[], source=SRC, step=1)
    v = StructureChecker().check(claim, store)
    assert (v.outcome, v.status, v.explanation) == (Outcome.FAIL, None, "inference has no premises")


def test_premise_in_terminal_status_invalidates():
    store = BeliefStore.open(":memory:")
    store.add(type=ClaimType.FACT, text="a", premises=[], source=SRC, step=1)
    store.add(type=ClaimType.FACT, text="ok", premises=[], source=SRC, step=1)
    store.retract("b1", "no", step=1)
    claim = store.add(type=ClaimType.FACT, text="b", premises=["b2", "b1"], source=SRC, step=2)
    v = StructureChecker().check(claim, store)
    assert (v.outcome, v.status) == (Outcome.FAIL, Status.INVALIDATED)
    assert v.explanation == "rests on b1, which is retracted"


def test_well_formed_claim_passes_without_verifying():
    store = BeliefStore.open(":memory:")
    store.add(type=ClaimType.FACT, text="a", premises=[], source=SRC, step=1)
    claim = store.add(type=ClaimType.INFERENCE, text="b", premises=["b1"], source=SRC, step=1)
    v = StructureChecker().check(claim, store)
    assert (v.outcome, v.status) == (Outcome.PASS, None)
