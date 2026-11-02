import pytest

from kengine.beliefs.status import Status
from kengine.beliefs.store import BeliefStore
from kengine.checkers.base import CheckerKind, Outcome, Verdict
from kengine.claims.schema import ClaimType, Source, SourceKind
from kengine.loop.gate import evaluate_gate
from kengine.loop.profiles import GatePolicy

SRC = Source(kind=SourceKind.MODEL)


def _set(s: BeliefStore, id: str, status: Status) -> None:
    v = Verdict(
        checker_id="t",
        checker_kind=CheckerKind.DETERMINISTIC,
        outcome=Outcome.PASS,
        status=status,
        explanation="",
    )
    s.apply_verdict(id, v, step=1)


@pytest.fixture
def s() -> BeliefStore:
    """b1 fact verified, b2 assumption, b3 inference(b1, b2) verified, b4 fact unchecked,
    b5 inference(b4) verified, b6 fact contradicted, b7 given (unchecked)."""
    store = BeliefStore.open(":memory:")
    add = store.add
    add(type=ClaimType.FACT, text="1", premises=[], source=SRC, step=1)
    add(type=ClaimType.ASSUMPTION, text="2", premises=[], source=SRC, step=1)
    add(type=ClaimType.INFERENCE, text="3", premises=["b1", "b2"], source=SRC, step=1)
    add(type=ClaimType.FACT, text="4", premises=[], source=SRC, step=1)
    add(type=ClaimType.INFERENCE, text="5", premises=["b4"], source=SRC, step=1)
    add(type=ClaimType.FACT, text="6", premises=[], source=SRC, step=1)
    add(type=ClaimType.GIVEN, text="7", premises=[], source=Source(kind=SourceKind.TASK), step=0)
    for id in ("b1", "b3", "b5"):
        _set(store, id, Status.VERIFIED)
    _set(store, "b6", Status.CONTRADICTED)
    return store


def test_standard_allows_assumed_premise_and_records_it(s: BeliefStore):
    d = evaluate_gate(["b3"], s, GatePolicy.STANDARD)
    assert d.allowed and d.acted_on_assumptions == ["b2"] and d.blocking == []


def test_strict_blocks_assumed_premise(s: BeliefStore):
    d = evaluate_gate(["b3"], s, GatePolicy.STRICT)
    assert not d.allowed and d.blocking == ["b2"] and d.reasons == ["b2 is assumed"]


def test_given_is_acceptable_even_strict(s: BeliefStore):
    assert evaluate_gate(["b7"], s, GatePolicy.STRICT).allowed


def test_unchecked_premise_blocks_verified_conclusion_in_standard(s: BeliefStore):
    d = evaluate_gate(["b5"], s, GatePolicy.STANDARD)
    assert not d.allowed and d.blocking == ["b4"] and d.reasons == ["b4 is unchecked"]


def test_terminal_blocks_even_permissive(s: BeliefStore):
    d = evaluate_gate(["b6"], s, GatePolicy.PERMISSIVE)
    assert not d.allowed and d.reasons == ["b6 is contradicted"]


def test_permissive_records_unchecked(s: BeliefStore):
    d = evaluate_gate(["b5", "b3"], s, GatePolicy.PERMISSIVE)
    assert d.allowed and d.acted_on_assumptions == ["b2", "b4"]


def test_empty_preconditions(s: BeliefStore):
    d = evaluate_gate([], s, GatePolicy.STANDARD)
    assert not d.allowed and d.reasons == ["final answer cites no supporting claims"]
    assert evaluate_gate([], s, GatePolicy.PERMISSIVE).allowed


def test_unknown_id_blocks(s: BeliefStore):
    d = evaluate_gate(["b99"], s, GatePolicy.PERMISSIVE)
    assert not d.allowed and d.blocking == ["b99"] and d.reasons == ["b99: unknown claim"]
