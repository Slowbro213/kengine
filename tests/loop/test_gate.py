import pytest

from kengine.beliefs.status import Status
from kengine.beliefs.store import BeliefStore
from kengine.checkers.base import CheckerKind, Outcome, Verdict
from kengine.claims.schema import ClaimType, Formal, Source, SourceKind
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
    s.add(type=ClaimType.INFERENCE, text="8", premises=["b7"], source=SRC, step=1)
    _set(s, "b8", Status.VERIFIED)
    assert evaluate_gate(["b8"], s, GatePolicy.STRICT).allowed


def test_an_answer_resting_only_on_givens_is_blocked(s: BeliefStore):
    """The givens are the task's premises; an answer needs a claim that states it."""
    d = evaluate_gate(["b7"], s, GatePolicy.STANDARD)
    assert not d.allowed and d.blocking == []
    assert d.reasons == [
        "final answer rests only on the task's givens; rest it on the claim that states it"
    ]
    assert evaluate_gate(["b7"], s, GatePolicy.PERMISSIVE).allowed


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


def _dogs() -> BeliefStore:
    """b1 "Rex is a dog", b2 "Dogs are mammals" (givens), b3 "Rex is a dog" (verified)."""
    store = BeliefStore.open(":memory:")
    task = Source(kind=SourceKind.TASK)
    for text, form in (("Rex is a dog", "dog(rex)"), ("Dogs are mammals", "dog(x) -> mammal(x)")):
        store.add(
            type=ClaimType.GIVEN,
            text=text,
            premises=[],
            source=task,
            step=0,
            formal=Formal(statements=(form,)),
        )
    store.add(
        type=ClaimType.INFERENCE,
        text="Rex is a dog",
        premises=["b1"],
        source=SRC,
        step=1,
        formal=Formal(statements=("dog(rex)",)),
    )
    _set(store, "b3", Status.VERIFIED)
    return store


MAMMAL = Formal(statements=("mammal(rex)",))


def test_an_answer_must_show_the_conclusion():
    s = _dogs()
    d = evaluate_gate(["b3"], s, GatePolicy.STANDARD, "True", MAMMAL)
    assert not d.allowed and d.blocking == []
    assert d.reasons[0].startswith("what it rests on (b1, b3) doesn't show the conclusion")
    assert evaluate_gate(["b3"], s, GatePolicy.STANDARD).allowed  # no conclusion: no check
    assert evaluate_gate(["b3"], s, GatePolicy.PERMISSIVE, "True", MAMMAL).allowed


def test_with_a_conclusion_an_answer_may_rest_on_givens():
    """The answer check proves the conclusion from them, so something was checked."""
    s = _dogs()
    assert evaluate_gate(["b1", "b2"], s, GatePolicy.STRICT, "True", MAMMAL).allowed
    d = evaluate_gate(["b1", "b2"], s, GatePolicy.STANDARD, "False", MAMMAL)
    assert d.reasons == ["the conclusion follows from b1, b2, so it isn't false"]
