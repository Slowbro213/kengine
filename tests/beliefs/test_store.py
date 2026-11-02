from pathlib import Path

import pytest

from kengine.beliefs.status import Status
from kengine.beliefs.store import BeliefStore, UnknownPremiseError, VerificationGuardError
from kengine.checkers.base import CheckerKind, Outcome, Verdict
from kengine.claims.schema import ClaimType, Computation, Source, SourceKind

SRC = Source(kind=SourceKind.MODEL)


def verdict(status: Status | None, kind: CheckerKind = CheckerKind.DETERMINISTIC) -> Verdict:
    outcome = Outcome.FAIL if status == Status.CONTRADICTED else Outcome.PASS
    return Verdict(
        checker_id="test", checker_kind=kind, outcome=outcome, status=status, explanation="why"
    )


def chain(s: BeliefStore) -> None:
    """b1 fact; b2 inference(b1); b3 inference(b2); b4 fact (independent)."""
    s.add(type=ClaimType.FACT, text="a", premises=[], source=SRC, step=1)
    s.add(type=ClaimType.INFERENCE, text="b", premises=["b1"], source=SRC, step=1)
    s.add(type=ClaimType.INFERENCE, text="c", premises=["b2"], source=SRC, step=1)
    s.add(type=ClaimType.FACT, text="d", premises=[], source=SRC, step=1)


def test_ids_sequential_and_initial_status():
    s = BeliefStore.open(":memory:")
    a = s.add(type=ClaimType.FACT, text="x", premises=[], source=SRC, step=1)
    b = s.add(type=ClaimType.ASSUMPTION, text="y", premises=[], source=SRC, step=1)
    assert (a.id, b.id) == ("b1", "b2")
    assert s.get("b1").status == Status.UNCHECKED and s.get("b2").status == Status.ASSUMED
    assert s.exists("b2") and not s.exists("b3")


def test_add_keeps_computation():
    s = BeliefStore.open(":memory:")
    comp = Computation(expression="2 * 3", claimed_result="6")
    c = s.add(
        type=ClaimType.COMPUTATION, text="2*3", premises=[], source=SRC, step=1, computation=comp
    )
    assert s.get(c.id).claim.computation == comp


def test_unknown_premise_rejected():
    s = BeliefStore.open(":memory:")
    with pytest.raises(UnknownPremiseError):
        s.add(type=ClaimType.INFERENCE, text="x", premises=["b9"], source=SRC, step=1)
    assert s.all() == []


def test_contradiction_propagates_transitively():
    s = BeliefStore.open(":memory:")
    chain(s)
    changes = s.apply_verdict("b1", verdict(Status.CONTRADICTED), step=2)
    assert [c.belief_id for c in changes] == ["b1", "b2", "b3"]
    assert changes[0].verdict is not None and changes[1].verdict is None
    assert s.get("b1").status == Status.CONTRADICTED and s.get("b1").status_reason == "why"
    assert s.get("b3").status == Status.INVALIDATED
    assert s.get("b3").status_reason == "depends on b1 (contradicted)"
    assert s.get("b4").status == Status.UNCHECKED


def test_retract_propagates():
    s = BeliefStore.open(":memory:")
    chain(s)
    changes = s.retract("b2", "wrong", step=3)
    assert [(c.belief_id, c.new) for c in changes] == [
        ("b2", Status.RETRACTED),
        ("b3", Status.INVALIDATED),
    ]
    assert s.get("b3").status_reason == "depends on b2 (retracted)"
    assert s.get("b1").status == Status.UNCHECKED


def test_propagation_skips_terminal_dependents():
    s = BeliefStore.open(":memory:")
    chain(s)
    s.retract("b3", "no", step=2)
    changes = s.apply_verdict("b1", verdict(Status.CONTRADICTED), step=3)
    assert [c.belief_id for c in changes] == ["b1", "b2"]
    assert s.get("b3").status == Status.RETRACTED


def test_dependents_and_premise_closure_order():
    s = BeliefStore.open(":memory:")
    chain(s)
    assert s.dependents("b1") == ["b2", "b3"]
    assert s.premise_closure(["b3"]) == ["b1", "b2", "b3"]
    assert s.premise_closure(["b4", "b2"]) == ["b1", "b2", "b4"]


def test_critic_cannot_verify_or_support():
    s = BeliefStore.open(":memory:")
    chain(s)
    for status in (Status.VERIFIED, Status.SUPPORTED):
        with pytest.raises(VerificationGuardError):
            s.apply_verdict("b1", verdict(status, CheckerKind.CRITIC), 1)
    s.apply_verdict("b1", verdict(Status.CRITIC_ACCEPTED, CheckerKind.CRITIC), 1)
    assert s.get("b1").status == Status.CRITIC_ACCEPTED


def test_terminal_status_is_sticky():
    s = BeliefStore.open(":memory:")
    chain(s)
    s.retract("b4", "no", step=1)
    assert s.apply_verdict("b4", verdict(Status.VERIFIED), 2) == []
    assert s.get("b4").status == Status.RETRACTED
    assert len(s.history("b4")) == 1
    assert len(s.verdicts("b4")) == 1


def test_none_status_verdict_records_without_change():
    s = BeliefStore.open(":memory:")
    chain(s)
    assert s.apply_verdict("b1", verdict(None), 2) == []
    assert s.get("b1").status == Status.UNCHECKED
    assert s.history("b1") == []
    assert [v.checker_id for v in s.verdicts("b1")] == ["test"]


def test_reopen_restores_state(tmp_path: Path):
    path = tmp_path / "beliefs.sqlite"
    s = BeliefStore.open(path)
    chain(s)
    s.apply_verdict("b1", verdict(Status.CONTRADICTED), step=2)
    s.close()
    s2 = BeliefStore.open(path)
    assert [r.status for r in s2.all()] == [
        Status.CONTRADICTED,
        Status.INVALIDATED,
        Status.INVALIDATED,
        Status.UNCHECKED,
    ]
    assert [c.belief_id for c in s2.history("b2")] == ["b2"]
    assert s2.dependents("b1") == ["b2", "b3"]
    assert s2.add(type=ClaimType.FACT, text="e", premises=[], source=SRC, step=3).id == "b5"
