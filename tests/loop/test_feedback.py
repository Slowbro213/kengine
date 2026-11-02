from kengine.beliefs.status import Status
from kengine.beliefs.store import StatusChange
from kengine.checkers.base import CheckerKind, Outcome, Verdict
from kengine.loop.feedback import build_feedback
from kengine.loop.gate import GateDecision


def verdict(outcome: Outcome, status: Status | None, explanation: str) -> Verdict:
    return Verdict(
        checker_id="arithmetic",
        checker_kind=CheckerKind.DETERMINISTIC,
        outcome=outcome,
        status=status,
        explanation=explanation,
    )


CONTRA = verdict(Outcome.FAIL, Status.CONTRADICTED, "`12*7` evaluates to 84, not 82")
CHANGES = [
    StatusChange(
        belief_id="b1",
        old=Status.UNCHECKED,
        new=Status.CONTRADICTED,
        reason=CONTRA.explanation,
        verdict=CONTRA,
        step=1,
    ),
    StatusChange(
        belief_id="b2",
        old=Status.UNCHECKED,
        new=Status.INVALIDATED,
        reason="depends on b1 (contradicted)",
        verdict=None,
        step=1,
    ),
    StatusChange(
        belief_id="b3",
        old=Status.UNCHECKED,
        new=Status.INVALIDATED,
        reason="depends on b1 (contradicted)",
        verdict=None,
        step=1,
    ),
]
GATE = GateDecision(
    allowed=False, blocking=["b4"], acted_on_assumptions=[], reasons=["b4 is unchecked"]
)


def test_feedback_priority_and_cap():
    verdicts = [
        ("b1", CONTRA),
        ("b5", verdict(Outcome.UNKNOWN, None, "could not evaluate: x")),
        ("b6", verdict(Outcome.PASS, Status.VERIFIED, "fine")),
    ]
    issues = ["issue one", "issue two", "issue three"]
    fb = build_feedback(CHANGES, verdicts, issues, GATE, max_items=5)
    assert fb == [
        "final answer blocked: b4 is unchecked",
        "b1 contradicted: `12*7` evaluates to 84, not 82",
        "2 claim(s) invalidated because b1 is contradicted",
        "b5: could not evaluate: x",
        "issue one",
    ]


def test_fail_without_change_and_structure_invalidation():
    fail = verdict(Outcome.FAIL, None, "inference has no premises")
    structural = verdict(Outcome.FAIL, Status.INVALIDATED, "rests on b1, which is retracted")
    change = StatusChange(
        belief_id="b7",
        old=Status.UNCHECKED,
        new=Status.INVALIDATED,
        reason=structural.explanation,
        verdict=structural,
        step=2,
    )
    fb = build_feedback([change], [("b8", fail), ("b7", structural)], [], None)
    assert fb == [
        "1 claim(s) invalidated because b1 is retracted",
        "b8: inference has no premises",
    ]


def test_allowed_gate_and_passes_say_nothing():
    gate = GateDecision(allowed=True, blocking=[], acted_on_assumptions=["b2"], reasons=[])
    ok = verdict(Outcome.PASS, Status.VERIFIED, "fine")
    assert build_feedback([], [("b1", ok)], [], gate) == []
