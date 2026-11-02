from kengine.beliefs.status import Status
from kengine.beliefs.store import BeliefStore
from kengine.checkers.arithmetic import ArithmeticChecker
from kengine.checkers.base import CheckerKind, Outcome
from kengine.claims.schema import Claim, ClaimType, Computation, Source, SourceKind

SRC = Source(kind=SourceKind.MODEL)


def computation(expression: str, claimed: str) -> Claim:
    return Claim(
        id="b1",
        type=ClaimType.COMPUTATION,
        text=f"{expression} = {claimed}",
        computation=Computation(expression=expression, claimed_result=claimed),
        source=SRC,
        step=1,
    )


def check(expression: str, claimed: str):
    return ArithmeticChecker().check(computation(expression, claimed), BeliefStore.open(":memory:"))


def test_identity_and_scope():
    checker = ArithmeticChecker()
    assert (checker.id, checker.kind) == ("arithmetic", CheckerKind.DETERMINISTIC)
    assert checker.applies_to(computation("1", "1"))
    assert not checker.applies_to(Claim(id="b1", type=ClaimType.FACT, text="x", source=SRC, step=1))


def test_match_verifies():
    v = check("12*7", "84")
    assert (v.outcome, v.status, v.explanation) == (Outcome.PASS, Status.VERIFIED, "`12*7` = 84")


def test_mismatch_contradicts():
    v = check("12*7", "82")
    assert (v.outcome, v.status) == (Outcome.FAIL, Status.CONTRADICTED)
    assert v.explanation == "`12*7` evaluates to 84, not 82"
    assert v.counterexample == "84"


def test_unevaluable_is_unknown():
    v = check("x*2", "4")
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert v.explanation.startswith("could not evaluate:")
    v = check("2*2", "four")
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert v.explanation.startswith("could not evaluate:")


def test_rounded_claim_passes():
    v = check("10/3", "3.3333333")
    assert (v.outcome, v.status) == (Outcome.PASS, Status.VERIFIED)
    assert v.explanation == "`10/3` = 3.333333"
