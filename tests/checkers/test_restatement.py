from kengine.beliefs.status import Status
from kengine.beliefs.store import BeliefStore
from kengine.checkers.arithmetic import ArithmeticChecker
from kengine.checkers.base import CheckerKind, Outcome
from kengine.checkers.restatement import RestatementChecker
from kengine.claims.schema import Claim, ClaimType, Computation, Source, SourceKind

TASK = Source(kind=SourceKind.TASK)
MODEL = Source(kind=SourceKind.MODEL)


def store_with(*givens: str) -> BeliefStore:
    store = BeliefStore.open(":memory:")
    for g in givens:
        store.add(type=ClaimType.GIVEN, text=g, premises=[], source=TASK, step=0)
    return store


def fact(store: BeliefStore, text: str, premises: list[str]) -> Claim:
    return store.add(type=ClaimType.FACT, text=text, premises=premises, source=MODEL, step=1)


def test_identity_and_scope():
    checker = RestatementChecker()
    assert (checker.id, checker.kind) == ("restatement", CheckerKind.EVIDENCE)
    s = store_with("A shop sells pens at $1.25.")
    assert checker.applies_to(fact(s, "Each pen costs $1.25", ["b1"]))
    assert not checker.applies_to(fact(s, "Pens are sold here", ["b1"]))


def test_cited_given_with_the_numbers_supports():
    s = store_with("A shop sells pens at $1.25.", "How much do 12 pens cost?")
    v = RestatementChecker().check(fact(s, "Each pen costs $1.25", ["b1"]), s)
    assert (v.outcome, v.status, v.evidence) == (Outcome.PASS, Status.SUPPORTED, ["b1"])
    assert v.explanation == "restates b1"


def test_number_words_count():
    s = store_with("She eats three eggs every morning.")
    v = RestatementChecker().check(fact(s, "She eats 3 eggs a day", ["b1"]), s)
    assert v.status == Status.SUPPORTED


def test_number_missing_from_cited_given_is_inconclusive():
    s = store_with("A shop sells pens at $1.25.", "How much do 12 pens cost?")
    v = RestatementChecker().check(fact(s, "Each pen costs $12", ["b1"]), s)
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert v.explanation.startswith("12 isn't in the task's givens")


def test_worked_out_number_points_to_a_calculation():
    s = store_with("Greg walks with his 3 younger siblings.")
    v = RestatementChecker().check(fact(s, "4 people share the money", ["b1"]), s)
    assert "`<expression> = 4`" in v.explanation


def test_non_given_premises_are_not_evidence():
    s = store_with()
    other = fact(s, "A pen costs $2", [])
    v = RestatementChecker().check(fact(s, "Each pen costs $2", [other.id]), s)
    assert (v.outcome, v.status, v.evidence) == (Outcome.UNKNOWN, None, [])


def test_verified_computation_is_citable():
    s = store_with("Ann buys 3 boxes of 12 pens.")
    comp = s.add(
        type=ClaimType.COMPUTATION,
        text="3 * 12 = 36",
        premises=["b1"],
        computation=Computation(expression="3 * 12", claimed_result="36"),
        source=MODEL,
        step=1,
    )
    s.apply_verdict(comp.id, ArithmeticChecker().check(comp, s), 1)
    v = RestatementChecker().check(fact(s, "Ann has 36 pens", [comp.id]), s)
    assert v.status == Status.SUPPORTED


def test_uncited_fact_needs_one_given_with_its_numbers_and_a_shared_word():
    s = store_with("A shop sells pens at $1.25.", "Bob has 12 apples.")
    checker = RestatementChecker()
    assert checker.check(fact(s, "A pen costs $1.25", []), s).evidence == ["b1"]
    # 12 is in b2, but b2 is about apples, not pens.
    assert checker.check(fact(s, "There are 12 pens", []), s).status is None


def test_unit_conversion_counts_when_both_units_are_named():
    s = store_with("How many badges does Wendy earn in a 1 year time frame?")
    v = RestatementChecker().check(fact(s, "The time frame is 12 months", ["b1"]), s)
    assert (v.outcome, v.status) == (Outcome.PASS, Status.SUPPORTED)
    assert v.explanation == "restates b1, with 1 year = 12 months"


def test_unit_conversion_needs_the_other_unit():
    s = store_with("Claire earns 1 badge every month.")
    v = RestatementChecker().check(fact(s, "She earns 12 badges in 12 months", ["b1"]), s)
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)


def test_unit_conversion_backs_an_uncited_fact():
    s = store_with("A movie lasts 2 hours.")
    v = RestatementChecker().check(fact(s, "An hour has 60 minutes and the movie 2 hours", []), s)
    assert (v.outcome, v.status, v.evidence) == (Outcome.PASS, Status.SUPPORTED, ["b1"])


def test_cent_does_not_match_percent():
    s = store_with("Prices rose 5 percent in dollars.")
    v = RestatementChecker().check(fact(s, "Each dollar is 100 percent", ["b1"]), s)
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
