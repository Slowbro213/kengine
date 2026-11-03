from kengine.beliefs.status import Status
from kengine.beliefs.store import BeliefStore
from kengine.checkers import default_checkers
from kengine.checkers.base import CheckerKind, Outcome, Verdict
from kengine.checkers.logic import LogicChecker
from kengine.claims.schema import Claim, ClaimType, Formal, FormalMode, Source, SourceKind

TASK = Source(kind=SourceKind.TASK)
MODEL = Source(kind=SourceKind.MODEL)


def given(s: BeliefStore, text: str, *formal: str) -> str:
    f = Formal(statements=formal) if formal else None
    return s.add(type=ClaimType.GIVEN, text=text, premises=[], source=TASK, step=0, formal=f).id


def add(
    s: BeliefStore,
    type_: ClaimType,
    premises: list[str],
    *formal: str,
    mode: FormalMode = "holds",
) -> Claim:
    return s.add(
        type=type_,
        text="claim",
        premises=premises,
        source=MODEL,
        step=1,
        formal=Formal(mode=mode, statements=formal),
    )


def check(s: BeliefStore, c: Claim) -> Verdict:
    return LogicChecker().check(c, s)


def store() -> BeliefStore:
    return BeliefStore.open(":memory:")


def test_valid_inference_is_verified():
    s = store()
    b1 = given(s, "Rex is a dog", "dog(rex)")
    b2 = given(s, "Dogs are mammals", "dog(x) -> mammal(x)")
    v = check(s, add(s, ClaimType.INFERENCE, [b1, b2], "mammal(rex)"))
    assert (v.outcome, v.status) == (Outcome.PASS, Status.VERIFIED)
    assert v.explanation == "follows from b1, b2" and v.evidence == ["b1", "b2"]
    assert v.checker_kind == CheckerKind.DETERMINISTIC


def test_inference_checked_only_against_what_it_cites():
    s = store()
    b1 = given(s, "Rex is a dog", "dog(rex)")
    given(s, "Dogs are mammals", "dog(x) -> mammal(x)")
    v = check(s, add(s, ClaimType.INFERENCE, [b1], "mammal(rex)"))
    assert (v.outcome, v.status) == (Outcome.FAIL, None)
    assert v.explanation.startswith("doesn't follow from b1: they allow a case where")
    assert "mammal(rex) is false" in v.explanation
    assert v.counterexample is not None and "mammal(rex) is false" in v.counterexample


def test_converse_error_fails_with_counterexample():
    s = store()
    b1 = given(s, "Dogs are mammals", "dog(x) -> mammal(x)")
    b2 = given(s, "Tom is a mammal", "mammal(tom)")
    v = check(s, add(s, ClaimType.INFERENCE, [b1, b2], "dog(tom)"))
    assert (v.outcome, v.status) == (Outcome.FAIL, None)
    assert "dog(tom) is false" in v.explanation


def test_refuted_inference_is_contradicted():
    s = store()
    b1 = given(s, "Rex is a dog", "dog(rex)")
    b2 = given(s, "No dog is a cat", "dog(x) -> not cat(x)")
    v = check(s, add(s, ClaimType.INFERENCE, [b1, b2], "cat(rex)"))
    assert (v.outcome, v.status) == (Outcome.FAIL, Status.CONTRADICTED)
    assert v.explanation == "contradicts b1, b2"


def test_informal_premise_makes_it_unknown():
    s = store()
    b1 = given(s, "Rex is a dog", "dog(rex)")
    b2 = given(s, "Dogs are mammals")
    v = check(s, add(s, ClaimType.INFERENCE, [b1, b2], "mammal(rex)"))
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert v.explanation.startswith("can't check: b2 has no formal form, and b1 alone allow")


def test_fact_is_checked_against_the_givens():
    s = store()
    given(s, "Rex is a dog", "dog(rex)")
    given(s, "Dogs are mammals", "dog(x) -> mammal(x)")
    v = check(s, add(s, ClaimType.FACT, [], "mammal(rex)"))
    assert (v.outcome, v.status) == (Outcome.PASS, Status.VERIFIED)
    v = check(s, add(s, ClaimType.FACT, [], "cat(rex)"))
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert v.explanation.startswith("the givens don't settle it")


def test_undetermined_is_verified_only_when_really_open():
    s = store()
    given(s, "Rex is a dog", "dog(rex)")
    given(s, "Dogs are mammals", "dog(x) -> mammal(x)")
    v = check(s, add(s, ClaimType.INFERENCE, [], "barks(rex)", mode="undetermined"))
    assert (v.outcome, v.status) == (Outcome.PASS, Status.VERIFIED)
    v = check(s, add(s, ClaimType.INFERENCE, [], "mammal(rex)", mode="undetermined"))
    assert (v.outcome, v.status) == (Outcome.FAIL, Status.CONTRADICTED)
    assert v.explanation == "it isn't open: it follows from b1, b2"
    v = check(s, add(s, ClaimType.INFERENCE, [], "not mammal(rex)", mode="undetermined"))
    assert v.explanation == "it isn't open: its negation follows from b1, b2"


def test_undetermined_needs_every_given_formalized():
    s = store()
    given(s, "Rex is a dog", "dog(rex)")
    given(s, "Every dog barks")
    v = check(s, add(s, ClaimType.INFERENCE, [], "barks(rex)", mode="undetermined"))
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert v.explanation == "can't confirm it's open: b2 has no formal form"


def test_inconsistent_givens_verify_nothing():
    s = store()
    b1 = given(s, "Rex is a dog", "dog(rex)")
    b2 = given(s, "Rex is not a dog", "not dog(rex)")
    v = check(s, add(s, ClaimType.INFERENCE, [b1, b2], "cat(rex)"))
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert v.explanation == "b1, b2 contradict each other, so this can't be checked"


def test_unknown_reason_is_reported():
    s = store()
    b1 = given(s, "x", "dog(rex)")
    v = check(s, add(s, ClaimType.INFERENCE, [b1], "dog(rex, tom)"))
    assert v.outcome == Outcome.UNKNOWN and v.explanation.startswith("logic check gave up:")


def test_applies_to_formal_facts_and_inferences_only():
    s = store()
    c = add(s, ClaimType.FACT, [], "dog(rex)")
    assert LogicChecker().applies_to(c)
    h = s.add(
        type=ClaimType.HYPOTHESIS,
        text="h",
        premises=[],
        source=MODEL,
        step=1,
        formal=Formal(statements=("dog(rex)",)),
    )
    assert not LogicChecker().applies_to(h)
    plain = s.add(type=ClaimType.FACT, text="f", premises=[], source=MODEL, step=1)
    assert not LogicChecker().applies_to(plain)


def test_default_checkers_include_logic():
    assert [c.id for c in default_checkers()] == [
        "structure",
        "arithmetic",
        "restatement",
        "logic",
    ]
