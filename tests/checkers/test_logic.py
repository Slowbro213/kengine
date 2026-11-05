from kengine.beliefs.status import Status
from kengine.beliefs.store import BeliefStore
from kengine.checkers import default_checkers
from kengine.checkers.base import CheckerKind, Outcome, Verdict
from kengine.checkers.logic import LogicChecker, answer_label, check_answer
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


def test_a_sum_naming_a_new_quantity_is_left_to_arithmetic():
    s = store()
    b1 = given(s, "10 on Saturday", "sold_sat = 10")
    b2 = given(s, "15 on Sunday", "sold_sun = 15")
    v = check(s, add(s, ClaimType.INFERENCE, [b1, b2], "sold_total = 25"))
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert v.explanation == (
        "can't check: nothing it cites says what sold_total is. If you worked it out, write "
        "the calculation as a computation; if a given defines it, cite that given"
    )


def test_a_wrong_number_for_a_known_quantity_still_fails():
    s = store()
    b1 = given(s, "10 on Saturday", "sold_sat = 10")
    b2 = given(s, "Sunday sold 5 more", "sold_sun = sold_sat + 5")
    v = check(s, add(s, ClaimType.INFERENCE, [b1, b2], "sold_sun = 14"))
    assert (v.outcome, v.status) == (Outcome.FAIL, Status.CONTRADICTED)


MAMMAL = Formal(statements=("mammal(rex)",))


def dogs(s: BeliefStore) -> tuple[str, str]:
    b1 = given(s, "Rex is a dog", "dog(rex)")
    b2 = given(s, "Dogs are mammals", "dog(x) -> mammal(x)")
    return b1, b2


def test_answer_labels():
    assert [answer_label(a) for a in ("True", "**false.**", " uncertain ", "Unknown")] == [
        "True",
        "False",
        "Uncertain",
        "Uncertain",
    ]
    assert answer_label("it's true") is None


def test_true_must_follow_from_what_the_answer_rests_on():
    s = store()
    b1, b2 = dogs(s)
    assert check_answer("True", MAMMAL, [b1, b2], s) is None
    why = check_answer("True", MAMMAL, [b1], s)
    assert why is not None and why.startswith("what it rests on (b1) doesn't show the conclusion")
    assert "mammal(rex) is false" in why
    assert why.endswith("Adding b2 shows it: rest the answer on it too")


def test_an_answer_the_givens_dont_settle_isnt_told_what_to_add():
    s = store()
    b1, _ = dogs(s)
    why = check_answer("True", Formal(statements=("cat(rex)",)), [b1], s)
    assert why is not None
    assert why.endswith("Rest the answer on a claim that states the conclusion")


def test_false_is_told_what_else_rules_the_conclusion_out():
    s = store()
    b1 = given(s, "Rex is a dog", "dog(rex)")
    b2 = given(s, "No dog is a cat", "dog(x) -> not cat(x)")
    why = check_answer("False", Formal(statements=("cat(rex)",)), [b1], s)
    assert why == (
        "what it rests on (b1) doesn't rule the conclusion out. Adding b2 rules it out: rest the "
        "answer on it too"
    )
    assert check_answer("False", Formal(statements=("cat(rex)",)), [b1, b2], s) is None


def test_a_valid_chain_to_somewhere_else_doesnt_answer_the_question():
    s = store()
    _, b2 = dogs(s)
    b3 = add(s, ClaimType.INFERENCE, [b2], "dog(rex) -> mammal(rex)").id
    why = check_answer("True", Formal(statements=("cat(rex)",)), [b3, b2], s)
    assert why is not None and "doesn't show the conclusion" in why


def test_false_must_contradict_what_it_rests_on():
    s = store()
    b1, b2 = dogs(s)
    assert check_answer("False", Formal(statements=("not mammal(rex)",)), [b1, b2], s) is None
    why = check_answer("False", MAMMAL, [b1, b2], s)
    assert why == "the conclusion follows from b1, b2, so it isn't false"
    why = check_answer("True", Formal(statements=("not mammal(rex)",)), [b1, b2], s)
    assert why == "the conclusion is false given b1, b2, not true"


def test_uncertain_must_be_left_open_by_the_givens():
    s = store()
    b1, _ = dogs(s)
    assert check_answer("Uncertain", Formal(statements=("cat(rex)",)), [b1], s) is None
    why = check_answer("Uncertain", MAMMAL, [b1], s)  # every given counts, not just the cited
    assert why == "the conclusion isn't open: it follows from b1, b2"
    given(s, "Rex is not a cat, or he's a lion")
    why = check_answer("Uncertain", Formal(statements=("cat(rex)",)), [b1], s)
    assert why == "can't confirm the conclusion is open: b3 has no formal form"


def test_a_verified_step_on_an_unchecked_claim_isnt_trusted():
    s = store()
    b1, _ = dogs(s)
    b3 = add(s, ClaimType.FACT, [b1], "cat(rex)").id  # unchecked
    b4 = add(s, ClaimType.INFERENCE, [b3], "cat(rex)")
    s.apply_verdict(b4.id, check(s, b4), step=1)
    assert s.get(b4.id).status == Status.VERIFIED
    assert check_answer("Uncertain", Formal(statements=("cat(rex)",)), [b1], s) is None
    v = check(s, add(s, ClaimType.FACT, [], "cat(rex)"))
    assert v.outcome != Outcome.PASS


def test_an_answer_that_isnt_a_label_is_refused():
    s = store()
    b1, b2 = dogs(s)
    assert check_answer("Rex is a mammal", MAMMAL, [b1, b2], s) == (
        "answer with exactly one of: True, False, Uncertain"
    )
