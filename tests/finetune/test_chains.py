import pytest

from kengine.claims.extraction import Extraction
from kengine.finetune.chains import (
    NoChain,
    Step,
    build_chain_pairs,
    check_chain,
    prove,
    row_pairs,
)
from kengine.prompts import load_prompt

DOGS = [["dog(rex)"], ["dog(x) -> mammal(x)"], ["mammal(x) -> animal(x)"]]


def test_a_true_conclusion_is_reached_one_rule_at_a_time():
    steps = prove(DOGS, ["animal(rex)"], "True")
    assert steps == [Step(("mammal(rex)",), ("b2", "b1")), Step(("animal(rex)",), ("b3", "1"))]
    assert check_chain(DOGS, ["animal(rex)"], "True", steps) is None


def test_only_the_steps_the_conclusion_needs_are_kept():
    givens = [*DOGS, ["cat(tom)"], ["cat(x) -> purrs(x)"]]
    steps = prove(givens, ["mammal(rex)"], "True")
    assert steps == [Step(("mammal(rex)",), ("b2", "b1"))]


def test_a_false_conclusion_ends_in_its_negation():
    givens = [["dog(rex)"], ["dog(x) -> not cat(x)"]]
    steps = prove(givens, ["cat(rex)"], "False")
    assert steps == [Step(("not cat(rex)",), ("b2", "b1"))]
    assert check_chain(givens, ["cat(rex)"], "False", steps) is None


def test_a_false_rule_is_broken_by_a_case():
    givens = [["dog(rex)"], ["dog(x) -> not barks(x)"]]
    steps = prove(givens, ["dog(x) -> barks(x)"], "False")
    assert steps[-1].statements == ("dog(rex)", "not barks(rex)")
    assert check_chain(givens, ["dog(x) -> barks(x)"], "False", steps) is None


def test_a_rule_conclusion_cites_what_shows_it():
    steps = prove(DOGS, ["dog(x) -> animal(x)"], "True")
    assert steps == [Step(("dog(x) -> animal(x)",), ("b2", "b3"))]
    assert check_chain(DOGS, ["dog(x) -> animal(x)"], "True", steps) is None


def test_uncertain_is_one_step_over_every_given():
    steps = prove(DOGS, ["barks(rex)"], "Uncertain")
    assert steps == [Step(("barks(rex)",), ("b1", "b2", "b3"), "undetermined")]
    assert check_chain(DOGS, ["barks(rex)"], "Uncertain", steps) is None


def test_a_label_the_givens_dont_give_has_no_chain():
    with pytest.raises(NoChain):
        prove(DOGS, ["barks(rex)"], "True")


def test_the_checker_refuses_a_bad_step():
    bad = [Step(("animal(rex)",), ("b1",))]
    assert check_chain(DOGS, ["animal(rex)"], "True", bad) == (
        "step 1: doesn't follow from b1: they allow a case where animal(rex) is false; cite "
        "what else it needs or fix the step"
    )


def test_a_rows_pairs_are_the_agents_calls_and_a_reply_it_accepts():
    r = {
        "story_id": "7",
        "example_id": "42",
        "premises": "Rex is a dog.\nAll dogs are mammals.\nAll mammals are animals.",
        "conclusion": "Rex is an animal.",
        "label": "True",
    }
    think, extract = row_pairs(r, 0, [*DOGS, ["animal(rex)"]], "train", budget=14000)
    assert (think.role, think.thinking, extract.thinking) == ("think", True, False)
    assert think.messages[0]["content"] == load_prompt("think")
    assert (
        "b1 [unchecked] (given) Rex is a dog. :: formal: dog(rex)" in think.messages[1]["content"]
    )
    reply = think.completion.split("</think>\n\n")[1]
    assert reply == (
        "Working:\n1. mammal(rex) (from b2 and b1).\n2. Rex is an animal (from b3 and 1).\n\n"
        "Final answer: True\nRests on: 2"
    )
    assert extract.messages[1]["content"].endswith(reply)
    got = Extraction.model_validate_json(extract.completion)
    assert [c.premises for c in got.claims] == [["b2", "b1"], ["b3", "c1"]]
    assert got.action.preconditions == ["c2"] and got.action.answer == "True"


def test_build_chain_pairs_counts_what_it_drops():
    rows = [
        {
            "story_id": "7",
            "example_id": str(i),
            "premises": "Rex is a dog.\nAll dogs are mammals.",
            "premises-FOL": "Dog(rex)\n∀x (Dog(x) → Mammal(x))",
            "conclusion": "Rex is a mammal.",
            "conclusion-FOL": fol,
            "label": "True",
        }
        for i, fol in enumerate(["Mammal(rex)", "∀x Mammal(x)"])
    ]
    pairs, dropped = build_chain_pairs(rows, budget=14000)
    assert [(p.item_id, p.role) for p in pairs] == [("folio-0", "think"), ("folio-0", "extract")]
    assert dropped == {"doesn't convert": 1}


def test_uncertain_train_rows_can_be_thinned_out():
    rows = [
        {
            "story_id": "7",
            "example_id": str(i),
            "premises": "Rex is a dog.\nAll dogs are mammals.",
            "premises-FOL": "Dog(rex)\n∀x (Dog(x) → Mammal(x))",
            "conclusion": "Rex barks.",
            "conclusion-FOL": "Barks(rex)",
            "label": "Uncertain",
        }
        for i in range(40)
    ]
    pairs, dropped = build_chain_pairs(rows, budget=14000, uncertain_percent=25)
    kept = len(pairs) // 2
    assert 0 < kept < 20 and dropped == {"Uncertain: thinned out": 40 - kept}
    assert build_chain_pairs(rows, budget=14000, uncertain_percent=0)[0] == []
