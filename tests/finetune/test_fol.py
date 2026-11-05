import pytest

from kengine.finetune.fol import Story, Unsupported, snake


def convert(text: str, *, conclusion: bool = False) -> list[str]:
    return Story().convert(text, conclusion=conclusion)


def test_a_rule_keeps_the_sentence_direction():
    assert convert("∀x (Dog(x) → Mammal(x))") == ["dog(x) -> mammal(x)"]
    assert convert("∀x (¬Want(x) → ¬Aware(x))") == ["not want(x) -> not aware(x)"]


def test_no_x_are_y():
    assert convert("∀x (Umpire(x) → ¬Player(x))") == ["umpire(x) -> not player(x)"]
    assert convert("¬∃x (Umpire(x) ∧ Player(x))") == ["umpire(x) -> not player(x)"]


def test_facts_and_choices():
    assert convert("WorldTourPlayer(nadal) ∨ GrandSlamChampion(nadal)") == [
        "world_tour_player(nadal) | grand_slam_champion(nadal)"
    ]
    assert convert("¬(Umpire(nadal) ∨ Champion(nadal))") == [
        "not umpire(nadal)",
        "not champion(nadal)",
    ]
    assert convert("Hot(a) ⊕ Cold(a)") == ["hot(a) | cold(a)", "hot(a) -> not cold(a)"]


def test_an_and_in_the_result_splits():
    assert convert("∀x (A(x) → B(x) ∧ C(x))") == ["a(x) -> b(x)", "a(x) -> c(x)"]


def test_some_names_the_individual():
    s = Story()
    assert s.convert("∃x (Pet(x) ∧ BrownSwiss(x))") == ["pet(some_pet)", "brown_swiss(some_pet)"]
    assert s.convert("∃x (Pet(x) ∧ Cute(x))") == ["pet(some_pet_2)", "cute(some_pet_2)"]


def test_a_story_keeps_one_spelling():
    s = Story()
    assert s.convert("ResidentOf(ann, paris)") == ["resident_of(ann, paris)"]
    assert s.convert("Residentof(bob, paris)") == ["resident_of(bob, paris)"]


def test_a_missing_forall_is_meant():
    assert convert("Dog(x) → Mammal(x)") == ["dog(x) -> mammal(x)"]


def test_unbalanced_parentheses_are_repaired():
    assert convert("∀x (Dog(x) → Mammal(x)") == ["dog(x) -> mammal(x)"]


@pytest.mark.parametrize(
    ("text", "conclusion", "why"),
    [
        ("∀x (Dog(x) → x = rex)", False, "equality"),
        ("∀x (Person(x) → ∃y Mother(y, x))", False, '"some" inside "every"'),
        ("∃x (Dog(x) ∧ Barks(x))", True, '"some" in a conclusion'),
        ("¬∀x (Dog(x) → Barks(x))", True, '"some" in a conclusion'),
        ("∃x (Fish(x) → Stings(x))", False, '"some" with an "if" part'),
    ],
)
def test_unsupported(text: str, conclusion: bool, why: str):
    with pytest.raises(Unsupported, match=why):
        convert(text, conclusion=conclusion)


def test_snake():
    assert snake("WildTurkey") == "wild_turkey"
    assert snake("Gould's") == "goulds"
    assert snake("NBA") == "nba"
    assert snake("IsDependentOn") == "is_dependent_on"
