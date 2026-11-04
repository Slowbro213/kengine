from kengine.claims.formal import Statement, parse_statements
from kengine.logic.entail import consistent, decide


def st(*texts: str) -> list[Statement]:
    return [s for t in texts for s in parse_statements(t)]


def test_modus_ponens_is_entailed_with_core():
    d = decide(st("dog(rex)", "cat(tom)", "dog(x) -> mammal(x)"), st("mammal(rex)"))
    assert d.kind == "entailed"
    assert d.core == (0, 2)


def test_missing_premise_is_open_with_counterexample():
    d = decide(st("dog(x) -> mammal(x)"), st("mammal(rex)"))
    assert d.kind == "open"
    assert ("mammal(rex)", "false") in d.world


def test_converse_error_is_open():
    d = decide(st("dog(x) -> mammal(x)", "mammal(tom)"), st("dog(tom)"))
    assert d.kind == "open"


def test_refuted():
    d = decide(st("dog(rex)", "dog(x) -> not cat(x)"), st("cat(rex)"))
    assert d.kind == "refuted"
    assert d.core == (0, 1)


def test_rule_goal_uses_a_witness():
    d = decide(st("dog(x) -> mammal(x)", "mammal(x) -> animal(x)"), st("dog(x) -> animal(x)"))
    assert d.kind == "entailed"
    d = decide(st("dog(x) -> mammal(x)"), st("mammal(x) -> dog(x)"))
    assert d.kind == "open"
    assert any("_w" in atom for atom, _ in d.world)


def test_disjunction():
    assert decide(st("cat(tom) | dog(tom)", "not cat(tom)"), st("dog(tom)")).kind == "entailed"
    assert decide(st("cat(tom) | dog(tom)"), st("dog(tom)")).kind == "open"


def test_inconsistent_premises_verify_nothing():
    d = decide(st("p", "q", "not p"), st("r"))
    assert d.kind == "inconsistent"
    assert d.core == (0, 2)


def test_universal_premises_contradict_without_constants():
    assert decide(st("p(x)", "not p(x)"), st("q")).kind == "inconsistent"


def test_propositional():
    assert decide(st("rain -> wet", "rain"), st("wet")).kind == "entailed"


def test_linear_arithmetic():
    prem = st("age(bob) = 30", "age(x) > 18 -> adult(x)")
    assert decide(prem, st("adult(bob)")).kind == "entailed"
    assert decide(st("price(a) + 2 = price(b)"), st("price(b) > price(a)")).kind == "entailed"
    open_ = decide(st("age(x) > 18 -> adult(x)"), st("adult(bob)"))
    assert open_.kind == "open"


def test_arity_clash_is_unknown():
    d = decide(st("dog(rex)", "dog(rex, tom)"), st("dog(tom)"))
    assert d.kind == "unknown" and "dog" in d.reason


def test_grounding_blowup_is_unknown_fast():
    constants = [f"likes(c{i}, c{i + 1})" for i in range(20)]
    rule = "likes(x, y) & likes(y, z) & likes(z, w) -> likes(x, w)"
    d = decide(st(*constants, rule), st("likes(c0, c20)"))
    assert d.kind == "unknown" and "too large" in d.reason


def test_empty_premises():
    assert decide([], st("dog(rex) | not dog(rex)")).kind == "entailed"
    assert decide([], st("dog(rex)")).kind == "open"


def test_consistent():
    assert consistent(st("dog(rex)", "dog(x) -> mammal(x)"))
    assert not consistent(st("dog(rex)", "not dog(rex)"))
