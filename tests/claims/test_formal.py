from fractions import Fraction

import pytest

from kengine.claims.formal import (
    Atom,
    Compare,
    FormalSyntaxError,
    JsonStatement,
    canonical,
    constants_of,
    is_variable,
    json_statement,
    json_statement_text,
    parse_statements,
    variables_of,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("dog(rex)", ["dog(rex)"]),
        ("raining", ["raining"]),
        ("not dog(rex)", ["not dog(rex)"]),
        ("dog(x) & not small(x) -> barks(x)", ["dog(x) & not small(x) -> barks(x)"]),
        ("cat(tom) | dog(tom)", ["cat(tom) | dog(tom)"]),
        ("pet(x) -> cat(x) | dog(x)", ["pet(x) -> cat(x) | dog(x)"]),
        ("likes(ann,bob)", ["likes(ann, bob)"]),
        ("a -> b & c", ["a -> b", "a -> c"]),
        ("a & b", ["a", "b"]),
        ("(a & b) & c -> d", ["a & b & c -> d"]),
        ("age(bob) >= 18", ["age(bob) >= 18"]),
        ("price(a) + 2 = price(b)", ["price(a) + 2 = price(b)"]),
        ("2*age(bob) - 3 < 0.5", ["2 * age(bob) - 3 < 1/2"]),
        ("age(x) > 18 -> adult(x)", ["age(x) > 18 -> adult(x)"]),
    ],
)
def test_canonical_forms(text: str, expected: list[str]):
    assert canonical(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("∀x (Dog(x) → Mammal(x))", ["dog(x) -> mammal(x)"]),
        ("forall x: dog(x) => mammal(x)", ["dog(x) -> mammal(x)"]),
        ("all x. Dog(x) -> Mammal(x)", ["dog(x) -> mammal(x)"]),
        ("¬Cat(Tom) ∧ Dog(Tom)", ["not cat(tom)", "dog(tom)"]),
        ("~cat(tom)", ["not cat(tom)"]),
        ("!cat(tom)", ["not cat(tom)"]),
        ("cat(tom) or dog(tom)", ["cat(tom) | dog(tom)"]),
        ("dog(rex) and mammal(rex).", ["dog(rex)", "mammal(rex)"]),
        ("age(bob) == 30", ["age(bob) = 30"]),
        ("not (dog(rex))", ["not dog(rex)"]),
        ("not not dog(rex)", ["dog(rex)"]),
    ],
)
def test_first_order_spellings_are_normalized(text: str, expected: list[str]):
    assert canonical(text) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("a -> b -> c", "more than one"),
        ("a | b -> c", "conditions"),
        ("a -> b & c | d", "both"),
        ("dog(f(x))", "arguments must be names"),
        ("x", "variable"),
        ("x(rex)", "variable"),
        ("age(x) * age(y) > 3", "linear"),
        ("x > 3", "individual"),
        ("1 / 0 > 3", "zero"),
        ("dog(rex", "unbalanced"),
        ("exists x: dog(x)", "exist"),
        ("a < b < c", "one comparison"),
        ("dog(rex) @", "unexpected"),
        ("a" * 301, "too long"),
    ],
)
def test_errors_are_specific(text: str, message: str):
    with pytest.raises(FormalSyntaxError, match=message):
        parse_statements(text)


def test_variables_and_constants():
    (s,) = parse_statements("likes(x, ann) & likes(y, x) -> friends(x, y)")
    assert variables_of(s) == ["x", "y"]
    assert constants_of(s) == {"ann"}
    assert is_variable("x2") and not is_variable("rex") and not is_variable("xavier")


def test_linear_terms_are_merged():
    (s,) = parse_statements("age(bob) + age(bob) - 2 = 1")
    lit = s.head[0]
    assert str(lit) == "2 * age(bob) - 2 = 1"
    assert isinstance(lit.atom, Compare)
    assert lit.atom.left.terms == ((Atom("age", ("bob",)), Fraction(2)),)


def test_json_statements_round_trip():
    js = JsonStatement.model_validate({"if": ["dog(x)", "not small(x)"], "then": "barks(x)"})
    assert json_statement_text(js) == "dog(x) & not small(x) -> barks(x)"
    assert json_statement_text(JsonStatement.model_validate({"then": "dog(rex)"})) == "dog(rex)"
    assert json_statement("pet(x) -> cat(x) | dog(x)").model_dump(by_alias=True) == {
        "if": ["pet(x)"],
        "then": "cat(x) | dog(x)",
    }
