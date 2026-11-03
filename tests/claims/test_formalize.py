# tests/claims/test_formalize.py
from kengine.claims.formal import JsonStatement
from kengine.claims.formalize import (
    Formalization,
    FormalizedGiven,
    formalize_messages,
    read_formalization,
)


def test_messages_number_the_statements():
    msgs = formalize_messages(
        "Is Rex a mammal?", ["All dogs are mammals.", "Rex is a dog."], "text"
    )
    assert msgs[0].role == "system" and "dog(x) -> mammal(x)" in msgs[0].content
    assert msgs[1].content.endswith("1. All dogs are mammals.\n2. Rex is a dog.")
    assert "## Task\nIs Rex a mammal?" in msgs[1].content


def test_json_syntax_prompt_shows_objects():
    msgs = formalize_messages("g", ["s"], "json")
    assert '{"if": ["dog(x)"], "then": "mammal(x)"}' in msgs[0].content


def test_read_formalization():
    parsed = Formalization(
        givens=[
            FormalizedGiven(given=1, statements=["dog(x) -> mammal(x)"]),
            FormalizedGiven(
                given=2, statements=[JsonStatement.model_validate({"then": "dog(rex)"})]
            ),
            FormalizedGiven(given=3, statements=["dog(rex"]),
            FormalizedGiven(given=4, statements=[]),
            FormalizedGiven(given=9, statements=["a"]),
        ]
    )
    formal, errors = read_formalization(parsed, 4)
    assert {k: v.statements for k, v in formal.items()} == {
        1: ("dog(x) -> mammal(x)",),
        2: ("dog(rex)",),
    }
    assert errors[0].startswith("given 3: unbalanced")
    assert errors[1] == "given 9 doesn't exist"
