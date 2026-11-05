import json

from kengine.backend.fake import ScriptedBackend, ScriptedReply
from kengine.claims.formalize import Formalization, formalize_messages
from kengine.config import RolesConfig
from kengine.eval.adapters import folio, gsm8k
from kengine.finetune.data import build_pairs, dev_story, distill_gsm8k


def row(
    premises: list[tuple[str, str]], conclusion: tuple[str, str], label: str, story: str = "7"
) -> dict[str, object]:
    return {
        "story_id": story,
        "example_id": "42",
        "premises": "\n".join(p for p, _ in premises),
        "premises-FOL": "\n".join(f for _, f in premises),
        "conclusion": conclusion[0],
        "conclusion-FOL": conclusion[1],
        "label": label,
    }


DOGS = [
    ("All dogs are mammals.", "∀x (Dog(x) → Mammal(x))"),
    ("Rex is a dog.", "Dog(rex)"),
]


def test_a_pair_is_the_agents_call_and_the_expert_answer():
    r = row(DOGS, ("Rex is a mammal.", "Mammal(rex)"), "True")
    pairs, dropped = build_pairs([r])
    assert not dropped
    (p,) = pairs
    item = folio.parse_row(r, 0)
    assert item.conclusion is not None
    want = formalize_messages(item.question, [*item.givens, item.conclusion], "text")
    assert p.messages == [{"role": m.role, "content": m.content} for m in want]
    assert Formalization.model_validate_json(p.completion).givens[2].statements == ["mammal(rex)"]
    assert json.loads(p.completion)["givens"][0] == {
        "given": 1,
        "statements": ["dog(x) -> mammal(x)"],
    }
    assert (p.label, p.item_id) == ("True", "folio-42")


def test_a_row_z3_disagrees_with_is_dropped():
    pairs, dropped = build_pairs([row(DOGS, ("Rex is a mammal.", "Mammal(rex)"), "Uncertain")])
    assert not pairs
    assert dropped == {"z3 says True": 1}


def test_a_statement_about_everything_is_dropped():
    r = row([("Everything is a dog.", "Dog(x)")], ("Rex is a dog.", "Dog(rex)"), "True")
    assert build_pairs([r])[1] == {"says everything": 1}


def test_the_dev_split_is_by_story():
    rows = [row(DOGS, ("Rex is a mammal.", "Mammal(rex)"), "True", story=str(i)) for i in range(50)]
    pairs, _ = build_pairs(rows, dev_percent=10)
    assert {p.split for p in pairs} == {"train", "dev"}
    assert all((p.split == "dev") == dev_story(p.story_id) for p in pairs)


def test_gsm8k_keeps_what_the_model_translates_today():
    item = gsm8k.parse_row(
        {"question": "Ann has 3 apples. Bob has 2 more. How many?", "answer": "#### 5"}, 0
    )
    reply = {
        "givens": [
            {"given": 1, "statements": ["apples(ann) = 3"]},
            {"given": 2, "statements": ["not a valid ( statement"]},
        ]
    }
    backend = ScriptedBackend([ScriptedReply(content=json.dumps(reply))])
    (p,) = distill_gsm8k(backend, RolesConfig().formalize, [item])
    assert json.loads(p.completion)["givens"] == [
        {"given": 1, "statements": ["apples(ann) = 3"]},
        {"given": 2, "statements": []},
        {"given": 3, "statements": []},
    ]
    assert (p.split, p.label) == ("train", "")
