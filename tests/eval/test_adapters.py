from kengine.eval.adapters import folio, gsm8k, hotpotqa


def test_gsm8k_row() -> None:
    item = gsm8k.parse_row({"question": "Q?", "answer": "steps...\n#### 1,234"}, 7)
    assert (item.id, item.gold, item.suite) == ("gsm8k-7", "1234", "gsm8k")
    assert item.answer_format == "Give the final answer as a number."


def test_hotpotqa_row_builds_documents() -> None:
    row: dict[str, object] = {
        "id": "5a8b",
        "question": "Who?",
        "answer": "Ann",
        "context": {"title": ["T1", "T2"], "sentences": [["a.", "b."], ["c."]]},
    }
    item = hotpotqa.parse_row(row, 0)
    assert item.id == "hotpotqa-5a8b" and item.documents == ["T1: a. b.", "T2: c."]
    assert item.answer_format == "Give the final answer as a short phrase."


def test_folio_row() -> None:
    row: dict[str, object] = {
        "example_id": 3,
        "premises": "All A are B.",
        "conclusion": "X is B.",
        "label": "Uncertain",
    }
    item = folio.parse_row(row, 0)
    assert item.id == "folio-3" and item.gold == "Uncertain"
    assert "Premises:\nAll A are B." in item.question and "Conclusion: X is B." in item.question
    assert item.answer_format == "Answer with exactly one of: True, False, Uncertain."


def test_folio_row_with_premise_list() -> None:
    row: dict[str, object] = {
        "premises": ["All A are B.", "X is A."],
        "conclusion": "X is B.",
        "label": "True",
    }
    item = folio.parse_row(row, 5)
    assert item.id == "folio-5" and "All A are B.\nX is A." in item.question
