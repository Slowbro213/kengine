from kengine.eval.answers import extract_final_answer, normalize_text


def test_extract_final_answer_takes_last_marker() -> None:
    assert extract_final_answer("Final answer: 3\n...\nfinal answer: **7**.") == "7"
    assert extract_final_answer("The answer is 7") is None
    assert extract_final_answer("Final Answer：  Paris ") == "Paris"  # full-width colon


def test_extract_final_answer_tolerates_markdown_marker() -> None:
    assert extract_final_answer("**Final answer:** 42") == "42"
    assert extract_final_answer("**Final answer**: Paris.") == "Paris"
    assert extract_final_answer("Final answer: ") is None


def test_normalize_text() -> None:
    assert normalize_text("The  Eiffel Tower!") == "eiffel tower"


def test_extract_final_answer_mid_line() -> None:
    assert extract_final_answer("So the final answer: 12 apples") == "12 apples"
