from kengine.eval.scoring import score_em_f1, score_label, score_numeric


def test_score_numeric() -> None:
    assert score_numeric("1234", "$1,234.00").correct
    assert not score_numeric("1234", "1235").correct
    assert not score_numeric("5", None).correct
    assert not score_numeric("5", "five").correct


def test_score_em_f1() -> None:
    s = score_em_f1("the Eiffel Tower", "Eiffel tower.")
    assert s.correct and s.detail == "f1=1.000"
    p = score_em_f1("Barack Obama", "Obama")
    assert not p.correct and p.detail == "f1=0.667"
    assert score_em_f1("x", None).detail == "f1=0.000"


def test_score_label_accepts_unknown_as_uncertain() -> None:
    assert score_label("Uncertain", "unknown").correct
    assert score_label("True", "true").correct
    assert score_label("True", "True.").correct
    assert not score_label("False", "True").correct
