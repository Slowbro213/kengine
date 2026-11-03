from kengine.eval.scoring import score_em_f1, score_label, score_numeric


def test_score_numeric() -> None:
    assert score_numeric("1234", "$1,234.00").correct
    assert not score_numeric("1234", "1235").correct
    assert not score_numeric("5", None).correct
    assert not score_numeric("5", "five").correct


def test_score_em_f1() -> None:
    s = score_em_f1("the Eiffel Tower", "Eiffel tower.")
    assert s.correct and s.credit == 1.0 and s.detail == "f1=1.000"
    p = score_em_f1("Barack Obama", "Obama")
    assert not p.correct and p.detail == "f1=0.667"
    assert abs(p.credit - 2 / 3) < 1e-9
    none = score_em_f1("x", None)
    assert (none.credit, none.detail) == (0.0, "f1=0.000")


def test_binary_scorers_give_full_or_no_credit() -> None:
    assert score_numeric("5", "5").credit == 1.0
    assert score_numeric("5", "6").credit == 0.0
    assert score_label("True", "False").credit == 0.0


def test_score_label_accepts_unknown_as_uncertain() -> None:
    assert score_label("Uncertain", "unknown").correct
    assert score_label("True", "true").correct
    assert score_label("True", "True.").correct
    assert not score_label("False", "True").correct
