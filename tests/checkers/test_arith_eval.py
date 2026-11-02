import time
from fractions import Fraction

import pytest

from kengine.checkers.arith_eval import ArithError, format_number, results_match, safe_eval


@pytest.mark.parametrize(
    "expr,value",
    [
        ("12*7", 84),
        ("1,000 + 234", 1234),
        ("200*15%", 30),
        ("2^10", 1024),
        ("7 × 6", 42),
        ("(3 - −2) ÷ 5", 1),
        ("max(3, 9) - abs(-1)", 8),
        ("0.1 + 0.2", Fraction(3, 10)),
        ("7 // 2 + 7 % 2", 4),
        ("2 ** -2", Fraction(1, 4)),
        ("round(2.567, 2)", Fraction("2.57")),
        ("min(1, 2, 3)", 1),
    ],
)
def test_safe_eval_values(expr: str, value: Fraction | int):
    assert safe_eval(expr) == value


def test_exact_results_are_fractions():
    assert isinstance(safe_eval("10 / 4"), Fraction)
    assert isinstance(safe_eval("round(7.5)"), Fraction)


def test_sqrt_and_fractional_powers_are_floats():
    assert safe_eval("sqrt(16)") == 4.0 and isinstance(safe_eval("sqrt(16)"), float)
    assert safe_eval("8 ** (1/3)") == pytest.approx(2.0)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "x + 1",
        "(1).real",
        "[1, 2]",
        "True + 1",
        "1/0",
        "lambda: 1",
        "5 % 0",
        "sqrt(-1)",
        "(-8) ** 0.5",
        "0 ** -1",
        "round(x=1)",
        "max()",
        "1 +",
        "1 < 2",
        "'a' * 3",
        "1j",
    ],
)
def test_safe_eval_rejects(expr: str):
    with pytest.raises(ArithError):
        safe_eval(expr)


def test_pathological_inputs_fail_fast():
    start = time.monotonic()
    for expr in ["9**9**9", "2**100000", "1+" * 50_000 + "1", "(" * 400 + "1" + ")" * 400]:
        with pytest.raises(ArithError):
            safe_eval(expr)
    assert time.monotonic() - start < 1.0


def test_large_but_bounded_power_is_allowed():
    assert safe_eval("2 ** 1000") == 2**1000
    with pytest.raises(ArithError, match="too large"):
        safe_eval("(10 ** 100) ** 200")


@pytest.mark.parametrize(
    "actual,claimed,ok",
    [
        (Fraction(10, 3), "3.33", True),
        (Fraction(10, 3), "3.34", False),
        (Fraction(10, 3), "3", False),
        (Fraction(84), "84", True),
        (Fraction(84), "82", False),
        (Fraction(30), "$30.00", True),
        (Fraction(1234), "1,234", True),
        (Fraction(-5), "−5", True),
        (Fraction(84), "eighty", None),
        (4.0, "4", True),
    ],
)
def test_results_match(actual: Fraction | float, claimed: str, ok: bool | None):
    assert results_match(actual, claimed) is ok


@pytest.mark.parametrize(
    "x,text",
    [
        (Fraction(84), "84"),
        (Fraction(10, 3), "3.333333"),
        (Fraction(1, 4), "0.25"),
        (Fraction(-1, 10**9), "0"),
        (2.5, "2.5"),
        (4.0, "4"),
    ],
)
def test_format_number(x: Fraction | float, text: str):
    assert format_number(x) == text
