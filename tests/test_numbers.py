from fractions import Fraction

import pytest

from kengine.numbers import decimal_places, parse_number


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1,234", Fraction(1234)),
        ("$30.00", Fraction(30)),
        ("−5", Fraction(-5)),
        ("15%", Fraction(15)),
        ("**42**", Fraction(42)),
        ("42.", Fraction(42)),
        ("18 dollars", Fraction(18)),
        ("3.25", Fraction(13, 4)),
        ("1 234", Fraction(1234)),
        ("1,234,567", Fraction(1234567)),
        ("abc", None),
        ("", None),
    ],
)
def test_parse_number(text: str, expected: Fraction | None) -> None:
    assert parse_number(text) == expected


def test_decimal_places() -> None:
    assert (decimal_places("3.30"), decimal_places("12"), decimal_places("x")) == (2, 0, None)
    assert decimal_places("$1,234.50") == 2
