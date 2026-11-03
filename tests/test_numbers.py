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


def test_numbers_in_reads_digits_and_words():
    from fractions import Fraction

    from kengine.numbers import numbers_in

    assert numbers_in("A shop sells pens at $1.25. How much do 12 pens cost?") == {
        Fraction("1.25"),
        Fraction(12),
    }
    assert numbers_in("Three dozen eggs, 1,000 hens, pages 5-6") == {
        Fraction(3),
        Fraction(12),
        Fraction(1000),
        Fraction(5),
        Fraction(6),
    }
    assert numbers_in("no numbers here") == set()
