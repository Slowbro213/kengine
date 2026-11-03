"""Number parsing shared by answer scoring and the arithmetic checker."""

from __future__ import annotations

import re
from fractions import Fraction

_MARKUP_RE = re.compile(r"[*_$€£]")
# A thousands separator: `,`, a space or a thin space between a digit and exactly three digits.
_THOUSANDS_RE = re.compile(r"(?<=\d)[,   ](?=\d{3}(?!\d))")
_NUMBER_RE = re.compile(r"-?\d+(?:\.(\d+))?")


def _clean(text: str) -> str:
    text = _MARKUP_RE.sub("", text).replace("−", "-")
    return _THOUSANDS_RE.sub("", text)


def parse_number(text: str) -> Fraction | None:
    """The first number in `text`, tolerant of `1,234`, `$30.00`, `−5`, `15%`, `**42**`, `42.`."""
    m = _NUMBER_RE.search(_clean(text))
    if m is None:
        return None
    return Fraction(m.group(0))


_WORD_NUMBERS = (
    {
        w: Fraction(i)
        for i, w in enumerate(
            "zero one two three four five six seven eight nine ten eleven twelve thirteen "
            "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
        )
    }
    | {
        w: Fraction(10 * i)
        for i, w in enumerate("thirty forty fifty sixty seventy eighty ninety".split(), 3)
    }
    | {
        "dozen": Fraction(12),
        "hundred": Fraction(100),
        "thousand": Fraction(1000),
    }
)
_WORD_RE = re.compile(r"\b(" + "|".join(_WORD_NUMBERS) + r")\b", re.IGNORECASE)


def numbers_in(text: str) -> set[Fraction]:
    """Every number in `text`: digits (as `parse_number` reads them) and the common number
    words (`three`, `twenty`, `dozen`). Hyphens count as separators, not minus signs, so
    `5-6` gives 5 and 6."""
    cleaned = _clean(text)
    found = {abs(Fraction(m.group(0))) for m in _NUMBER_RE.finditer(cleaned)}
    return found | {_WORD_NUMBERS[m.group(1).lower()] for m in _WORD_RE.finditer(cleaned)}


def decimal_places(text: str) -> int | None:
    """Digits after the decimal point in the first number of `text`."""
    m = _NUMBER_RE.search(_clean(text))
    if m is None:
        return None
    return len(m.group(1)) if m.group(1) else 0
