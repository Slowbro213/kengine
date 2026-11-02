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


def decimal_places(text: str) -> int | None:
    """Digits after the decimal point in the first number of `text`."""
    m = _NUMBER_RE.search(_clean(text))
    if m is None:
        return None
    return len(m.group(1)) if m.group(1) else 0
