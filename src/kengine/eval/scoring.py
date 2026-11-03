"""Per-suite answer scorers."""

from __future__ import annotations

from collections import Counter

from kengine.eval.answers import normalize_text
from kengine.eval.types import ItemScore
from kengine.numbers import parse_number


def _binary(correct: bool, detail: str = "") -> ItemScore:
    return ItemScore(correct=correct, credit=float(correct), detail=detail)


def score_numeric(gold: str, answer: str | None) -> ItemScore:
    if answer is None:
        return _binary(False, "no answer")
    g, a = parse_number(gold), parse_number(answer)
    if g is None:
        raise ValueError(f"gold answer is not numeric: {gold!r}")
    if a is None:
        return _binary(False, "answer not numeric")
    return _binary(g == a)


def _f1(gold: str, answer: str) -> float:
    g, a = normalize_text(gold).split(), normalize_text(answer).split()
    common = sum((Counter(g) & Counter(a)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(a), common / len(g)
    return 2 * precision * recall / (precision + recall)


def score_em_f1(gold: str, answer: str | None) -> ItemScore:
    """Partial credit: the credit is token F1; `correct` stays exact match."""
    if answer is None:
        return ItemScore(correct=False, credit=0.0, detail="f1=0.000")
    em = normalize_text(gold) == normalize_text(answer)
    f1 = 1.0 if em else _f1(gold, answer)
    return ItemScore(correct=em, credit=f1, detail=f"f1={f1:.3f}")


_LABEL_ALIASES = {"unknown": "uncertain"}


def _label(s: str) -> str:
    n = normalize_text(s)
    return _LABEL_ALIASES.get(n, n)


def score_label(gold: str, answer: str | None) -> ItemScore:
    if answer is None:
        return _binary(False, "no answer")
    return _binary(_label(gold) == _label(answer))
