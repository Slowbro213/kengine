"""Per-suite answer scorers."""

from __future__ import annotations

from collections import Counter

from kengine.eval.answers import normalize_text
from kengine.eval.types import ItemScore
from kengine.numbers import parse_number


def score_numeric(gold: str, answer: str | None) -> ItemScore:
    if answer is None:
        return ItemScore(correct=False, detail="no answer")
    g, a = parse_number(gold), parse_number(answer)
    if g is None:
        raise ValueError(f"gold answer is not numeric: {gold!r}")
    if a is None:
        return ItemScore(correct=False, detail="answer not numeric")
    return ItemScore(correct=g == a)


def _f1(gold: str, answer: str) -> float:
    g, a = normalize_text(gold).split(), normalize_text(answer).split()
    common = sum((Counter(g) & Counter(a)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(a), common / len(g)
    return 2 * precision * recall / (precision + recall)


def score_em_f1(gold: str, answer: str | None) -> ItemScore:
    if answer is None:
        return ItemScore(correct=False, detail="f1=0.000")
    em = normalize_text(gold) == normalize_text(answer)
    return ItemScore(correct=em, detail=f"f1={_f1(gold, answer):.3f}")


_LABEL_ALIASES = {"unknown": "uncertain"}


def _label(s: str) -> str:
    n = normalize_text(s)
    return _LABEL_ALIASES.get(n, n)


def score_label(gold: str, answer: str | None) -> ItemScore:
    if answer is None:
        return ItemScore(correct=False, detail="no answer")
    return ItemScore(correct=_label(gold) == _label(answer))
