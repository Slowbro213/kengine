from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from kengine.eval.types import EvalItem
from kengine.loop.task import JUDGE_FORMAT, judge_question

ANSWER_FORMAT = JUDGE_FORMAT


def parse_row(row: Mapping[str, object], index: int) -> EvalItem:
    premises = row["premises"]
    if isinstance(premises, str):
        lines = [p.strip() for p in premises.splitlines()]
    else:
        lines = [str(p).strip() for p in cast(Sequence[object], premises)]
    lines = [p for p in lines if p]
    conclusion = str(row["conclusion"]).strip()
    question = judge_question(lines, conclusion)
    example_id = row.get("example_id", index)
    return EvalItem(
        id=f"folio-{example_id}",
        suite="folio",
        question=question,
        gold=str(row["label"]).strip(),
        answer_format=ANSWER_FORMAT,
        givens=lines,  # the premises; never the conclusion, which is what's being judged
        conclusion=conclusion,
    )
