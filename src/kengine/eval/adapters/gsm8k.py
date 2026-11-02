from __future__ import annotations

from collections.abc import Mapping

from kengine.eval.types import EvalItem

ANSWER_FORMAT = "Give the final answer as a number."


def parse_row(row: Mapping[str, object], index: int) -> EvalItem:
    answer = str(row["answer"])
    if "####" not in answer:
        raise ValueError(f"gsm8k row {index} has no '####' answer line")
    gold = answer.rsplit("####", 1)[1].strip().replace(",", "").replace(" ", "")
    return EvalItem(
        id=f"gsm8k-{index}",
        suite="gsm8k",
        question=str(row["question"]),
        gold=gold,
        answer_format=ANSWER_FORMAT,
    )
