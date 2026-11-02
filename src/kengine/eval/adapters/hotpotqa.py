from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from kengine.eval.types import EvalItem

ANSWER_FORMAT = "Give the final answer as a short phrase."


def parse_row(row: Mapping[str, object], index: int) -> EvalItem:
    context = cast(Mapping[str, Sequence[object]], row["context"])
    titles = [str(t) for t in context["title"]]
    sentences = [[str(s) for s in cast(Sequence[object], ss)] for ss in context["sentences"]]
    documents = [
        f"{title}: {' '.join(s.strip() for s in sents)}"
        for title, sents in zip(titles, sentences, strict=True)
    ]
    return EvalItem(
        id=f"hotpotqa-{row['id']}",
        suite="hotpotqa",
        question=str(row["question"]),
        documents=documents,
        gold=str(row["answer"]),
        answer_format=ANSWER_FORMAT,
    )
