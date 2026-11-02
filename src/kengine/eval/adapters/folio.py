from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from kengine.eval.types import EvalItem

ANSWER_FORMAT = "Answer with exactly one of: True, False, Uncertain."


def parse_row(row: Mapping[str, object], index: int) -> EvalItem:
    premises = row["premises"]
    if isinstance(premises, str):
        premises_text = premises.strip()
    else:
        premises_text = "\n".join(str(p).strip() for p in cast(Sequence[object], premises))
    question = (
        f"Premises:\n{premises_text}\n\n"
        f"Conclusion: {str(row['conclusion']).strip()}\n\n"
        "Based only on the premises, is the conclusion True, False, or Uncertain?"
    )
    example_id = row.get("example_id", index)
    return EvalItem(
        id=f"folio-{example_id}",
        suite="folio",
        question=question,
        gold=str(row["label"]).strip(),
        answer_format=ANSWER_FORMAT,
    )
