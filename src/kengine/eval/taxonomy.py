"""Error labels for hand-labelling baseline failures (docs/eval/error-taxonomy.md)."""

from __future__ import annotations

import json
from collections import Counter
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter


class ErrorLabel(StrEnum):
    FABRICATED_FACT = "fabricated_fact"
    INVALID_INFERENCE = "invalid_inference"
    ARITHMETIC_ERROR = "arithmetic_error"
    INCONSISTENCY = "inconsistency"
    FORGOTTEN_CONSTRAINT = "forgotten_constraint"
    MISREAD_QUESTION = "misread_question"
    FORMAT_FAILURE = "format_failure"
    TRUNCATED = "truncated"
    OTHER = "other"


class TaxonomyRow(BaseModel):
    item_id: str
    labels: list[ErrorLabel] = Field(min_length=1)
    note: str = ""


_ROWS = TypeAdapter(dict[str, TaxonomyRow])


def summarize_taxonomy(path: Path) -> dict[ErrorLabel, int]:
    """Count labels in a taxonomy.jsonl file. Validation errors name the offending line."""
    counts: Counter[ErrorLabel] = Counter()
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {n}: invalid JSON: {exc}") from exc
        row = _ROWS.validate_python({f"line {n}": data})[f"line {n}"]
        counts.update(row.labels)
    return dict(counts.most_common())
