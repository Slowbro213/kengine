"""Pinned benchmark sources and seeded subset selection."""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable, Mapping
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from kengine.eval.adapters import folio, gsm8k, hotpotqa
from kengine.eval.scoring import score_em_f1, score_label, score_numeric
from kengine.eval.types import SIZE_COUNTS, EvalItem, ItemScore, Size, SuiteName


class SuiteSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: SuiteName
    hf_id: str
    hf_config: str | None
    split: str
    parse_row: Callable[[Mapping[str, object], int], EvalItem]
    score: Callable[[str, str | None], ItemScore]
    fallback_hf_id: str | None = None


SUITES: dict[SuiteName, SuiteSpec] = {
    "gsm8k": SuiteSpec(
        name="gsm8k",
        hf_id="openai/gsm8k",
        hf_config="main",
        split="test",
        parse_row=gsm8k.parse_row,
        score=score_numeric,
    ),
    "hotpotqa": SuiteSpec(
        name="hotpotqa",
        hf_id="hotpotqa/hotpot_qa",
        hf_config="distractor",
        split="validation",
        parse_row=hotpotqa.parse_row,
        score=score_em_f1,
    ),
    "folio": SuiteSpec(
        name="folio",
        hf_id="yale-nlp/FOLIO",
        hf_config=None,
        split="validation",
        parse_row=folio.parse_row,
        score=score_label,
        fallback_hf_id="tasksource/folio",
    ),
}


def select_indices(total: int, size: Size, seed: int) -> list[int]:
    n = SIZE_COUNTS.get(size)
    if n is None or n >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), n))


def _load_dataset(hf_id: str, hf_config: str | None, split: str) -> Any:
    # Imported dynamically: `datasets` is an optional extra and ships no usable types.
    try:
        datasets = importlib.import_module("datasets")
    except ImportError as exc:
        raise RuntimeError("install with: uv sync --extra eval") from exc
    load_dataset: Callable[..., Any] = datasets.load_dataset
    return load_dataset(hf_id, hf_config, split=split)


def load_rows(hf_id: str, hf_config: str | None, split: str) -> list[Mapping[str, object]]:
    """Every row of a dataset split. Needs the `eval` extra."""
    ds = _load_dataset(hf_id, hf_config, split)
    return [cast(Mapping[str, object], ds[i]) for i in range(len(ds))]


def load_items(suite: SuiteName, size: Size, seed: int = 0) -> list[EvalItem]:
    """Load a fixed, seeded subset of a suite. Needs the `eval` extra and network access."""
    spec = SUITES[suite]
    prefix: str | None = None
    try:
        ds = _load_dataset(spec.hf_id, spec.hf_config, spec.split)
    except Exception:
        if spec.fallback_hf_id is None:
            raise
        ds = _load_dataset(spec.fallback_hf_id, spec.hf_config, spec.split)
        prefix = spec.fallback_hf_id.split("/")[0]
    items: list[EvalItem] = []
    for i in select_indices(len(ds), size, seed):
        item = spec.parse_row(cast(Mapping[str, object], ds[i]), i)
        if prefix is not None:
            item = item.model_copy(update={"id": f"{suite}-{prefix}-{item.id[len(suite) + 1 :]}"})
        items.append(item)
    return items
