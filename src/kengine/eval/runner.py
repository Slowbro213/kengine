"""Resumable eval runner. Writes config.json, items.jsonl, failures_sample.jsonl, metrics.json."""

from __future__ import annotations

import datetime
import json
import random
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

from kengine.eval.suites import SuiteSpec
from kengine.eval.types import EvalItem, RunMetrics, Solver, SolveResult, SolveStatus


class ItemLine(BaseModel):
    item_id: str
    status: SolveStatus
    answer: str | None
    gold: str
    correct: bool
    detail: str
    model_calls: int
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    session_dir: str | None
    error: str | None
    transcript_excerpt: str | None = None


def results_dir_name(date: datetime.date, suite: str, solver: str, size: str, profile: str) -> str:
    return f"{date.isoformat()}-{suite}-{solver}-{size}-{profile}"


def _read_lines(path: Path) -> list[ItemLine]:
    if not path.exists():
        return []
    lines: list[ItemLine] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            lines.append(ItemLine.model_validate_json(raw))
    return lines


def _solve(solver: Solver, item: EvalItem) -> SolveResult:
    start = time.monotonic()
    try:
        return solver.solve(item)
    except Exception as exc:  # one bad item must not end a multi-hour run
        detail = "".join(traceback.format_exception_only(exc)).strip()
        return SolveResult(
            answer=None,
            status="error",
            model_calls=0,
            prompt_tokens=0,
            completion_tokens=0,
            wall_s=time.monotonic() - start,
            error=detail,
        )


def _metrics(suite: str, solver: str, lines: Sequence[ItemLine]) -> RunMetrics:
    n = len(lines)
    answered = [x for x in lines if x.status == "answered"]
    correct = sum(x.correct for x in lines)

    def mean(values: Sequence[float]) -> float:
        return sum(values) / n if n else 0.0

    return RunMetrics(
        suite=suite,
        solver=solver,
        n=n,
        headline_accuracy=correct / n if n else 0.0,
        abstention_rate=(n - len(answered)) / n if n else 0.0,
        answered_accuracy=(sum(x.correct for x in answered) / len(answered)) if answered else None,
        status_counts=dict(Counter(x.status for x in lines)),
        mean_model_calls=mean([x.model_calls for x in lines]),
        mean_prompt_tokens=mean([x.prompt_tokens for x in lines]),
        mean_completion_tokens=mean([x.completion_tokens for x in lines]),
        mean_wall_s=mean([x.wall_s for x in lines]),
    )


def run_eval(
    spec: SuiteSpec,
    items: Sequence[EvalItem],
    solver: Solver,
    out_dir: Path,
    run_config: Mapping[str, object],
    failure_sample_size: int = 25,
    seed: int = 0,
) -> RunMetrics:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(dict(run_config), indent=2, default=str) + "\n", encoding="utf-8"
    )
    items_path = out_dir / "items.jsonl"
    done = {line.item_id for line in _read_lines(items_path)}
    with items_path.open("a", encoding="utf-8") as f:
        for item in items:
            if item.id in done:
                continue
            result = _solve(solver, item)
            score = spec.score(item.gold, result.answer) if result.status == "answered" else None
            line = ItemLine(
                item_id=item.id,
                status=result.status,
                answer=result.answer,
                gold=item.gold,
                correct=score.correct if score else False,
                detail=score.detail if score else "",
                model_calls=result.model_calls,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                wall_s=result.wall_s,
                session_dir=result.session_dir,
                error=result.error,
                transcript_excerpt=result.transcript_excerpt,
            )
            f.write(line.model_dump_json() + "\n")
            f.flush()
            done.add(item.id)

    lines = _read_lines(items_path)
    _write_failures(out_dir, lines, items, failure_sample_size, seed)
    metrics = _metrics(spec.name, solver.name, lines)
    (out_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2) + "\n", "utf-8")
    return metrics


def _write_failures(
    out_dir: Path,
    lines: Sequence[ItemLine],
    items: Sequence[EvalItem],
    size: int,
    seed: int,
) -> None:
    questions = {i.id: i.question for i in items}
    wrong = [x for x in lines if not x.correct]
    chosen = set(x.item_id for x in random.Random(seed).sample(wrong, min(size, len(wrong))))
    with (out_dir / "failures_sample.jsonl").open("w", encoding="utf-8") as f:
        for x in wrong:
            if x.item_id not in chosen:
                continue
            row = {
                "item_id": x.item_id,
                "question": questions.get(x.item_id),
                "gold": x.gold,
                "answer": x.answer,
                "status": x.status,
                "error": x.error,
                "transcript_excerpt": x.transcript_excerpt,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
