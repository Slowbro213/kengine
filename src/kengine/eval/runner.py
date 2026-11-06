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
    # None in results written before partial credit; read as `float(correct)`.
    credit: float | None = None
    detail: str
    model_calls: int
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    session_dir: str | None
    error: str | None
    transcript_excerpt: str | None = None


def _credit(line: ItemLine) -> float:
    return float(line.correct) if line.credit is None else line.credit


def results_dir_name(
    date: datetime.date,
    suite: str,
    solver: str,
    size: str,
    profile: str,
    limit: int | None = None,
) -> str:
    name = f"{date.isoformat()}-{suite}-{solver}-{size}-{profile}"
    return name if limit is None else f"{name}-first{limit}"


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

    def mean(values: Sequence[float]) -> float:
        return sum(values) / n if n else 0.0

    return RunMetrics(
        suite=suite,
        solver=solver,
        n=n,
        headline_accuracy=mean([_credit(x) for x in lines]),
        exact_accuracy=mean([float(x.correct) for x in lines]),
        abstention_rate=(n - len(answered)) / n if n else 0.0,
        answered_accuracy=(sum(_credit(x) for x in answered) / len(answered)) if answered else None,
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
                credit=score.credit if score else 0.0,
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
    questions = {i.id: i.question for i in items}
    _write_failures(out_dir, lines, questions, failure_sample_size, seed)
    return _write_metrics(out_dir, spec.name, solver.name, lines)


def shard_dir_name(k: int, n: int) -> str:
    return f"shard-{k}of{n}"


def merge_shards(
    spec: SuiteSpec, out_dir: Path, failure_sample_size: int = 25, seed: int = 0
) -> RunMetrics:
    """Join the finished shard-KofN runs under OUT_DIR into one run, written as an unsharded
    run would have written it. Shard K holds items K-1, K-1+N, ... of the full list, and must
    keep every failure in its failures_sample.jsonl, so the merged sample is the same."""
    found = set(out_dir.glob("shard-*of*"))
    if not found:
        raise ValueError(f"no shard-KofN dirs in {out_dir}")
    n = int(min(found).name.rsplit("of", 1)[1])
    dirs = [out_dir / shard_dir_name(k, n) for k in range(1, n + 1)]
    if found != set(dirs):
        raise ValueError(f"expected exactly {', '.join(d.name for d in dirs)} in {out_dir}")
    configs: list[dict[str, object]] = []
    shards: list[list[ItemLine]] = []
    for d in dirs:
        config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        lines = _read_lines(d / "items.jsonl")
        if len(lines) != config["n_items"]:
            raise ValueError(f"{d.name} is unfinished: {len(lines)} of {config['n_items']} items")
        configs.append(config)
        shards.append(lines)
    longest = max(len(s) for s in shards)
    lines = [s[j] for j in range(longest) for s in shards if j < len(s)]
    run_config = {k: v for k, v in configs[0].items() if k != "shard"}
    run_config["n_items"] = len(lines)
    run_config["shards"] = n
    (out_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / "items.jsonl").write_text(
        "".join(x.model_dump_json() + "\n" for x in lines), "utf-8"
    )
    questions: dict[str, str | None] = {}
    for d in dirs:
        for raw in (d / "failures_sample.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(raw)
            questions[row["item_id"]] = row["question"]
    _write_failures(out_dir, lines, questions, failure_sample_size, seed)
    return _write_metrics(out_dir, spec.name, str(run_config["solver"]), lines)


def _write_metrics(out_dir: Path, suite: str, solver: str, lines: Sequence[ItemLine]) -> RunMetrics:
    metrics = _metrics(suite, solver, lines)
    (out_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2) + "\n", "utf-8")
    return metrics


def rescore(spec: SuiteSpec, out_dir: Path) -> RunMetrics:
    """Re-score a finished run's answers with the suite's current scorer, rewriting
    items.jsonl and metrics.json. failures_sample.jsonl is left alone, so a hand-made
    taxonomy.jsonl still matches it."""
    items_path = out_dir / "items.jsonl"
    lines = [line.model_copy(update=_score_fields(spec, line)) for line in _read_lines(items_path)]
    items_path.write_text("".join(x.model_dump_json() + "\n" for x in lines), "utf-8")
    solver = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))["solver"]
    return _write_metrics(out_dir, spec.name, str(solver), lines)


def _score_fields(spec: SuiteSpec, line: ItemLine) -> dict[str, object]:
    if line.status != "answered":
        return {"correct": False, "credit": 0.0, "detail": ""}
    score = spec.score(line.gold, line.answer)
    return {"correct": score.correct, "credit": score.credit, "detail": score.detail}


def _write_failures(
    out_dir: Path,
    lines: Sequence[ItemLine],
    questions: Mapping[str, str | None],
    size: int,
    seed: int,
) -> None:
    wrong = [x for x in lines if _credit(x) < 1.0]
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
