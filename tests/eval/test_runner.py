import json
from datetime import date
from pathlib import Path

from kengine.backend.types import BackendError
from kengine.eval.runner import rescore, results_dir_name, run_eval
from kengine.eval.suites import SUITES
from kengine.eval.types import EvalItem, SolveResult, SolveStatus

SPEC = SUITES["gsm8k"]


def item(i: int, gold: str = "5") -> EvalItem:
    return EvalItem(
        id=f"gsm8k-{i}", suite="gsm8k", question=f"q{i}", gold=gold, answer_format="number"
    )


def result(answer: str | None, status: SolveStatus = "answered") -> SolveResult:
    return SolveResult(
        answer=answer,
        status=status,
        model_calls=1,
        prompt_tokens=10,
        completion_tokens=20,
        wall_s=0.5,
        transcript_excerpt="t",
    )


class StubSolver:
    name = "stub"

    def __init__(self, results: dict[str, SolveResult | Exception]) -> None:
        self.results = results
        self.seen: list[str] = []

    def solve(self, item: EvalItem) -> SolveResult:
        self.seen.append(item.id)
        r = self.results[item.id]
        if isinstance(r, Exception):
            raise r
        return r


def read_items(path: Path) -> list[dict[str, object]]:
    return [json.loads(x) for x in (path / "items.jsonl").read_text().splitlines()]


def test_metrics_count_non_answered_as_wrong(tmp_path: Path) -> None:
    items = [item(i) for i in range(4)]
    solver = StubSolver(
        {
            "gsm8k-0": result("5"),
            "gsm8k-1": result("6"),
            "gsm8k-2": result("5", "blocked"),  # right answer, but blocked: still wrong
            "gsm8k-3": result(None, "abstained"),
        }
    )
    m = run_eval(SPEC, items, solver, tmp_path, {"k": 1})
    assert m.headline_accuracy == 0.25
    assert m.abstention_rate == 0.5
    assert m.answered_accuracy == 0.5
    assert m.status_counts == {"answered": 2, "blocked": 1, "abstained": 1}
    assert m.n == 4 and m.mean_completion_tokens == 20
    assert json.loads((tmp_path / "config.json").read_text()) == {"k": 1}
    assert json.loads((tmp_path / "metrics.json").read_text())["n"] == 4


def test_backend_error_recorded_and_run_continues(tmp_path: Path) -> None:
    items = [item(i) for i in range(3)]
    solver = StubSolver(
        {
            "gsm8k-0": result("5"),
            "gsm8k-1": BackendError("HTTP 500"),
            "gsm8k-2": result("5"),
        }
    )
    run_eval(SPEC, items, solver, tmp_path, {})
    lines = read_items(tmp_path)
    assert len(lines) == 3 and lines[1]["status"] == "error"
    assert "HTTP 500" in str(lines[1]["error"])


def test_resume_skips_completed_items(tmp_path: Path) -> None:
    items = [item(i) for i in range(3)]
    run_eval(SPEC, items[:1], StubSolver({"gsm8k-0": result("5")}), tmp_path, {})
    solver = StubSolver({"gsm8k-1": result("5"), "gsm8k-2": result("1")})
    m = run_eval(SPEC, items, solver, tmp_path, {})
    assert solver.seen == ["gsm8k-1", "gsm8k-2"]
    assert m.n == 3 and m.headline_accuracy == 2 / 3


def test_failures_sample_capped_and_seeded(tmp_path: Path) -> None:
    items = [item(i) for i in range(40)]
    results: dict[str, SolveResult | Exception] = {i.id: result("0") for i in items}
    run_eval(SPEC, items, StubSolver(results), tmp_path / "a", {}, failure_sample_size=25)
    run_eval(SPEC, items, StubSolver(results), tmp_path / "b", {}, failure_sample_size=25)
    a = (tmp_path / "a" / "failures_sample.jsonl").read_text().splitlines()
    b = (tmp_path / "b" / "failures_sample.jsonl").read_text().splitlines()
    assert len(a) == 25 and a == b
    row = json.loads(a[0])
    assert row["question"].startswith("q") and row["transcript_excerpt"] == "t"


def test_results_dir_name() -> None:
    assert (
        results_dir_name(date(2026, 11, 2), "gsm8k", "baseline", "smoke", "none")
        == "2026-11-02-gsm8k-baseline-smoke-none"
    )


def hotpot(i: int, gold: str) -> EvalItem:
    return EvalItem(
        id=f"hotpotqa-{i}", suite="hotpotqa", question=f"q{i}", gold=gold, answer_format="phrase"
    )


def test_partial_credit_headline_and_exact_accuracy(tmp_path: Path) -> None:
    items = [hotpot(0, "Barack Obama"), hotpot(1, "Paris"), hotpot(2, "Rome")]
    solver = StubSolver(
        {
            "hotpotqa-0": result("Obama"),
            "hotpotqa-1": result("Paris"),
            "hotpotqa-2": result(None, "partial"),
        }
    )
    m = run_eval(SUITES["hotpotqa"], items, solver, tmp_path, {"solver": "stub"})
    assert abs(m.headline_accuracy - (2 / 3 + 1) / 3) < 1e-9
    assert m.exact_accuracy == 1 / 3
    assert m.answered_accuracy is not None and abs(m.answered_accuracy - (2 / 3 + 1) / 2) < 1e-9
    failures = (tmp_path / "failures_sample.jsonl").read_text().splitlines()
    assert sorted(json.loads(x)["item_id"] for x in failures) == ["hotpotqa-0", "hotpotqa-2"]


def test_rescore_fills_credit_for_old_results(tmp_path: Path) -> None:
    items = [hotpot(0, "Barack Obama"), hotpot(1, "Paris")]
    solver = StubSolver({"hotpotqa-0": result("Obama"), "hotpotqa-1": result("Paris")})
    run_eval(SUITES["hotpotqa"], items, solver, tmp_path, {"solver": "stub"})
    old = [{k: v for k, v in x.items() if k != "credit"} for x in read_items(tmp_path)]
    (tmp_path / "items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in old))
    sample = (tmp_path / "failures_sample.jsonl").read_text()
    m = rescore(SUITES["hotpotqa"], tmp_path)
    assert abs(m.headline_accuracy - (2 / 3 + 1) / 2) < 1e-9
    assert m.solver == "stub" and m.exact_accuracy == 0.5
    assert [x["credit"] for x in read_items(tmp_path)][1] == 1.0
    assert (tmp_path / "failures_sample.jsonl").read_text() == sample
