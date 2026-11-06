import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kengine.backend.fake import ScriptedBackend, ScriptedReply
from kengine.backend.types import BackendHealth, ModelBackend
from kengine.cli import main
from kengine.config import BackendConfig
from kengine.eval import suites
from kengine.eval.types import EvalItem, Size, SuiteName
from kengine.loop.agent import RunResult
from kengine.loop.task import JUDGE_FORMAT, Task, judge_question

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "default_config_paths", lambda: [tmp_path / "kengine.toml"])


def use_backend(monkeypatch: pytest.MonkeyPatch, backend: ScriptedBackend) -> None:
    def factory(config: BackendConfig) -> ModelBackend:
        return backend

    monkeypatch.setattr(main, "backend_factory", factory)


def gsm_items(suite: SuiteName, size: Size, seed: int = 0) -> list[EvalItem]:
    return [
        EvalItem(id=f"gsm8k-{i}", suite="gsm8k", question="q", gold="1234", answer_format="n")
        for i in range(2)
    ]


def test_cli_doctor_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = ScriptedBackend(
        [ScriptedReply(content='{"ok": true}'), ScriptedReply(content="5", reasoning="r")],
        health=BackendHealth(ok=True, detail="ok"),
    )
    use_backend(monkeypatch, ok)
    r = runner.invoke(main.app, ["doctor", "--json", "doc.json"])
    assert r.exit_code == 0, r.output
    assert json.loads(Path("doc.json").read_text())["ok"] is True

    use_backend(monkeypatch, ScriptedBackend([], health=BackendHealth(ok=False, detail="down")))
    assert runner.invoke(main.app, ["doctor"]).exit_code == 1


def test_cli_eval_baseline_writes_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_backend(monkeypatch, ScriptedBackend([ScriptedReply(content="Final answer: 1234")] * 2))
    monkeypatch.setattr(suites, "load_items", gsm_items)
    r = runner.invoke(
        main.app,
        ["eval", "gsm8k", "--solver", "baseline", "--size", "smoke", "--out", str(tmp_path / "o")],
    )
    assert r.exit_code == 0, r.output
    dirs = list((tmp_path / "o").iterdir())
    assert len(dirs) == 1 and dirs[0].name.endswith("-gsm8k-baseline-smoke-none")
    metrics = json.loads((dirs[0] / "metrics.json").read_text())
    assert metrics["headline_accuracy"] == 1.0
    assert json.loads((dirs[0] / "config.json").read_text())["solver"] == "baseline"


# `kengine run` formalizes the task's givens first (logic.formalize_givens is on by default).
FORMALIZE_NONE = ScriptedReply(content='{"givens": []}')


def think_reply() -> ScriptedReply:
    return ScriptedReply(content="Working:\n1. something\nFinal answer: x")


def extract_reply(
    claims: Sequence[Mapping[str, object]], answer: str, pre: list[str]
) -> ScriptedReply:
    action = {"kind": "final_answer", "answer": answer, "preconditions": pre}
    return ScriptedReply(content=json.dumps({"claims": claims, "action": action}))


def comp(ref: str, expr: str, result: str, premises: list[str]) -> dict[str, object]:
    computation = {"expression": expr, "claimed_result": result}
    text = f"{expr} = {result}"
    return {
        "ref": ref,
        "type": "computation",
        "text": text,
        "premises": premises,
        "computation": computation,
    }


def use_data_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    (tmp_path / "kengine.toml").write_text(f'data_dir = "{data}"\n')
    return data


def test_cli_eval_kengine_writes_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    step = [
        think_reply(),
        *[extract_reply([comp("c1", "1000+234", "1234", [])], "1234", ["c1"])] * 3,
    ]
    use_backend(monkeypatch, ScriptedBackend(step * 2))
    monkeypatch.setattr(suites, "load_items", gsm_items)
    out = tmp_path / "o"
    r = runner.invoke(main.app, ["eval", "gsm8k", "--solver", "kengine", "--out", str(out)])
    assert r.exit_code == 0, r.output
    dirs = list(out.iterdir())
    assert len(dirs) == 1 and dirs[0].name.endswith("-gsm8k-kengine-smoke-standard")
    assert json.loads((dirs[0] / "metrics.json").read_text())["headline_accuracy"] == 1.0
    assert len(list((dirs[0] / "sessions").iterdir())) == 2


def test_cli_eval_rejects_unknown_profile_and_baseline_profile() -> None:
    assert (
        runner.invoke(
            main.app, ["eval", "gsm8k", "--solver", "kengine", "--profile", "x"]
        ).exit_code
        == 2
    )
    assert runner.invoke(main.app, ["eval", "gsm8k", "--profile", "quick"]).exit_code == 2


def test_cli_run_prints_answer_and_assumptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = use_data_dir(tmp_path)
    assumption = {"ref": "c1", "type": "assumption", "text": "a pen costs $2"}
    reply = extract_reply([assumption, comp("c2", "12*2", "24", ["c1"])], "24", ["c2"])
    use_backend(monkeypatch, ScriptedBackend([FORMALIZE_NONE, think_reply(), reply]))
    r = runner.invoke(main.app, ["run", "12 pens?", "--profile", "quick"])
    assert r.exit_code == 0, r.output
    assert "status: answered" in r.output and "answer: 24" in r.output
    assert "ASSUMED: b2 a pen costs $2" in r.output  # b1 is the given "12 pens?"
    assert f"session: {data / 'sessions'}" in r.output


def test_cli_run_blocked_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_data_dir(tmp_path)
    fact = {"ref": "c1", "type": "fact", "text": "Paris is the capital of France"}
    use_backend(
        monkeypatch,
        ScriptedBackend([FORMALIZE_NONE, think_reply(), extract_reply([fact], "Paris", ["c1"])]),
    )
    r = runner.invoke(
        main.app, ["run", "capital of France?", "--profile", "quick", "--max-steps", "1"]
    )
    assert r.exit_code == 3, r.output
    assert "blocked answer (unverified): Paris" in r.output


def test_cli_eval_summarize(tmp_path: Path) -> None:
    (tmp_path / "taxonomy.jsonl").write_text(
        '{"item_id": "a", "labels": ["arithmetic_error"]}\n'
        '{"item_id": "b", "labels": ["arithmetic_error", "truncated"]}\n'
    )
    r = runner.invoke(main.app, ["eval-summarize", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert json.loads((tmp_path / "taxonomy_summary.json").read_text()) == {
        "arithmetic_error": 2,
        "truncated": 1,
    }


def test_cli_eval_rescore(tmp_path: Path) -> None:
    line = {
        "item_id": "hotpotqa-0",
        "status": "answered",
        "answer": "Obama",
        "gold": "Barack Obama",
        "correct": False,
        "detail": "f1=0.667",
        "model_calls": 1,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "wall_s": 1.0,
        "session_dir": None,
        "error": None,
    }
    (tmp_path / "items.jsonl").write_text(json.dumps(line) + "\n")
    (tmp_path / "config.json").write_text('{"suite": "hotpotqa", "solver": "baseline"}')
    r = runner.invoke(main.app, ["eval-rescore", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "headline=0.667 exact=0.000" in r.output
    assert json.loads((tmp_path / "metrics.json").read_text())["headline_accuracy"] > 0.66


def test_cli_eval_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_backend(monkeypatch, ScriptedBackend([ScriptedReply(content="Final answer: 1234")]))
    monkeypatch.setattr(suites, "load_items", gsm_items)
    out = tmp_path / "o"
    r = runner.invoke(main.app, ["eval", "gsm8k", "--limit", "1", "--out", str(out)])
    assert r.exit_code == 0, r.output
    (d,) = list(out.iterdir())
    assert d.name.endswith("-gsm8k-baseline-smoke-none-first1")
    assert len((d / "items.jsonl").read_text().splitlines()) == 1
    assert json.loads((d / "config.json").read_text())["limit"] == 1


def test_cli_eval_shards_and_merge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    replies = [ScriptedReply(content="Final answer: 1234")] * 2
    use_backend(monkeypatch, ScriptedBackend(replies))
    monkeypatch.setattr(suites, "load_items", gsm_items)
    out = tmp_path / "o"
    for shard in ["1/2", "2/2"]:
        r = runner.invoke(main.app, ["eval", "gsm8k", "--shard", shard, "--out", str(out)])
        assert r.exit_code == 0, r.output
    (d,) = list(out.iterdir())
    assert sorted(x.name for x in d.iterdir()) == ["shard-1of2", "shard-2of2"]
    shard2 = [json.loads(x) for x in (d / "shard-2of2" / "items.jsonl").read_text().splitlines()]
    assert [x["item_id"] for x in shard2] == ["gsm8k-1"]
    r = runner.invoke(main.app, ["eval-merge", str(d)])
    assert r.exit_code == 0, r.output
    assert "n=2 headline=1.000" in r.output
    assert json.loads((d / "config.json").read_text())["shards"] == 2


def test_cli_eval_rejects_bad_shard(tmp_path: Path) -> None:
    r = runner.invoke(main.app, ["eval", "gsm8k", "--shard", "3/2", "--out", str(tmp_path)])
    assert r.exit_code == 2
    assert "--shard must be K/N" in r.output


def test_cli_eval_compare(tmp_path: Path) -> None:
    for name, correct in [("base", [True, False]), ("other", [True, True])]:
        d = tmp_path / name
        d.mkdir()
        rows = [
            {"item_id": f"i{n}", "status": "answered", "correct": c} for n, c in enumerate(correct)
        ]
        (d / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = runner.invoke(main.app, ["eval-compare", str(tmp_path / "base"), str(tmp_path / "other")])
    assert r.exit_code == 0, r.output
    assert "fixed=1" in r.output and "errors_caught=1.000" in r.output
    assert (tmp_path / "other" / "compare-base.json").exists()


def test_cli_run_with_a_conclusion_judges_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_data_dir(tmp_path)
    tasks: list[Task] = []
    run = main.Agent.run

    def record(self: main.Agent, task: Task) -> RunResult:
        tasks.append(task)
        return run(self, task)

    monkeypatch.setattr(main.Agent, "run", record)
    use_backend(monkeypatch, ScriptedBackend([FORMALIZE_NONE, think_reply()] * 2))
    args = ["run", "All cats purr. Tom is a cat.", "--conclusion", "Tom purrs.", "--max-steps", "1"]
    runner.invoke(main.app, args)
    assert tasks == [
        Task(
            goal=judge_question(["All cats purr.", "Tom is a cat."], "Tom purrs."),
            answer_format=JUDGE_FORMAT,
            givens=["All cats purr.", "Tom is a cat."],
            conclusion="Tom purrs.",
        )
    ]
