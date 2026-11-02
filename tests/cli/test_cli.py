import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kengine.backend.fake import ScriptedBackend, ScriptedReply
from kengine.backend.types import BackendHealth, ModelBackend
from kengine.cli import main
from kengine.config import BackendConfig
from kengine.eval import suites
from kengine.eval.types import EvalItem, Size, SuiteName

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


def test_cli_eval_kengine_not_yet(monkeypatch: pytest.MonkeyPatch) -> None:
    r = runner.invoke(main.app, ["eval", "gsm8k", "--solver", "kengine"])
    assert r.exit_code == 2


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
