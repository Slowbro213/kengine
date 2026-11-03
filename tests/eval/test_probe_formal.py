import json
from pathlib import Path

from kengine.backend.fake import ScriptedBackend, ScriptedReply
from kengine.config import RolesConfig
from kengine.eval.adapters.folio import parse_row
from kengine.eval.probe_formal import probe_item, run_probe

ROW = {
    "premises": "All dogs are mammals.\nRex is a dog.",
    "conclusion": "Rex is a mammal.",
    "label": "True",
    "example_id": 7,
}


def reply(*items: tuple[int, list[str]]) -> ScriptedReply:
    return ScriptedReply(
        content=json.dumps({"givens": [{"given": g, "statements": s} for g, s in items]})
    )


def test_probe_item_decides_the_conclusion():
    backend = ScriptedBackend(
        [reply((1, ["dog(x) -> mammal(x)"]), (2, ["dog(rex)"]), (3, ["mammal(rex)"]))]
    )
    line = probe_item(backend, RolesConfig().formalize, "text", parse_row(ROW, 0))
    assert (line.predicted, line.correct, line.decision) == ("True", True, "entailed")
    assert (line.n_premises, line.n_formalized, line.conclusion_ok) == (2, 2, True)
    assert "3. Rex is a mammal." in backend.calls[0].messages[-1].content


def test_probe_item_without_conclusion_form():
    backend = ScriptedBackend([reply((1, ["dog(x) -> mammal(x)"]), (2, ["dog(rex)"]))])
    line = probe_item(backend, RolesConfig().formalize, "text", parse_row(ROW, 0))
    assert line.predicted is None and not line.correct and not line.conclusion_ok


def test_run_probe_writes_summary(tmp_path: Path):
    backend = ScriptedBackend(
        [reply((1, ["dog(x) -> mammal(x)"]), (2, ["dog(rex)"]), (3, ["mammal(rex)"])),
         ScriptedReply(error="down")]
    )  # fmt: skip
    items = [parse_row(ROW, 0), parse_row({**ROW, "example_id": 8}, 1)]
    s = run_probe(backend, RolesConfig().formalize, "text", items, tmp_path / "out")
    assert (s.n, s.all_formalized, s.decided, s.correct) == (2, 1, 1, 1)
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["correct"] == 1
    assert len((tmp_path / "out" / "items.jsonl").read_text().splitlines()) == 2
