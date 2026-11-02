import json
from pathlib import Path

from kengine.backend.fake import ScriptedBackend, ScriptedReply
from kengine.config import KengineConfig
from kengine.eval.kengine_solver import KengineSolver
from kengine.eval.types import EvalItem
from kengine.loop.profiles import get_profile

ITEM = EvalItem(id="gsm8k-1", suite="gsm8k", question="12*7?", gold="84", answer_format="n")
THINK = ScriptedReply(content="Working:\n1. 12*7 = 84\nFinal answer: 84\nRests on: 1")


def extraction(answer: str | None) -> ScriptedReply:
    action = {"kind": "continue"}
    if answer is not None:
        action = {"kind": "final_answer", "answer": answer, "preconditions": ["c1"]}
    claim = {
        "ref": "c1",
        "type": "computation",
        "text": "12*7 = 84",
        "computation": {"expression": "12*7", "claimed_result": "84"},
    }
    return ScriptedReply(content=json.dumps({"claims": [claim], "action": action}))


def solver(backend: ScriptedBackend, tmp_path: Path, profile: str, budget: int) -> KengineSolver:
    return KengineSolver(
        backend, KengineConfig(), get_profile(profile), tmp_path / "sessions", budget
    )


def test_solver_maps_status_and_passes_budget(tmp_path: Path):
    backend = ScriptedBackend([THINK, *[extraction("84")] * 3])
    r = solver(backend, tmp_path, "standard", 32768).solve(ITEM)
    assert (r.status, r.answer, r.model_calls) == ("answered", "84", 4)
    assert r.session_dir is not None and Path(r.session_dir).parent == tmp_path / "sessions"
    assert r.transcript_excerpt is not None
    assert r.transcript_excerpt.startswith(THINK.content + "\n---\n")


def test_solver_budget_exhausted_maps_to_partial(tmp_path: Path):
    backend = ScriptedBackend([THINK, extraction(None)] * 3)
    r = solver(backend, tmp_path, "quick", budget=10).solve(ITEM)
    assert (r.status, r.answer, r.model_calls) == ("partial", None, 2)
