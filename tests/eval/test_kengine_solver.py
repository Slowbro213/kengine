import json
from pathlib import Path

from kengine.backend.fake import ScriptedBackend, ScriptedReply
from kengine.config import KengineConfig
from kengine.eval.kengine_solver import KengineSolver
from kengine.eval.types import EvalItem
from kengine.loop.profiles import get_profile
from kengine.retrieval.fake import FakeEmbedder, FakeEntailment
from kengine.retrieval.models import Models, NLIScores

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


def test_solver_with_models_verifies_a_document_fact_and_answers(tmp_path: Path):
    sentence = "Heat was directed by Michael Mann."
    fact_text = "Heat was directed by Michael Mann (document 1)"
    models = Models(
        FakeEntailment([(sentence, fact_text, NLIScores(0.97, 0.02, 0.01))]), FakeEmbedder()
    )
    claim = {"ref": "c1", "type": "fact", "text": fact_text}
    action = {"kind": "final_answer", "answer": "Michael Mann", "preconditions": ["c1"]}
    reply = ScriptedReply(content=json.dumps({"claims": [claim], "action": action}))
    think = ScriptedReply(content="Heat was directed by Michael Mann.\nFinal answer: Michael Mann")
    item = EvalItem(
        id="hp-1",
        suite="hotpotqa",
        question="Who directed Heat?",
        documents=[f"Heat: {sentence}"],
        gold="Michael Mann",
        answer_format="a name",
    )
    backend = ScriptedBackend([think, *[reply] * 3])
    s = KengineSolver(
        backend, KengineConfig(), get_profile("standard"), tmp_path / "s", 32768, models=models
    )
    r = s.solve(item)
    assert (r.status, r.answer) == ("answered", "Michael Mann"), r.transcript_excerpt
