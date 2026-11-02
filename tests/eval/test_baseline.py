from kengine.backend.fake import ScriptedBackend, ScriptedReply
from kengine.config import load_config
from kengine.eval.baseline import BaselineSolver
from kengine.eval.types import EvalItem

THINK = load_config([]).roles.think
ITEM = EvalItem(
    id="gsm8k-1",
    suite="gsm8k",
    question="What is 6*7?",
    gold="42",
    answer_format="Give the final answer as a number.",
)


def test_baseline_single_call_with_budget_and_answer_extraction() -> None:
    backend = ScriptedBackend([ScriptedReply(content="... Final answer: 42", reasoning="r")])
    r = BaselineSolver(backend, THINK, token_budget=1000).solve(ITEM)
    assert r.status == "answered" and r.answer == "42" and r.model_calls == 1
    assert backend.calls[0].role.max_tokens == 1000 and backend.calls[0].role.thinking
    assert backend.calls[0].messages[0].role == "system"
    user = backend.calls[0].messages[-1].content
    assert ITEM.question in user and ITEM.answer_format in user
    assert r.transcript_excerpt == "r\n---\n... Final answer: 42"


def test_baseline_truncated_without_answer_is_partial() -> None:
    backend = ScriptedBackend([ScriptedReply(content="", reasoning="...", finish_reason="length")])
    r = BaselineSolver(backend, THINK, 1000).solve(ITEM)
    assert r.status == "partial" and r.answer is None


def test_baseline_no_answer_line_is_abstained() -> None:
    backend = ScriptedBackend([ScriptedReply(content="I am not sure.")])
    assert BaselineSolver(backend, THINK, 1000).solve(ITEM).status == "abstained"


def test_baseline_documents_are_numbered_in_prompt() -> None:
    backend = ScriptedBackend([ScriptedReply(content="Final answer: x")])
    BaselineSolver(backend, THINK, 1000).solve(ITEM.model_copy(update={"documents": ["A", "B"]}))
    user = backend.calls[0].messages[-1].content
    assert "## Documents\n[1] A\n[2] B" in user and user.index("## Documents") < user.index(
        "## Question"
    )
