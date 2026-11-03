import pytest

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import Status
from kengine.beliefs.store import BeliefStore
from kengine.checkers.base import CheckerKind, Outcome, Verdict
from kengine.claims.schema import Claim, ClaimType, Computation, Formal, Source, SourceKind
from kengine.loop.context import (
    ContextOverflowError,
    StepSummary,
    approx_tokens,
    assemble_extract_context,
    assemble_think_context,
    belief_line,
)
from kengine.loop.task import Task

SRC = Source(kind=SourceKind.MODEL)


def _set(s: BeliefStore, id: str, status: Status) -> None:
    v = Verdict(
        checker_id="t",
        checker_kind=CheckerKind.DETERMINISTIC,
        outcome=Outcome.PASS,
        status=status,
        explanation="",
    )
    s.apply_verdict(id, v, step=1)


@pytest.fixture
def s() -> BeliefStore:
    """b1 given, b2 assumption, b3 fact unchecked, b4 fact contradicted."""
    store = BeliefStore.open(":memory:")
    store.add(type=ClaimType.GIVEN, text="pens cost 2", premises=[], source=SRC, step=0)
    store.add(type=ClaimType.ASSUMPTION, text="no tax", premises=[], source=SRC, step=1)
    store.add(type=ClaimType.FACT, text="12 pens bought", premises=[], source=SRC, step=1)
    store.add(type=ClaimType.FACT, text="wrong one", premises=[], source=SRC, step=1)
    _set(store, "b4", Status.CONTRADICTED)
    return store


def test_approx_tokens():
    assert (approx_tokens(""), approx_tokens("abc"), approx_tokens("a" * 8)) == (0, 1, 3)


def test_think_sections_order_and_exclusions(s: BeliefStore):
    task = Task(goal="How much?", documents=["doc one"], answer_format="a number")
    history = [StepSummary(step=1, action_kind="continue", note="started")]
    ctx = assemble_think_context(task, s, ["b9 contradicted: x"], history, 10_000, approx_tokens)
    assert ctx.messages[0].role == "system" and "Working:" in ctx.messages[0].content
    user = ctx.messages[-1].content
    headers = [
        "## Task",
        "## Documents",
        "## Problems found last step",
        "## Established",
        "## Assumptions in play",
        "## Unchecked claims",
        "## Your previous steps",
    ]
    positions = [user.index(h) for h in headers]
    assert positions == sorted(positions)
    assert "Answer format: a number" in user and "[1] doc one" in user
    assert "step 1: continue — started" in user
    assert "b4" not in user and "wrong one" not in user
    assert ctx.included == ["b1", "b2", "b3"] and ctx.dropped == []
    assert ctx.prompt_tokens == approx_tokens(ctx.messages[0].content) + approx_tokens(user)


def test_empty_sections_are_left_out():
    ctx = assemble_think_context(
        Task(goal="x"), BeliefStore.open(":memory:"), [], [], 10_000, approx_tokens
    )
    assert ctx.messages[-1].content == "## Task\nx"


def test_budget_drops_lowest_priority_oldest_first():
    store = BeliefStore.open(":memory:")
    store.add(type=ClaimType.GIVEN, text="the given fact", premises=[], source=SRC, step=0)
    for i in range(20):
        store.add(
            type=ClaimType.FACT,
            text=f"unchecked claim number {i:02d}",
            premises=[],
            source=SRC,
            step=1,
        )
    base = assemble_think_context(
        Task(goal="g"), BeliefStore.open(":memory:"), [], [], 10**6, approx_tokens
    ).prompt_tokens
    line_tokens = approx_tokens("b10 [unchecked] (fact) unchecked claim number 08\n")
    budget = base + 40 + 5 * line_tokens
    ctx = assemble_think_context(Task(goal="g"), store, [], [], budget, approx_tokens)
    assert "b1" in ctx.included
    assert "b21" in ctx.included and "b2" in ctx.dropped
    assert set(ctx.included) | set(ctx.dropped) == {f"b{i}" for i in range(1, 22)}
    assert 1 < len(ctx.included) < 21
    assert ctx.prompt_tokens <= budget
    user = ctx.messages[-1].content
    shown = [line.split(" ")[0] for line in user.splitlines() if line.startswith("b")]
    assert shown == sorted(shown, key=lambda i: int(i[1:]))


def test_history_left_out_when_it_does_not_fit(s: BeliefStore):
    history = [StepSummary(step=i, action_kind="continue", note="n" * 200) for i in range(10)]
    full = assemble_think_context(Task(goal="g"), s, [], history, 10**6, approx_tokens)
    assert full.messages[-1].content.count("step ") == 5
    tight = assemble_think_context(
        Task(goal="g"), s, [], history, full.prompt_tokens - 50, approx_tokens
    )
    assert "## Your previous steps" not in tight.messages[-1].content
    assert tight.included == ["b1", "b2", "b3"]


def test_mandatory_overflow_raises(s: BeliefStore):
    with pytest.raises(ContextOverflowError):
        assemble_think_context(Task(goal="x" * 10_000), s, [], [], 100, approx_tokens)


def test_belief_line_format():
    comp = Claim(
        id="b2",
        type=ClaimType.COMPUTATION,
        text="Total cost",
        computation=Computation(expression="12*7", claimed_result="84"),
        source=SRC,
        step=1,
    )
    rec = BeliefRecord(claim=comp, status=Status.VERIFIED, status_reason="")
    assert belief_line(rec) == "b2 [verified] (computation) Total cost :: 12*7 = 84"


def test_extract_context(s: BeliefStore):
    ctx = assemble_extract_context(s, "Working:\n1. x", 10_000, approx_tokens)
    user = ctx.messages[-1].content
    assert user.index("## Existing claims you may reference") < user.index(
        "## Reasoning to convert"
    )
    assert user.endswith("Working:\n1. x")
    assert "b4" not in user and ctx.included == ["b1", "b2", "b3"]
    assert "computation" in ctx.messages[0].content


def test_extract_context_without_beliefs():
    ctx = assemble_extract_context(BeliefStore.open(":memory:"), "r", 10_000, approx_tokens)
    assert ctx.messages[-1].content == "## Reasoning to convert\nr"


def test_belief_line_shows_formal():
    rec = BeliefRecord(
        claim=Claim(
            id="b2",
            type=ClaimType.INFERENCE,
            text="Rex barks",
            premises=("b1",),
            source=Source(kind=SourceKind.MODEL),
            step=1,
            formal=Formal(mode="undetermined", statements=("barks(rex)",)),
        ),
        status=Status.UNCHECKED,
        status_reason="",
    )
    assert belief_line(rec) == (
        "b2 [unchecked] (inference) Rex barks :: formal (undetermined): barks(rex)"
    )
