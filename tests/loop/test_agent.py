import json
from pathlib import Path

from kengine.audit.session import Session
from kengine.backend.fake import ScriptedBackend, ScriptedReply
from kengine.backend.types import FinishReason
from kengine.beliefs.status import Status
from kengine.checkers import default_checkers
from kengine.checkers.arithmetic import ArithmeticChecker
from kengine.checkers.base import Checker
from kengine.checkers.structure import StructureChecker
from kengine.claims.schema import ClaimType, SourceKind
from kengine.config import RolesConfig
from kengine.loop.agent import NO_LIMITS, Agent, Limits
from kengine.loop.profiles import get_profile
from kengine.loop.task import Task, split_sentences

Claim = dict[str, object]
Action = dict[str, object]


def think(text: str = "Working:\n1. something\nNext: continue", finish: FinishReason = "stop"):
    return ScriptedReply(content=text, finish_reason=finish)


def extract_json(claims: list[Claim], action: Action) -> ScriptedReply:
    return ScriptedReply(content=json.dumps({"claims": claims, "action": action}))


def comp(ref: str, expr: str, result: str, premises: list[str] | None = None) -> Claim:
    return {
        "ref": ref,
        "type": "computation",
        "text": f"{expr} = {result}",
        "premises": premises or [],
        "computation": {"expression": expr, "claimed_result": result},
    }


def claim(ref: str, type_: str, text: str, premises: list[str] | None = None) -> Claim:
    return {"ref": ref, "type": type_, "text": text, "premises": premises or []}


def final(answer: str, preconditions: list[str]) -> Action:
    return {"kind": "final_answer", "answer": answer, "preconditions": preconditions}


CONTINUE: Action = {"kind": "continue"}


class Sleeps(list[float]):
    def __call__(self, s: float) -> None:
        self.append(s)


def make_agent(
    script: list[ScriptedReply],
    profile: str,
    tmp: Path,
    limits: Limits = NO_LIMITS,
    sleep: Sleeps | None = None,
    checkers: list[Checker] | None = None,
) -> tuple[Agent, ScriptedBackend]:
    backend = ScriptedBackend(script)
    agent = Agent(
        backend=backend,
        roles=RolesConfig(),
        profile=get_profile(profile),
        context_budget=8000,
        checkers=[StructureChecker(), ArithmeticChecker()] if checkers is None else checkers,
        session=Session.create(tmp),
        limits=limits,
        sleep=Sleeps() if sleep is None else sleep,
    )
    return agent, backend


def think_prompts(backend: ScriptedBackend) -> list[str]:
    return [c.messages[-1].content for c in backend.calls if c.schema_name is None]


def events(session_dir: Path) -> list[dict[str, object]]:
    lines = (session_dir / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_happy_path_verified_computation_answers(tmp_path: Path):
    script = [
        think("Working:\n1. 12*7 = 84\nFinal answer: 84\nRests on: 1"),
        *[extract_json([comp("c1", "12*7", "84")], final("84", ["c1"]))] * 3,
    ]
    agent, _ = make_agent(script, "standard", tmp_path)
    r = agent.run(Task(goal="12*7?"))
    assert (r.status, r.answer, r.steps, r.model_calls) == ("answered", "84", 1, 4)
    assert r.blocked_answer is None and r.completion_tokens > 0
    assert Session.open(r.session_dir).store.get("b1").status == Status.VERIFIED
    kinds = [e["kind"] for e in events(r.session_dir)]
    assert kinds[0] == "session_started" and kinds[-1] == "run_finished"
    assert {"think", "extraction", "vote", "belief_added", "verdict", "gate"} <= set(kinds)


def test_wrong_arithmetic_is_contradicted_and_fed_back(tmp_path: Path):
    script = [
        think(),
        *[extract_json([comp("c1", "12*7", "82")], final("82", ["c1"]))] * 3,
        think(),
        *[extract_json([comp("c1", "12*7", "84")], final("84", ["c1"]))] * 3,
    ]
    agent, backend = make_agent(script, "standard", tmp_path)
    r = agent.run(Task(goal="12*7?"))
    assert (r.status, r.answer, r.steps) == ("answered", "84", 2)
    step2 = think_prompts(backend)[1]
    assert "b1 contradicted: `12*7` evaluates to 84, not 82" in step2
    assert "final answer blocked: b1 is contradicted" in step2


def test_assumption_answer_recorded_under_standard(tmp_path: Path):
    # The plan's version rested the answer on an unchecked inference, which its own
    # standard-gate rules block; a verified computation on the assumption keeps the intent.
    script = [
        think(),
        *[
            extract_json(
                [claim("c1", "assumption", "a pen costs 2"), comp("c2", "3*2", "6", ["c1"])],
                final("6", ["c2"]),
            )
        ]
        * 3,
    ]
    agent, _ = make_agent(script, "standard", tmp_path)
    r = agent.run(Task(goal="3 pens?"))
    assert (r.status, r.answer, r.acted_on_assumptions) == ("answered", "6", ["b1"])


def test_all_extractions_invalid_gives_feedback_and_continues(tmp_path: Path):
    script = [
        think(),
        *[ScriptedReply(content="{not json")] * 3,
        think(),
        *[extract_json([comp("c1", "2+2", "4")], final("4", ["c1"]))] * 3,
    ]
    agent, backend = make_agent(script, "standard", tmp_path)
    r = agent.run(Task(goal="2+2?"))
    assert (r.status, r.answer, r.steps) == ("answered", "4", 2)
    assert "could not turn your reasoning into claims" in think_prompts(backend)[1]
    failed = [e for e in events(r.session_dir) if e["kind"] == "extraction" and not e["ok"]]
    assert len(failed) == 3


def test_think_cutoff_skips_extraction(tmp_path: Path):
    script = [
        think("", finish="length"),
        think(),
        *[extract_json([comp("c1", "2+2", "4")], final("4", ["c1"]))] * 3,
    ]
    agent, backend = make_agent(script, "standard", tmp_path)
    r = agent.run(Task(goal="2+2?"))
    assert r.status == "answered"
    assert backend.calls[1].schema_name is None
    assert "reasoning was cut off" in think_prompts(backend)[1]


def test_backend_failure_retries_once_then_errors(tmp_path: Path):
    sleeps = Sleeps()
    script = [ScriptedReply(error="boom"), ScriptedReply(error="boom again")]
    agent, _ = make_agent(script, "standard", tmp_path, sleep=sleeps)
    r = agent.run(Task(goal="x"))
    assert r.status == "error" and r.error is not None and "boom again" in r.error
    assert sleeps == [5.0]
    assert events(r.session_dir)[-1]["kind"] == "run_finished"
    assert (r.session_dir / "beliefs.sqlite").exists()


def test_backend_failure_recovers_after_retry(tmp_path: Path):
    script = [
        ScriptedReply(error="boom"),
        think(),
        *[extract_json([comp("c1", "2+2", "4")], final("4", ["c1"]))] * 3,
    ]
    agent, _ = make_agent(script, "standard", tmp_path)
    assert agent.run(Task(goal="2+2?")).status == "answered"


def test_every_extract_call_failing_is_retried(tmp_path: Path):
    sleeps = Sleeps()
    script = [
        think(),
        ScriptedReply(error="down"),
        *[extract_json([comp("c1", "2+2", "4")], final("4", ["c1"]))],
    ]
    agent, _ = make_agent(script, "quick", tmp_path, sleep=sleeps)
    r = agent.run(Task(goal="2+2?"))
    assert (r.status, sleeps) == ("answered", [5.0])


def test_loop_detection_stops_partial(tmp_path: Path):
    step = [think(), *[extract_json([claim("c1", "inference", "so x")], CONTINUE)] * 3]
    agent, _ = make_agent(step * 4, "standard", tmp_path)
    r = agent.run(Task(goal="x"))
    assert (r.status, r.steps) == ("partial", 3)


def test_step_limit_budget_exhausted(tmp_path: Path):
    step = [think(), extract_json([claim("c1", "fact", "x is 1")], CONTINUE)]
    agent, _ = make_agent(step * 5, "quick", tmp_path, limits=Limits(max_steps=2))
    r = agent.run(Task(goal="x"))
    assert (r.status, r.steps, r.answer) == ("budget_exhausted", 2, None)


def test_completion_token_limit(tmp_path: Path):
    step = [think("x" * 400), extract_json([claim("c1", "fact", "x is 1")], CONTINUE)]
    limits = Limits(max_completion_tokens=50)
    agent, _ = make_agent(step * 5, "quick", tmp_path, limits=limits)
    r = agent.run(Task(goal="x"))
    assert (r.status, r.steps) == ("budget_exhausted", 1)


def test_blocked_answer_reported_when_limit_hit(tmp_path: Path):
    script = [
        think(),
        extract_json([claim("c1", "fact", "Paris is the capital")], final("Paris", ["c1"])),
    ]
    agent, _ = make_agent(script, "quick", tmp_path, limits=Limits(max_steps=1))
    r = agent.run(Task(goal="capital of France?"))
    assert (r.status, r.answer, r.blocked_answer) == ("blocked", None, "Paris")


def test_empty_final_answer_is_an_issue(tmp_path: Path):
    script = [
        think(),
        extract_json([comp("c1", "2+2", "4")], final("  ", ["c1"])),
        think(),
        extract_json([], final("4", ["b1"])),
    ]
    agent, backend = make_agent(script, "quick", tmp_path)
    r = agent.run(Task(goal="2+2?"))
    assert (r.status, r.answer) == ("answered", "4")
    assert "final answer had no text" in think_prompts(backend)[1]


def test_load_bearing_scope_only_checks_answer_closure(tmp_path: Path):
    script = [
        think(),
        extract_json([comp("c1", "2+2", "4"), comp("c2", "3*3", "9")], final("9", ["c2"])),
    ]
    agent, _ = make_agent(script, "quick", tmp_path)
    r = agent.run(Task(goal="3*3?"))
    store = Session.open(r.session_dir).store
    assert r.status == "answered"
    assert store.get("b2").status == Status.VERIFIED
    assert store.get("b1").status == Status.UNCHECKED


def test_load_bearing_scope_checks_a_skipped_claim_once_an_answer_rests_on_it(tmp_path: Path):
    script = [
        think(),
        extract_json([comp("c1", "2+2", "4")], CONTINUE),
        think(),
        extract_json([], final("4", ["b1"])),
    ]
    agent, _ = make_agent(script, "quick", tmp_path)
    r = agent.run(Task(goal="2+2?"))
    assert (r.status, r.answer) == ("answered", "4")
    assert Session.open(r.session_dir).store.get("b1").status == Status.VERIFIED


def test_unresolvable_premise_link_removed_with_issue(tmp_path: Path):
    script = [
        think(),
        extract_json([claim("c1", "inference", "so x", ["b42", "c9"])], CONTINUE),
        think(),
        extract_json([comp("c1", "2+2", "4")], final("4", ["c1"])),
    ]
    agent, backend = make_agent(script, "quick", tmp_path)
    r = agent.run(Task(goal="x"))
    assert r.status == "answered"
    assert Session.open(r.session_dir).store.get("b1").claim.premises == ()
    step2 = think_prompts(backend)[1]
    assert "c1 cited unknown b42; link removed" in step2
    assert "c1 cited unknown c9; link removed" in step2


def test_premises_inserted_before_dependents(tmp_path: Path):
    script = [
        think(),
        extract_json([comp("c2", "4*2", "8", ["c1"]), comp("c1", "2+2", "4")], final("8", ["c2"])),
    ]
    agent, _ = make_agent(script, "quick", tmp_path)
    r = agent.run(Task(goal="x"))
    store = Session.open(r.session_dir).store
    assert r.status == "answered"
    assert store.get("b1").claim.text == "2+2 = 4"
    assert store.get("b2").claim.premises == ("b1",)


def test_task_givens_are_seeded_and_restating_facts_supported(tmp_path: Path):
    script = [
        think("Working:\n1. Each pen costs $1.25 (from b1).\n2. 1.25 * 12 = 15 (from 1 and b2)"),
        extract_json(
            [
                claim("c1", "fact", "Each pen costs $1.25", ["b1"]),
                comp("c2", "1.25 * 12", "15", ["c1", "b2"]),
            ],
            final("15", ["c2"]),
        ),
    ]
    agent, backend = make_agent(script, "quick", tmp_path, checkers=default_checkers())
    givens = split_sentences("A shop sells pens at $1.25. How much do 12 pens cost?")
    r = agent.run(Task(goal="pens", givens=givens))
    assert (r.status, r.answer) == ("answered", "15")
    store = Session.open(r.session_dir).store
    assert [store.get(i).claim.type for i in ("b1", "b2")] == [ClaimType.GIVEN] * 2
    assert store.get("b1").claim.source.kind == SourceKind.TASK
    assert (store.get("b3").status, store.get("b4").status) == (Status.SUPPORTED, Status.VERIFIED)
    assert "b1 [unchecked] (given) A shop sells pens at $1.25." in think_prompts(backend)[0]


def test_unsupported_fact_still_blocks(tmp_path: Path):
    script = [
        think(),
        extract_json(
            [claim("c1", "fact", "Each pen costs $2", ["b1"]), comp("c2", "2 * 12", "24", ["c1"])],
            final("24", ["c2"]),
        ),
    ]
    limits = Limits(max_steps=1)
    agent, _ = make_agent(script, "quick", tmp_path, limits=limits, checkers=default_checkers())
    r = agent.run(Task(goal="pens", givens=["A shop sells pens at $1.25."]))
    assert (r.status, r.blocked_answer) == ("blocked", "24")


def test_bad_formal_form_is_dropped_but_claim_kept(tmp_path: Path):
    bad = claim("c1", "fact", "Rex is a dog")
    bad["formal"] = {"statements": ["dog(rex"]}
    agent, backend = make_agent(
        [think(), extract_json([bad], CONTINUE), think(), extract_json([], CONTINUE)],
        "quick",
        tmp_path,
        limits=Limits(max_steps=2),
    )
    agent.run(Task(goal="g"))
    rec = agent.store.get("b1")
    assert rec.claim.text == "Rex is a dog" and rec.claim.formal is None
    assert "c1's formal form was dropped: unbalanced" in think_prompts(backend)[1]


def formalize_json(*items: tuple[int, list[str]]) -> ScriptedReply:
    return ScriptedReply(
        content=json.dumps({"givens": [{"given": g, "statements": s} for g, s in items]})
    )


def test_givens_are_formalized_and_a_logical_answer_verified(tmp_path: Path):
    inference = claim("c1", "inference", "Rex is a mammal", ["b1", "b2"])
    inference["formal"] = {"statements": ["mammal(rex)"]}
    backend = ScriptedBackend(
        [
            formalize_json((1, ["dog(x) -> mammal(x)"]), (2, ["dog(rex)"]), (3, [])),
            think(),
            extract_json([inference], final("True", ["c1"])),
        ]
    )
    agent = Agent(
        backend=backend,
        roles=RolesConfig(),
        profile=get_profile("standard").model_copy(update={"n_samples": 1}),
        context_budget=8000,
        checkers=default_checkers(),
        session=Session.create(tmp_path),
        sleep=Sleeps(),
        formalize_givens=True,
    )
    r = agent.run(
        Task(goal="g", givens=["All dogs are mammals.", "Rex is a dog.", "Is Rex a mammal?"])
    )
    assert r.status == "answered" and r.answer == "True"
    assert agent.store.get("b4").status == Status.VERIFIED
    assert [c.schema_name for c in backend.calls][0] == "Formalization"
    kinds = [e.kind for e in agent.session.log.read()]
    assert kinds.index("formalize") < kinds.index("belief_added")
    assert "formal: dog(rex)" in backend.calls[2].messages[-1].content  # extract sees it


def test_formalize_failure_does_not_stop_the_run(tmp_path: Path):
    backend = ScriptedBackend(
        [
            ScriptedReply(content="{not json"),
            think(),
            extract_json([], CONTINUE),
        ]
    )
    agent = Agent(
        backend=backend,
        roles=RolesConfig(),
        profile=get_profile("quick"),
        context_budget=8000,
        checkers=default_checkers(),
        session=Session.create(tmp_path),
        limits=Limits(max_steps=1),
        sleep=Sleeps(),
        formalize_givens=True,
    )
    r = agent.run(Task(goal="g", givens=["Rex is a dog."]))
    assert r.status == "budget_exhausted"
    assert agent.store.get("b1").claim.formal is None
    events = [e for e in agent.session.log.read() if e.kind == "formalize"]
    assert len(events) == 1 and not events[0].ok  # type: ignore[union-attr]


def test_formalize_backend_error_retries_once(tmp_path: Path):
    sleeps = Sleeps()
    backend = ScriptedBackend(
        [
            ScriptedReply(error="down"),
            ScriptedReply(error="down"),
            think(),
            extract_json([], CONTINUE),
        ]
    )
    agent = Agent(
        backend=backend,
        roles=RolesConfig(),
        profile=get_profile("quick"),
        context_budget=8000,
        checkers=default_checkers(),
        session=Session.create(tmp_path),
        limits=Limits(max_steps=1),
        sleep=sleeps,
        formalize_givens=True,
    )
    assert agent.run(Task(goal="g", givens=["Rex is a dog."])).status == "budget_exhausted"
    assert sleeps == [5.0]


def formal_agent(script: list[ScriptedReply], tmp: Path) -> tuple[Agent, ScriptedBackend]:
    backend = ScriptedBackend(script)
    agent = Agent(
        backend=backend,
        roles=RolesConfig(),
        profile=get_profile("standard").model_copy(update={"n_samples": 1}),
        context_budget=8000,
        checkers=default_checkers(),
        session=Session.create(tmp),
        sleep=Sleeps(),
        formalize_givens=True,
    )
    return agent, backend


SHAPES = ["All four-sided things are shapes.", "Tile is four-sided.", "Is Tile a shape?"]


def test_a_given_that_says_everything_is_translated_again(tmp_path: Path):
    inference = claim("c1", "inference", "Tile is a shape", ["b1", "b2"])
    inference["formal"] = {"statements": ["shape(tile)"]}
    agent, backend = formal_agent(
        [
            formalize_json((1, ["four_sided(x)"]), (2, ["four_sided(tile)"]), (3, [])),
            formalize_json((1, ["four_sided(x) -> shape(x)"])),
            think(),
            extract_json([inference], final("True", ["c1"])),
        ],
        tmp_path,
    )
    r = agent.run(Task(goal="g", givens=SHAPES))
    assert (r.status, r.answer) == ("answered", "True")
    asked = backend.calls[1]
    assert asked.schema_name == "Formalization"
    assert (
        "1. All four-sided things are shapes.\n   now: `four_sided(x)`"
        in asked.messages[-1].content
    )
    assert "Statement 1: `four_sided(x)` says it holds for every individual" in (
        asked.messages[-1].content
    )
    formal = agent.store.get("b1").claim.formal
    assert formal is not None and formal.statements == ("four_sided(x) -> shape(x)",)
    (event,) = [e for e in agent.session.log.read() if e.kind == "given_reformalized"]
    assert event.belief_id == "b1"  # type: ignore[union-attr]


def test_a_given_is_translated_again_when_a_step_resting_on_it_alone_fails(tmp_path: Path):
    restated = claim("c1", "inference", "Four-sided things are shapes", ["b1"])
    restated["formal"] = {"statements": ["four_sided(x) -> shape(x)"]}
    agent, backend = formal_agent(
        [
            formalize_json(
                (1, ["four_sided(x) -> polygon(x)"]), (2, ["four_sided(tile)"]), (3, [])
            ),
            think(),
            extract_json([restated], CONTINUE),
            formalize_json((1, ["four_sided(x) -> shape(x)"])),
            think(),
            extract_json([], CONTINUE),
        ],
        tmp_path,
    )
    agent.limits = Limits(max_steps=2)
    agent.run(Task(goal="g", givens=SHAPES))
    assert [c.schema_name for c in backend.calls][:4] == [
        "Formalization",
        None,
        "Extraction",
        "Formalization",
    ]
    assert "a step resting on it alone didn't follow" in backend.calls[3].messages[-1].content
    assert "Four-sided things are shapes" not in backend.calls[3].messages[-1].content
    assert agent.store.get("b4").status == Status.VERIFIED


def test_a_translation_that_makes_the_givens_contradict_is_rejected(tmp_path: Path):
    restated = claim("c1", "inference", "Tile is a shape", ["b1"])
    restated["formal"] = {"statements": ["shape(tile)"]}
    agent, _ = formal_agent(
        [
            formalize_json((1, ["polygon(tile)"]), (2, ["not shape(tile)"]), (3, [])),
            think(),
            extract_json([restated], CONTINUE),
            formalize_json((1, ["shape(tile)"])),
            think(),
            extract_json([], CONTINUE),
        ],
        tmp_path,
    )
    agent.limits = Limits(max_steps=2)
    agent.run(Task(goal="g", givens=SHAPES))
    formal = agent.store.get("b1").claim.formal
    assert formal is not None and formal.statements == ("polygon(tile)",)
    assert not [e for e in agent.session.log.read() if e.kind == "given_reformalized"]
    assert agent.store.get("b4").status == Status.UNCHECKED


def test_a_retranslation_that_still_says_everything_is_rejected(tmp_path: Path):
    agent, _ = formal_agent(
        [
            formalize_json((1, ["four_sided(x)"]), (2, ["four_sided(tile)"]), (3, [])),
            formalize_json((1, ["four_sided(x) | shape(x)"])),
            think(),
            extract_json([], CONTINUE),
        ],
        tmp_path,
    )
    agent.limits = Limits(max_steps=1)
    agent.run(Task(goal="g", givens=SHAPES))
    formal = agent.store.get("b1").claim.formal
    assert formal is not None and formal.statements == ("four_sided(x)",)


def test_the_answer_is_checked_against_the_conclusion(tmp_path: Path):
    inference = claim("c1", "inference", "Tile is a shape", ["b1", "b2"])
    inference["formal"] = {"statements": ["shape(tile)"]}
    agent, backend = formal_agent(
        [
            formalize_json(
                (1, ["four_sided(x) -> shape(x)"]), (2, ["four_sided(tile)"]), (3, ["shape(tile)"])
            ),
            think(),
            extract_json([inference], final("False", ["c1"])),
            think(),
            extract_json([], final("True", ["b3"])),
        ],
        tmp_path,
    )
    task = Task(goal="g", givens=SHAPES[:2], conclusion="Tile is a shape.")
    r = agent.run(task)
    assert (r.status, r.answer) == ("answered", "True")
    assert "3. Tile is a shape." in backend.calls[0].messages[-1].content
    assert [c.claim.type for c in agent.store.all()].count(ClaimType.GIVEN) == 2
    (event,) = [e for e in agent.session.log.read() if e.kind == "conclusion_formalized"]
    assert event.formal is not None and event.formal.statements == ("shape(tile)",)  # type: ignore[union-attr]
    gates = [e for e in agent.session.log.read() if e.kind == "gate"]
    assert gates[0].decision["reasons"] == [  # type: ignore[union-attr]
        "the conclusion follows from b3, so it isn't false"
    ]
