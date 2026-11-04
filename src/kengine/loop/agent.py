"""The agent loop (spec §5.2): think, extract, vote, insert, check, act, feed back."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kengine.audit.events import (
    BeliefAddedEvent,
    Event,
    ExtractionEvent,
    FeedbackEvent,
    FormalizeEvent,
    GateEvent,
    GivenReformalizedEvent,
    ModelCallEvent,
    RunFinishedEvent,
    SessionStartedEvent,
    StatusChangedEvent,
    ThinkEvent,
    VerdictEvent,
    VoteEvent,
)
from kengine.audit.session import Session
from kengine.backend.types import (
    BackendError,
    Completion,
    Message,
    ModelBackend,
    RoleConfig,
    StructuredOutputError,
)
from kengine.beliefs.status import TERMINAL
from kengine.beliefs.store import BeliefStore, StatusChange
from kengine.checkers.base import Checker, Outcome, Verdict
from kengine.claims.extraction import ExtractedClaim, Extraction, to_formal
from kengine.claims.formal import FormalSyntax, Statement, parse_statements
from kengine.claims.formalize import (
    Formalization,
    formalize_messages,
    read_formalization,
    reformalize_messages,
    says_everything,
)
from kengine.claims.schema import Claim, ClaimType, Formal, Source, SourceKind
from kengine.claims.voting import VoteResult, default_k, vote
from kengine.config import RolesConfig
from kengine.logic.entail import consistent
from kengine.loop.context import (
    ContextOverflowError,
    StepSummary,
    approx_tokens,
    assemble_extract_context,
    assemble_think_context,
)
from kengine.loop.feedback import build_feedback
from kengine.loop.gate import GateDecision, evaluate_gate
from kengine.loop.profiles import CheckScope, Profile
from kengine.loop.task import Task

RunStatus = Literal["answered", "blocked", "partial", "budget_exhausted", "error"]

LOOP_REPEATS = 3
MAX_GIVEN_CHARS = 400  # ClaimText's limit
CUT_OFF = "your reasoning was cut off before the Working section; be more concise"
NO_CLAIMS = "could not turn your reasoning into claims; keep the Working list short and numbered"
NO_ANSWER_TEXT = "final answer had no text"
_BELIEF_ID_RE = re.compile(r"\bb\d+\b")
LONE_GIVEN_FAILED = (
    "a step resting on it alone didn't follow from this translation; check that it says "
    "everything the sentence says"
)


class Limits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_steps: int | None = None
    max_completion_tokens: int | None = None
    max_wall_s: float | None = None


NO_LIMITS = Limits()


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: RunStatus
    answer: str | None
    blocked_answer: str | None
    session_dir: Path
    steps: int
    model_calls: int
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    acted_on_assumptions: list[str]
    error: str | None = None


class _Finished(Exception):
    """Raised inside the loop to stop it with a status."""

    def __init__(
        self, status: RunStatus, answer: str | None = None, assumed: list[str] | None = None
    ):
        self.status: RunStatus = status
        self.answer = answer
        self.assumed = assumed or []


@dataclass
class _Step:
    """What one step produced, for feedback and the step summary."""

    n: int
    issues: list[str] = field(default_factory=list[str])
    changes: list[StatusChange] = field(default_factory=list[StatusChange])
    verdicts: list[tuple[str, Verdict]] = field(default_factory=list[tuple[str, Verdict]])
    gate: GateDecision | None = None
    action_kind: str = "none"
    new_ids: list[str] = field(default_factory=list[str])


class Agent:
    def __init__(
        self,
        backend: ModelBackend,
        roles: RolesConfig,
        profile: Profile,
        context_budget: int,
        checkers: Sequence[Checker],
        session: Session,
        limits: Limits = NO_LIMITS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        retry_delay_s: float = 5.0,
        formal_syntax: FormalSyntax = "text",
        formalize_givens: bool = False,
    ) -> None:
        self.backend = backend
        self.roles = roles
        self.profile = profile
        self.context_budget = context_budget
        self.checkers = list(checkers)
        self.session = session
        self.limits = limits
        self.clock = clock
        self.sleep = sleep
        self.retry_delay_s = retry_delay_s
        self.formal_syntax: FormalSyntax = formal_syntax
        self.formalize_givens = formalize_givens
        self._reset()

    def _reset(self) -> None:
        self._start = self.clock()
        self._steps = 0
        self._calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._last_feedback: list[str] = []
        self._history: list[StepSummary] = []
        self._signatures: list[frozenset[str]] = []
        self._blocked_answer: str | None = None
        self._last_blocked = False
        # Under the load-bearing scope: claims not checked yet because nothing rested on them.
        self._skipped: set[str] = set()
        self._goal = ""
        self._reformalized: set[str] = set()  # givens translated again; each gets one retry

    @property
    def store(self) -> BeliefStore:
        return self.session.store

    # Driver

    def run(self, task: Task) -> RunResult:
        self._reset()
        self._log(SessionStartedEvent(step=0, task=task.model_dump(mode="json")))
        try:
            self._loop(task)
            raise AssertionError("unreachable: the loop only exits by _Finished")
        except _Finished as done:
            result = self._result(done.status, done.answer, done.assumed)
        except (BackendError, ContextOverflowError) as e:
            result = self._result("error", None, [], error=f"{type(e).__name__}: {e}")
        self._log(RunFinishedEvent(step=self._steps, result=result.model_dump(mode="json")))
        return result

    def _seed_givens(self, task: Task) -> None:
        """The task's statements become `given` claims b1, b2, ... before the first step,
        each with a formal form when FORMALIZE gives one. A statement too long for a claim
        is left out; it's still in the task text.

        A translation that says something holds of every individual is asked for again, once:
        it's usually "All X are Y" without its "if" part."""
        self._goal = task.goal
        kept = [t for t in task.givens if len(t) <= MAX_GIVEN_CHARS]
        formal: dict[int, Formal] = {}
        fixed: dict[int, tuple[Formal, str]] = {}  # number -> (old form, why)
        if self.formalize_givens and kept:
            formal = self._formalize(
                formalize_messages(task.goal, kept, self.formal_syntax), len(kept), 0
            )
            redo = {i: why for i, f in formal.items() if (why := says_everything(f)) is not None}
            if redo:
                messages = reformalize_messages(task.goal, kept, formal, redo, self.formal_syntax)
                again = self._formalize(messages, len(kept), 0)
                for i, why in redo.items():
                    new = again.get(i)
                    if new is not None and _accepts(formal, i, new):
                        fixed[i] = (formal[i], why)
                        formal[i] = new
        ids: dict[int, str] = {}
        for i, text in enumerate(kept, 1):
            claim = self.store.add(
                type=ClaimType.GIVEN,
                text=text,
                premises=[],
                source=Source(kind=SourceKind.TASK, ref=f"given:{i}"),
                step=0,
                formal=formal.get(i),
            )
            ids[i] = claim.id
            self._log(BeliefAddedEvent(step=0, claim=claim))
        for i, (old, why) in fixed.items():
            self._reformalized.add(ids[i])
            self._log(
                GivenReformalizedEvent(step=0, belief_id=ids[i], old=old, new=formal[i], reason=why)
            )

    def _formalize(self, messages: list[Message], n: int, step: int) -> dict[int, Formal]:
        """One FORMALIZE call over `n` statements. Any failure leaves the givens informal; it
        never ends the run."""
        for attempt in (1, 2):
            t0 = self.clock()
            try:
                parsed, completion = self.backend.complete_structured(
                    messages, self.roles.formalize, Formalization
                )
            except StructuredOutputError as e:
                if e.completion is not None:
                    self._record_call(messages, e.completion, "formalize", step, t0)
                self._log(FormalizeEvent(step=step, ok=False, raw=e.raw, errors=[str(e)]))
                return {}
            except BackendError as e:
                if attempt == 1:
                    self.sleep(self.retry_delay_s)
                    continue
                self._log(FormalizeEvent(step=step, ok=False, raw=None, errors=[str(e)]))
                return {}
            self._record_call(messages, completion, "formalize", step, t0)
            formal, errors = read_formalization(parsed, n)
            self._log(FormalizeEvent(step=step, ok=True, raw=completion.content, errors=errors))
            return formal
        return {}

    def _reformalize(self, given: str, failed: Verdict, step: _Step) -> bool:
        """Translate `given` again, without showing the step that failed against it. Returns
        whether the new form was taken."""
        self._reformalized.add(given)
        givens = [
            r
            for r in self.store.all()
            if r.claim.type == ClaimType.GIVEN and r.claim.source.kind == SourceKind.TASK
        ]
        i = [r.claim.id for r in givens].index(given) + 1
        current = {n: r.claim.formal for n, r in enumerate(givens, 1) if r.claim.formal}
        messages = reformalize_messages(
            self._goal,
            [r.claim.text for r in givens],
            current,
            {i: LONE_GIVEN_FAILED},
            self.formal_syntax,
        )
        new = self._formalize(messages, len(givens), step.n).get(i)
        if new is None or not _accepts(current, i, new):
            return False
        self.store.set_formal(given, new)
        reason = f"a step resting on it alone didn't follow: {failed.explanation}"
        self._log(
            GivenReformalizedEvent(
                step=step.n, belief_id=given, old=current.get(i), new=new, reason=reason
            )
        )
        return True

    def _lone_given(self, claim: Claim, verdict: Verdict) -> str | None:
        """The given to translate again: the only premise of an inference the logic checker
        found doesn't follow, when that given hasn't been translated again yet."""
        if verdict.checker_id != "logic" or verdict.outcome != Outcome.FAIL:
            return None
        if verdict.status is not None or claim.type != ClaimType.INFERENCE:
            return None
        if len(claim.premises) != 1 or claim.premises[0] in self._reformalized:
            return None
        given = self.store.get(claim.premises[0])
        if given.claim.type != ClaimType.GIVEN or given.claim.formal is None:
            return None
        if given.status in TERMINAL:
            return None
        return given.claim.id

    def _loop(self, task: Task) -> None:
        self._seed_givens(task)
        while True:
            self._check_limits()
            self._steps += 1
            step = _Step(n=self._steps)
            think = self._think(task, step)
            if think is not None:
                samples = self._extract(think, step)
                if samples:
                    voted = vote(samples, default_k(len(samples)))
                    self._log(
                        VoteEvent(
                            step=step.n,
                            kept=[c.ref for c in voted.claims],
                            dropped=voted.dropped,
                            n_valid=voted.n_valid,
                        )
                    )
                    refs = self._insert(voted.claims, step)
                    preconditions = self._translate_preconditions(voted, refs, step)
                    self._check(preconditions, step)
                    self._act(voted, preconditions, step)
            self._feedback(step)

    # Stages

    def _think(self, task: Task, step: _Step) -> str | None:
        """The THINK text, or None when it was cut off before saying anything."""
        ctx = assemble_think_context(
            task, self.store, self._last_feedback, self._history, self.context_budget, approx_tokens
        )
        completion = self._with_retry(
            lambda: self._complete(ctx.messages, self.roles.think, "think", step.n)
        )
        self._log(ThinkEvent(step=step.n, text=completion.content, reasoning=completion.reasoning))
        if not completion.content.strip() and completion.finish_reason == "length":
            step.issues.append(CUT_OFF)
            return None
        return completion.content

    def _extract(self, think_text: str, step: _Step) -> list[Extraction]:
        ctx = assemble_extract_context(
            self.store, think_text, self.context_budget, approx_tokens, self.formal_syntax
        )
        samples, backend_failures = self._extract_round(ctx.messages, step)
        if not samples and backend_failures == self.profile.n_samples:
            self.sleep(self.retry_delay_s)
            samples, backend_failures = self._extract_round(ctx.messages, step)
            if not samples and backend_failures == self.profile.n_samples:
                raise BackendError("every extraction call failed twice")
        if not samples:
            step.issues.append(NO_CLAIMS)
        return samples

    def _extract_round(self, messages: list[Message], step: _Step) -> tuple[list[Extraction], int]:
        samples: list[Extraction] = []
        backend_failures = 0
        for i in range(self.profile.n_samples):
            t0 = self.clock()
            try:
                parsed, completion = self.backend.complete_structured(
                    messages, self.roles.extract, Extraction
                )
            except StructuredOutputError as e:
                if e.completion is not None:
                    self._record_call(messages, e.completion, "extract", step.n, t0)
                self._log(
                    ExtractionEvent(step=step.n, sample_index=i, ok=False, raw=e.raw, error=str(e))
                )
                continue
            except BackendError as e:
                backend_failures += 1
                self._log(
                    ExtractionEvent(step=step.n, sample_index=i, ok=False, raw=None, error=str(e))
                )
                continue
            self._record_call(messages, completion, "extract", step.n, t0)
            self._log(
                ExtractionEvent(
                    step=step.n, sample_index=i, ok=True, raw=completion.content, error=None
                )
            )
            samples.append(parsed)
        return samples, backend_failures

    def _insert(self, claims: Sequence[ExtractedClaim], step: _Step) -> dict[str, str]:
        """Add the voted claims, premises first. Returns the map from `c#` refs to ids."""
        refs: dict[str, str] = {}
        kept = {c.ref for c in claims}
        pending = list(claims)
        while pending:
            # The first claim whose in-step premises are all inserted (or will never be).
            ready = next(
                (c for c in pending if all(p in refs or p not in kept for p in _c_refs(c))),
                pending[0],  # only reachable through a cycle: its pending links get cut
            )
            pending.remove(ready)
            premises: list[str] = []
            for p in ready.premises:
                resolved = (
                    refs.get(p) if p.startswith("c") else (p if self.store.exists(p) else None)
                )
                if resolved is None:
                    step.issues.append(f"{ready.ref} cited unknown {p}; link removed")
                elif resolved not in premises:
                    premises.append(resolved)
            formal: Formal | None = None
            if ready.formal is not None:
                try:
                    formal = to_formal(ready.formal)
                except ValueError as e:
                    step.issues.append(f"{ready.ref}'s formal form was dropped: {e}")
            claim = self.store.add(
                type=ClaimType(ready.type),
                text=ready.text,
                premises=premises,
                source=Source(kind=SourceKind.MODEL, ref=f"step:{step.n}"),
                step=step.n,
                computation=ready.computation,
                formal=formal,
            )
            refs[ready.ref] = claim.id
            step.new_ids.append(claim.id)
            self._log(BeliefAddedEvent(step=step.n, claim=claim))
        return refs

    def _translate_preconditions(
        self, voted: VoteResult, refs: dict[str, str], step: _Step
    ) -> list[str]:
        if voted.action is None:
            return []
        out: list[str] = []
        for p in voted.action.preconditions:
            resolved = refs.get(p) if p.startswith("c") else (p if self.store.exists(p) else None)
            if resolved is None:
                step.issues.append(f"{voted.action.kind} cited unknown {p}; link removed")
            elif resolved not in out:
                out.append(resolved)
        return out

    def _check(self, preconditions: Sequence[str], step: _Step) -> None:
        scope = step.new_ids
        if self.profile.check_scope == CheckScope.LOAD_BEARING:
            # A claim skipped because nothing rested on it yet is checked once something does.
            self._skipped.update(step.new_ids)
            closure = self.store.premise_closure(preconditions)
            scope = [i for i in closure if i in self._skipped]
            self._skipped.difference_update(scope)
        for id in scope:
            for checker in self.checkers:
                record = self.store.get(id)
                if record.status in TERMINAL:
                    break
                if not checker.applies_to(record.claim):
                    continue
                verdict = checker.check(record.claim, self.store)
                given = self._lone_given(record.claim, verdict)
                if given is not None and self._reformalize(given, verdict, step):
                    verdict = checker.check(record.claim, self.store)
                self._log(VerdictEvent(step=step.n, belief_id=id, verdict=verdict))
                step.verdicts.append((id, verdict))
                for change in self.store.apply_verdict(id, verdict, step.n):
                    step.changes.append(change)
                    self._log(StatusChangedEvent(step=step.n, change=change))

    def _act(self, voted: VoteResult, preconditions: list[str], step: _Step) -> None:
        action = voted.action
        if action is None:
            return
        step.action_kind = action.kind
        if action.kind == "continue":
            self._last_blocked = False
            return
        answer = (action.answer or "").strip()
        if not answer:
            step.issues.append(NO_ANSWER_TEXT)
            self._last_blocked = False
            return
        decision = evaluate_gate(preconditions, self.store, self.profile.final_gate)
        self._log(GateEvent(step=step.n, decision=decision.model_dump(mode="json")))
        if decision.allowed:
            raise _Finished("answered", answer, decision.acted_on_assumptions)
        step.gate = decision
        self._blocked_answer = answer
        self._last_blocked = True

    def _feedback(self, step: _Step) -> None:
        items = build_feedback(step.changes, step.verdicts, step.issues, step.gate)
        self._log(FeedbackEvent(step=step.n, items=items))
        self._last_feedback = items
        note = f"{len(step.new_ids)} new claim(s)"
        if step.gate is not None:
            note += f"; final answer {self._blocked_answer!r} blocked"
        if items:
            note += f"; {len(items)} problem(s)"
        self._history.append(StepSummary(step=step.n, action_kind=step.action_kind, note=note))

        signature = frozenset(_BELIEF_ID_RE.sub("b#", item) for item in items)
        self._signatures.append(signature)
        recent = self._signatures[-LOOP_REPEATS:]
        if len(recent) == LOOP_REPEATS and signature and all(s == signature for s in recent):
            raise _Finished("blocked" if self._blocked_answer is not None else "partial")

    # Helpers

    def _check_limits(self) -> None:
        max_steps = self.profile.max_steps
        if self.limits.max_steps is not None:
            max_steps = min(max_steps, self.limits.max_steps)
        over = (
            self._steps >= max_steps
            or (
                self.limits.max_completion_tokens is not None
                and self._completion_tokens >= self.limits.max_completion_tokens
            )
            or (
                self.limits.max_wall_s is not None
                and self.clock() - self._start >= self.limits.max_wall_s
            )
        )
        if over:
            raise _Finished("blocked" if self._last_blocked else "budget_exhausted")

    def _with_retry(self, call: Callable[[], Completion]) -> Completion:
        try:
            return call()
        except StructuredOutputError:
            raise
        except BackendError:
            self.sleep(self.retry_delay_s)
            return call()

    def _complete(
        self, messages: list[Message], role: RoleConfig, name: str, step: int
    ) -> Completion:
        t0 = self.clock()
        completion = self.backend.complete(messages, role)
        self._record_call(messages, completion, name, step, t0)
        return completion

    def _record_call(
        self, messages: list[Message], completion: Completion, role: str, step: int, t0: float
    ) -> None:
        self._calls += 1
        self._prompt_tokens += completion.usage.prompt_tokens
        self._completion_tokens += completion.usage.completion_tokens
        self._log(
            ModelCallEvent(
                step=step,
                role=role,
                messages=messages,
                completion=completion,
                wall_s=self.clock() - t0,
            )
        )

    def _log(self, event: Event) -> None:
        self.session.log.append(event)

    def _result(
        self,
        status: RunStatus,
        answer: str | None,
        assumed: list[str],
        error: str | None = None,
    ) -> RunResult:
        return RunResult(
            status=status,
            answer=answer,
            blocked_answer=None if status == "answered" else self._blocked_answer,
            session_dir=self.session.dir,
            steps=self._steps,
            model_calls=self._calls,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            wall_s=self.clock() - self._start,
            acted_on_assumptions=assumed,
            error=error,
        )


def _statements(formals: Iterable[Formal]) -> list[Statement]:
    return [s for f in formals for text in f.statements for s in parse_statements(text)]


def _accepts(current: Mapping[int, Formal], i: int, new: Formal) -> bool:
    """A new translation of statement `i` is taken if it changes the form and doesn't make
    consistent givens contradict each other."""
    if new == current.get(i):
        return False
    after = {**current, i: new}
    return consistent(_statements(after.values())) or not consistent(_statements(current.values()))


def _c_refs(claim: ExtractedClaim) -> list[str]:
    return [p for p in claim.premises if p.startswith("c")]
