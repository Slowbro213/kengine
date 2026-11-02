"""The kengine solver: the agent loop with checks and the final-answer gate (§6)."""

from __future__ import annotations

from pathlib import Path

from kengine.audit.events import FeedbackEvent, ThinkEvent
from kengine.audit.session import Session
from kengine.backend.types import ModelBackend
from kengine.checkers import default_checkers
from kengine.config import KengineConfig
from kengine.eval.baseline import EXCERPT_CHARS
from kengine.eval.types import EvalItem, SolveResult, SolveStatus
from kengine.loop.agent import Agent, Limits, RunStatus
from kengine.loop.profiles import Profile
from kengine.loop.task import Task

STATUS_MAP: dict[RunStatus, SolveStatus] = {
    "answered": "answered",
    "blocked": "blocked",
    "partial": "partial",
    "budget_exhausted": "partial",
    "error": "error",
}


class KengineSolver:
    name = "kengine"

    def __init__(
        self,
        backend: ModelBackend,
        config: KengineConfig,
        profile: Profile,
        sessions_root: Path,
        token_budget: int,
    ) -> None:
        self._backend = backend
        self._config = config
        self._profile = profile
        self._sessions_root = sessions_root
        self._token_budget = token_budget

    def solve(self, item: EvalItem) -> SolveResult:
        session = Session.create(self._sessions_root)
        try:
            agent = Agent(
                backend=self._backend,
                roles=self._config.roles,
                profile=self._profile,
                context_budget=self._config.context.step_prompt_tokens,
                checkers=default_checkers(),
                session=session,
                limits=Limits(max_completion_tokens=self._token_budget),
            )
            task = Task(
                goal=item.question, documents=item.documents, answer_format=item.answer_format
            )
            r = agent.run(task)
            excerpt = session_excerpt(session)
        finally:
            session.close()
        return SolveResult(
            answer=r.answer,
            status=STATUS_MAP[r.status],
            model_calls=r.model_calls,
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
            wall_s=r.wall_s,
            session_dir=str(r.session_dir),
            transcript_excerpt=excerpt,
            error=r.error,
        )


def session_excerpt(session: Session) -> str:
    """The last THINK text, then `---`, then the last feedback items."""
    think = ""
    feedback: list[str] = []
    for event in session.log.read():
        if isinstance(event, ThinkEvent):
            think = event.text
        elif isinstance(event, FeedbackEvent):
            feedback = event.items
    return f"{think[-EXCERPT_CHARS:]}\n---\n" + "\n".join(feedback)
