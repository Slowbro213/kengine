"""The session event log: one JSON line per event (spec §5.4)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from kengine.backend.types import Completion, Message
from kengine.beliefs.store import StatusChange
from kengine.checkers.base import Verdict
from kengine.claims.schema import Claim


def _now() -> datetime:
    return datetime.now(UTC)


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: datetime = Field(default_factory=_now)
    step: int


class SessionStartedEvent(_Event):
    kind: Literal["session_started"] = "session_started"
    # The Task model, dumped; it doesn't exist until Task 13.
    task: dict[str, object]


class ModelCallEvent(_Event):
    kind: Literal["model_call"] = "model_call"
    role: str
    messages: list[Message]
    completion: Completion
    wall_s: float


class ThinkEvent(_Event):
    kind: Literal["think"] = "think"
    text: str
    reasoning: str | None


class ExtractionEvent(_Event):
    kind: Literal["extraction"] = "extraction"
    sample_index: int
    ok: bool
    raw: str | None
    error: str | None


class VoteEvent(_Event):
    kind: Literal["vote"] = "vote"
    kept: list[str]
    dropped: list[str]
    n_valid: int


class BeliefAddedEvent(_Event):
    kind: Literal["belief_added"] = "belief_added"
    claim: Claim


class VerdictEvent(_Event):
    kind: Literal["verdict"] = "verdict"
    belief_id: str
    verdict: Verdict


class StatusChangedEvent(_Event):
    kind: Literal["status_changed"] = "status_changed"
    change: StatusChange


class GateEvent(_Event):
    kind: Literal["gate"] = "gate"
    decision: dict[str, object]


class FeedbackEvent(_Event):
    kind: Literal["feedback"] = "feedback"
    items: list[str]


class FormalizeEvent(_Event):
    kind: Literal["formalize"] = "formalize"
    ok: bool
    raw: str | None
    errors: list[str]


class RunFinishedEvent(_Event):
    kind: Literal["run_finished"] = "run_finished"
    result: dict[str, object]


Event = Annotated[
    SessionStartedEvent
    | ModelCallEvent
    | ThinkEvent
    | ExtractionEvent
    | VoteEvent
    | BeliefAddedEvent
    | VerdictEvent
    | StatusChangedEvent
    | GateEvent
    | FeedbackEvent
    | FormalizeEvent
    | RunFinishedEvent,
    Field(discriminator="kind"),
]
_EVENT: TypeAdapter[Event] = TypeAdapter(Event)


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: Event) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
            f.flush()

    def read(self) -> list[Event]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [_EVENT.validate_json(line) for line in lines if line.strip()]
