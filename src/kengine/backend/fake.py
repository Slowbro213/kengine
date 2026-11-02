"""A scripted backend for tests: replies are consumed in order and every call is recorded."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from kengine.backend.types import (
    BackendError,
    BackendHealth,
    Completion,
    FinishReason,
    Message,
    RoleConfig,
    ScriptExhausted,
    Usage,
    parse_structured,
)


class ScriptedReply(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str = ""
    reasoning: str | None = None
    finish_reason: FinishReason = "stop"
    error: str | None = None


class RecordedCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: list[Message]
    role: RoleConfig
    schema_name: str | None


class ScriptedBackend:
    def __init__(
        self, script: Sequence[ScriptedReply], health: BackendHealth | None = None
    ) -> None:
        self._script = list(script)
        self._next = 0
        self._health = health or BackendHealth(ok=True, model="scripted", n_ctx=32768)
        self.calls: list[RecordedCall] = []

    def _reply(
        self, messages: Sequence[Message], role: RoleConfig, schema_name: str | None
    ) -> Completion:
        self.calls.append(RecordedCall(messages=list(messages), role=role, schema_name=schema_name))
        if self._next >= len(self._script):
            raise ScriptExhausted(f"script exhausted after {len(self._script)} replies")
        reply = self._script[self._next]
        self._next += 1
        if reply.error is not None:
            raise BackendError(reply.error)
        usage = Usage(
            prompt_tokens=sum(self.count_tokens(m.content) for m in messages),
            completion_tokens=self.count_tokens(reply.content + (reply.reasoning or "")),
        )
        return Completion(
            content=reply.content,
            reasoning=reply.reasoning,
            usage=usage,
            finish_reason=reply.finish_reason,
        )

    def complete(self, messages: Sequence[Message], role: RoleConfig) -> Completion:
        return self._reply(messages, role, None)

    def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], role: RoleConfig, schema: type[T]
    ) -> tuple[T, Completion]:
        completion = self._reply(messages, role, schema.__name__)
        return parse_structured(completion, schema), completion

    def count_tokens(self, text: str) -> int:
        return math.ceil(len(text) / 4)

    def health(self) -> BackendHealth:
        return self._health

    @property
    def remaining(self) -> int:
        return len(self._script) - self._next
