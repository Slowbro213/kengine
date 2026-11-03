"""Types shared by every model backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SamplingParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int = Field(ge=0)
    min_p: float = Field(ge=0.0, le=1.0)
    presence_penalty: float
    repetition_penalty: float = 1.0


class RoleConfig(BaseModel):
    """Sampling settings, thinking switch and output limit for one model role."""

    model_config = ConfigDict(frozen=True)

    sampling: SamplingParams
    thinking: bool
    max_tokens: int = Field(gt=0)
    # Thinking tokens before the server closes the thinking block, leaving the rest of
    # `max_tokens` for the answer. None: think until done or out of `max_tokens`.
    thinking_budget: int | None = Field(default=None, gt=0)


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0


FinishReason = Literal["stop", "length", "other"]


class Completion(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    reasoning: str | None = None
    usage: Usage = Usage()
    finish_reason: FinishReason = "stop"


class BackendHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    model: str | None = None
    n_ctx: int | None = None
    detail: str = ""


class BackendError(Exception):
    """The backend could not produce a completion (network, HTTP, server error)."""


class StructuredOutputError(BackendError):
    """A structured completion was truncated or did not validate against the schema.

    `completion` is set when the backend returned one, so its usage still counts.
    """

    def __init__(self, message: str, raw: str, completion: Completion | None = None) -> None:
        super().__init__(message)
        self.raw = raw
        self.completion = completion


class ScriptExhausted(BackendError):
    """The scripted fake backend ran out of replies."""


class ModelBackend(Protocol):
    def complete(self, messages: Sequence[Message], role: RoleConfig) -> Completion: ...

    def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], role: RoleConfig, schema: type[T]
    ) -> tuple[T, Completion]: ...

    def count_tokens(self, text: str) -> int: ...

    def health(self) -> BackendHealth: ...


def parse_structured[T: BaseModel](completion: Completion, schema: type[T]) -> T:
    """Shared structured-output semantics: truncation or invalid JSON raises."""
    if completion.finish_reason == "length":
        raise StructuredOutputError(
            "structured output truncated (finish_reason=length)", completion.content, completion
        )
    try:
        return schema.model_validate_json(completion.content)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        raise StructuredOutputError(
            f"structured output invalid for {schema.__name__}: {loc}: {first['msg']}",
            completion.content,
            completion,
        ) from exc
