"""Client for llama.cpp's `llama-server` (OpenAI-compatible chat endpoint plus native extras)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import PurePath
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from kengine.backend.types import (
    BackendError,
    BackendHealth,
    Completion,
    FinishReason,
    Message,
    RoleConfig,
    Usage,
    parse_structured,
)
from kengine.config import BackendConfig

_THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)


class _RespMessage(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None


class _RespChoice(BaseModel):
    message: _RespMessage
    finish_reason: str | None = None


class _RespUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class _ChatResponse(BaseModel):
    choices: list[_RespChoice]
    usage: _RespUsage | None = None


class _TokenizeResponse(BaseModel):
    tokens: list[Any]


class _GenSettings(BaseModel):
    n_ctx: int | None = None


class _Props(BaseModel):
    n_ctx: int | None = None
    model_path: str | None = None
    default_generation_settings: _GenSettings | None = None


def _finish_reason(raw: str | None) -> FinishReason:
    if raw == "stop":
        return "stop"
    if raw == "length":
        return "length"
    return "other"


def _split_think(content: str, reasoning: str | None) -> tuple[str, str | None]:
    """Move an inline `<think>...</think>` prefix into `reasoning`."""
    m = _THINK_RE.match(content)
    if m is None:
        return content, reasoning
    inline = m.group(1).strip()
    merged = inline if not reasoning else f"{reasoning}\n{inline}"
    return content[m.end() :], merged


class LlamaCppBackend:
    def __init__(self, config: BackendConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        self._client = client or httpx.Client(base_url=config.base_url, timeout=config.timeout_s)

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        try:
            resp = self._client.post(path, json=body)
        except httpx.HTTPError as exc:
            raise BackendError(f"POST {path} failed: {type(exc).__name__}: {exc}") from exc
        if resp.status_code // 100 != 2:
            raise BackendError(f"POST {path} returned HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise BackendError(f"POST {path} returned non-JSON: {resp.text[:300]}") from exc

    def _body(self, messages: Sequence[Message], role: RoleConfig) -> dict[str, Any]:
        s = role.sampling
        return {
            "model": self._config.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": s.temperature,
            "top_p": s.top_p,
            "top_k": s.top_k,
            "min_p": s.min_p,
            "presence_penalty": s.presence_penalty,
            "repeat_penalty": s.repetition_penalty,
            "max_tokens": role.max_tokens,
            "chat_template_kwargs": {"enable_thinking": role.thinking},
        }

    def _chat(self, body: dict[str, Any]) -> Completion:
        data = self._post("/v1/chat/completions", body)
        try:
            parsed = _ChatResponse.model_validate(data)
        except ValidationError as exc:
            raise BackendError(f"unexpected chat response shape: {str(data)[:300]}") from exc
        if not parsed.choices:
            raise BackendError("chat response had no choices")
        choice = parsed.choices[0]
        content, reasoning = _split_think(
            choice.message.content or "", choice.message.reasoning_content or None
        )
        usage = parsed.usage or _RespUsage()
        return Completion(
            content=content,
            reasoning=reasoning,
            usage=Usage(
                prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens
            ),
            finish_reason=_finish_reason(choice.finish_reason),
        )

    def complete(self, messages: Sequence[Message], role: RoleConfig) -> Completion:
        return self._chat(self._body(messages, role))

    def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], role: RoleConfig, schema: type[T]
    ) -> tuple[T, Completion]:
        body = self._body(messages, role)
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
        completion = self._chat(body)
        return parse_structured(completion, schema), completion

    def count_tokens(self, text: str) -> int:
        data = self._post("/tokenize", {"content": text})
        try:
            return len(_TokenizeResponse.model_validate(data).tokens)
        except ValidationError as exc:
            raise BackendError(f"unexpected tokenize response: {str(data)[:300]}") from exc

    def health(self) -> BackendHealth:
        try:
            h = self._client.get("/health")
            if h.status_code != 200:
                return BackendHealth(
                    ok=False, detail=f"/health HTTP {h.status_code}: {h.text[:200]}"
                )
            p = self._client.get("/props")
            if p.status_code != 200:
                return BackendHealth(ok=False, detail=f"/props HTTP {p.status_code}")
            props = _Props.model_validate(p.json())
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            return BackendHealth(ok=False, detail=f"{type(exc).__name__}: {exc}")
        n_ctx = props.n_ctx
        if props.default_generation_settings and props.default_generation_settings.n_ctx:
            n_ctx = props.default_generation_settings.n_ctx
        model = PurePath(props.model_path).name if props.model_path else None
        return BackendHealth(ok=True, model=model, n_ctx=n_ctx, detail="ok")
