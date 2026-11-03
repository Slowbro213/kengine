import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from kengine.backend.llamacpp import LlamaCppBackend
from kengine.backend.types import (
    BackendError,
    Message,
    RoleConfig,
    SamplingParams,
    StructuredOutputError,
)
from kengine.config import BackendConfig

THINK = RoleConfig(
    sampling=SamplingParams(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5),
    thinking=True,
    max_tokens=512,
)
EXTRACT = THINK.model_copy(update={"thinking": False})
MSGS = [Message(role="user", content="hi")]
CFG = BackendConfig(base_url="http://llm.test", model="qwen-test")


class Probe(BaseModel):
    ok: bool


Handler = Callable[[httpx.Request], httpx.Response]


def make(handler: Handler) -> LlamaCppBackend:
    client = httpx.Client(base_url=CFG.base_url, transport=httpx.MockTransport(handler))
    return LlamaCppBackend(CFG, client=client)


def chat_response(
    content: str, reasoning: str | None = None, finish: str = "stop"
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {
        "choices": [{"message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


class Capture:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=self.response)

    def body(self, i: int = 0) -> dict[str, Any]:
        return json.loads(self.requests[i].content)


def test_complete_sends_sampling_and_thinking_flag() -> None:
    cap = Capture(chat_response("x"))
    c = make(cap).complete(MSGS, THINK)
    req = cap.requests[0]
    assert req.method == "POST" and req.url.path == "/v1/chat/completions"
    body = cap.body()
    assert body["model"] == "qwen-test"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert (body["temperature"], body["top_p"], body["top_k"], body["min_p"]) == (
        1.0,
        0.95,
        20,
        0.0,
    )
    assert body["presence_penalty"] == 1.5 and body["repeat_penalty"] == 1.0
    assert body["max_tokens"] == 512
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert "response_format" not in body
    assert "thinking_budget_tokens" not in body
    assert c.usage.prompt_tokens == 11 and c.usage.completion_tokens == 7


def test_thinking_budget_is_sent_with_a_wrap_up_message() -> None:
    cap = Capture(chat_response("x"))
    make(cap).complete(MSGS, THINK.model_copy(update={"thinking_budget": 300}))
    body = cap.body()
    assert body["thinking_budget_tokens"] == 300
    assert "Working" in body["reasoning_budget_message"]


def test_thinking_budget_is_ignored_when_thinking_is_off() -> None:
    cap = Capture(chat_response("x"))
    role = THINK.model_copy(update={"thinking": False, "thinking_budget": 300})
    make(cap).complete(MSGS, role)
    assert "thinking_budget_tokens" not in cap.body()


def test_reasoning_content_returned_separately() -> None:
    c = make(Capture(chat_response("Final answer: 5", "2+3..."))).complete(MSGS, THINK)
    assert c.content == "Final answer: 5" and c.reasoning == "2+3..."


def test_inline_think_tags_are_split_out() -> None:
    c = make(Capture(chat_response("<think>abc</think>\n\nhello"))).complete(MSGS, THINK)
    assert c.content == "hello" and c.reasoning == "abc"


def test_no_reasoning_is_none() -> None:
    c = make(Capture(chat_response("plain", ""))).complete(MSGS, THINK)
    assert c.reasoning is None


def test_structured_sends_json_schema_response_format() -> None:
    cap = Capture(chat_response('{"ok": true}'))
    parsed, _ = make(cap).complete_structured(MSGS, EXTRACT, Probe)
    assert parsed == Probe(ok=True)
    body = cap.body()
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Probe", "schema": Probe.model_json_schema()},
    }
    assert body["chat_template_kwargs"]["enable_thinking"] is False


def test_structured_length_finish_raises() -> None:
    b = make(Capture(chat_response('{"ok": true}', finish="length")))
    with pytest.raises(StructuredOutputError):
        b.complete_structured(MSGS, EXTRACT, Probe)


def test_structured_schema_violation_raises() -> None:
    b = make(Capture(chat_response('{"wrong": 1}')))
    with pytest.raises(StructuredOutputError) as exc:
        b.complete_structured(MSGS, EXTRACT, Probe)
    assert exc.value.raw == '{"wrong": 1}'


def test_http_500_and_connect_error_raise_backend_error() -> None:
    b500 = make(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(BackendError, match="HTTP 500"):
        b500.complete(MSGS, THINK)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(BackendError, match="connection refused"):
        make(refuse).complete(MSGS, THINK)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(BackendError, match="ReadTimeout"):
        make(slow).complete(MSGS, THINK)


def test_malformed_response_raises_backend_error() -> None:
    with pytest.raises(BackendError):
        make(Capture({"nope": 1})).complete(MSGS, THINK)


@pytest.mark.parametrize(
    "raw,mapped", [("stop", "stop"), ("length", "length"), ("tool_calls", "other"), (None, "other")]
)
def test_finish_reason_mapping(raw: str | None, mapped: str) -> None:
    resp = chat_response("x")
    resp["choices"][0]["finish_reason"] = raw
    assert make(Capture(resp)).complete(MSGS, THINK).finish_reason == mapped


def test_count_tokens_uses_tokenize_endpoint() -> None:
    cap = Capture({"tokens": [1, 2, 3]})
    assert make(cap).count_tokens("hi") == 3
    assert cap.requests[0].url.path == "/tokenize" and cap.body() == {"content": "hi"}


def props_handler(props: dict[str, Any]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json=props)

    return handler


def test_health_reads_props() -> None:
    h = make(
        props_handler(
            {"default_generation_settings": {"n_ctx": 32768}, "model_path": "/m/qwen.gguf"}
        )
    ).health()
    assert (h.ok, h.n_ctx, h.model) == (True, 32768, "qwen.gguf")
    h2 = make(props_handler({"n_ctx": 4096})).health()
    assert h2.ok and h2.n_ctx == 4096


def test_health_unreachable_is_not_ok() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    h = make(refuse).health()
    assert not h.ok and "connection refused" in h.detail
