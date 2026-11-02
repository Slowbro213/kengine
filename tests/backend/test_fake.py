import pytest
from pydantic import BaseModel

from kengine.backend.fake import ScriptedBackend, ScriptedReply
from kengine.backend.types import (
    BackendError,
    Message,
    RoleConfig,
    SamplingParams,
    ScriptExhausted,
    StructuredOutputError,
)

ROLE = RoleConfig(
    sampling=SamplingParams(temperature=0.3, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=0),
    thinking=False,
    max_tokens=100,
)
MSGS = [Message(role="user", content="hello there")]


class Probe(BaseModel):
    ok: bool


def test_replies_consumed_in_order_and_calls_recorded() -> None:
    b = ScriptedBackend([ScriptedReply(content="one"), ScriptedReply(content="two")])
    assert b.complete(MSGS, ROLE).content == "one"
    assert b.complete(MSGS, ROLE).content == "two"
    assert len(b.calls) == 2 and b.calls[1].schema_name is None
    assert b.calls[0].messages == MSGS and b.calls[0].role == ROLE


def test_structured_valid_json_parses() -> None:
    b = ScriptedBackend([ScriptedReply(content='{"ok": true}')])
    parsed, completion = b.complete_structured(MSGS, ROLE, Probe)
    assert parsed == Probe(ok=True) and completion.content == '{"ok": true}'
    assert b.calls[0].schema_name == "Probe"


def test_structured_invalid_json_raises_with_raw() -> None:
    b = ScriptedBackend([ScriptedReply(content='{"ok": ')])
    with pytest.raises(StructuredOutputError) as exc:
        b.complete_structured(MSGS, ROLE, Probe)
    assert exc.value.raw == '{"ok": '
    assert exc.value.completion is not None


def test_structured_truncated_raises_even_if_parseable() -> None:
    b = ScriptedBackend([ScriptedReply(content='{"ok": true}', finish_reason="length")])
    with pytest.raises(StructuredOutputError):
        b.complete_structured(MSGS, ROLE, Probe)


def test_error_reply_raises_backend_error() -> None:
    b = ScriptedBackend([ScriptedReply(error="connection refused")])
    with pytest.raises(BackendError, match="connection refused"):
        b.complete(MSGS, ROLE)


def test_exhausted_script_raises() -> None:
    with pytest.raises(ScriptExhausted):
        ScriptedBackend([]).complete(MSGS, ROLE)


def test_count_tokens_is_ceil_chars_over_4() -> None:
    b = ScriptedBackend([])
    assert b.count_tokens("abcde") == 2 and b.count_tokens("") == 0


def test_usage_counts_tokens_of_reply() -> None:
    b = ScriptedBackend([ScriptedReply(content="abcdefgh", reasoning="xyz")])
    c = b.complete(MSGS, ROLE)
    assert c.usage.completion_tokens == b.count_tokens("abcdefghxyz")
    assert c.usage.prompt_tokens == sum(b.count_tokens(m.content) for m in MSGS)
