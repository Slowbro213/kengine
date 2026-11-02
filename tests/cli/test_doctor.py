from kengine.backend.fake import ScriptedBackend, ScriptedReply
from kengine.backend.types import BackendHealth
from kengine.cli.doctor import run_doctor
from kengine.config import load_config

CFG = load_config([])
HEALTHY = BackendHealth(ok=True, model="qwen.gguf", n_ctx=32768, detail="ok")


def test_doctor_ok_with_healthy_fake() -> None:
    backend = ScriptedBackend(
        [ScriptedReply(content='{"ok": true}'), ScriptedReply(content="5", reasoning="r")],
        health=HEALTHY,
    )
    rep = run_doctor(CFG, [], backend, nvidia_smi=lambda: "8151MiB / 8188MiB")
    assert rep.ok and rep.structured_ok and rep.thinking_separated is True
    assert rep.gpu is not None and rep.gpu.startswith("8151")
    assert backend.calls[0].schema_name == "Probe" and not backend.calls[0].role.thinking
    assert backend.calls[1].role.thinking


def test_doctor_not_ok_when_unreachable() -> None:
    backend = ScriptedBackend([], health=BackendHealth(ok=False, detail="connection refused"))
    rep = run_doctor(CFG, [], backend, nvidia_smi=lambda: None)
    assert rep.ok is False and backend.calls == [] and rep.gpu is None


def test_doctor_not_ok_when_structured_fails() -> None:
    backend = ScriptedBackend(
        [ScriptedReply(content='{"ok": '), ScriptedReply(content="5")], health=HEALTHY
    )
    rep = run_doctor(CFG, [], backend, nvidia_smi=lambda: None)
    assert rep.ok is False and "StructuredOutputError" in rep.structured_detail
    assert rep.thinking_separated is False
