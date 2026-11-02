"""`kengine doctor`: is the backend up, does structured output work, is thinking separated?"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel

from kengine.backend.types import BackendError, BackendHealth, Message, ModelBackend
from kengine.config import KengineConfig

THINK_PROBE_MAX_TOKENS = 2048


class Probe(BaseModel):
    ok: bool


class DoctorReport(BaseModel):
    config_paths: list[str]
    health: BackendHealth
    structured_ok: bool
    structured_detail: str
    thinking_separated: bool | None
    thinking_detail: str = ""
    gpu: str | None
    ok: bool


def nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip() or None


def run_doctor(
    config: KengineConfig,
    config_paths: Sequence[Path],
    backend: ModelBackend,
    nvidia_smi: Callable[[], str | None],
) -> DoctorReport:
    paths = [str(p) for p in config_paths if p.is_file()]
    health = backend.health()
    gpu = nvidia_smi()
    if not health.ok:
        return DoctorReport(
            config_paths=paths,
            health=health,
            structured_ok=False,
            structured_detail="skipped: backend not healthy",
            thinking_separated=None,
            gpu=gpu,
            ok=False,
        )

    structured_ok, structured_detail = False, ""
    probe_msgs = [Message(role="user", content='Reply with the JSON object {"ok": true}.')]
    try:
        parsed, _ = backend.complete_structured(probe_msgs, config.roles.extract, Probe)
        structured_ok, structured_detail = True, f"parsed {parsed.model_dump_json()}"
    except BackendError as exc:
        structured_detail = f"{type(exc).__name__}: {exc}"

    thinking_separated: bool | None = None
    thinking_detail = ""
    role = config.roles.think.model_copy(
        update={"max_tokens": min(config.roles.think.max_tokens, THINK_PROBE_MAX_TOKENS)}
    )
    try:
        c = backend.complete([Message(role="user", content="What is 2+3? Answer briefly.")], role)
        thinking_separated = c.reasoning is not None
        thinking_detail = f"content={c.content[:60]!r} finish={c.finish_reason}"
    except BackendError as exc:
        thinking_detail = f"{type(exc).__name__}: {exc}"

    return DoctorReport(
        config_paths=paths,
        health=health,
        structured_ok=structured_ok,
        structured_detail=structured_detail,
        thinking_separated=thinking_separated,
        thinking_detail=thinking_detail,
        gpu=gpu,
        ok=health.ok and structured_ok,
    )


def render_report(rep: DoctorReport) -> str:
    def mark(ok: bool | None) -> str:
        return {True: "ok  ", False: "FAIL", None: "??  "}[ok]

    h = rep.health
    lines = [
        f"config files : {', '.join(rep.config_paths) or '(defaults only)'}",
        f"[{mark(h.ok)}] backend    : {h.detail}  model={h.model} n_ctx={h.n_ctx}",
        f"[{mark(rep.structured_ok)}] structured : {rep.structured_detail}",
        f"[{mark(rep.thinking_separated)}] thinking   : separated={rep.thinking_separated} "
        f"{rep.thinking_detail}",
        f"       gpu        : {rep.gpu or 'nvidia-smi not available'}",
        f"overall: {'OK' if rep.ok else 'NOT OK'}",
    ]
    return "\n".join(lines)
