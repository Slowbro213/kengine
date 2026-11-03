"""Which formal serialization does the model fill in more reliably? (spec §9)

Each FOLIO item's premises and conclusion are formalized in one call. Z3 then decides the
conclusion: entailed is True, refuted is False, open is Uncertain. That measures the
translation, which is what the logic checker depends on, without the rest of the loop.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from kengine.backend.types import BackendError, ModelBackend, RoleConfig
from kengine.claims.formal import FormalSyntax, Statement, parse_statements
from kengine.claims.formalize import Formalization, formalize_messages, read_formalization
from kengine.eval.types import EvalItem
from kengine.logic.entail import decide

_CONCLUSION_RE = re.compile(r"^Conclusion: (.+)$", re.M)
LABELS = {"entailed": "True", "refuted": "False", "open": "Uncertain"}


class ProbeLine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    gold: str
    predicted: str | None
    correct: bool
    n_premises: int
    n_formalized: int
    conclusion_ok: bool
    decision: str | None
    errors: list[str]
    wall_s: float


class ProbeSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    syntax: FormalSyntax
    n: int
    all_formalized: int
    decided: int
    correct: int
    mean_wall_s: float


def probe_item(
    backend: ModelBackend,
    role: RoleConfig,
    syntax: FormalSyntax,
    item: EvalItem,
    clock: Callable[[], float] = time.monotonic,
) -> ProbeLine:
    m = _CONCLUSION_RE.search(item.question)
    conclusion = m.group(1).strip() if m else ""
    texts = [*item.givens, conclusion]
    n = len(item.givens)
    t0 = clock()

    def line(**kw: object) -> ProbeLine:
        fields: dict[str, object] = {
            "item_id": item.id, "gold": item.gold, "predicted": None, "correct": False,
            "n_premises": n, "n_formalized": 0, "conclusion_ok": False, "decision": None,
            "errors": [], "wall_s": clock() - t0,
        }  # fmt: skip
        return ProbeLine.model_validate({**fields, **kw})

    try:
        parsed, _ = backend.complete_structured(
            formalize_messages(item.question, texts, syntax), role, Formalization
        )
    except BackendError as e:
        return line(errors=[f"{type(e).__name__}: {e}"])
    formal, errors = read_formalization(parsed, len(texts))
    premises: list[Statement] = [
        s for i in range(1, n + 1) if i in formal for t in formal[i].statements
        for s in parse_statements(t)
    ]  # fmt: skip
    n_formalized = sum(1 for i in range(1, n + 1) if i in formal)
    goal = formal.get(n + 1)
    if goal is None:
        return line(n_formalized=n_formalized, errors=errors)
    d = decide(premises, [s for t in goal.statements for s in parse_statements(t)])
    predicted = LABELS.get(d.kind)
    if d.reason:
        errors.append(d.reason)
    return line(
        predicted=predicted,
        correct=predicted == item.gold,
        n_formalized=n_formalized,
        conclusion_ok=True,
        decision=d.kind,
        errors=errors,
    )


def run_probe(
    backend: ModelBackend,
    role: RoleConfig,
    syntax: FormalSyntax,
    items: Sequence[EvalItem],
    out_dir: Path,
) -> ProbeSummary:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[ProbeLine] = []
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as f:
        for item in items:
            ln = probe_item(backend, role, syntax, item)
            lines.append(ln)
            f.write(ln.model_dump_json() + "\n")
            f.flush()
    summary = ProbeSummary(
        syntax=syntax,
        n=len(lines),
        all_formalized=sum(ln.conclusion_ok and ln.n_formalized == ln.n_premises for ln in lines),
        decided=sum(ln.predicted is not None for ln in lines),
        correct=sum(ln.correct for ln in lines),
        mean_wall_s=sum(ln.wall_s for ln in lines) / len(lines) if lines else 0.0,
    )
    (out_dir / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n")
    return summary
