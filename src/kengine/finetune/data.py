"""Training pairs for the FORMALIZE call, from FOLIO's training split.

Each pair is exactly what the agent sends (the system prompt, the task, the premises and
the conclusion numbered), and the JSON it should get back: FOLIO's expert logic, converted
by `fol`. A row is kept only when every premise and the conclusion convert, no statement
says something of every individual, and z3, on the converted forms alone, gives FOLIO's
label. That also filters out the expert logic's own mistakes.

Stories are split into train and dev by a hash of the story id, so the split is the same on
every run and no story is in both.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from kengine.backend.types import BackendError, ModelBackend, RoleConfig
from kengine.claims.formal import parse_statements, variables_of
from kengine.claims.formalize import Formalization, formalize_messages, read_formalization
from kengine.eval.adapters import folio
from kengine.eval.types import EvalItem
from kengine.finetune.fol import Story, Unsupported
from kengine.logic.entail import decide

MAX_GIVEN_CHARS = 400  # the agent leaves longer givens out (loop.agent)
LABELS = {"entailed": "True", "refuted": "False", "open": "Uncertain"}


class Pair(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    story_id: str
    label: str
    split: str  # train | dev
    messages: list[dict[str, str]]  # system, user
    completion: str  # the JSON answer


def dev_story(story_id: str, dev_percent: int = 10) -> bool:
    h = int(hashlib.sha256(story_id.encode()).hexdigest(), 16)
    return h % 100 < dev_percent


def dev_items(rows: Iterable[Mapping[str, object]], dev_percent: int = 10) -> list[EvalItem]:
    """Every row of the held-back stories, converted or not, as an eval item: puzzles that
    neither training nor the evals have used."""
    items: list[EvalItem] = []
    for i, row in enumerate(rows):
        if dev_story(str(row["story_id"]), dev_percent):
            item = folio.parse_row(row, i)
            items.append(item.model_copy(update={"id": f"folio-train-{item.id[len('folio-') :]}"}))
    return items


def completion_json(forms: list[list[str]]) -> str:
    """The answer in FORMALIZE's schema, spaced as in the prompt's example."""
    return json.dumps(
        {"givens": [{"given": i, "statements": s} for i, s in enumerate(forms, 1)]},
        ensure_ascii=False,
    )


def convert_row(row: Mapping[str, object]) -> list[list[str]]:
    """Each premise's statements, then the conclusion's. Raises Unsupported with the reason a
    row is dropped."""
    texts = [p for p in str(row["premises-FOL"]).splitlines() if p.strip()]
    item = folio.parse_row(row, 0)
    if len(texts) != len(item.givens):
        raise Unsupported("premise count doesn't match")
    if any(len(g) > MAX_GIVEN_CHARS for g in item.givens):
        raise Unsupported("a premise is too long")
    story = Story()
    forms = [story.convert(t) for t in texts]
    goal = story.convert(str(row["conclusion-FOL"]), conclusion=True)
    for f in [*forms, goal]:
        if any(not s.body and variables_of(s) for t in f for s in parse_statements(t)):
            # The agent drops a given like this (formalize.says_everything): it's
            # nearly always a lost "if" part, here too.
            raise Unsupported("says everything")
    premises = [s for f in forms for t in f for s in parse_statements(t)]
    d = decide(premises, [s for t in goal for s in parse_statements(t)])
    got = LABELS.get(d.kind, d.kind)
    if got != str(row["label"]).strip():
        raise Unsupported(f"z3 says {got}")
    return [*forms, goal]


def build_pairs(
    rows: Iterable[Mapping[str, object]], dev_percent: int = 10
) -> tuple[list[Pair], Counter[str]]:
    """The kept pairs, and how many rows were dropped for each reason."""
    pairs: list[Pair] = []
    dropped: Counter[str] = Counter()
    for i, row in enumerate(rows):
        try:
            forms = convert_row(row)
        except Unsupported as e:
            dropped[_reason(str(e))] += 1
            continue
        except Exception as e:  # noqa: BLE001  the formal parser's own errors
            dropped[f"malformed: {type(e).__name__}"] += 1
            continue
        item = folio.parse_row(row, i)
        assert item.conclusion is not None
        messages = formalize_messages(item.question, [*item.givens, item.conclusion], "text")
        story_id = str(row["story_id"])
        pairs.append(
            Pair(
                item_id=item.id,
                story_id=story_id,
                label=item.gold,
                split="dev" if dev_story(story_id, dev_percent) else "train",
                messages=[{"role": m.role, "content": m.content} for m in messages],
                completion=completion_json(forms),
            )
        )
    return pairs, dropped


def distill_gsm8k(backend: ModelBackend, role: RoleConfig, items: Iterable[EvalItem]) -> list[Pair]:
    """Pairs that keep what the model does today on arithmetic word problems.

    FORMALIZE also runs on gsm8k, where it should mostly translate nothing. Mixing in the
    current model's own answers keeps the tuned model from translating arithmetic. Only
    translations the fragment accepts are kept, as written; the rest become empty lists.
    """
    pairs: list[Pair] = []
    for item in items:
        givens = [g for g in item.givens if len(g) <= MAX_GIVEN_CHARS]
        if not givens:
            continue
        messages = formalize_messages(item.question, givens, "text")
        try:
            parsed, _ = backend.complete_structured(messages, role, Formalization)
        except BackendError:
            continue
        formal, _ = read_formalization(parsed, len(givens))
        # As the model wrote them (`3.79`), not canonical (`379/100`).
        raw = {g.given: [str(t) for t in g.statements] for g in reversed(parsed.givens)}
        forms = [raw[i] if i in formal else [] for i in range(1, len(givens) + 1)]
        pairs.append(
            Pair(
                item_id=item.id,
                story_id=item.id,
                label="",
                split="train",
                messages=[{"role": m.role, "content": m.content} for m in messages],
                completion=completion_json(forms),
            )
        )
    return pairs


def _reason(e: str) -> str:
    if "isn't bound" in e:
        return "a free variable"
    if e.startswith("needs"):
        return "needs too many statements"
    if e.startswith(("unexpected", "expected", "bad name")):
        return "malformed"
    return e
