"""Worked solutions for FOLIO-train rows, one rule at a time (stage-2 fine-tuning data).

From the expert logic (converted by `fol`), each True or False row gets a chain of steps
that ends in the conclusion or its negation. A step applies one given rule to facts
already known (unit resolution over the rule's ground instances): `dog(rex)` and
`dog(x) -> mammal(x)` give `mammal(rex)`. Only the steps the end needs are kept. When the
conclusion is a rule, or needs reasoning by cases, the last step cites what z3 needs to
show it instead. An Uncertain row gets one step: the givens don't settle the conclusion.

Every chain is checked by the real logic checker, as the agent would check it, and a row
is dropped if any step or the answer fails.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kengine.beliefs.store import BeliefStore
from kengine.checkers.base import Outcome
from kengine.checkers.logic import LogicChecker, check_answer
from kengine.claims.formal import Atom, Lit, Statement, constants_of, parse_statements, variables_of
from kengine.claims.schema import ClaimType, Formal, FormalMode, Source, SourceKind
from kengine.eval.adapters import folio
from kengine.finetune.data import convert_row, dev_story
from kengine.logic.entail import decide
from kengine.loop.context import approx_tokens, assemble_extract_context, assemble_think_context
from kengine.loop.task import Task

MAX_INSTANCES = 5_000


class NoChain(Exception):
    pass


@dataclass(frozen=True)
class Step:
    statements: tuple[str, ...]
    cites: tuple[str, ...]  # given ids (`b2`) and earlier steps, numbered from 1 (`3`)
    mode: FormalMode = "holds"


def _neg(lit: Lit) -> Lit:
    return Lit(lit.atom, not lit.positive)


def _ground(s: Statement) -> Lit | None:
    """The one literal `s` states outright, if it's a ground fact."""
    if s.body or len(s.head) != 1 or variables_of(s) or not isinstance(s.head[0].atom, Atom):
        return None
    return s.head[0]


def _instances(s: Statement, constants: Sequence[str]) -> Iterator[list[Lit]]:
    """`s` over `constants`, each as a disjunction of literals."""
    if any(not isinstance(lit.atom, Atom) for lit in (*s.body, *s.head)):
        return
    names = variables_of(s)
    for combo in itertools.product(constants, repeat=len(names)):
        env = dict(zip(names, combo, strict=True))
        yield [_neg(_sub(lit, env)) for lit in s.body] + [_sub(lit, env) for lit in s.head]


def _sub(lit: Lit, env: dict[str, str]) -> Lit:
    a = lit.atom
    assert isinstance(a, Atom)
    return Lit(Atom(a.pred, tuple(env.get(x, x) for x in a.args)), lit.positive)


def _statements(texts: Sequence[str]) -> list[Statement]:
    return [s for t in texts for s in parse_statements(t)]


def prove(givens: Sequence[Sequence[str]], goal: Sequence[str], label: str) -> list[Step]:
    """The steps that show `label` (True, False or Uncertain) for `goal` from `givens`,
    whose ids are b1, b2, … Raises NoChain if there's no chain this can write."""
    ids = [f"b{i}" for i in range(1, len(givens) + 1)]
    if label == "Uncertain":
        return [Step(tuple(goal), tuple(ids), "undetermined")]
    given_stmts = {id: _statements(g) for id, g in zip(ids, givens, strict=True)}
    goal_stmts = _statements(goal)
    constants = sorted(
        {c for ss in [*given_stmts.values(), goal_stmts] for s in ss for c in constants_of(s)}
    )

    # Forward: what each rule gives, one at a time, until nothing new follows.
    known: dict[Lit, str] = {}  # literal -> the given or step that states it
    for id, ss in given_stmts.items():
        for s in ss:
            lit = _ground(s)
            if lit is not None:
                known.setdefault(lit, id)
    rules = [
        (id, lits)
        for id, ss in given_stmts.items()
        for s in ss
        if _ground(s) is None
        for lits in _instances(s, constants)
    ]
    if len(rules) > MAX_INSTANCES:
        raise NoChain("too many instances")
    derived: list[tuple[Lit, tuple[str, ...]]] = []  # step literal, what it cites
    grew = True
    while grew:
        grew = False
        for id, lits in rules:
            unsettled = [lit for lit in lits if _neg(lit) not in known]
            if len(unsettled) != 1 or unsettled[0] in known:
                continue
            new = unsettled[0]
            cites = (id, *(known[_neg(lit)] for lit in lits if lit != new))
            derived.append((new, tuple(dict.fromkeys(cites))))
            known[new] = str(len(derived))
            grew = True

    # Backward: the last step, then only the steps it needs.
    target = _ground(goal_stmts[0]) if len(goal_stmts) == 1 else None
    want = None if target is None else (target if label == "True" else _neg(target))
    if want is not None and want in known:
        last_cites: tuple[str, ...] = (known[want],)
        last = (str(want),)
    else:
        sources = [(id, s) for id, ss in given_stmts.items() for s in ss]
        sources += [(str(i), Statement((), (lit,))) for i, (lit, _) in enumerate(derived, 1)]
        d = decide([s for _, s in sources], goal_stmts)
        if d.kind != ("entailed" if label == "True" else "refuted"):
            raise NoChain(f"z3 says {d.kind}")
        last_cites = tuple(dict.fromkeys(sources[i][0] for i in d.core))
        if label == "True":
            last = tuple(goal)
        else:
            # A case the conclusion gets wrong, from facts already known: for `dog(x) ->
            # barks(x)`, `dog(rex)` and `not barks(rex)`.
            case = next(
                (
                    lits
                    for s in goal_stmts
                    for lits in _instances(s, constants)
                    if all(_neg(lit) in known for lit in lits)
                ),
                None,
            )
            if case is None:
                raise NoChain("no case that breaks the conclusion")
            last = tuple(dict.fromkeys(str(_neg(lit)) for lit in case))
    needed: set[int] = set()
    todo = [c for c in last_cites if not c.startswith("b")]
    while todo:
        n = int(todo.pop())
        if n not in needed:
            needed.add(n)
            todo += [c for c in derived[n - 1][1] if not c.startswith("b")]
    renumber = {str(n): str(i) for i, n in enumerate(sorted(needed), 1)}

    def cite(cs: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(renumber.get(c, c) for c in cs)

    steps = [Step((str(derived[n - 1][0]),), cite(derived[n - 1][1])) for n in sorted(needed)]
    if last == (steps[-1].statements if steps else None) and last_cites == (str(max(needed)),):
        return steps  # the last derived fact is the answer itself
    return [*steps, Step(last, cite(last_cites))]


def check_chain(
    givens: Sequence[Sequence[str]], goal: Sequence[str], label: str, steps: Sequence[Step]
) -> str | None:
    """Why the agent's checker wouldn't accept `steps` as an answer of `label`, or None."""
    store = BeliefStore.open(":memory:")
    task = Source(kind=SourceKind.TASK)
    for i, g in enumerate(givens, 1):
        store.add(
            type=ClaimType.GIVEN,
            text=f"given {i}",
            premises=[],
            source=task,
            step=0,
            formal=Formal(statements=tuple(g)) if g else None,
        )
    ids: list[str] = []
    checker = LogicChecker()
    for step in steps:
        premises = [c if c.startswith("b") else ids[int(c) - 1] for c in step.cites]
        claim = store.add(
            type=ClaimType.INFERENCE,
            text="step",
            premises=premises,
            source=Source(kind=SourceKind.MODEL),
            step=1,
            formal=Formal(mode=step.mode, statements=step.statements),
        )
        v = checker.check(claim, store)
        if v.outcome != Outcome.PASS:
            return f"step {len(ids) + 1}: {v.explanation}"
        store.apply_verdict(claim.id, v, step=1)
        ids.append(claim.id)
    rests_on = store.premise_closure([ids[-1]])
    return check_answer(label, Formal(statements=tuple(goal)), rests_on, store)


def _from(cites: Sequence[str]) -> str:
    return " and ".join(cites)


def working(steps: Sequence[Step], label: str, conclusion: str, n_givens: int) -> str:
    """The end of the THINK reply the agent asks for: the `Working:` items, the answer and
    what it rests on."""
    lines: list[str] = []
    for i, step in enumerate(steps, 1):
        if i < len(steps):
            text = "; ".join(step.statements)
        elif label == "Uncertain":
            text = f"b1 to b{n_givens} don't settle whether this holds: {conclusion.rstrip('.')}"
        elif label == "True":
            text = conclusion.rstrip(".")
        else:
            text = f"The conclusion is false: {' and '.join(step.statements)}"
        lines.append(f"{i}. {text} (from {_from(step.cites)}).")
    return "Working:\n" + "\n".join(lines) + f"\n\nFinal answer: {label}\nRests on: {len(steps)}"


def extraction(steps: Sequence[Step], working_text: str, label: str) -> str:
    """The extract role's JSON for `working_text`."""
    texts = [
        line.split(". ", 1)[1].rsplit(" (from ", 1)[0]
        for line in working_text.splitlines()
        if line[:1].isdigit()
    ]
    claims = [
        {
            "ref": f"c{i}",
            "type": "inference",
            "text": text,
            "premises": [c if c.startswith("b") else f"c{c}" for c in step.cites],
            "formal": {"mode": step.mode, "statements": list(step.statements)},
        }
        for i, (step, text) in enumerate(zip(steps, texts, strict=True), 1)
    ]
    action = {"kind": "final_answer", "answer": label, "preconditions": [f"c{len(steps)}"]}
    return json.dumps({"claims": claims, "action": action}, ensure_ascii=False)


class ChainPair(BaseModel):
    """One training pair for the THINK or the extract call, in `data.Pair`'s shape. A THINK
    pair is trained with thinking on: `completion` holds the thinking, then the reply."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    story_id: str
    label: str
    split: str  # train | dev
    role: Literal["think", "extract"]
    thinking: bool
    messages: list[dict[str, str]]
    completion: str


def _thinking(steps: Sequence[Step], label: str) -> str:
    lines = [
        f"From {_from(s.cites)}: {'; '.join(s.statements)}." for s in steps if s.mode == "holds"
    ]
    if label == "Uncertain":
        lines.append("Nothing in the givens settles the conclusion either way.")
    return "\n".join([*lines, f"So the answer is {label}."])


def row_pairs(
    row: Mapping[str, object],
    index: int,
    forms: Sequence[Sequence[str]],
    split: str,
    budget: int,
) -> list[ChainPair]:
    """The THINK and extract pairs for a converted row (`data.convert_row`), with prompts
    built as the agent builds them within `budget` tokens. Raises NoChain."""
    item = folio.parse_row(row, index)
    assert item.conclusion is not None
    givens, goal, label = forms[:-1], forms[-1], item.gold
    steps = prove(givens, goal, label)
    why = check_chain(givens, goal, label, steps)
    if why is not None:
        raise NoChain(f"the checker refuses it: {why}")
    store = BeliefStore.open(":memory:")
    for text, g in zip(item.givens, givens, strict=True):
        store.add(
            type=ClaimType.GIVEN,
            text=text,
            premises=[],
            source=Source(kind=SourceKind.TASK),
            step=0,
            formal=Formal(statements=tuple(g)) if g else None,
        )
    task = Task(
        goal=item.question,
        answer_format=item.answer_format,
        givens=item.givens,
        conclusion=item.conclusion,
    )
    think = assemble_think_context(task, store, [], [], budget, approx_tokens)
    reply = working(steps, label, item.conclusion, len(givens))
    extract = assemble_extract_context(store, reply, budget, approx_tokens, "text")
    common = {"item_id": item.id, "story_id": str(row["story_id"]), "label": label, "split": split}
    return [
        ChainPair(
            **common,
            role="think",
            thinking=True,
            messages=[{"role": m.role, "content": m.content} for m in think.messages],
            completion=f"{_thinking(steps, label)}\n</think>\n\n{reply}",
        ),
        ChainPair(
            **common,
            role="extract",
            thinking=False,
            messages=[{"role": m.role, "content": m.content} for m in extract.messages],
            completion=extraction(steps, reply, label),
        ),
    ]


def build_chain_pairs(
    rows: Iterable[Mapping[str, object]],
    budget: int,
    dev_percent: int = 10,
    uncertain_percent: int = 100,
) -> tuple[list[ChainPair], Counter[str]]:
    """The pairs for every row that converts and has a chain, and how many rows were dropped
    for each reason. Stories split into train and dev as for FORMALIZE (`data.dev_story`).
    Only about `uncertain_percent` of the train rows labelled Uncertain are kept, picked by a
    hash of the row; dev keeps them all."""
    pairs: list[ChainPair] = []
    dropped: Counter[str] = Counter()
    for i, row in enumerate(rows):
        try:
            forms = convert_row(row)
        except Exception:  # noqa: BLE001  counted by `kengine finetune-data`
            dropped["doesn't convert"] += 1
            continue
        split = "dev" if dev_story(str(row["story_id"]), dev_percent) else "train"
        if (
            split == "train"
            and row["label"] == "Uncertain"
            and not dev_story(f"uncertain:{row['example_id']}", uncertain_percent)
        ):
            dropped["Uncertain: thinned out"] += 1
            continue
        try:
            pairs += row_pairs(row, i, forms, split, budget)
        except NoChain as e:
            dropped[f"{row['label']}: {e}"] += 1
    return pairs, dropped
