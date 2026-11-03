# kengine M2 (formal fragment + Z3 logic checker) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let kengine *verify* logical steps. Claims carry an optional formal form in the spec's flat fragment. The task's givens are translated into it once per run. A Z3-backed checker decides whether a formal claim follows from what it rests on, and renders counterexamples as feedback. The plan ends with Checkpoint C, owner review point 2: the first measured with-vs-without comparison, on FOLIO.

**Architecture:**
- `claims/formal.py` parses and prints the fragment, a small text language. The AST is internal. Claims, events and sqlite carry canonical text.
- `logic/entail.py` grounds statements over the problem's constants (plus fresh witnesses) and asks Z3 for one of: entailed, refuted, open, inconsistent or unknown. `logic/render.py` turns a Z3 model into a short sentence.
- `checkers/logic.py` maps that decision to a verdict. Only `entailed` (or, for "settled neither way" claims, `open`) reaches `verified`.
- The agent gains a FORMALIZE call before step 1. It's one schema-constrained call that gives each `given` a formal form. EXTRACT can attach a formal form to `fact` and `inference` claims. Both reuse one vocabulary because the extractor sees the givens' formal forms.
- Eval gains `--limit`, `eval-compare` (errors caught, false alarms) and a `probe-formal` command. The probe measures which serialization the model fills in more reliably (spec §9).

**Tech Stack:** as M1, plus `z3-solver` (PyPI wheel; works under the repo's `shell.nix`).

**Spec:** `docs/superpowers/specs/2026-11-02-kengine-design.md`, mainly §5.3 (formal fragment), §5.5 #3 (logic checker), §6 (metrics), §7 (M2, review point 2), §9 (serialization choice).

**Scope:**
- In: the formal fragment, the Z3 logic checker, counterexample rendering, givens formalization, the serialization probe, eval comparison tooling, and Checkpoint C.
- Out: the critic for informal inferences (M5); until then those stay `unchecked` and block under `standard`. Also out: consistency and fact checking (M3), and skill-supplied formal givens (M5, §5.9).

## Global Constraints

- Everything in the M1 plan's Global Constraints still holds. In particular: pyright strict **0 errors**, `ruff check` and `ruff format --check` clean, and unit tests never need a network, GPU or model.
- **`verified` only from a deterministic or evidence checker.** The logic checker is `deterministic`. It never marks a `hypothesis` or an `assumption` (§5.3 table).
- **An undecidable or too-large problem is `unknown`, never a pass.** Grounding is capped at `MAX_INSTANCES = 20_000` ground instances. Z3 gets a `TIMEOUT_MS = 5_000` timeout per check.
- **Inconsistent premises verify nothing.** Ex falso is reported as `inconsistent`, not `entailed`.
- **A formal form that doesn't parse never costs the claim or the sample.** Only the `formal` field is dropped, with feedback.
- `z3` has no type stubs. Only `src/kengine/logic/entail.py` imports it, and that file starts with `# pyright: basic`.
- The fragment's AST (frozen dataclasses) never leaves `kengine.claims.formal` and `kengine.logic`. Anything persisted or logged holds canonical statement text. That's the one deliberate exception to "pydantic at every boundary".
- Names in the fragment are case-insensitive and stored lowercase. `x y z u v w` (optionally followed by digits) are variables.

## Review Focus

1. **The model writes textbook first-order logic** (`∀x (Dog(x) → Mammal(x))`, `forall x:`, `¬`, `∧`, `=>`, trailing `.`). The parser normalizes it to canonical text. *Tests in Task 1.*
2. **A grounding blowup** (many constants, a 4-variable rule) returns `unknown` quickly. **The same predicate with two arities** returns `unknown`, not a crash. *Tests in Task 2.*
3. **Contradictory givens** (FOLIO 1244 has some) give `inconsistent`. Nothing is verified from them. *Tests in Task 2 and Task 4.*
4. **A malformed formal form inside an otherwise good extraction**: the claim is kept without it and the model is told why. *Test in Task 3.*
5. **FORMALIZE fails** (invalid JSON, truncation, backend down). The run carries on with plain givens. *Tests in Task 5.*
6. **"Uncertain" answers.** An `undetermined` claim is only verified when the premises really settle it neither way, *and* every given has a formal form. If one doesn't, a given we couldn't translate might settle it. *Tests in Task 4.*

---

## File Structure

```
pyproject.toml                         + z3-solver dependency
kengine.example.toml                   + [logic], [roles.formalize]
src/kengine/
  claims/
    formal.py            NEW  FormalSyntax, AST, parse_statements, canonical, JsonStatement, json_statement
    schema.py                 + Formal, Claim.formal
    extraction.py             + ExtractedFormal, ExtractedClaim.formal, to_formal
    formalize.py         NEW  Formalization schema, formalize_messages, read_formalization
  logic/
    __init__.py          NEW
    entail.py            NEW  decide() -> Decision (z3; `# pyright: basic`)
    render.py            NEW  render_world
  checkers/
    logic.py             NEW  LogicChecker
    __init__.py               + LogicChecker in default_checkers()
  beliefs/store.py            + add(formal=...)
  loop/
    context.py                + formal forms in belief lines; extract prompt per syntax
    agent.py                  + FORMALIZE before step 1; formal on inserted claims
  audit/events.py             + FormalizeEvent
  config.py                   + LogicConfig, roles.formalize
  prompts/
    __init__.py               + system_prompt(name, syntax), formal_value()
    formal_text.md       NEW  the text serialization, for prompts
    formal_json.md       NEW  the JSON serialization, for prompts
    formalize.md         NEW
    extract.md, think.md      + formal instructions
  eval/
    compare.py           NEW  compare_runs -> Comparison
    probe_formal.py      NEW  probe_item, run_probe
    kengine_solver.py         + pass logic config to Agent
  cli/main.py                 + eval --limit, eval-compare, probe-formal; run passes logic config
docs/real-machine-runbook.md  + Checkpoint C
eval/results/CHECKPOINT-C.md  written by Task 8
```

---

## Task 1: The formal fragment: parser and canonical printer

**Files:**
- Create: `src/kengine/claims/formal.py`
- Test: `tests/claims/test_formal.py`

**Interfaces:**
- Produces:
  - `FormalSyntax = Literal["text", "json"]`
  - `class FormalSyntaxError(ValueError)`
  - frozen dataclasses `Atom(pred: str, args: tuple[str, ...])`, `Linear(const: Fraction, terms: tuple[tuple[Atom, Fraction], ...])`, `Compare(left: Linear, op: CmpOp, right: Linear)`, `Lit(atom: Atom | Compare, positive: bool)` and `Statement(body: tuple[Lit, ...], head: tuple[Lit, ...])`, where the head is a disjunction. `str()` of each gives canonical text.
  - `is_variable(name: str) -> bool`
  - `parse_statements(text: str) -> tuple[Statement, ...]`. A conjunctive head (`a -> b & c`) or a conjunctive fact (`a & b`) yields several statements.
  - `canonical(text: str) -> list[str]`
  - `atoms_of(lit: Lit) -> list[Atom]`, which covers predicate atoms and the numeric atoms inside comparisons
  - `variables_of(s: Statement) -> list[str]` (order of first appearance) and `constants_of(s: Statement) -> set[str]`
  - `class JsonStatement(BaseModel)` with `if_: list[str]` (alias `"if"`) and `then: str`; `json_statement_text(js) -> str`; `json_statement(text: str) -> JsonStatement`

- [ ] **Step 1: Write the failing tests**

```python
from fractions import Fraction

import pytest

from kengine.claims.formal import (
    Atom,
    FormalSyntaxError,
    JsonStatement,
    canonical,
    constants_of,
    is_variable,
    json_statement,
    json_statement_text,
    parse_statements,
    variables_of,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("dog(rex)", ["dog(rex)"]),
        ("raining", ["raining"]),
        ("not dog(rex)", ["not dog(rex)"]),
        ("dog(x) & not small(x) -> barks(x)", ["dog(x) & not small(x) -> barks(x)"]),
        ("cat(tom) | dog(tom)", ["cat(tom) | dog(tom)"]),
        ("pet(x) -> cat(x) | dog(x)", ["pet(x) -> cat(x) | dog(x)"]),
        ("likes(ann,bob)", ["likes(ann, bob)"]),
        ("a -> b & c", ["a -> b", "a -> c"]),
        ("a & b", ["a", "b"]),
        ("(a & b) & c -> d", ["a & b & c -> d"]),
        ("age(bob) >= 18", ["age(bob) >= 18"]),
        ("price(a) + 2 = price(b)", ["price(a) + 2 = price(b)"]),
        ("2*age(bob) - 3 < 0.5", ["2 * age(bob) - 3 < 1/2"]),
        ("age(x) > 18 -> adult(x)", ["age(x) > 18 -> adult(x)"]),
    ],
)
def test_canonical_forms(text: str, expected: list[str]):
    assert canonical(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("∀x (Dog(x) → Mammal(x))", ["dog(x) -> mammal(x)"]),
        ("forall x: dog(x) => mammal(x)", ["dog(x) -> mammal(x)"]),
        ("all x. Dog(x) -> Mammal(x)", ["dog(x) -> mammal(x)"]),
        ("¬Cat(Tom) ∧ Dog(Tom)", ["not cat(tom)", "dog(tom)"]),
        ("~cat(tom)", ["not cat(tom)"]),
        ("!cat(tom)", ["not cat(tom)"]),
        ("cat(tom) or dog(tom)", ["cat(tom) | dog(tom)"]),
        ("dog(rex) and mammal(rex).", ["dog(rex)", "mammal(rex)"]),
        ("age(bob) == 30", ["age(bob) = 30"]),
        ("not (dog(rex))", ["not dog(rex)"]),
        ("not not dog(rex)", ["dog(rex)"]),
    ],
)
def test_first_order_spellings_are_normalized(text: str, expected: list[str]):
    assert canonical(text) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("a -> b -> c", "more than one"),
        ("a | b -> c", "conditions"),
        ("a -> b & c | d", "both"),
        ("dog(f(x))", "arguments must be names"),
        ("x", "variable"),
        ("x(rex)", "variable"),
        ("age(x) * age(y) > 3", "linear"),
        ("x > 3", "individual"),
        ("1 / 0 > 3", "zero"),
        ("dog(rex", "unbalanced"),
        ("exists x: dog(x)", "exist"),
        ("a < b < c", "one comparison"),
        ("dog(rex) @", "unexpected"),
        ("a" * 301, "too long"),
    ],
)
def test_errors_are_specific(text: str, message: str):
    with pytest.raises(FormalSyntaxError, match=message):
        parse_statements(text)


def test_variables_and_constants():
    (s,) = parse_statements("likes(x, ann) & likes(y, x) -> friends(x, y)")
    assert variables_of(s) == ["x", "y"]
    assert constants_of(s) == {"ann"}
    assert is_variable("x2") and not is_variable("rex") and not is_variable("xavier")


def test_linear_terms_are_merged():
    (s,) = parse_statements("age(bob) + age(bob) - 2 = 1")
    lit = s.head[0]
    assert str(lit) == "2 * age(bob) - 2 = 1"
    assert lit.atom != Atom("age", ("bob",))
    assert Fraction(1) == Fraction(1)


def test_json_statements_round_trip():
    js = JsonStatement.model_validate({"if": ["dog(x)", "not small(x)"], "then": "barks(x)"})
    assert json_statement_text(js) == "dog(x) & not small(x) -> barks(x)"
    assert json_statement_text(JsonStatement.model_validate({"then": "dog(rex)"})) == "dog(rex)"
    assert json_statement("pet(x) -> cat(x) | dog(x)").model_dump(by_alias=True) == {
        "if": ["pet(x)"],
        "then": "cat(x) | dog(x)",
    }
```

- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/claims/test_formal.py -q`. Expected: FAIL (`ModuleNotFoundError: kengine.claims.formal`).

- [ ] **Step 3: Implement `src/kengine/claims/formal.py`**

```python
"""The formal fragment (spec §5.3) as text: parsing, canonical printing and names.

    statement := [body "->"] head
    body      := literal ("&" literal)*
    head      := literal ("|" literal)*        a disjunction
    literal   := ["not"] atom | ["not"] term CMP term
    atom      := name ["(" name ("," name)* ")"]
    term      := a linear sum of numbers and numeric atoms, e.g. `2 * age(bob) + 3`

Names are case-insensitive. `x`, `y`, `z`, `u`, `v` and `w`, optionally followed by
digits, are variables, universally quantified over their statement. Every other name is a
constant, predicate or function.

Two conveniences go beyond the spec's "conjunction implies one literal". A head may be a
disjunction ("either A or B" is common, and `not A -> B` is an unnatural way to write it).
A conjunctive head or fact is split into one statement per conjunct. Textbook first-order
spellings (`∀x`, `→`, `∧`, `¬`, `Dog(x)`) are accepted, so the model can write what it
knows.

The AST is internal to `kengine.claims` and `kengine.logic`. Claims, events and the
database carry the canonical text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

FormalSyntax = Literal["text", "json"]
CmpOp = Literal["<", "<=", ">", ">=", "=", "!="]

MAX_STATEMENT_CHARS = 300
_CMP_OPS: frozenset[str] = frozenset({"<", "<=", ">", ">=", "=", "!="})
_VARIABLE_RE = re.compile(r"^[uvwxyz][0-9]*$")
_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d+)?)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op><=|>=|!=|==|->|=>|[-+*/()<>=,&|~!:.¬∧∨→∀∃]))"
)
_OP_ALIASES = {
    "=>": "->", "→": "->", "∧": "&", "∨": "|", "¬": "not", "~": "not", "!": "not",
    "==": "=", "∀": "forall", "∃": "exists",
}  # fmt: skip
_NAME_KEYWORDS = {"and": "&", "or": "|", "not": "not", "forall": "forall", "exists": "exists"}


class FormalSyntaxError(ValueError):
    pass


def is_variable(name: str) -> bool:
    return _VARIABLE_RE.match(name) is not None


def _num(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


@dataclass(frozen=True)
class Atom:
    pred: str
    args: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.pred}({', '.join(self.args)})" if self.args else self.pred


@dataclass(frozen=True)
class Linear:
    """`const + sum(coef * value)` over numeric atoms such as `age(bob)`."""

    const: Fraction
    terms: tuple[tuple[Atom, Fraction], ...] = ()

    def __str__(self) -> str:
        parts: list[tuple[bool, str]] = []  # (negative, magnitude text)
        for atom, coef in self.terms:
            mag = abs(coef)
            parts.append((coef < 0, str(atom) if mag == 1 else f"{_num(mag)} * {atom}"))
        if self.const != 0 or not parts:
            parts.append((self.const < 0, _num(abs(self.const))))
        negative, first = parts[0]
        out = ("-" if negative else "") + first
        for negative, text in parts[1:]:
            out += f" {'-' if negative else '+'} {text}"
        return out


@dataclass(frozen=True)
class Compare:
    left: Linear
    op: CmpOp
    right: Linear

    def __str__(self) -> str:
        return f"{self.left} {self.op} {self.right}"


@dataclass(frozen=True)
class Lit:
    atom: Atom | Compare
    positive: bool = True

    def __str__(self) -> str:
        return str(self.atom) if self.positive else f"not {self.atom}"


@dataclass(frozen=True)
class Statement:
    body: tuple[Lit, ...]
    head: tuple[Lit, ...]

    def __str__(self) -> str:
        head = " | ".join(str(lit) for lit in self.head)
        if not self.body:
            return head
        return f"{' & '.join(str(lit) for lit in self.body)} -> {head}"


def atoms_of(lit: Lit) -> list[Atom]:
    if isinstance(lit.atom, Atom):
        return [lit.atom]
    return [a for a, _ in lit.atom.left.terms] + [a for a, _ in lit.atom.right.terms]


def variables_of(s: Statement) -> list[str]:
    seen: dict[str, None] = {}
    for lit in (*s.body, *s.head):
        for atom in atoms_of(lit):
            for arg in atom.args:
                if is_variable(arg):
                    seen.setdefault(arg)
    return list(seen)


def constants_of(s: Statement) -> set[str]:
    return {
        arg
        for lit in (*s.body, *s.head)
        for atom in atoms_of(lit)
        for arg in atom.args
        if not is_variable(arg)
    }


# Tokens and parenthesized groups

type _Tok = tuple[str, str]  # (kind, text); kind is "num", "name" or "op"

_LPAREN: _Tok = ("op", "(")
_RPAREN: _Tok = ("op", ")")


def _tokenize(text: str) -> list[_Tok]:
    out: list[_Tok] = []
    text = text.strip()
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            raise FormalSyntaxError(f"unexpected {text[pos : pos + 10].strip()!r}")
        pos = m.end()
        if m.group("num") is not None:
            out.append(("num", m.group("num")))
        elif m.group("name") is not None:
            name = m.group("name").lower()
            keyword = _NAME_KEYWORDS.get(name)
            out.append(("op", keyword) if keyword else ("name", name))
        else:
            op = m.group("op")
            out.append(("op", _OP_ALIASES.get(op, op)))
    return out


def _show(toks: list[_Tok]) -> str:
    return " ".join(t[1] for t in toks)


def _closing(toks: list[_Tok], i: int) -> int:
    depth = 0
    for j in range(i, len(toks)):
        if toks[j] == _LPAREN:
            depth += 1
        elif toks[j] == _RPAREN:
            depth -= 1
            if depth == 0:
                return j
    raise FormalSyntaxError(f"unbalanced parentheses in {_show(toks)!r}")


def _strip(toks: list[_Tok]) -> list[_Tok]:
    while len(toks) >= 2 and toks[0] == _LPAREN and _closing(toks, 0) == len(toks) - 1:
        toks = toks[1:-1]
    return toks


def _depth0(toks: list[_Tok]) -> list[int]:
    """Indices of the tokens outside any parentheses."""
    out: list[int] = []
    depth = 0
    for i, t in enumerate(toks):
        if t == _LPAREN:
            depth += 1
        elif t == _RPAREN:
            depth -= 1
            if depth < 0:
                raise FormalSyntaxError(f"unbalanced parentheses in {_show(toks)!r}")
        elif depth == 0:
            out.append(i)
    if depth != 0:
        raise FormalSyntaxError(f"unbalanced parentheses in {_show(toks)!r}")
    return out


def _has(toks: list[_Tok], op: str) -> bool:
    return any(toks[i] == ("op", op) for i in _depth0(toks))


def _split(toks: list[_Tok], op: str) -> list[list[_Tok]]:
    cuts = [i for i in _depth0(toks) if toks[i] == ("op", op)]
    bounds = [-1, *cuts, len(toks)]
    return [toks[a + 1 : b] for a, b in zip(bounds, bounds[1:], strict=False)]


def _flatten(toks: list[_Tok], op: str) -> list[list[_Tok]]:
    """Split on `op`, also inside parenthesized groups that are themselves `op`-joined."""
    out: list[list[_Tok]] = []
    for part in _split(_strip(toks), op):
        part = _strip(part)
        out.extend(_flatten(part, op) if _has(part, op) else [part])
    return out


def _unquantify(toks: list[_Tok]) -> list[_Tok]:
    """Drop leading `forall x y:` / `all x.` prefixes; refuse existentials."""
    while True:
        toks = _strip(toks)
        if toks and (toks[0] == ("op", "exists") or toks[0] == ("name", "some")):
            if toks[0] == ("op", "exists") or (len(toks) > 1 and is_variable(toks[1][1])):
                raise FormalSyntaxError(
                    "existentials aren't supported; name the individual instead"
                )
        quantified = toks and (toks[0] == ("op", "forall") or toks[0] == ("name", "all"))
        if not (quantified and len(toks) > 1 and is_variable(toks[1][1])):
            return toks
        i = 1
        while i < len(toks) and (is_variable(toks[i][1]) or toks[i] == ("op", ",")):
            i += 1
        if i < len(toks) and toks[i] in (("op", ":"), ("op", ".")):
            i += 1
        toks = toks[i:]


# Statements and literals


def parse_statements(text: str) -> tuple[Statement, ...]:
    if len(text) > MAX_STATEMENT_CHARS:
        raise FormalSyntaxError(f"statement too long ({len(text)} > {MAX_STATEMENT_CHARS})")
    toks = _tokenize(text)
    if toks and toks[-1] == ("op", "."):
        toks = toks[:-1]
    toks = _unquantify(toks)
    if not toks:
        raise FormalSyntaxError("empty statement")
    sides = _split(toks, "->")
    if len(sides) > 2:
        raise FormalSyntaxError("more than one '->' in a statement")
    body: tuple[Lit, ...] = ()
    if len(sides) == 2:
        conditions = _strip(sides[0])
        if _has(conditions, "|"):
            raise FormalSyntaxError(
                "a rule's conditions can't use '|'; write one rule per alternative"
            )
        body = tuple(_literal(p) for p in _flatten(conditions, "&"))
    head = _strip(sides[-1])
    if _has(head, "&") and _has(head, "|"):
        raise FormalSyntaxError("a result can't use both '&' and '|'; split the statement")
    if _has(head, "&"):
        return tuple(Statement(body, (_literal(p),)) for p in _flatten(head, "&"))
    return (Statement(body, tuple(_literal(p) for p in _flatten(head, "|"))),)


def canonical(text: str) -> list[str]:
    return [str(s) for s in parse_statements(text)]


def _literal(toks: list[_Tok]) -> Lit:
    toks = _strip(toks)
    if not toks:
        raise FormalSyntaxError("empty statement")
    if toks[0] == ("op", "not"):
        inner = _literal(toks[1:])
        return Lit(inner.atom, not inner.positive)
    cmps = [i for i in _depth0(toks) if toks[i][0] == "op" and toks[i][1] in _CMP_OPS]
    if len(cmps) > 1:
        raise FormalSyntaxError(f"one comparison per literal: {_show(toks)!r}")
    if cmps:
        i = cmps[0]
        op = cast(CmpOp, toks[i][1])
        return Lit(Compare(_term(toks[:i]), op, _term(toks[i + 1 :])))
    return Lit(_atom(toks))


def _args(toks: list[_Tok]) -> tuple[str, ...]:
    args: list[str] = []
    for part in _split(toks, ","):
        if len(part) != 1 or part[0][0] != "name":
            raise FormalSyntaxError(f"arguments must be names, not {_show(part)!r}")
        args.append(part[0][1])
    return tuple(args)


def _atom(toks: list[_Tok]) -> Atom:
    kind, name = toks[0]
    if kind != "name":
        raise FormalSyntaxError(f"unexpected {name!r} in {_show(toks)!r}")
    if is_variable(name):
        raise FormalSyntaxError(f"{name} is a variable; it can't be a statement or predicate")
    if len(toks) == 1:
        return Atom(name)
    if toks[1] != _LPAREN or _closing(toks, 1) != len(toks) - 1:
        raise FormalSyntaxError(f"unexpected {_show(toks[1:])!r} after {name}")
    return Atom(name, _args(toks[2:-1]))


# Linear terms


def _linear(const: Fraction, terms: dict[Atom, Fraction]) -> Linear:
    kept = sorted(((a, c) for a, c in terms.items() if c != 0), key=lambda t: str(t[0]))
    return Linear(const, tuple(kept))


def _combine(a: Linear, b: Linear, sign: int) -> Linear:
    terms = dict(a.terms)
    for atom, coef in b.terms:
        terms[atom] = terms.get(atom, Fraction(0)) + sign * coef
    return _linear(a.const + sign * b.const, terms)


def _scale(a: Linear, k: Fraction) -> Linear:
    return _linear(a.const * k, {atom: coef * k for atom, coef in a.terms})


class _Terms:
    def __init__(self, toks: list[_Tok]) -> None:
        self.toks = toks
        self.i = 0

    def peek(self) -> _Tok | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> _Tok:
        tok = self.peek()
        if tok is None:
            raise FormalSyntaxError(f"incomplete term {_show(self.toks)!r}")
        self.i += 1
        return tok

    def sum(self) -> Linear:
        value = self.product()
        while self.peek() in (("op", "+"), ("op", "-")):
            sign = 1 if self.take()[1] == "+" else -1
            value = _combine(value, self.product(), sign)
        return value

    def product(self) -> Linear:
        value = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            other = self.unary()
            if op == "*":
                if value.terms and other.terms:
                    raise FormalSyntaxError(f"not linear: ({value}) * ({other})")
                value = _scale(other, value.const) if not value.terms else _scale(value, other.const)
            else:
                if other.terms:
                    raise FormalSyntaxError(f"not linear: division by {other}")
                if other.const == 0:
                    raise FormalSyntaxError("division by zero")
                value = _scale(value, 1 / other.const)
        return value

    def unary(self) -> Linear:
        if self.peek() == ("op", "-"):
            self.take()
            return _scale(self.unary(), Fraction(-1))
        if self.peek() == ("op", "+"):
            self.take()
            return self.unary()
        return self.primary()

    def primary(self) -> Linear:
        kind, text = self.take()
        if kind == "num":
            return Linear(Fraction(text))
        if (kind, text) == _LPAREN:
            value = self.sum()
            if self.take() != _RPAREN:
                raise FormalSyntaxError(f"unbalanced parentheses in {_show(self.toks)!r}")
            return value
        if kind != "name":
            raise FormalSyntaxError(f"unexpected {text!r} in {_show(self.toks)!r}")
        if self.peek() == _LPAREN:
            start = self.i
            end = _closing(self.toks, start)
            self.i = end + 1
            return Linear(Fraction(0), ((Atom(text, _args(self.toks[start + 1 : end])), Fraction(1)),))
        if is_variable(text):
            raise FormalSyntaxError(f"{text} stands for an individual, not a number")
        return Linear(Fraction(0), ((Atom(text), Fraction(1)),))


def _term(toks: list[_Tok]) -> Linear:
    if not toks:
        raise FormalSyntaxError("a comparison is missing a side")
    parser = _Terms(toks)
    value = parser.sum()
    rest = parser.peek()
    if rest is not None:
        raise FormalSyntaxError(f"unexpected {rest[1]!r} in {_show(toks)!r}")
    return value


# The JSON serialization (spec §9 leaves the choice to measurement; see Task 7)


class JsonStatement(BaseModel):
    """`{"if": [literal, ...], "then": literal}`; a plain fact has an empty `if`."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    if_: list[str] = Field(default_factory=list[str], alias="if", max_length=6)
    then: str


def json_statement_text(js: JsonStatement) -> str:
    return f"{' & '.join(js.if_)} -> {js.then}" if js.if_ else js.then


def json_statement(text: str) -> JsonStatement:
    """`text` (one statement) in the JSON serialization."""
    (s,) = parse_statements(text)
    # model_validate, not the constructor: pyright sees the alias `if`, a keyword.
    return JsonStatement.model_validate(
        {"if": [str(lit) for lit in s.body], "then": " | ".join(str(lit) for lit in s.head)}
    )
```

Note: `test_linear_terms_are_merged` contains two trivial assertions that the implementer may delete. The meaningful one is the `str(lit)` check.

- [ ] **Step 4: Run the tests and make them pass.** Run: `uv run pytest tests/claims/test_formal.py -q && uv run pyright src/kengine/claims/formal.py tests/claims/test_formal.py && ruff check src tests && ruff format --check src tests`. Expected: all pass, `0 errors`. If a parametrized case fails because a message differs, fix the code, not the test, unless the test's expectation is wrong about the grammar above.

- [ ] **Step 5: Commit**

```bash
git add src/kengine/claims/formal.py tests/claims/test_formal.py
git commit -m "feat: formal fragment parser and canonical printer"
```

---

## Task 2: Grounding, Z3 entailment and counterexample rendering

**Files:**
- Modify: `pyproject.toml` (add `"z3-solver>=4.13"` to `dependencies`), `uv.lock` (via `uv lock`)
- Create: `src/kengine/logic/__init__.py` (docstring only), `src/kengine/logic/entail.py`, `src/kengine/logic/render.py`
- Test: `tests/logic/__init__.py`, `tests/logic/test_entail.py`, `tests/logic/test_render.py`

**Interfaces:**
- Consumes: `Statement`, `Atom`, `Compare`, `Linear`, `Lit`, `atoms_of`, `variables_of`, `constants_of`, `parse_statements` (Task 1).
- Produces:
  - `DecisionKind = Literal["entailed", "refuted", "open", "inconsistent", "unknown"]`
  - `class Decision(BaseModel)` with `kind`, `core: tuple[int, ...]` (indices into `premises`), `world: tuple[tuple[str, str], ...]` (for `open`: ground atom or term, then `"true"`, `"false"` or a number) and `reason: str`
  - `decide(premises: Sequence[Statement], goal: Sequence[Statement]) -> Decision`
  - `MAX_INSTANCES = 20_000`, `TIMEOUT_MS = 5_000`
  - `render_world(world: Sequence[tuple[str, str]]) -> str`

**Semantics** (write this in the module docstring):
- Statements are universally quantified clauses with no function symbols over individuals. Herbrand's theorem makes grounding complete:
  - Ground over the constants that appear, plus one fresh witness per variable of each goal statement. The witnesses are the Skolem constants of the negated goal.
  - If nothing has a constant, add one individual. First-order domains are never empty.
- Steps:
  1. `premises ∧ ¬goal` unsat: check `premises` alone. If that's unsat too, the result is `inconsistent`; otherwise `entailed`. The core is the premises in the unsat core.
  2. Otherwise, `premises ∧ goal` unsat means `refuted`.
  3. Otherwise the result is `open`, and the first model is the counterexample.
- Numeric atoms (`age(bob)`) are uninterpreted reals. Comparisons are linear real arithmetic.

- [ ] **Step 1: Write the failing tests** (`tests/logic/test_entail.py`)

```python
from kengine.claims.formal import Statement, parse_statements
from kengine.logic.entail import decide


def st(*texts: str) -> list[Statement]:
    return [s for t in texts for s in parse_statements(t)]


def test_modus_ponens_is_entailed_with_core():
    d = decide(st("dog(rex)", "cat(tom)", "dog(x) -> mammal(x)"), st("mammal(rex)"))
    assert d.kind == "entailed"
    assert d.core == (0, 2)


def test_missing_premise_is_open_with_counterexample():
    d = decide(st("dog(x) -> mammal(x)"), st("mammal(rex)"))
    assert d.kind == "open"
    assert ("mammal(rex)", "false") in d.world


def test_converse_error_is_open():
    d = decide(st("dog(x) -> mammal(x)", "mammal(tom)"), st("dog(tom)"))
    assert d.kind == "open"


def test_refuted():
    d = decide(st("dog(rex)", "dog(x) -> not cat(x)"), st("cat(rex)"))
    assert d.kind == "refuted"
    assert d.core == (0, 1)


def test_rule_goal_uses_a_witness():
    d = decide(st("dog(x) -> mammal(x)", "mammal(x) -> animal(x)"), st("dog(x) -> animal(x)"))
    assert d.kind == "entailed"
    d = decide(st("dog(x) -> mammal(x)"), st("mammal(x) -> dog(x)"))
    assert d.kind == "open"
    assert any("_w" in atom for atom, _ in d.world)


def test_disjunction():
    assert decide(st("cat(tom) | dog(tom)", "not cat(tom)"), st("dog(tom)")).kind == "entailed"
    assert decide(st("cat(tom) | dog(tom)"), st("dog(tom)")).kind == "open"


def test_inconsistent_premises_verify_nothing():
    d = decide(st("p", "q", "not p"), st("r"))
    assert d.kind == "inconsistent"
    assert d.core == (0, 2)


def test_universal_premises_contradict_without_constants():
    assert decide(st("p(x)", "not p(x)"), st("q")).kind == "inconsistent"


def test_propositional():
    assert decide(st("rain -> wet", "rain"), st("wet")).kind == "entailed"


def test_linear_arithmetic():
    prem = st("age(bob) = 30", "age(x) > 18 -> adult(x)")
    assert decide(prem, st("adult(bob)")).kind == "entailed"
    assert decide(st("price(a) + 2 = price(b)"), st("price(b) > price(a)")).kind == "entailed"
    open_ = decide(st("age(x) > 18 -> adult(x)"), st("adult(bob)"))
    assert open_.kind == "open"


def test_arity_clash_is_unknown():
    d = decide(st("dog(rex)", "dog(rex, tom)"), st("dog(tom)"))
    assert d.kind == "unknown" and "dog" in d.reason


def test_grounding_blowup_is_unknown_fast():
    constants = [f"likes(c{i}, c{i + 1})" for i in range(20)]
    rule = "likes(x, y) & likes(y, z) & likes(z, w) -> likes(x, w)"
    d = decide(st(*constants, rule), st("likes(c0, c20)"))
    assert d.kind == "unknown" and "too large" in d.reason


def test_empty_premises():
    assert decide([], st("dog(rex) | not dog(rex)")).kind == "entailed"
    assert decide([], st("dog(rex)")).kind == "open"
```

`tests/logic/test_render.py`:

```python
from kengine.logic.render import render_world


def test_render_world():
    assert render_world([("dog(rex)", "true"), ("mammal(rex)", "false")]) == (
        "a case where dog(rex) is true and mammal(rex) is false"
    )
    assert render_world([("age(bob)", "17")]) == "a case where age(bob) = 17"
    assert render_world([("a", "true"), ("b", "false"), ("c", "true")]) == (
        "a case where a is true, b is false and c is true"
    )


def test_render_world_names_witnesses():
    out = render_world([("mammal(_w1)", "true"), ("dog(_w1)", "false")])
    assert out == (
        "a case where mammal(_w1) is true and dog(_w1) is false, for some individual _w1"
    )


def test_render_empty_world():
    assert render_world([]) == "a case the premises allow"
```

- [ ] **Step 2: Add the dependency and confirm the tests fail.** Run: `uv add 'z3-solver>=4.13' && uv run pytest tests/logic -q`. Expected: FAIL (`ModuleNotFoundError: kengine.logic`).

- [ ] **Step 3: Implement `src/kengine/logic/entail.py`**

```python
# pyright: basic
# z3 ships no type stubs; this is the only module that imports it.
"""Deciding entailment in the formal fragment with Z3 (spec §5.3, §5.5 #3).

Statements are universally quantified clauses with no function symbols over individuals,
so grounding them over the constants that appear is complete (Herbrand). Add one fresh
witness per variable of each goal statement (the Skolem constants of the negated goal),
and one individual if there are no constants at all, since first-order domains are never
empty. The result is quantifier-free, and Z3 always finishes it, up to the timeout.

    premises ∧ ¬goal unsat  →  premises alone unsat?  inconsistent : entailed
    premises ∧ goal unsat   →  refuted
    otherwise               →  open, with the first model as the counterexample

Numeric atoms such as `age(bob)` are uninterpreted reals, and comparisons are linear real
arithmetic.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from fractions import Fraction
from typing import Literal

import z3
from pydantic import BaseModel, ConfigDict

from kengine.claims.formal import (
    Atom,
    Compare,
    Linear,
    Lit,
    Statement,
    atoms_of,
    constants_of,
    variables_of,
)

MAX_INSTANCES = 20_000
TIMEOUT_MS = 5_000
MAX_WORLD = 8

DecisionKind = Literal["entailed", "refuted", "open", "inconsistent", "unknown"]


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: DecisionKind
    # Indices into `premises` of an unsat core (entailed, refuted, inconsistent).
    core: tuple[int, ...] = ()
    # open: a case where the premises hold and the goal fails, as (ground atom, value).
    world: tuple[tuple[str, str], ...] = ()
    reason: str = ""


class _Unsupported(Exception):
    pass


type _Env = dict[str, str]


def decide(premises: Sequence[Statement], goal: Sequence[Statement]) -> Decision:
    try:
        return _decide(premises, goal)
    except _Unsupported as e:
        return Decision(kind="unknown", reason=str(e))


def _decide(premises: Sequence[Statement], goal: Sequence[Statement]) -> Decision:
    statements = [*premises, *goal]
    _check_signature(statements)
    constants = sorted(set().union(*(constants_of(s) for s in statements)))
    taken = set(constants)
    witnesses: list[_Env] = []
    for g in goal:
        env: _Env = {}
        for v in variables_of(g):
            env[v] = _fresh(taken)
        witnesses.append(env)
    domain = constants + [w for env in witnesses for w in env.values()]
    if not domain and any(variables_of(s) for s in statements):
        domain = [_fresh(taken)]
    total = sum(len(domain) ** len(variables_of(s)) for s in statements)
    if total > MAX_INSTANCES:
        raise _Unsupported(f"too large to check: {total} ground instances")

    enc = _Encoder()
    tracked = [
        (z3.Bool(f"__p{i}"), _and([enc.statement(p, env) for env in _envs(p, domain)]))
        for i, p in enumerate(premises)
    ]
    negated = _or([z3.Not(enc.statement(g, env)) for g, env in zip(goal, witnesses, strict=True)])
    asserted = _and([enc.statement(g, env) for g in goal for env in _envs(g, domain)])

    first, core, model = _check(tracked, negated)
    if first == "unsat":
        alone, alone_core, _ = _check(tracked, None)
        if alone == "unsat":
            return Decision(kind="inconsistent", core=alone_core)
        if alone == "unknown":
            return Decision(kind="unknown", reason="solver gave up (timeout)")
        return Decision(kind="entailed", core=core)
    if first == "unknown" or model is None:
        return Decision(kind="unknown", reason="solver gave up (timeout)")
    second, core, _ = _check(tracked, asserted)
    if second == "unsat":
        return Decision(kind="refuted", core=core)
    if second == "unknown":
        return Decision(kind="unknown", reason="solver gave up (timeout)")
    return Decision(kind="open", world=_world(model, enc, goal, witnesses))


def _fresh(taken: set[str]) -> str:
    n = 1
    while f"_w{n}" in taken:
        n += 1
    taken.add(f"_w{n}")
    return f"_w{n}"


def _check_signature(statements: Sequence[Statement]) -> None:
    seen: dict[str, tuple[str, int]] = {}
    for s in statements:
        for lit in (*s.body, *s.head):
            role = "predicate" if isinstance(lit.atom, Atom) else "number"
            for atom in atoms_of(lit):
                use = (role, len(atom.args))
                prev = seen.setdefault(atom.pred, use)
                if prev != use:
                    raise _Unsupported(
                        f"{atom.pred} is used as a {prev[0]} of {prev[1]} argument(s) and as "
                        f"a {use[0]} of {use[1]}"
                    )


def _envs(s: Statement, domain: Sequence[str]) -> list[_Env]:
    vs = variables_of(s)
    return [dict(zip(vs, combo, strict=True)) for combo in itertools.product(domain, repeat=len(vs))]


def _ground(atom: Atom, env: _Env) -> Atom:
    return Atom(atom.pred, tuple(env.get(a, a) for a in atom.args))


def _and(xs: Sequence[z3.BoolRef]) -> z3.BoolRef:
    if not xs:
        return z3.BoolVal(True)
    return xs[0] if len(xs) == 1 else z3.And(*xs)


def _or(xs: Sequence[z3.BoolRef]) -> z3.BoolRef:
    if not xs:
        return z3.BoolVal(False)
    return xs[0] if len(xs) == 1 else z3.Or(*xs)


def _q(f: Fraction) -> z3.ArithRef:
    return z3.Q(f.numerator, f.denominator)


class _Encoder:
    def __init__(self) -> None:
        self.bools: dict[str, z3.BoolRef] = {}
        self.reals: dict[str, z3.ArithRef] = {}

    def atom(self, atom: Atom, env: _Env) -> z3.BoolRef:
        key = str(_ground(atom, env))
        return self.bools.setdefault(key, z3.Bool(key))

    def real(self, atom: Atom, env: _Env) -> z3.ArithRef:
        key = str(_ground(atom, env))
        return self.reals.setdefault(key, z3.Real(key))

    def linear(self, t: Linear, env: _Env) -> z3.ArithRef:
        expr = _q(t.const)
        for atom, coef in t.terms:
            expr = expr + _q(coef) * self.real(atom, env)
        return expr

    def compare(self, c: Compare, env: _Env) -> z3.BoolRef:
        a, b = self.linear(c.left, env), self.linear(c.right, env)
        match c.op:
            case "<":
                return a < b
            case "<=":
                return a <= b
            case ">":
                return a > b
            case ">=":
                return a >= b
            case "=":
                return a == b
            case "!=":
                return a != b

    def lit(self, lit: Lit, env: _Env) -> z3.BoolRef:
        e = self.atom(lit.atom, env) if isinstance(lit.atom, Atom) else self.compare(lit.atom, env)
        return e if lit.positive else z3.Not(e)

    def statement(self, s: Statement, env: _Env) -> z3.BoolRef:
        head = _or([self.lit(lit, env) for lit in s.head])
        if not s.body:
            return head
        return z3.Implies(_and([self.lit(lit, env) for lit in s.body]), head)


def _check(
    tracked: Sequence[tuple[z3.BoolRef, z3.BoolRef]], extra: z3.BoolRef | None
) -> tuple[Literal["sat", "unsat", "unknown"], tuple[int, ...], z3.ModelRef | None]:
    solver = z3.Solver()
    solver.set("timeout", TIMEOUT_MS)
    solver.set("core.minimize", True)  # so explanations cite only what's needed
    for label, expr in tracked:
        solver.assert_and_track(expr, label)
    if extra is not None:
        solver.add(extra)
    result = solver.check()
    if result == z3.unsat:
        core = tuple(sorted(int(str(c).removeprefix("__p")) for c in solver.unsat_core()))
        return "unsat", core, None
    if result == z3.sat:
        return "sat", (), solver.model()
    return "unknown", (), None


def _value(model: z3.ModelRef, expr: z3.ExprRef) -> str:
    v = model.eval(expr, model_completion=True)
    if z3.is_true(v):
        return "true"
    if z3.is_false(v):
        return "false"
    if z3.is_rational_value(v):
        f = Fraction(v.numerator_as_long(), v.denominator_as_long())
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
    return str(v)


def _world(
    model: z3.ModelRef, enc: _Encoder, goal: Sequence[Statement], witnesses: Sequence[_Env]
) -> tuple[tuple[str, str], ...]:
    """The goal's atoms in the goal statements the model falsifies."""
    out: dict[str, str] = {}
    for g, env in zip(goal, witnesses, strict=True):
        if not z3.is_false(model.eval(enc.statement(g, env), model_completion=True)):
            continue
        for lit in (*g.body, *g.head):
            for atom in atoms_of(lit):
                ground = _ground(atom, env)
                key = str(ground)
                expr = enc.atom(atom, env) if isinstance(lit.atom, Atom) else enc.real(atom, env)
                out.setdefault(key, _value(model, expr))
    return tuple(list(out.items())[:MAX_WORLD])
```

`src/kengine/logic/render.py`:

```python
"""Counterexamples as short sentences for feedback and the audit trail (spec §5.5 #3)."""

from __future__ import annotations

import re
from collections.abc import Sequence

_WITNESS_RE = re.compile(r"\b_w\d+\b")


def render_world(world: Sequence[tuple[str, str]]) -> str:
    if not world:
        return "a case the premises allow"
    parts = [
        f"{atom} is {value}" if value in ("true", "false") else f"{atom} = {value}"
        for atom, value in world
    ]
    text = parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"
    out = f"a case where {text}"
    witnesses = sorted({w for atom, _ in world for w in _WITNESS_RE.findall(atom)})
    if witnesses:
        out += f", for some individual {', '.join(witnesses)}"
    return out
```

- [ ] **Step 4: Run the tests and make them pass.** Run: `uv run pytest tests/logic -q && uv run pyright && ruff check src tests && ruff format --check src tests`. Expected: all pass, `0 errors`. (`# pyright: basic` covers `entail.py`. If pyright still reports `reportMissingTypeStubs` for `z3`, add `# pyright: reportMissingTypeStubs=false` under it.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/kengine/logic tests/logic
git commit -m "feat: ground the formal fragment and decide entailment with z3"
```

---

## Task 3: Formal forms on claims, end to end

**Files:**
- Modify: `src/kengine/claims/schema.py`, `src/kengine/claims/extraction.py`, `src/kengine/beliefs/store.py` (`add`), `src/kengine/loop/context.py` (`belief_line`), `src/kengine/loop/agent.py` (`_insert`)
- Test: `tests/claims/test_schema.py`, `tests/claims/test_extraction.py` (create if missing; otherwise add to the file that tests `Extraction`), `tests/beliefs/test_store.py`, `tests/loop/test_context.py`, `tests/loop/test_agent.py`

**Interfaces:**
- Consumes: `canonical`, `FormalSyntaxError`, `JsonStatement`, `json_statement_text` (Task 1).
- Produces:
  - `FormalMode = Literal["holds", "undetermined"]`, and `class Formal(BaseModel)` with `mode: FormalMode = "holds"` and `statements: tuple[str, ...]` (1..12, canonicalized on validation), in `claims/schema.py`
  - `Claim.formal: Formal | None = None`
  - `class ExtractedFormal(BaseModel)` with `mode: FormalMode = "holds"` and `statements: list[str | JsonStatement]` (1..8, raw, not validated), plus `ExtractedClaim.formal: ExtractedFormal | None = None`
  - `to_formal(f: ExtractedFormal) -> Formal`, which raises `FormalSyntaxError` (a `ValueError`)
  - `BeliefStore.add(..., formal: Formal | None = None)`
  - `belief_line` appends ` :: formal: s1; s2` (mode `undetermined` shows as ` :: formal (undetermined): ...`)
  - In `Agent._insert`, a formal form that fails `to_formal` is dropped, with issue `f"{ref}'s formal form was dropped: {error}"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/claims/test_schema.py (add)
import pytest
from pydantic import ValidationError

from kengine.claims.schema import Claim, ClaimType, Formal, Source, SourceKind


def test_formal_is_canonicalized_and_split():
    f = Formal(statements=("∀x (Dog(x) → Mammal(x))", "a & b"))
    assert f.statements == ("dog(x) -> mammal(x)", "a", "b")
    assert f.mode == "holds"


def test_formal_rejects_bad_syntax():
    with pytest.raises(ValidationError, match="unbalanced"):
        Formal(statements=("dog(rex",))


def test_claim_round_trips_formal():
    c = Claim(
        id="b1",
        type=ClaimType.GIVEN,
        text="Rex is a dog",
        source=Source(kind=SourceKind.TASK),
        step=0,
        formal=Formal(statements=("dog(rex)",)),
    )
    assert Claim.model_validate_json(c.model_dump_json()) == c
```

```python
# tests/claims/test_extraction.py (add)
import pytest

from kengine.claims.extraction import ExtractedClaim, ExtractedFormal, Extraction, to_formal
from kengine.claims.formal import FormalSyntaxError, JsonStatement


def test_extracted_formal_accepts_both_serializations():
    e = Extraction.model_validate(
        {
            "claims": [
                {
                    "ref": "c1",
                    "type": "inference",
                    "text": "Rex is a mammal",
                    "premises": ["b1", "b2"],
                    "formal": {"statements": ["mammal(rex)", {"if": [], "then": "dog(rex)"}]},
                }
            ],
            "action": {"kind": "continue"},
        }
    )
    f = e.claims[0].formal
    assert f is not None and isinstance(f.statements[1], JsonStatement)
    assert to_formal(f).statements == ("mammal(rex)", "dog(rex)")


def test_extracted_formal_is_not_validated_until_converted():
    c = ExtractedClaim.model_validate(
        {"ref": "c1", "type": "fact", "text": "t", "formal": {"statements": ["dog(rex"]}}
    )
    assert c.formal is not None
    with pytest.raises(FormalSyntaxError, match="unbalanced"):
        to_formal(c.formal)


def test_to_formal_keeps_mode():
    f = to_formal(ExtractedFormal(mode="undetermined", statements=["barks(rex)"]))
    assert f.mode == "undetermined"
```

```python
# tests/beliefs/test_store.py (add)
def test_add_with_formal_persists(tmp_path):
    path = tmp_path / "b.sqlite"
    s = BeliefStore.open(path)
    s.add(
        type=ClaimType.GIVEN,
        text="Rex is a dog",
        premises=[],
        source=Source(kind=SourceKind.TASK),
        step=0,
        formal=Formal(statements=("dog(rex)",)),
    )
    s.close()
    again = BeliefStore.open(path)
    formal = again.get("b1").claim.formal
    assert formal is not None and formal.statements == ("dog(rex)",)
```

```python
# tests/loop/test_context.py (add)
def test_belief_line_shows_formal():
    rec = BeliefRecord(
        claim=Claim(
            id="b2",
            type=ClaimType.INFERENCE,
            text="Rex barks",
            premises=("b1",),
            source=Source(kind=SourceKind.MODEL),
            step=1,
            formal=Formal(mode="undetermined", statements=("barks(rex)",)),
        ),
        status=Status.UNCHECKED,
        status_reason="",
    )
    assert belief_line(rec) == (
        "b2 [unchecked] (inference) Rex barks :: formal (undetermined): barks(rex)"
    )
```

```python
# tests/loop/test_agent.py (add)
def test_bad_formal_form_is_dropped_but_claim_kept(tmp_path):
    bad = claim("c1", "fact", "Rex is a dog")
    bad["formal"] = {"statements": ["dog(rex"]}
    agent, _ = make_agent(
        [think(), extract_json([bad], CONTINUE), think(), extract_json([], CONTINUE)],
        "quick",
        tmp_path,
        limits=Limits(max_steps=2),
    )
    agent.run(Task(goal="g"))
    rec = agent.store.get("b1")
    assert rec.claim.text == "Rex is a dog" and rec.claim.formal is None
    prompts = think_prompts(agent.backend)  # type: ignore[arg-type]
    assert "c1's formal form was dropped: unbalanced" in prompts[1]
```

(If `make_agent` returns the backend separately, as it does, use the returned backend instead of `agent.backend`: `agent, backend = make_agent(...)`, then `think_prompts(backend)`.)

- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/claims tests/beliefs tests/loop -q`. Expected: FAIL (`ImportError: cannot import name 'Formal'`).

- [ ] **Step 3: Implement**

`claims/schema.py`. Add these imports and definitions, then the `formal` field on `Claim`:

```python
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from kengine.claims.formal import canonical

FormalMode = Literal["holds", "undetermined"]


class Formal(BaseModel):
    """A claim's formal form (spec §5.3). `holds`: the statements are true. `undetermined`:
    the premises settle them neither way (how "Uncertain" answers get checked)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: FormalMode = "holds"
    statements: tuple[str, ...] = Field(min_length=1, max_length=12)

    @field_validator("statements", mode="before")
    @classmethod
    def _canonical(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        out: list[str] = []
        for text in value:  # pyright: ignore[reportUnknownVariableType]
            out.extend(canonical(str(text)))  # pyright: ignore[reportUnknownArgumentType]
        return tuple(out)
```

In `Claim`, add `formal: Formal | None = None` after `computation`. (A `FormalSyntaxError` raised in the validator is a `ValueError`, so pydantic reports it as a `ValidationError` that includes the message.)

`claims/extraction.py`:

```python
from kengine.claims.formal import FormalSyntaxError, JsonStatement, canonical, json_statement_text
from kengine.claims.schema import ClaimText, Computation, Formal, FormalMode, PremiseRef, check_computation


class ExtractedFormal(BaseModel):
    """Raw formal form from the model. Parsed by `to_formal`, so a typo costs only this
    field, not the whole sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: FormalMode = "holds"
    statements: list[str | JsonStatement] = Field(min_length=1, max_length=8)


def to_formal(f: ExtractedFormal) -> Formal:
    texts = [s if isinstance(s, str) else json_statement_text(s) for s in f.statements]
    statements = [c for t in texts for c in canonical(t)]  # raises FormalSyntaxError
    if len(statements) > 12:
        raise FormalSyntaxError(f"too many statements ({len(statements)} > 12)")
    return Formal(mode=f.mode, statements=tuple(statements))
```

Add `formal: ExtractedFormal | None = None` to `ExtractedClaim` after `computation`.

`beliefs/store.py` `add`: add the parameter `formal: Formal | None = None` (import `Formal` from `kengine.claims.schema`) and pass `formal=formal` to `Claim(...)`.

`loop/context.py` `belief_line`:

```python
    if claim.formal is not None:
        label = "formal" if claim.formal.mode == "holds" else f"formal ({claim.formal.mode})"
        line += f" :: {label}: {'; '.join(claim.formal.statements)}"
```

`loop/agent.py` `_insert`. Before `self.store.add(...)`:

```python
            formal: Formal | None = None
            if ready.formal is not None:
                try:
                    formal = to_formal(ready.formal)
                except ValueError as e:
                    step.issues.append(f"{ready.ref}'s formal form was dropped: {e}")
```

Then pass `formal=formal` to `store.add`. Import `to_formal` from `kengine.claims.extraction` and `Formal` from `kengine.claims.schema`.

- [ ] **Step 4: Run everything.** Run: `uv run pytest -q && uv run pyright && ruff check src tests && ruff format --check src tests`. Expected: all pass, `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "feat: claims carry an optional formal form"
```

---

## Task 4: The logic checker

**Files:**
- Create: `src/kengine/checkers/logic.py`
- Modify: `src/kengine/checkers/__init__.py`
- Test: `tests/checkers/test_logic.py`

**Interfaces:**
- Consumes: `decide`, `Decision` (Task 2), `render_world` (Task 2), `parse_statements` (Task 1), `Formal`, `Claim.formal` (Task 3), `BeliefView`, `Verdict`, `CheckerKind`, `Outcome` (M1).
- Produces: `LogicChecker`, with `id = "logic"` and `kind = CheckerKind.DETERMINISTIC`. `default_checkers()` returns `[StructureChecker(), ArithmeticChecker(), RestatementChecker(), LogicChecker()]`.

**Rules:**
- **Applies to** `fact` and `inference` claims that have a formal form.
- **The base** (the premises it is checked against):
  - **`holds` inference:** its cited premises that have a `holds` formal form and aren't terminal. "Does the step follow from what it says it follows from?"
  - **`holds` fact, and any `undetermined` claim:** every trusted formal belief, plus its cited premises with formal forms. Trusted means a non-terminal `given` or `verified` claim with a `holds` formal form, other than the claim itself.

| Mode | Decision | Outcome | Status | Explanation |
|---|---|---|---|---|
| holds | entailed | PASS | VERIFIED | `follows from b1, b3` (core owners); `is true in every case` if the core is empty |
| holds | refuted | FAIL | CONTRADICTED | `contradicts b1, b3` |
| holds, inference | open, every cited premise formal | FAIL | None | `doesn't follow from b1, b2: they allow {world}; cite what else it needs or fix the step` (counterexample = world) |
| holds, inference | open, some cited premise informal | UNKNOWN | None | `can't check: b4 has no formal form, and b1, b2 alone allow {world}` |
| holds, fact | open | UNKNOWN | None | `the givens don't settle it: they allow {world}` |
| undetermined | open | PASS | VERIFIED | `the premises settle it neither way` (evidence = base ids) |
| undetermined | open, but some live given has no formal form | UNKNOWN | None | `can't confirm it's open: b2, b5 have no formal form` |
| undetermined | entailed | FAIL | CONTRADICTED | `it isn't open: it follows from b1, b3` |
| undetermined | refuted | FAIL | CONTRADICTED | `it isn't open: its negation follows from b1, b3` |
| any | inconsistent | UNKNOWN | None | `b1, b4 contradict each other, so this can't be checked` |
| any | unknown | UNKNOWN | None | `logic check gave up: {reason}` |

Ids in explanations are deduplicated and in insertion order.

- [ ] **Step 1: Write the failing tests** (`tests/checkers/test_logic.py`)

```python
from kengine.beliefs.status import Status
from kengine.beliefs.store import BeliefStore
from kengine.checkers import default_checkers
from kengine.checkers.base import CheckerKind, Outcome
from kengine.checkers.logic import LogicChecker
from kengine.claims.schema import Claim, ClaimType, Formal, FormalMode, Source, SourceKind
from kengine.checkers.base import Verdict

TASK = Source(kind=SourceKind.TASK)
MODEL = Source(kind=SourceKind.MODEL)


def given(s: BeliefStore, text: str, *formal: str) -> str:
    f = Formal(statements=formal) if formal else None
    return s.add(type=ClaimType.GIVEN, text=text, premises=[], source=TASK, step=0, formal=f).id


def add(
    s: BeliefStore,
    type_: ClaimType,
    premises: list[str],
    *formal: str,
    mode: FormalMode = "holds",
) -> Claim:
    return s.add(
        type=type_,
        text="claim",
        premises=premises,
        source=MODEL,
        step=1,
        formal=Formal(mode=mode, statements=formal),
    )


def check(s: BeliefStore, c: Claim) -> Verdict:
    return LogicChecker().check(c, s)


def store() -> BeliefStore:
    return BeliefStore.open(":memory:")


def test_valid_inference_is_verified():
    s = store()
    b1 = given(s, "Rex is a dog", "dog(rex)")
    b2 = given(s, "Dogs are mammals", "dog(x) -> mammal(x)")
    v = check(s, add(s, ClaimType.INFERENCE, [b1, b2], "mammal(rex)"))
    assert (v.outcome, v.status) == (Outcome.PASS, Status.VERIFIED)
    assert v.explanation == "follows from b1, b2" and v.evidence == ["b1", "b2"]
    assert v.checker_kind == CheckerKind.DETERMINISTIC


def test_inference_checked_only_against_what_it_cites():
    s = store()
    b1 = given(s, "Rex is a dog", "dog(rex)")
    given(s, "Dogs are mammals", "dog(x) -> mammal(x)")
    v = check(s, add(s, ClaimType.INFERENCE, [b1], "mammal(rex)"))
    assert (v.outcome, v.status) == (Outcome.FAIL, None)
    assert v.explanation.startswith("doesn't follow from b1: they allow a case where")
    assert "mammal(rex) is false" in v.explanation
    assert v.counterexample is not None and "mammal(rex) is false" in v.counterexample


def test_converse_error_fails_with_counterexample():
    s = store()
    b1 = given(s, "Dogs are mammals", "dog(x) -> mammal(x)")
    b2 = given(s, "Tom is a mammal", "mammal(tom)")
    v = check(s, add(s, ClaimType.INFERENCE, [b1, b2], "dog(tom)"))
    assert (v.outcome, v.status) == (Outcome.FAIL, None)
    assert "dog(tom) is false" in v.explanation


def test_refuted_inference_is_contradicted():
    s = store()
    b1 = given(s, "Rex is a dog", "dog(rex)")
    b2 = given(s, "No dog is a cat", "dog(x) -> not cat(x)")
    v = check(s, add(s, ClaimType.INFERENCE, [b1, b2], "cat(rex)"))
    assert (v.outcome, v.status) == (Outcome.FAIL, Status.CONTRADICTED)
    assert v.explanation == "contradicts b1, b2"


def test_informal_premise_makes_it_unknown():
    s = store()
    b1 = given(s, "Rex is a dog", "dog(rex)")
    b2 = given(s, "Dogs are mammals")
    v = check(s, add(s, ClaimType.INFERENCE, [b1, b2], "mammal(rex)"))
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert v.explanation.startswith("can't check: b2 has no formal form, and b1 alone allow")


def test_fact_is_checked_against_the_givens():
    s = store()
    given(s, "Rex is a dog", "dog(rex)")
    given(s, "Dogs are mammals", "dog(x) -> mammal(x)")
    v = check(s, add(s, ClaimType.FACT, [], "mammal(rex)"))
    assert (v.outcome, v.status) == (Outcome.PASS, Status.VERIFIED)
    v = check(s, add(s, ClaimType.FACT, [], "cat(rex)"))
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert v.explanation.startswith("the givens don't settle it")


def test_undetermined_is_verified_only_when_really_open():
    s = store()
    given(s, "Rex is a dog", "dog(rex)")
    given(s, "Dogs are mammals", "dog(x) -> mammal(x)")
    v = check(s, add(s, ClaimType.INFERENCE, [], "barks(rex)", mode="undetermined"))
    assert (v.outcome, v.status) == (Outcome.PASS, Status.VERIFIED)
    v = check(s, add(s, ClaimType.INFERENCE, [], "mammal(rex)", mode="undetermined"))
    assert (v.outcome, v.status) == (Outcome.FAIL, Status.CONTRADICTED)
    assert v.explanation == "it isn't open: it follows from b1, b2"
    v = check(s, add(s, ClaimType.INFERENCE, [], "not mammal(rex)", mode="undetermined"))
    assert v.explanation == "it isn't open: its negation follows from b1, b2"


def test_undetermined_needs_every_given_formalized():
    s = store()
    given(s, "Rex is a dog", "dog(rex)")
    given(s, "Every dog barks")
    v = check(s, add(s, ClaimType.INFERENCE, [], "barks(rex)", mode="undetermined"))
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert v.explanation == "can't confirm it's open: b2 has no formal form"


def test_inconsistent_givens_verify_nothing():
    s = store()
    b1 = given(s, "Rex is a dog", "dog(rex)")
    b2 = given(s, "Rex is not a dog", "not dog(rex)")
    v = check(s, add(s, ClaimType.INFERENCE, [b1, b2], "cat(rex)"))
    assert (v.outcome, v.status) == (Outcome.UNKNOWN, None)
    assert v.explanation == "b1, b2 contradict each other, so this can't be checked"


def test_unknown_reason_is_reported():
    s = store()
    b1 = given(s, "x", "dog(rex)")
    v = check(s, add(s, ClaimType.INFERENCE, [b1], "dog(rex, tom)"))
    assert v.outcome == Outcome.UNKNOWN and v.explanation.startswith("logic check gave up:")


def test_applies_to_formal_facts_and_inferences_only():
    s = store()
    c = add(s, ClaimType.FACT, [], "dog(rex)")
    assert LogicChecker().applies_to(c)
    h = s.add(
        type=ClaimType.HYPOTHESIS,
        text="h",
        premises=[],
        source=MODEL,
        step=1,
        formal=Formal(statements=("dog(rex)",)),
    )
    assert not LogicChecker().applies_to(h)
    plain = s.add(type=ClaimType.FACT, text="f", premises=[], source=MODEL, step=1)
    assert not LogicChecker().applies_to(plain)


def test_default_checkers_include_logic():
    assert [c.id for c in default_checkers()] == [
        "structure",
        "arithmetic",
        "restatement",
        "logic",
    ]
```

- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/checkers/test_logic.py -q`. Expected: FAIL (`ModuleNotFoundError: kengine.checkers.logic`).

- [ ] **Step 3: Implement `src/kengine/checkers/logic.py`**

```python
"""The logic checker (spec §5.5 #3): does a formal claim follow from what it rests on?

An inference is checked against the premises it cites, so a step that silently needs
another premise fails with a counterexample. A fact is checked against everything trusted
(the givens and verified claims). An `undetermined` claim ("the premises settle X neither
way") is verified only if X is open, and every given has a formal form; otherwise a given
we couldn't translate might settle it.

The givens' formal forms are the model's translation (FORMALIZE). A `verified` here means
"follows in the fragment from those forms". The audit trail records them, so a
mistranslation can be found.
"""

from __future__ import annotations

from collections.abc import Sequence

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import TERMINAL, Status
from kengine.checkers.base import BeliefView, CheckerKind, Outcome, Verdict
from kengine.claims.formal import FormalSyntaxError, Statement, parse_statements
from kengine.claims.schema import Claim, ClaimType, Formal
from kengine.logic.entail import decide
from kengine.logic.render import render_world


def _ids(ids: Sequence[str]) -> str:
    return ", ".join(dict.fromkeys(ids))


def _has(ids: Sequence[str]) -> str:
    return "has" if len(set(ids)) == 1 else "have"


def _statements(formal: Formal) -> list[Statement]:
    return [s for text in formal.statements for s in parse_statements(text)]


def _holds(rec: BeliefRecord) -> bool:
    return rec.claim.formal is not None and rec.claim.formal.mode == "holds"


def _trusted(view: BeliefView, exclude: str) -> list[BeliefRecord]:
    return [
        r
        for r in view.all()
        if r.claim.id != exclude
        and r.status not in TERMINAL
        and _holds(r)
        and (r.claim.type == ClaimType.GIVEN or r.status == Status.VERIFIED)
    ]


class LogicChecker:
    id = "logic"
    kind = CheckerKind.DETERMINISTIC

    def applies_to(self, claim: Claim) -> bool:
        return claim.formal is not None and claim.type in (ClaimType.FACT, ClaimType.INFERENCE)

    def check(self, claim: Claim, view: BeliefView) -> Verdict:
        formal = claim.formal
        if formal is None:
            return self._verdict(Outcome.UNKNOWN, None, "no formal form")
        cited = [r for r in (view.get(p) for p in claim.premises) if r.status not in TERMINAL]
        step_check = formal.mode == "holds" and claim.type == ClaimType.INFERENCE
        if step_check:
            base = [r for r in cited if _holds(r)]
        else:
            base = _trusted(view, claim.id)
            base += [r for r in cited if _holds(r) and r not in base]
        try:
            premises: list[Statement] = []
            owners: list[str] = []
            for rec in base:
                assert rec.claim.formal is not None
                for s in _statements(rec.claim.formal):
                    premises.append(s)
                    owners.append(rec.claim.id)
            goal = _statements(formal)
        except FormalSyntaxError as e:  # stored forms are canonical; this is a safety net
            return self._verdict(Outcome.UNKNOWN, None, f"logic check gave up: {e}")
        d = decide(premises, goal)
        core = [owners[i] for i in d.core]

        if d.kind == "inconsistent":
            return self._verdict(
                Outcome.UNKNOWN,
                None,
                f"{_ids(core)} contradict each other, so this can't be checked",
                core,
            )
        if d.kind == "unknown":
            return self._verdict(Outcome.UNKNOWN, None, f"logic check gave up: {d.reason}")

        if formal.mode == "undetermined":
            if d.kind == "entailed":
                return self._verdict(
                    Outcome.FAIL, Status.CONTRADICTED, f"it isn't open: it follows from {_ids(core)}", core
                )
            if d.kind == "refuted":
                return self._verdict(
                    Outcome.FAIL,
                    Status.CONTRADICTED,
                    f"it isn't open: its negation follows from {_ids(core)}",
                    core,
                )
            informal = [
                r.claim.id
                for r in view.all()
                if r.claim.type == ClaimType.GIVEN and r.status not in TERMINAL and r.claim.formal is None
            ]
            if informal:
                return self._verdict(
                    Outcome.UNKNOWN,
                    None,
                    f"can't confirm it's open: {_ids(informal)} {_has(informal)} no formal form",
                )
            return self._verdict(
                Outcome.PASS,
                Status.VERIFIED,
                "the premises settle it neither way",
                [r.claim.id for r in base],
            )

        if d.kind == "entailed":
            text = f"follows from {_ids(core)}" if core else "is true in every case"
            return self._verdict(Outcome.PASS, Status.VERIFIED, text, core)
        if d.kind == "refuted":
            return self._verdict(Outcome.FAIL, Status.CONTRADICTED, f"contradicts {_ids(core)}", core)
        world = render_world(d.world)
        if not step_check:
            return self._verdict(Outcome.UNKNOWN, None, f"the givens don't settle it: they allow {world}")
        informal = [r.claim.id for r in cited if r.claim.formal is None]
        formal_ids = [r.claim.id for r in base]
        if informal:
            alone = _ids(formal_ids) if formal_ids else "the rest"
            return self._verdict(
                Outcome.UNKNOWN,
                None,
                f"can't check: {_ids(informal)} {_has(informal)} no formal form, and {alone} "
                f"alone allow {world}",
            )
        cites = _ids(formal_ids) if formal_ids else "its premises"
        return self._verdict(
            Outcome.FAIL,
            None,
            f"doesn't follow from {cites}: they allow {world}; cite what else it needs or fix "
            "the step",
            formal_ids,
            counterexample=world,
        )

    def _verdict(
        self,
        outcome: Outcome,
        status: Status | None,
        explanation: str,
        evidence: Sequence[str] = (),
        counterexample: str | None = None,
    ) -> Verdict:
        return Verdict(
            checker_id=self.id,
            checker_kind=self.kind,
            outcome=outcome,
            status=status,
            explanation=explanation,
            evidence=list(dict.fromkeys(evidence)),
            counterexample=counterexample,
        )
```

Add `LogicChecker()` last in `default_checkers()`, and update its docstring to "The M2 checkers ...".

- [ ] **Step 4: Run everything.** Run: `uv run pytest -q && uv run pyright && ruff check src tests && ruff format src tests`. Expected: all pass, `0 errors`. Format, then check that the line lengths in the code above are within 100.

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "feat: z3 logic checker with counterexample feedback"
```

---

## Task 5: FORMALIZE the givens, and the prompts

**Files:**
- Create: `src/kengine/claims/formalize.py`, `src/kengine/prompts/formalize.md`, `src/kengine/prompts/formal_text.md`, `src/kengine/prompts/formal_json.md`
- Modify:
  - `src/kengine/prompts/__init__.py`, `src/kengine/prompts/extract.md`, `src/kengine/prompts/think.md`
  - `src/kengine/config.py`, `kengine.example.toml`
  - `src/kengine/audit/events.py`, `src/kengine/loop/agent.py`, `src/kengine/loop/context.py` (`assemble_extract_context` takes `syntax`)
  - `src/kengine/eval/kengine_solver.py`, `src/kengine/cli/main.py` (`run`)
- Test: `tests/claims/test_formalize.py`, `tests/test_prompts.py`, `tests/test_config.py`, `tests/loop/test_agent.py`, `tests/cli/test_cli.py`

**Interfaces:**
- Consumes: `ExtractedFormal`, `to_formal` (Task 3), `JsonStatement`, `json_statement`, `FormalSyntax` (Task 1), `Formal`.
- Produces:
  - `claims/formalize.py`:
    - `FormalizedGiven(BaseModel)`, with `given: int (ge=1)` and `statements: list[str | JsonStatement]` (max 8)
    - `Formalization(BaseModel)`, with `givens: list[FormalizedGiven]` (max 60)
    - `formalize_messages(goal: str, statements: Sequence[str], syntax: FormalSyntax) -> list[Message]`
    - `read_formalization(parsed: Formalization, n: int) -> tuple[dict[int, Formal], list[str]]`. The keys are 1-based statement numbers; the list holds errors.
  - `prompts/__init__.py`: `formal_value(statements: Sequence[str], syntax: FormalSyntax, mode: str = "holds") -> dict[str, object]` and `system_prompt(name: Literal["extract", "formalize"], syntax: FormalSyntax) -> str`
  - `config.py`: `LogicConfig` with `syntax: FormalSyntax = "text"` and `formalize_givens: bool = True`; `KengineConfig.logic`; and `RolesConfig.formalize` (default: same as `extract`)
  - `audit/events.py`: `FormalizeEvent(kind="formalize", ok: bool, raw: str | None, errors: list[str])`, added to the `Event` union
  - `Agent(..., formal_syntax: FormalSyntax = "text", formalize_givens: bool = False)`. The default is off, so M1 tests and scripts don't change. `KengineSolver` and `kengine run` pass `cfg.logic` (on by default).
  - `assemble_extract_context(view, think_text, budget_tokens, count_tokens, syntax: FormalSyntax = "text")`

**Behaviour:**
- `_seed_givens` keeps the M1 rule: a given longer than 400 characters is skipped. If `formalize_givens` is set and any givens are kept, it makes **one** structured call (role `formalize`) on the kept statements before adding them, and gives each its `Formal`.
- The call is logged as a `model_call` (role `"formalize"`, step 0), plus a `FormalizeEvent`.
- On `StructuredOutputError`, a `FormalizeEvent(ok=False)` is logged and the givens go in without formal forms.
- On `BackendError`, it sleeps `retry_delay_s` and retries once; if that fails too, the same as above. **Formalization never ends a run.**

- [ ] **Step 1: Write the failing tests**

```python
# tests/claims/test_formalize.py
from kengine.claims.formalize import (
    Formalization,
    FormalizedGiven,
    formalize_messages,
    read_formalization,
)
from kengine.claims.formal import JsonStatement


def test_messages_number_the_statements():
    msgs = formalize_messages("Is Rex a mammal?", ["All dogs are mammals.", "Rex is a dog."], "text")
    assert msgs[0].role == "system" and "dog(x) -> mammal(x)" in msgs[0].content
    assert msgs[1].content.endswith("1. All dogs are mammals.\n2. Rex is a dog.")
    assert "## Task\nIs Rex a mammal?" in msgs[1].content


def test_json_syntax_prompt_shows_objects():
    msgs = formalize_messages("g", ["s"], "json")
    assert '{"if": ["dog(x)"], "then": "mammal(x)"}' in msgs[0].content


def test_read_formalization():
    parsed = Formalization(
        givens=[
            FormalizedGiven(given=1, statements=["dog(x) -> mammal(x)"]),
            FormalizedGiven(given=2, statements=[JsonStatement.model_validate({"then": "dog(rex)"})]),
            FormalizedGiven(given=3, statements=["dog(rex"]),
            FormalizedGiven(given=4, statements=[]),
            FormalizedGiven(given=9, statements=["a"]),
        ]
    )
    formal, errors = read_formalization(parsed, 4)
    assert {k: v.statements for k, v in formal.items()} == {
        1: ("dog(x) -> mammal(x)",),
        2: ("dog(rex)",),
    }
    assert errors[0].startswith("given 3: unbalanced")
    assert errors[1] == "given 9 doesn't exist"
```

```python
# tests/test_prompts.py (add)
from kengine.prompts import formal_value, system_prompt


def test_system_prompts_fill_the_syntax():
    for syntax in ("text", "json"):
        for name in ("extract", "formalize"):
            text = system_prompt(name, syntax)
            assert "{formal" not in text and "{example" not in text
    assert formal_value(["mammal(rex)"], "json") == {
        "mode": "holds",
        "statements": [{"if": [], "then": "mammal(rex)"}],
    }
```

```python
# tests/test_config.py (add)
def test_logic_defaults():
    cfg = KengineConfig()
    assert cfg.logic.syntax == "text" and cfg.logic.formalize_givens
    assert cfg.roles.formalize == cfg.roles.extract
```

```python
# tests/loop/test_agent.py (add)
def formalize_json(*items: tuple[int, list[str]]) -> ScriptedReply:
    return ScriptedReply(
        content=json.dumps({"givens": [{"given": g, "statements": s} for g, s in items]})
    )


def test_givens_are_formalized_and_a_logical_answer_verified(tmp_path):
    inference = claim("c1", "inference", "Rex is a mammal", ["b1", "b2"])
    inference["formal"] = {"statements": ["mammal(rex)"]}
    backend = ScriptedBackend(
        [
            formalize_json((1, ["dog(x) -> mammal(x)"]), (2, ["dog(rex)"]), (3, [])),
            think(),
            extract_json([inference], final("True", ["c1"])),
        ]
    )
    agent = Agent(
        backend=backend,
        roles=RolesConfig(),
        profile=get_profile("standard").model_copy(update={"n_samples": 1}),
        context_budget=8000,
        checkers=default_checkers(),
        session=Session.create(tmp_path),
        sleep=Sleeps(),
        formalize_givens=True,
    )
    r = agent.run(
        Task(goal="g", givens=["All dogs are mammals.", "Rex is a dog.", "Is Rex a mammal?"])
    )
    assert r.status == "answered" and r.answer == "True"
    assert agent.store.get("b4").status == Status.VERIFIED
    assert [c.schema_name for c in backend.calls][0] == "Formalization"
    kinds = [e.kind for e in agent.session.log.read()]
    assert kinds.index("formalize") < kinds.index("belief_added")
    assert "formal: dog(rex)" in backend.calls[2].messages[-1].content  # extract sees it


def test_formalize_failure_does_not_stop_the_run(tmp_path):
    backend = ScriptedBackend(
        [
            ScriptedReply(content="{not json"),
            think(),
            extract_json([], CONTINUE),
        ]
    )
    agent = Agent(
        backend=backend,
        roles=RolesConfig(),
        profile=get_profile("quick"),
        context_budget=8000,
        checkers=default_checkers(),
        session=Session.create(tmp_path),
        limits=Limits(max_steps=1),
        sleep=Sleeps(),
        formalize_givens=True,
    )
    r = agent.run(Task(goal="g", givens=["Rex is a dog."]))
    assert r.status == "budget_exhausted"
    assert agent.store.get("b1").claim.formal is None
    events = [e for e in agent.session.log.read() if e.kind == "formalize"]
    assert len(events) == 1 and not events[0].ok  # type: ignore[union-attr]


def test_formalize_backend_error_retries_once(tmp_path):
    sleeps = Sleeps()
    backend = ScriptedBackend(
        [
            ScriptedReply(error="down"),
            ScriptedReply(error="down"),
            think(),
            extract_json([], CONTINUE),
        ]
    )
    agent = Agent(
        backend=backend,
        roles=RolesConfig(),
        profile=get_profile("quick"),
        context_budget=8000,
        checkers=default_checkers(),
        session=Session.create(tmp_path),
        limits=Limits(max_steps=1),
        sleep=sleeps,
        formalize_givens=True,
    )
    assert agent.run(Task(goal="g", givens=["Rex is a dog."])).status == "budget_exhausted"
    assert sleeps == [5.0]
```

In `tests/cli/test_cli.py`, the `kengine run` test's script needs a formalize reply first, because the config default turns formalization on. Prepend `ScriptedReply(content='{"givens": []}')` to that script (and to any other CLI or solver test whose task has givens). Check that every assertion still holds.

- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest -q`. Expected: FAIL (import errors in the new tests).

- [ ] **Step 3: Write the prompts**

`src/kengine/prompts/formal_text.md`:

```
`formal` is {"mode": "holds" or "undetermined", "statements": [...]}. Each statement is a string in this language:

- an atom: `dog(rex)`, `likes(ann, bob)`, `raining`
- a negation: `not dog(rex)`
- a comparison of numbers: `age(bob) >= 18`, `price(a) + 2 = price(b)`
- a rule, read "for every x, if ... then ...": `dog(x) & not small(x) -> barks(x)`
- either/or: `cat(tom) | dog(tom)`, also as a rule's result: `pet(x) -> cat(x) | dog(x)`

`x`, `y`, `z`, `u`, `v` and `w` are variables: they stand for any individual. Every other name is a specific individual, property or relation. There's no "some" or "exists": name the individual instead. Names are lowercase, with underscores: `rex`, `has_fur`.
```

`src/kengine/prompts/formal_json.md`:

```
`formal` is {"mode": "holds" or "undetermined", "statements": [...]}. Each statement is an object {"if": [conditions], "then": result}, read "for every x, if all the conditions hold, then the result holds". For a plain statement, leave "if" empty: {"if": [], "then": "dog(rex)"}.

Conditions and results are literals:

- an atom: `dog(rex)`, `likes(ann, bob)`, `raining`
- a negation: `not dog(rex)`
- a comparison of numbers: `age(bob) >= 18`, `price(a) + 2 = price(b)`

A result can also be either/or: `cat(x) | dog(x)`. For example, "dogs that aren't small bark" is {"if": ["dog(x)", "not small(x)"], "then": "barks(x)"}.

`x`, `y`, `z`, `u`, `v` and `w` are variables: they stand for any individual. Every other name is a specific individual, property or relation. There's no "some" or "exists": name the individual instead. Names are lowercase, with underscores: `rex`, `has_fur`.
```

`src/kengine/prompts/formalize.md`:

```
You translate a task's statements into a small formal language, as JSON matching the given schema. Translate what each statement says, nothing more, and don't solve the task.

For each numbered statement, write {"given": <its number>, "statements": [...]}. Use an empty list for a statement that doesn't fit the language: a question, an instruction, a calculation, or something only loosely logical.

{formal_syntax}

Use one vocabulary for the whole task: the same name for the same individual, property or relation everywhere. Choose names that also fit the task's question or conclusion, because later steps reuse yours.

Example. Statements:
1. All dogs are mammals.
2. Rex is a dog or a cat.
3. Rex is not a cat.
4. What is Rex?

Output:

{example}
```

`extract.md`. Insert this before the line `Set \`premises\` to ...`:

```
For a `fact` or `inference` that's a logical statement about the problem's individuals, properties and relations (not a calculation), also set `formal`.

{formal_syntax}

- Reuse the names in the existing claims' `formal:` forms. Invent a new name only for something new.
- An `inference`'s `premises` must list every claim it follows from.
- If the reasoning says the premises settle a statement neither way, record that statement with `"mode": "undetermined"`. Otherwise use `"mode": "holds"`.
- Leave `formal` out when the item doesn't fit the language. Never force it.

For example, the item "Rex is a mammal (from b1 and b2)" becomes {"ref": "c1", "type": "inference", "text": "Rex is a mammal", "premises": ["b1", "b2"], "formal": {formal_example}}.
```

The pen example stays the **last line** of `extract.md`; `test_extract_example_is_valid_extraction` relies on that.

`think.md`. Add to the Rules list:

```
- In a logic problem (premises and a conclusion to judge), make each item one inference and name everything it uses, for example "Rex is a mammal (from b1 and b2)". If the premises settle the conclusion neither way, write an item that says so, for example "b1 to b4 don't settle whether Rex barks", and rest the answer on it.
```

- [ ] **Step 4: Implement the code**

`prompts/__init__.py`:

```python
"""Prompt templates shipped as package data."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib.resources import files
from typing import Literal

from kengine.claims.formal import FormalSyntax, json_statement


def load_prompt(name: str) -> str:
    return (files("kengine.prompts") / f"{name}.md").read_text(encoding="utf-8")


def formal_value(
    statements: Sequence[str], syntax: FormalSyntax, mode: str = "holds"
) -> dict[str, object]:
    """A `formal` object in `syntax`, for prompt examples."""
    if syntax == "text":
        return {"mode": mode, "statements": list(statements)}
    return {
        "mode": mode,
        "statements": [json_statement(s).model_dump(by_alias=True) for s in statements],
    }


_FORMALIZE_EXAMPLE: list[tuple[int, list[str]]] = [
    (1, ["dog(x) -> mammal(x)"]),
    (2, ["dog(rex) | cat(rex)"]),
    (3, ["not cat(rex)"]),
    (4, []),
]


def system_prompt(name: Literal["extract", "formalize"], syntax: FormalSyntax) -> str:
    text = load_prompt(name).replace("{formal_syntax}", load_prompt(f"formal_{syntax}").strip())
    if name == "extract":
        return text.replace("{formal_example}", json.dumps(formal_value(["mammal(rex)"], syntax)))
    example = {
        "givens": [
            {"given": n, "statements": formal_value(s, syntax)["statements"]}
            for n, s in _FORMALIZE_EXAMPLE
        ]
    }
    return text.replace("{example}", json.dumps(example))
```

`claims/formalize.py`:

```python
"""FORMALIZE: one call that gives the task's givens formal forms (spec §5.3)."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from kengine.backend.types import Message
from kengine.claims.extraction import ExtractedFormal, to_formal
from kengine.claims.formal import FormalSyntax, JsonStatement
from kengine.claims.schema import Formal
from kengine.prompts import system_prompt


class FormalizedGiven(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    given: int = Field(ge=1)
    statements: list[str | JsonStatement] = Field(
        default_factory=list[str | JsonStatement], max_length=8
    )


class Formalization(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    givens: list[FormalizedGiven] = Field(max_length=60)


def formalize_messages(
    goal: str, statements: Sequence[str], syntax: FormalSyntax
) -> list[Message]:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))
    user = f"## Task\n{goal}\n\n## Statements to translate\n{numbered}"
    return [
        Message(role="system", content=system_prompt("formalize", syntax)),
        Message(role="user", content=user),
    ]


def read_formalization(parsed: Formalization, n: int) -> tuple[dict[int, Formal], list[str]]:
    out: dict[int, Formal] = {}
    errors: list[str] = []
    for item in parsed.givens:
        if not 1 <= item.given <= n:
            errors.append(f"given {item.given} doesn't exist")
            continue
        if not item.statements or item.given in out:
            continue
        try:
            out[item.given] = to_formal(ExtractedFormal(statements=item.statements))
        except ValueError as e:
            errors.append(f"given {item.given}: {e}")
    return out, errors
```

(Check `Message`'s import path in `backend/types.py`. If `prompts` importing `claims.formal` and `claims.formalize` importing `prompts` causes a cycle, it won't: `claims/formal.py` imports neither.)

`config.py`:

```python
class LogicConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    syntax: FormalSyntax = "text"  # set from the Checkpoint C probe (spec §9)
    formalize_givens: bool = True
```

Add `formalize: RoleConfig = Field(default_factory=_extract_role)` to `RolesConfig`, and `logic: LogicConfig = Field(default_factory=LogicConfig)` to `KengineConfig`. Add matching `[logic]` and `[roles.formalize]` tables to `kengine.example.toml`; copy `[roles.extract]` and change the comment to `# translates the task's givens into the formal fragment`. If `tests/test_config.py` checks that the example file matches the defaults, it will catch a mismatch.

`audit/events.py`:

```python
class FormalizeEvent(_Event):
    kind: Literal["formalize"] = "formalize"
    ok: bool
    raw: str | None
    errors: list[str]
```

Add it to the `Event` union.

`loop/context.py`: `assemble_extract_context(..., syntax: FormalSyntax = "text")` uses `system_prompt("extract", syntax)` instead of `load_prompt("extract")`.

`loop/agent.py`:
- Add the constructor parameters `formal_syntax: FormalSyntax = "text"` and `formalize_givens: bool = False` and store them.
- Pass `self.formal_syntax` to `assemble_extract_context`.
- Replace `_seed_givens` with:

```python
    def _seed_givens(self, task: Task) -> None:
        """The task's statements become `given` claims b1, b2, ... before the first step,
        each with a formal form when FORMALIZE gives one. A statement too long for a claim
        is left out; it's still in the task text."""
        kept = [t for t in task.givens if len(t) <= MAX_GIVEN_CHARS]
        formal = self._formalize(task, kept) if self.formalize_givens and kept else {}
        for i, text in enumerate(kept, 1):
            claim = self.store.add(
                type=ClaimType.GIVEN,
                text=text,
                premises=[],
                source=Source(kind=SourceKind.TASK, ref=f"given:{i}"),
                step=0,
                formal=formal.get(i),
            )
            self._log(BeliefAddedEvent(step=0, claim=claim))

    def _formalize(self, task: Task, statements: list[str]) -> dict[int, Formal]:
        """One FORMALIZE call. Any failure leaves the givens informal; it never ends the run."""
        messages = formalize_messages(task.goal, statements, self.formal_syntax)
        for attempt in (1, 2):
            t0 = self.clock()
            try:
                parsed, completion = self.backend.complete_structured(
                    messages, self.roles.formalize, Formalization
                )
            except StructuredOutputError as e:
                if e.completion is not None:
                    self._record_call(messages, e.completion, "formalize", 0, t0)
                self._log(FormalizeEvent(step=0, ok=False, raw=e.raw, errors=[str(e)]))
                return {}
            except BackendError as e:
                if attempt == 1:
                    self.sleep(self.retry_delay_s)
                    continue
                self._log(FormalizeEvent(step=0, ok=False, raw=None, errors=[str(e)]))
                return {}
            self._record_call(messages, completion, "formalize", 0, t0)
            formal, errors = read_formalization(parsed, len(statements))
            self._log(FormalizeEvent(step=0, ok=True, raw=completion.content, errors=errors))
            return formal
        return {}
```

Note that the M1 version numbered `ref=f"given:{i}"` over *all* givens, including skipped ones. It now numbers the kept ones, so the ref matches the statement number FORMALIZE saw. Check that no test depends on the old numbering.

`eval/kengine_solver.py` and `cli/main.py` `run`: pass `formal_syntax=cfg.logic.syntax, formalize_givens=cfg.logic.formalize_givens` to `Agent`. (`KengineSolver` has `self._config`.)

- [ ] **Step 5: Run everything.** Run: `uv run pytest -q && uv run pyright && ruff check src tests && ruff format --check src tests`. Expected: all pass, `0 errors`.

- [ ] **Step 6: Commit**

```bash
git add -A src tests kengine.example.toml
git commit -m "feat: formalize the task's givens and extract formal forms"
```

---

## Task 6: Eval tooling: `--limit`, `eval-compare`, `probe-formal`

**Files:**
- Create: `src/kengine/eval/compare.py`, `src/kengine/eval/probe_formal.py`
- Modify: `src/kengine/cli/main.py`, `src/kengine/eval/runner.py` (`results_dir_name` gains `limit`)
- Test: `tests/eval/test_compare.py`, `tests/eval/test_probe_formal.py`, `tests/cli/test_cli.py`

**Interfaces:**
- Consumes: `decide` (Task 2), `formalize_messages`, `read_formalization`, `Formalization` (Task 5), `parse_statements` (Task 1), `EvalItem` (M0).
- Produces:
  - `results_dir_name(date, suite, solver, size, profile, limit: int | None = None)`, which appends `-first{limit}` when `limit` is set
  - `kengine eval ... --limit N`: the first N items of the seeded selection, recorded as `"limit"` in `config.json`
  - `compare.py`: `class Comparison(BaseModel)` with `n`, `base_correct`, `other_correct`, `both_right`, `fixed`, `broke`, `caught`, `false_alarm` and `both_wrong` (all ints), plus `ids: dict[str, list[str]]` per category; `compare_runs(base_dir: Path, other_dir: Path) -> Comparison`
  - `kengine eval-compare BASE OTHER`, which prints the comparison and writes `OTHER/compare-<basename(BASE)>.json`
  - `probe_formal.py`: `class ProbeLine(BaseModel)` and `class ProbeSummary(BaseModel)` (fields below), `probe_item(backend, role, syntax, item) -> ProbeLine` and `run_probe(backend, role, syntax, items, out_dir) -> ProbeSummary`
  - `kengine probe-formal --syntax text|json --size smoke|small [--limit N]`, writing `eval/results/<date>-folio-probe-<syntax>-<size>[-firstN]/` with `items.jsonl` and `summary.json`

**Comparison categories.** Each item id in both runs is classed by the base run's `correct` and the other run's `status`/`correct`:
- `both_right`: base right, other right
- `broke`: base right, other answered wrong
- `false_alarm`: base right, other not answered
- `fixed`: base wrong, other right
- `caught`: base wrong, other not answered
- `both_wrong`: base wrong, other answered wrong

The printout is one line: `n=… base_acc=… other_acc=… fixed=… broke=… caught=… false_alarm=… both_right=… both_wrong=… errors_caught=… false_alarm_rate=…`, where:
- `errors_caught = (fixed + caught) / (n - base_correct)`
- `false_alarm_rate = false_alarm / base_correct`

Print `n/a` when a denominator is 0.

**Probe.** For each FOLIO item:
1. Take the conclusion from `item.question` with `re.search(r"^Conclusion: (.+)$", question, re.M)`.
2. Formalize `item.givens + [conclusion]` in one call, using `formalize_messages(item.question, ...)` and `roles.formalize`.
3. Decide the premises' statements against the conclusion's: `entailed` → `True`, `refuted` → `False`, `open` → `Uncertain`, anything else → `None`.

`ProbeLine` has `item_id`, `gold`, `predicted: str | None`, `correct: bool`, `n_premises`, `n_formalized` (premises that got a formal form), `conclusion_ok: bool`, `decision: str | None`, `errors: list[str]` and `wall_s: float`. A `BackendError` gives `predicted=None` and records the error.

`ProbeSummary` has `syntax`, `n`, `all_formalized` (every premise and the conclusion), `decided` (predicted is not None), `correct` and `mean_wall_s`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_compare.py
import json
from pathlib import Path

from kengine.eval.compare import compare_runs


def write(d: Path, rows: list[tuple[str, str, bool]]) -> Path:
    d.mkdir()
    with (d / "items.jsonl").open("w") as f:
        for item_id, status, correct in rows:
            f.write(json.dumps({"item_id": item_id, "status": status, "correct": correct}) + "\n")
    return d


def test_compare_categories(tmp_path: Path):
    base = write(
        tmp_path / "base",
        [("a", "answered", True), ("b", "answered", True), ("c", "answered", True),
         ("d", "answered", False), ("e", "partial", False), ("f", "answered", False),
         ("only-base", "answered", True)],
    )  # fmt: skip
    other = write(
        tmp_path / "other",
        [("a", "answered", True), ("b", "answered", False), ("c", "blocked", False),
         ("d", "answered", True), ("e", "blocked", False), ("f", "answered", False)],
    )  # fmt: skip
    c = compare_runs(base, other)
    assert (c.n, c.base_correct, c.other_correct) == (6, 3, 2)
    assert (c.both_right, c.broke, c.false_alarm) == (1, 1, 1)
    assert (c.fixed, c.caught, c.both_wrong) == (1, 1, 1)
    assert c.ids["false_alarm"] == ["c"] and c.ids["caught"] == ["e"]
```

```python
# tests/eval/test_probe_formal.py
import json

from kengine.backend.fake import ScriptedBackend, ScriptedReply
from kengine.config import RolesConfig
from kengine.eval.adapters.folio import parse_row
from kengine.eval.probe_formal import probe_item, run_probe

ROW = {
    "premises": "All dogs are mammals.\nRex is a dog.",
    "conclusion": "Rex is a mammal.",
    "label": "True",
    "example_id": 7,
}


def reply(*items: tuple[int, list[str]]) -> ScriptedReply:
    return ScriptedReply(
        content=json.dumps({"givens": [{"given": g, "statements": s} for g, s in items]})
    )


def test_probe_item_decides_the_conclusion():
    backend = ScriptedBackend(
        [reply((1, ["dog(x) -> mammal(x)"]), (2, ["dog(rex)"]), (3, ["mammal(rex)"]))]
    )
    line = probe_item(backend, RolesConfig().formalize, "text", parse_row(ROW, 0))
    assert (line.predicted, line.correct, line.decision) == ("True", True, "entailed")
    assert (line.n_premises, line.n_formalized, line.conclusion_ok) == (2, 2, True)
    assert "3. Rex is a mammal." in backend.calls[0].messages[-1].content


def test_probe_item_without_conclusion_form():
    backend = ScriptedBackend([reply((1, ["dog(x) -> mammal(x)"]), (2, ["dog(rex)"]))])
    line = probe_item(backend, RolesConfig().formalize, "text", parse_row(ROW, 0))
    assert line.predicted is None and not line.correct and not line.conclusion_ok


def test_run_probe_writes_summary(tmp_path):
    backend = ScriptedBackend(
        [reply((1, ["dog(x) -> mammal(x)"]), (2, ["dog(rex)"]), (3, ["mammal(rex)"])),
         ScriptedReply(error="down")]
    )  # fmt: skip
    items = [parse_row(ROW, 0), parse_row({**ROW, "example_id": 8}, 1)]
    s = run_probe(backend, RolesConfig().formalize, "text", items, tmp_path / "out")
    assert (s.n, s.all_formalized, s.decided, s.correct) == (2, 1, 1, 1)
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["correct"] == 1
    assert len((tmp_path / "out" / "items.jsonl").read_text().splitlines()) == 2
```

CLI tests (add to `tests/cli/test_cli.py`, following the existing `eval` test's pattern of monkeypatching `suites.load_items` and `backend_factory`):
- `test_cli_eval_limit`: `--limit 1` on a 2-item scripted suite writes `*-first1/` with one line in `items.jsonl`, and `config.json["limit"] == 1`.
- `test_cli_eval_compare`: writes two tiny results dirs and checks the output contains `fixed=` and that `compare-<base>.json` exists.

- [ ] **Step 2: Run them and confirm they fail.** Run: `uv run pytest tests/eval tests/cli -q`. Expected: FAIL (import errors).

- [ ] **Step 3: Implement**

`eval/compare.py`:

```python
"""With vs. without: which baseline errors a run avoided, and what it broke (spec §6)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

CATEGORIES = ("both_right", "broke", "false_alarm", "fixed", "caught", "both_wrong")


class Comparison(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    base_correct: int
    other_correct: int
    both_right: int
    broke: int
    false_alarm: int
    fixed: int
    caught: int
    both_wrong: int
    ids: dict[str, list[str]]

    def line(self) -> str:
        wrong = self.n - self.base_correct
        caught = f"{(self.fixed + self.caught) / wrong:.3f}" if wrong else "n/a"
        alarms = f"{self.false_alarm / self.base_correct:.3f}" if self.base_correct else "n/a"
        acc = (lambda k: f"{k / self.n:.3f}") if self.n else (lambda k: "n/a")
        return (
            f"n={self.n} base_acc={acc(self.base_correct)} other_acc={acc(self.other_correct)} "
            f"fixed={self.fixed} broke={self.broke} caught={self.caught} "
            f"false_alarm={self.false_alarm} both_right={self.both_right} "
            f"both_wrong={self.both_wrong} errors_caught={caught} false_alarm_rate={alarms}"
        )


def _rows(d: Path) -> dict[str, tuple[str, bool]]:
    out: dict[str, tuple[str, bool]] = {}
    for raw in (d / "items.jsonl").read_text(encoding="utf-8").splitlines():
        if raw.strip():
            row = json.loads(raw)
            out[str(row["item_id"])] = (str(row["status"]), bool(row["correct"]))
    return out


def compare_runs(base_dir: Path, other_dir: Path) -> Comparison:
    base, other = _rows(base_dir), _rows(other_dir)
    ids: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    common = [i for i in base if i in other]
    for item_id in common:
        base_right = base[item_id][1]
        status, right = other[item_id]
        if right:
            ids["both_right" if base_right else "fixed"].append(item_id)
        elif status != "answered":
            ids["false_alarm" if base_right else "caught"].append(item_id)
        else:
            ids["broke" if base_right else "both_wrong"].append(item_id)
    counts = {c: len(v) for c, v in ids.items()}
    return Comparison(
        n=len(common),
        base_correct=sum(base[i][1] for i in common),
        other_correct=sum(other[i][1] for i in common),
        ids=ids,
        **counts,
    )
```

`eval/probe_formal.py`:

```python
"""Which formal serialization does the model fill in more reliably? (spec §9)

Each FOLIO item's premises and conclusion are formalized in one call. Z3 then decides the
conclusion: entailed is True, refuted is False, open is Uncertain. That measures the
translation, which is what the logic checker depends on, without the rest of the loop.
"""

from __future__ import annotations

import json
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
```

(`test_run_probe_writes_summary` expects `all_formalized == 1`: the second item's backend error gives `conclusion_ok=False`.)

`cli/main.py`:
- In `eval`, add `limit: Annotated[int | None, typer.Option(min=1, help="Only the first N items.")] = None`. Then `items = items[:limit]` when it's set, `results_dir_name(..., limit)`, and `"limit": limit` in `run_config`.
- Add the `eval-compare` command:

```python
@app.command("eval-compare")
def eval_compare(
    base: Annotated[Path, typer.Argument(help="Results dir of the reference run (baseline)")],
    other: Annotated[Path, typer.Argument(help="Results dir to compare with it")],
) -> None:
    """Compare two runs item by item: errors caught, fixed, broken and false alarms."""
    c = compare_runs(base, other)
    typer.echo(c.line())
    out = other / f"compare-{base.name}.json"
    out.write_text(c.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(f"wrote {out}")
```

- Add the `probe-formal` command:

```python
@app.command("probe-formal")
def probe_formal(
    syntax: Annotated[FormalSyntax, typer.Option(help="text | json")] = "text",
    size: Annotated[Size, typer.Option()] = "small",
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    seed: Annotated[int, typer.Option()] = 0,
    out: Annotated[Path, typer.Option()] = Path("eval/results"),
) -> None:
    """Formalize FOLIO items in SYNTAX and score the Z3 verdicts (spec §9)."""
    cfg, _ = _config()
    items = suites.load_items("folio", size, seed)
    if limit is not None:
        items = items[:limit]
    name = f"{datetime.date.today().isoformat()}-folio-probe-{syntax}-{size}"
    out_dir = out / (name + (f"-first{limit}" if limit is not None else ""))
    s = run_probe(backend_factory(cfg.backend), cfg.roles.formalize, syntax, items, out_dir)
    typer.echo(
        f"{syntax}: n={s.n} all_formalized={s.all_formalized} decided={s.decided} "
        f"correct={s.correct} mean_wall_s={s.mean_wall_s:.1f} -> {out_dir}"
    )
```

- [ ] **Step 4: Run everything.** Run: `uv run pytest -q && uv run pyright && ruff check src tests && ruff format --check src tests`. Expected: all pass, `0 errors`.

- [ ] **Step 5: Commit**

```bash
git add -A src tests
git commit -m "feat: eval --limit, eval-compare and the formal serialization probe"
```

---

## Task 7: Real-model probe, serialization choice and live check

This runs on the GPU machine with `llama-server` up (runbook Setup). Nothing else may be using the server.

- [ ] **Step 1:** `uv sync --extra eval && uv run kengine doctor`. Expected: OK.
- [ ] **Step 2:** Run the probe in both serializations on the same 40 items:

```bash
uv run kengine probe-formal --syntax text --size small --limit 40
uv run kengine probe-formal --syntax json --size small --limit 40
```

- [ ] **Step 3: Choose.** The serialization with more `correct` wins; on a tie, the one with more `all_formalized`; if still tied, `text`. If `json` wins, change `LogicConfig.syntax`'s default to `"json"` and update `kengine.example.toml` and `test_logic_defaults`. Record both summaries and the decision for CHECKPOINT-C.md.
- [ ] **Step 4: Read 5 probe failures** (`items.jsonl` lines with `correct: false`) in the chosen syntax, and their formalize outputs in the raw completion. If one prompt fix would plausibly fix several (a recurring naming inconsistency, say), make it, re-run the winning syntax's probe once, and record before and after. Don't iterate further; the checkpoint measures the system, not prompt tuning.
- [ ] **Step 5: Live check:**

```bash
uv run kengine run "All dogs are mammals. Rex is a dog. Based only on these premises, is the conclusion 'Rex is a mammal' True, False, or Uncertain?" --profile standard
```

  Expected: `status: answered`, `answer: True`. In the session's `events.jsonl` there's a `formalize` event, givens with `formal`, and an inference whose `logic` verdict is `verified`. If it's blocked instead, read the trail and fix the cause (prompt or code) before Task 8, with a regression test if the cause is in code.
- [ ] **Step 6: Commit** the probe results dirs, plus any syntax or prompt change (`git add eval/results/*probe* src tests kengine.example.toml`), with the message `eval: formal serialization probe (<winner> chosen)`.

---

## Task 8: Checkpoint C (owner review point 2)

**Files:**
- Modify: `docs/real-machine-runbook.md` (add Checkpoint C, describing steps 1–6 below)
- Create: `eval/results/CHECKPOINT-C.md`

- [ ] **Step 1: Size the runs.** Run folio smoke under both profiles:

```bash
uv run kengine eval folio --solver kengine --size smoke --profile quick
uv run kengine eval folio --solver kengine --size smoke --profile standard
```

  Let `t = mean_wall_s(quick) + mean_wall_s(standard)`. Then `N = min(100, floor(36000 / t / 10) * 10)`: a 10-hour budget for the small runs, rounded down to a multiple of 10. N is at least 10.
- [ ] **Step 2: Main runs:**

```bash
uv run kengine eval folio --solver kengine --size small --profile quick --limit N
uv run kengine eval folio --solver kengine --size small --profile standard --limit N
```

  If N = 100, omit `--limit`.
- [ ] **Step 3: Arithmetic regression** (formalization is on for every suite):

```bash
uv run kengine eval gsm8k --solver kengine --size smoke --profile quick
uv run kengine eval gsm8k --solver kengine --size smoke --profile standard
```

  Compare these with the Checkpoint B gsm8k smoke runs, using `eval-compare` with the Checkpoint B dir as base.
- [ ] **Step 4: Compare.** Run `uv run kengine eval-compare eval/results/2026-11-02-folio-baseline-small-none <run dir>` for each folio run. (Use the real baseline dir name; `ls eval/results`.)
- [ ] **Step 5: Write `eval/results/CHECKPOINT-C.md`**, with these sections:
  - **Setup:** commit, server command, chosen syntax, N and why.
  - **Probe:** both summaries; the decision; the formalize-and-solve accuracy as a reference point.
  - **Results table** per run: n, headline, abstention, answered acc., mean wall s/item, and the baseline's accuracy on the same items (`base_acc` from `eval-compare`).
  - **With vs. without** (§6 metrics): fixed, broke, caught, false alarms, errors caught and false-alarm rate per profile.
  - **Where the logic checker acted:** count `logic` verdicts by outcome and status over all folio sessions (a short Python one-off over `sessions/*/events.jsonl`; put the script in the doc).
  - **Three audit-trail excerpts** from `events.jsonl`: one item kengine fixed, one it broke or falsely blocked, and one blocked by an informal inference (the M5 critic's job).
  - **gsm8k regression:** the comparison with Checkpoint B.
  - **Insights for the owner (review point 2):** what M2 bought and cost on FOLIO, whether formalization is reliable enough to lean on, what blocks most (expected: informal inferences, so M5 matters), and whether M3 or M5 should come next. M3 is the spec's order; scope can't change, only order (§7).
- [ ] **Step 6: Commit and push** everything except `sessions/` (gitignored): `git add eval/results docs/real-machine-runbook.md && git commit -m "eval: checkpoint C (logic checker, with vs without on folio)" && git push`. Tell the owner the review point is ready.

---

## After this plan

- **Owner review point 2** reads CHECKPOINT-C.md. It may reorder M3 (retrieval, entailment, fact and consistency checkers) and M5 (the critic, which unblocks informal inferences), but not change the scope.
- **Known limits to carry forward:**
  - The givens' formal forms are the model's translation, and nothing checks that translation yet. M3's consistency checker and M5's critic are the natural places to cross-check it.
  - Existentials aren't in the fragment. A FOLIO premise like "some dogs bark" stays informal.
