"""FOLIO's expert first-order logic, converted to the formal fragment (fine-tuning data).

FOLIO writes each premise and conclusion in textbook notation: `∀x (Dog(x) → Mammal(x))`.
The conversion is the standard one. Push negations inward, give each "some" individual a
name, and split the result into clauses, each written as `body -> head`.

Some formulas can't be converted:

- **Equality**, which the fragment doesn't have.
- **A "some" inside an "every"**, which would need a function where the fragment only
  names individuals.
- **A "some" in a conclusion.** A named individual would claim one specific thing, not
  that some thing exists.
- **A "some" whose body is an "if"**, which is almost always a slip for "and".
- **A formula that needs more than 8 statements or 6 variables.**

These raise `Unsupported`.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from kengine.claims.formal import is_variable, parse_statements

BinOp = Literal["&", "|", "->", "<->", "^"]


@dataclass(frozen=True)
class Pred:
    name: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Not:
    f: Formula


@dataclass(frozen=True)
class Ante:
    """A literal from an "if" part. It goes in a rule's body, so a rule reads in the
    sentence's direction: `not a(x) -> not b(x)`, not `b(x) -> a(x)`."""

    f: Pred | Not


@dataclass(frozen=True)
class Bin:
    op: BinOp
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Quant:
    q: Literal["all", "some"]
    var: str
    body: Formula


Formula = Pred | Not | Ante | Bin | Quant
Literal_ = tuple[bool, Pred, bool]  # (positive, atom, from an "if" part)

MAX_STATEMENTS = 8  # FormalizedGiven's limit
_VARS = ("x", "y", "z", "u", "v", "w")


class Unsupported(ValueError):
    """The formula is outside what the fragment can say, or isn't well formed."""


_TOKEN_RE = re.compile(r"\s*(∀|∃|¬|∧|∨|→|↔|⊕|\(|\)|,|[A-Za-z0-9_'.\-]+)")
_PREC = {"↔": 1, "→": 2, "⊕": 3, "∨": 3, "∧": 4}
_OPS: dict[str, BinOp] = {"↔": "<->", "→": "->", "⊕": "^", "∨": "|", "∧": "&"}


def _tokens(text: str) -> list[str]:
    if "=" in text or "≠" in text:
        raise Unsupported("uses equality")
    out: list[str] = []
    i, text = 0, text.strip()
    while i < len(text):
        m = _TOKEN_RE.match(text, i)
        if m is None:
            raise Unsupported(f"unexpected {text[i]!r}")
        out.append(m.group(1))
        i = m.end()
        while i < len(text) and text[i].isspace():
            i += 1
    # FOLIO's parentheses are sometimes unbalanced at the end.
    depth = out.count("(") - out.count(")")
    out += [")"] * max(depth, 0)
    while depth < 0 and out and out[-1] == ")":
        out.pop()
        depth += 1
    return out


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens, self.i = tokens, 0

    def peek(self) -> str | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def eat(self, want: str | None = None) -> str:
        tok = self.peek()
        if tok is None or (want is not None and tok != want):
            raise Unsupported(f"expected {want or 'more'}, got {tok}")
        self.i += 1
        return tok

    def formula(self, min_prec: int = 1) -> Formula:
        left = self.unary()
        while (op := self.peek()) in _PREC and _PREC[op] >= min_prec:
            self.eat()
            # → and ↔ group to the right.
            right = self.formula(_PREC[op] if op in ("→", "↔") else _PREC[op] + 1)
            left = Bin(_OPS[op], left, right)
        return left

    def unary(self) -> Formula:
        tok = self.peek()
        if tok == "¬":
            self.eat()
            return Not(self.unary())
        if tok in ("∀", "∃"):
            self.eat()
            var = self.eat()
            # A quantifier's body runs as far right as it can.
            return Quant("all" if tok == "∀" else "some", var, self.formula())
        if tok == "(":
            self.eat()
            f = self.formula()
            self.eat(")")
            return f
        name = self.eat()
        if not name[0].isalpha():
            raise Unsupported(f"bad name {name}")
        args: list[str] = []
        if self.peek() == "(":
            self.eat()
            args.append(self.eat())
            while self.peek() == ",":
                self.eat()
                args.append(self.eat())
            self.eat(")")
        return Pred(name, tuple(args))


def parse_fol(text: str) -> Formula:
    p = _Parser(_tokens(text))
    f = p.formula()
    if p.peek() is not None:
        raise Unsupported(f"unexpected {p.peek()}")
    return f


def nnf(f: Formula, neg: bool = False) -> Formula:
    """Negations only on atoms, and only `&`, `|` and quantifiers above them. The literals
    of an implication's "if" part are marked `Ante`."""
    match f:
        case Pred():
            return Not(f) if neg else f
        case Not(g):
            return nnf(g, not neg)
        case Ante():
            return f
        case Quant(q, var, body):
            flip: Literal["all", "some"] = "some" if q == "all" else "all"
            return Quant(flip if neg else q, var, nnf(body, neg))
        case Bin("->", a, b) if not neg:
            return Bin("|", _mark(nnf(a, True)), nnf(b))
        case Bin("->", a, b):
            return nnf(Bin("|", Not(a), b), neg)
        case Bin("<->", a, b):
            return nnf(Bin("&", Bin("->", a, b), Bin("->", b, a)), neg)
        case Bin("^", a, b):
            return nnf(Bin("&", Bin("|", a, b), Not(Bin("&", a, b))), neg)
        case Bin(op, a, b):
            flipped: BinOp = "|" if op == "&" else "&"
            return Bin(flipped if neg else op, nnf(a, neg), nnf(b, neg))


def _mark(f: Formula) -> Formula:
    match f:
        case Pred() | Not():
            return Ante(f)
        case Ante():
            return f
        case Quant(q, var, body):
            return Quant(q, var, _mark(body))
        case Bin(op, a, b):
            return Bin(op, _mark(a), _mark(b))


def _some_if(f: Formula) -> bool:
    match f:
        case Quant("some", _, body):
            while isinstance(body, Quant):
                body = body.body
            return (isinstance(body, Bin) and body.op == "->") or _some_if(body)
        case Quant(_, _, body) | Not(body) | Ante(body):
            return _some_if(body)
        case Bin(_, a, b):
            return _some_if(a) or _some_if(b)
        case Pred():
            return False


def free_variables(f: Formula, bound: frozenset[str] = frozenset()) -> list[str]:
    """Variables used without a quantifier, in order. FOLIO leaves out a leading `∀x`
    sometimes, and means it."""
    match f:
        case Pred(_, args):
            return [a for a in args if a not in bound and is_variable(a)]
        case Not(g) | Ante(g):
            return free_variables(g, bound)
        case Quant(_, var, body):
            return free_variables(body, bound | {var})
        case Bin(_, a, b):
            left = free_variables(a, bound)
            return left + [v for v in free_variables(b, bound) if v not in left]


def snake(name: str) -> str:
    """`WildTurkey` -> `wild_turkey`, `Gould's` -> `goulds`."""
    name = name.replace("'", "").replace(".", "_").replace("-", "_")
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name)
    s = re.sub(r"(?<=[A-Z])([A-Z][a-z])", r"_\1", s)
    return re.sub(r"_+", "_", s.lower()).strip("_")


class Story:
    """Names for one story: one spelling per name, and fresh names for "some" individuals.

    FOLIO sometimes spells a name two ways in one story (`ResidentOf`, `Residentof`). The
    first spelling is kept, so the premises and the conclusion still talk about one thing.
    """

    def __init__(self) -> None:
        self._spelling: dict[str, str] = {}
        self._used: set[str] = set()

    def name(self, raw: str) -> str:
        key = raw.lower().replace("_", "")
        s = self._spelling.setdefault(key, snake(raw))
        self._used.add(s)
        return s

    def fresh(self, hint: str) -> str:
        base = f"some_{hint}" if hint else "something"
        name, n = base, 1
        while name in self._used:
            n += 1
            name = f"{base}_{n}"
        self._used.add(name)
        return name

    def convert(self, text: str, *, conclusion: bool = False) -> list[str]:
        """The statements for one FOLIO formula, or `Unsupported`."""
        f = parse_fol(text)
        if _some_if(f):
            # "Some fish sting people" as `∃x (Fish(x) → Sting(x))`: an expert slip, true
            # of anything that isn't a fish.
            raise Unsupported('"some" with an "if" part')
        for v in reversed(free_variables(f)):
            f = Quant("all", v, f)
        f = self._rename(nnf(f), {})
        clauses = _cnf(self._skolem(f, {}, [], conclusion))
        out: list[str] = []
        for c in clauses:
            t = _clause_text(c)
            if t is not None and t not in out:
                out.append(t)
        if not out:
            raise Unsupported("says nothing")
        if len(out) > MAX_STATEMENTS:
            raise Unsupported(f"needs {len(out)} statements")
        for t in out:
            parse_statements(t)  # the fragment's own check
        return out

    def _rename(self, f: Formula, bound: dict[str, str]) -> Formula:
        match f:
            case Pred(name, args):
                return Pred(self.name(name), tuple(bound.get(a) or self._const(a) for a in args))
            case Not(g):
                return Not(self._rename(g, bound))
            case Ante(g):
                g = self._rename(g, bound)
                assert isinstance(g, Pred | Not)
                return Ante(g)
            case Quant(q, var, body):
                return Quant(q, var, self._rename(body, {**bound, var: f"?{var}"}))
            case Bin(op, a, b):
                return Bin(op, self._rename(a, bound), self._rename(b, bound))

    def _const(self, raw: str) -> str:
        c = self.name(raw)
        if is_variable(c):
            raise Unsupported(f"{raw} isn't bound by a quantifier")
        return c

    def _skolem(
        self, f: Formula, env: dict[str, str], universals: list[str], conclusion: bool
    ) -> Formula:
        match f:
            case Pred(name, args):
                return Pred(name, tuple(env.get(a, a) for a in args))
            case Not(g):
                return Not(self._skolem(g, env, universals, conclusion))
            case Ante(g):
                g = self._skolem(g, env, universals, conclusion)
                assert isinstance(g, Pred | Not)
                return Ante(g)
            case Quant("some", var, body):
                if conclusion:
                    raise Unsupported('"some" in a conclusion')
                if universals:
                    raise Unsupported('"some" inside "every"')
                c = self.fresh(_hint(body, f"?{var}"))
                return self._skolem(body, {**env, f"?{var}": c}, universals, conclusion)
            case Quant(_, var, body):
                v = f"?{len(universals)}"
                return self._skolem(body, {**env, f"?{var}": v}, [*universals, v], conclusion)
            case Bin(op, a, b):
                return Bin(
                    op,
                    self._skolem(a, env, universals, conclusion),
                    self._skolem(b, env, universals, conclusion),
                )


def _hint(f: Formula, var: str) -> str:
    """The first one-place property of `var` in `f`, to name a "some" individual by."""
    match f:
        case Pred(name, (arg,)) if arg == var:
            return name
        case Pred():
            return ""
        case Not(g) | Ante(g):
            return _hint(g, var)
        case Quant(_, _, body):
            return _hint(body, var)
        case Bin(_, a, b):
            return _hint(a, var) or _hint(b, var)


def _cnf(f: Formula) -> list[list[Literal_]]:
    match f:
        case Pred():
            return [[(True, f, False)]]
        case Not(Pred() as a):
            return [[(False, a, False)]]
        case Ante(Pred() as a):
            return [[(True, a, True)]]
        case Ante(Not(Pred() as a)):
            return [[(False, a, True)]]
        case Bin("&", a, b):
            return _cnf(a) + _cnf(b)
        case Bin("|", a, b):
            left, right = _cnf(a), _cnf(b)
            if len(left) * len(right) > 64:
                raise Unsupported("too large")
            return [x + y for x, y in itertools.product(left, right)]
        case _:
            raise Unsupported("not in negation normal form")


def _clause_text(clause: Sequence[Literal_]) -> str | None:
    """A clause as a rule. Literals from an "if" part go in the body; when there are none,
    the negative literals do: `not a | not b | c` is `a & b -> c`. None for a clause that
    always holds."""
    lits: list[Literal_] = []
    for item in clause:
        if all(item[:2] != seen[:2] for seen in lits):
            lits.append(item)
    if any((not pos, a) in {(p, b) for p, b, _ in lits} for pos, a, _ in lits):
        return None
    names: dict[str, str] = {}
    for _, a, _ in lits:
        for arg in a.args:
            if arg.startswith("?") and arg not in names:
                if len(names) == len(_VARS):
                    raise Unsupported("too many variables")
                names[arg] = _VARS[len(names)]

    def lit(pos: bool, a: Pred) -> str:
        args = [names.get(x, x) for x in a.args]
        text = f"{a.name}({', '.join(args)})" if args else a.name
        return text if pos else f"not {text}"

    # The body holds what the clause's literal is the negation of.
    if any(ante for _, _, ante in lits):
        body = [lit(not pos, a) for pos, a, ante in lits if ante]
        head = [lit(pos, a) for pos, a, ante in lits if not ante]
    else:
        body = [lit(True, a) for pos, a, _ in lits if not pos]
        head = [lit(True, a) for pos, a, _ in lits if pos]
    if not head:
        last = body.pop()
        head = [last[4:] if last.startswith("not ") else f"not {last}"]
    return f"{' & '.join(body)} -> {' | '.join(head)}" if body else " | ".join(head)
