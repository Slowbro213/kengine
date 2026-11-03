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
                value = (
                    _scale(other, value.const) if not value.terms else _scale(value, other.const)
                )
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
            return Linear(
                Fraction(0), ((Atom(text, _args(self.toks[start + 1 : end])), Fraction(1)),)
            )
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
