# pyright: basic, reportAttributeAccessIssue=false, reportReturnType=false
# pyright: reportOperatorIssue=false, reportArgumentType=false
# z3's bundled annotations are too loose to check against (every expression is a union of
# all its sorts); this is the only module that imports it.
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
    return [
        dict(zip(vs, combo, strict=True)) for combo in itertools.product(domain, repeat=len(vs))
    ]


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
