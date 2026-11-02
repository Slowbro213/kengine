"""Short, specific feedback for the next THINK step (spec §5.7)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from kengine.beliefs.status import Status
from kengine.beliefs.store import StatusChange
from kengine.checkers.base import Outcome, Verdict
from kengine.loop.gate import GateDecision

# Reasons the store and the structure checker give for an invalidation.
_ROOT_RES = (
    re.compile(r"^depends on (b\d+) \((\w+)\)$"),
    re.compile(r"^rests on (b\d+), which is (\w+)$"),
)


def build_feedback(
    changes: Sequence[StatusChange],
    verdicts: Sequence[tuple[str, Verdict]],
    issues: Sequence[str],
    gate: GateDecision | None,
    max_items: int = 5,
) -> list[str]:
    """Most important first: gate blocks, contradictions, invalidations, failed and
    inconclusive checks, then other issues."""
    items: list[str] = []
    if gate is not None and not gate.allowed:
        items += [f"final answer blocked: {reason}" for reason in gate.reasons]

    for c in changes:
        if c.new == Status.CONTRADICTED:
            explanation = c.verdict.explanation if c.verdict is not None else c.reason
            items.append(f"{c.belief_id} contradicted: {explanation}")

    roots: dict[str, int] = {}
    unexplained: list[str] = []
    for c in changes:
        if c.new == Status.INVALIDATED:
            root = _root(c.reason)
            if root is None:
                unexplained.append(f"{c.belief_id} invalidated: {c.reason}")
            else:
                roots[root] = roots.get(root, 0) + 1
    items += [f"{n} claim(s) invalidated because {root}" for root, n in roots.items()]
    items += unexplained

    changed = [(c.belief_id, c.verdict) for c in changes if c.verdict is not None]
    items += [
        f"{bid}: {v.explanation}"
        for bid, v in verdicts
        if v.outcome == Outcome.FAIL and (bid, v) not in changed
    ]
    items += [f"{bid}: {v.explanation}" for bid, v in verdicts if v.outcome == Outcome.UNKNOWN]
    items += issues
    return items[:max_items]


def _root(reason: str) -> str | None:
    for pattern in _ROOT_RES:
        m = pattern.match(reason)
        if m:
            return f"{m.group(1)} is {m.group(2)}"
    return None
