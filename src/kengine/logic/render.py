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
