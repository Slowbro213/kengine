"""Cross-sample voting: keep the claims most extraction samples agree on (spec §5.3)."""

from __future__ import annotations

import difflib
from collections import Counter
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from kengine.claims.extraction import ExtractedClaim, Extraction, ProposedAction
from kengine.eval.answers import normalize_text


class VoteResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    claims: list[ExtractedClaim]
    action: ProposedAction | None
    dropped: list[str]
    n_valid: int


def default_k(n: int) -> int:
    """A strict majority of `n` samples."""
    return n // 2 + 1


def vote(samples: Sequence[Extraction], k: int, similarity: float = 0.8) -> VoteResult:
    """Keep the anchor sample's claims that at least `k` samples contain.

    The anchor is the sample with the most such claims (ties go to the earliest). Claims
    match when they have the same type and their normalized texts are at least
    `similarity` alike.
    """
    if not samples:
        return VoteResult(claims=[], action=None, dropped=[], n_valid=0)

    def matches(a: ExtractedClaim, b: ExtractedClaim) -> bool:
        if a.type != b.type:
            return False
        ratio = difflib.SequenceMatcher(None, normalize_text(a.text), normalize_text(b.text))
        return ratio.ratio() >= similarity

    def support(claim: ExtractedClaim) -> int:
        return sum(any(matches(claim, other) for other in s.claims) for s in samples)

    supported = [[support(x) >= k for x in s.claims] for s in samples]
    anchor_i = max(range(len(samples)), key=lambda i: (sum(supported[i]), -i))
    anchor = samples[anchor_i]
    kept_raw = [x for x, ok in zip(anchor.claims, supported[anchor_i], strict=True) if ok]
    dropped = [x.ref for x, ok in zip(anchor.claims, supported[anchor_i], strict=True) if not ok]
    gone = set(dropped)

    claims = [
        x.model_copy(update={"premises": [p for p in x.premises if p not in gone]})
        for x in kept_raw
    ]
    return VoteResult(
        claims=claims,
        action=_vote_action(samples, anchor_i, kept_raw, gone, matches),
        dropped=dropped,
        n_valid=len(samples),
    )


def _vote_action(
    samples: Sequence[Extraction],
    anchor_i: int,
    kept: list[ExtractedClaim],
    gone: set[str],
    matches: Callable[[ExtractedClaim, ExtractedClaim], bool],
) -> ProposedAction:
    counts = Counter(s.action.kind for s in samples)
    anchor_kind = samples[anchor_i].action.kind
    best = max(counts.values())
    kind = anchor_kind if counts[anchor_kind] == best else _first_with(samples, counts, best)
    if kind == anchor_kind:
        action = samples[anchor_i].action
        preconditions = [p for p in action.preconditions if p not in gone]
    else:
        source = next(s for s in samples if s.action.kind == kind)
        action = source.action
        # The source sample numbers its claims its own way: map each `c#` to the kept
        # anchor claim it matches, and drop it if there's none.
        by_ref = {x.ref: x for x in source.claims}
        preconditions: list[str] = []
        for p in action.preconditions:
            if not p.startswith("c"):
                preconditions.append(p)
                continue
            claim = by_ref.get(p)
            match = next((x for x in kept if claim is not None and matches(claim, x)), None)
            if match is not None:
                preconditions.append(match.ref)
    return action.model_copy(update={"preconditions": preconditions})


def _first_with(samples: Sequence[Extraction], counts: Counter[str], best: int) -> str:
    return next(s.action.kind for s in samples if counts[s.action.kind] == best)
