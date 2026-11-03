"""Facts that restate the task's numbers (spec §5.5 #5, with the task as the only corpus).

A `fact` such as "each pen costs $1.25 (from b1)", where b1 is a `given` from the task,
is supported when every number it states appears in what it cites: givens, or verified
computations. With no premises it may instead be backed by a single given that has all
its numbers and shares a word with it. A standard unit conversion such as 12 months in a
year counts too, when the fact names one unit and the fact or what it cites names the
other. It's a lexical check, so it reaches `supported`, never `verified`. Facts without
numbers are left to the fact checker (M3).
"""

from __future__ import annotations

import re
from fractions import Fraction

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import Status
from kengine.checkers.arith_eval import format_number
from kengine.checkers.base import BeliefView, CheckerKind, Outcome, Verdict
from kengine.claims.schema import Claim, ClaimType
from kengine.numbers import numbers_in

_WORD_RE = re.compile(r"[a-z]{3,}")
_STOPWORDS = frozenset(
    "the and for are was were has have had that this with from each per there their they "
    "she her his him its into than then what which who how much many does did is".split()
)


# (factor, larger unit, smaller unit): 1 larger unit is `factor` smaller ones. Each unit
# is matched at the start of a word, so "cent" doesn't match "percent".
_CONVERSIONS: tuple[tuple[int, str, str], ...] = (
    (12, "year", "month"),
    (52, "year", "week"),
    (365, "year", "day"),
    (7, "week", "day"),
    (24, "day", "hour"),
    (60, "hour", "minute"),
    (60, "minute", "second"),
    (100, "dollar", "cent"),
    (12, "foot", "inch"),
    (12, "feet", "inch"),
    (3, "yard", "feet"),
    (3, "yard", "foot"),
    (16, "pound", "ounce"),
    (1000, "kilogram", "gram"),
    (1000, "kilometer", "meter"),
    (100, "meter", "centimeter"),
    (1000, "liter", "milliliter"),
    (12, "dozen", "dozen"),
)


def _names(unit: str, text: str) -> bool:
    return re.search(rf"\b{unit}", text) is not None


def _conversions(claim_text: str, pool: str) -> dict[Fraction, str]:
    """Conversion factors the claim may use: it names one unit and `pool` names the other."""
    claim_text = claim_text.lower()
    pool = f"{claim_text} {pool.lower()}"
    found: dict[Fraction, str] = {}
    for factor, big, small in _CONVERSIONS:
        named = _names(big, claim_text) or _names(small, claim_text)
        if named and _names(big, pool) and _names(small, pool):
            label = f"1 {big} = {factor}" if big == small else f"1 {big} = {factor} {small}s"
            found.setdefault(Fraction(factor), label)
    return found


def _stems(text: str) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    return {w.removesuffix("s") for w in words if w not in _STOPWORDS}


# Ordinals in what a fact cites, so "Week 1" can restate "the first week".
_ORDINALS = {
    w: Fraction(i)
    for i, w in enumerate(
        "first second third fourth fifth sixth seventh eighth ninth tenth".split(), 1
    )
}
_ORDINAL_RE = re.compile(r"\b(" + "|".join(_ORDINALS) + r")\b", re.IGNORECASE)
# Claim ids such as `b3` or `c2`, which a claim's text may cite; their digits aren't numbers.
_ID_RE = re.compile(r"\b[bc]\d+\b")


def _stated(text: str) -> set[Fraction]:
    return numbers_in(_ID_RE.sub(" ", text))


def _numbers(rec: BeliefRecord) -> set[Fraction]:
    claim = rec.claim
    found = _stated(claim.text)
    found |= {_ORDINALS[m.group(1).lower()] for m in _ORDINAL_RE.finditer(claim.text)}
    if claim.computation is not None:
        found |= numbers_in(claim.computation.expression)
        found |= numbers_in(claim.computation.claimed_result)
    return found


def _citable(rec: BeliefRecord) -> bool:
    return rec.claim.type == ClaimType.GIVEN or (
        rec.claim.type == ClaimType.COMPUTATION and rec.status == Status.VERIFIED
    )


class RestatementChecker:
    id = "restatement"
    kind = CheckerKind.EVIDENCE

    def applies_to(self, claim: Claim) -> bool:
        return claim.type == ClaimType.FACT and bool(_stated(claim.text))

    def check(self, claim: Claim, view: BeliefView) -> Verdict:
        wanted = _stated(claim.text)
        if claim.premises:
            cited = [r for r in (view.get(p) for p in claim.premises) if _citable(r)]
            have: set[Fraction] = set()
            for rec in cited:
                have |= _numbers(rec)
            evidence = [r.claim.id for r in cited]
            units = _conversions(claim.text, " ".join(r.claim.text for r in cited))
        else:
            givens = [r for r in view.all() if r.claim.type == ClaimType.GIVEN]
            words = _stems(claim.text)
            match, units = None, {}
            for g in givens:
                g_units = _conversions(claim.text, g.claim.text)
                if wanted <= _numbers(g) | g_units.keys() and words & _stems(g.claim.text):
                    match, units = g, g_units
                    break
            have = _numbers(match) if match is not None else set()
            evidence = [match.claim.id] if match is not None else []
        used = sorted({units[n] for n in wanted - have if n in units})
        missing = sorted(wanted - have - units.keys())
        if not missing and evidence:
            return self._verdict(
                Outcome.PASS,
                Status.SUPPORTED,
                f"restates {', '.join(evidence)}" + (f", with {'; '.join(used)}" if used else ""),
                evidence,
            )
        shown = ", ".join(format_number(n) for n in missing)
        return self._verdict(
            Outcome.UNKNOWN,
            None,
            f"{shown} isn't in the task's givens or verified calculations it cites. If a given "
            "states it, cite that given. If you worked it out, even by counting, write the "
            f"calculation as its own item (`<expression> = {format_number(missing[0])}`) and "
            "build on that item instead of this fact",
            evidence,
        )

    def _verdict(
        self, outcome: Outcome, status: Status | None, explanation: str, evidence: list[str]
    ) -> Verdict:
        return Verdict(
            checker_id=self.id,
            checker_kind=self.kind,
            outcome=outcome,
            status=status,
            explanation=explanation,
            evidence=evidence,
        )
