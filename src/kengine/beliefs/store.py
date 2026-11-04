"""The belief store: a small truth-maintenance system on sqlite (spec §5.4)."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import TERMINAL, Status
from kengine.checkers.base import CheckerKind, Verdict
from kengine.claims.schema import Claim, ClaimType, Computation, Formal, Source

_SCHEMA = """
CREATE TABLE IF NOT EXISTS beliefs (
    id TEXT PRIMARY KEY, seq INTEGER NOT NULL UNIQUE,
    claim_json TEXT NOT NULL, status TEXT NOT NULL, reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS changes (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, belief_id TEXT NOT NULL,
    old TEXT NOT NULL, new TEXT NOT NULL, reason TEXT NOT NULL,
    verdict_json TEXT, step INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, belief_id TEXT NOT NULL,
    verdict_json TEXT NOT NULL, step INTEGER NOT NULL
);
"""

# Only checkers that can show a claim true may mark it verified or supported.
_MAY_VERIFY = frozenset({CheckerKind.DETERMINISTIC, CheckerKind.EVIDENCE})


class StatusChange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    belief_id: str
    old: Status
    new: Status
    reason: str
    verdict: Verdict | None
    step: int


class UnknownPremiseError(ValueError):
    pass


class VerificationGuardError(ValueError):
    pass


class BeliefStore:
    """Beliefs, their statuses and every status change, written through to sqlite.

    Every mutating call commits, so a session survives a crash.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._records: dict[str, BeliefRecord] = {}
        self._children: dict[str, list[str]] = {}
        for claim_json, status, reason in conn.execute(
            "SELECT claim_json, status, reason FROM beliefs ORDER BY seq"
        ):
            claim = Claim.model_validate_json(claim_json)
            self._remember(BeliefRecord(claim=claim, status=Status(status), status_reason=reason))

    @classmethod
    def open(cls, path: Path | Literal[":memory:"]) -> BeliefStore:
        conn = sqlite3.connect(str(path))
        conn.executescript(_SCHEMA)
        conn.commit()
        return cls(conn)

    def close(self) -> None:
        self._conn.close()

    # BeliefView

    def get(self, id: str) -> BeliefRecord:
        try:
            return self._records[id]
        except KeyError:
            raise KeyError(f"no belief {id}") from None

    def exists(self, id: str) -> bool:
        return id in self._records

    def all(self) -> list[BeliefRecord]:
        return list(self._records.values())

    def premise_closure(self, ids: Sequence[str]) -> list[str]:
        """`ids` and all their transitive premises, in insertion order."""
        seen: set[str] = set()
        todo = list(ids)
        while todo:
            current = todo.pop()
            if current not in seen:
                seen.add(current)
                todo.extend(self.get(current).claim.premises)
        return [i for i in self._records if i in seen]

    # Mutations

    def add(
        self,
        *,
        type: ClaimType,
        text: str,
        premises: Sequence[str],
        source: Source,
        step: int,
        computation: Computation | None = None,
        confidence: float | None = None,
        formal: Formal | None = None,
    ) -> Claim:
        missing = [p for p in premises if p not in self._records]
        if missing:
            raise UnknownPremiseError(f"unknown premises: {', '.join(missing)}")
        claim = Claim(
            id=f"b{len(self._records) + 1}",
            type=type,
            text=text,
            premises=tuple(premises),
            computation=computation,
            formal=formal,
            confidence=confidence,
            source=source,
            step=step,
        )
        status = Status.ASSUMED if type == ClaimType.ASSUMPTION else Status.UNCHECKED
        record = BeliefRecord(claim=claim, status=status, status_reason="")
        self._conn.execute(
            "INSERT INTO beliefs (id, seq, claim_json, status, reason) VALUES (?, ?, ?, ?, ?)",
            (claim.id, len(self._records), claim.model_dump_json(), status, ""),
        )
        self._conn.commit()
        self._remember(record)
        return claim

    def apply_verdict(self, id: str, verdict: Verdict, step: int) -> list[StatusChange]:
        """Record `verdict` and apply its status, propagating terminal statuses."""
        record = self.get(id)
        new = verdict.status
        if new in (Status.VERIFIED, Status.SUPPORTED) and verdict.checker_kind not in _MAY_VERIFY:
            raise VerificationGuardError(
                f"{verdict.checker_kind} checker {verdict.checker_id} can't set {new}"
            )
        self._conn.execute(
            "INSERT INTO verdicts (belief_id, verdict_json, step) VALUES (?, ?, ?)",
            (id, verdict.model_dump_json(), step),
        )
        if new is None or record.status in TERMINAL or new == record.status:
            self._conn.commit()
            return []
        return self._change(id, new, verdict.explanation, verdict, step)

    def set_formal(self, id: str, formal: Formal) -> Claim:
        """Replace a given's formal form with a new translation of the task's sentence.
        Verdicts already given against the old form stand; the event log has both."""
        record = self.get(id)
        if record.claim.type != ClaimType.GIVEN:
            raise ValueError(f"only a given's formal form can be replaced, not {id}'s")
        claim = record.claim.model_copy(update={"formal": formal})
        self._conn.execute(
            "UPDATE beliefs SET claim_json = ? WHERE id = ?", (claim.model_dump_json(), id)
        )
        self._conn.commit()
        self._records[id] = record.model_copy(update={"claim": claim})
        return claim

    def retract(self, id: str, reason: str, step: int) -> list[StatusChange]:
        if self.get(id).status in TERMINAL:
            return []
        return self._change(id, Status.RETRACTED, reason, None, step)

    def dependents(self, id: str) -> list[str]:
        """Every belief that rests on `id`, directly or transitively, in insertion order."""
        seen: set[str] = set()
        todo = list(self._children.get(id, []))
        while todo:
            current = todo.pop()
            if current not in seen:
                seen.add(current)
                todo.extend(self._children.get(current, []))
        return [i for i in self._records if i in seen]

    def history(self, id: str) -> list[StatusChange]:
        rows = self._conn.execute(
            "SELECT old, new, reason, verdict_json, step FROM changes"
            " WHERE belief_id = ? ORDER BY seq",
            (id,),
        )
        return [
            StatusChange(
                belief_id=id,
                old=Status(old),
                new=Status(new),
                reason=reason,
                verdict=Verdict.model_validate_json(vj) if vj is not None else None,
                step=step,
            )
            for old, new, reason, vj, step in rows
        ]

    def verdicts(self, id: str) -> list[Verdict]:
        """Every verdict recorded for `id`, including ones that changed nothing."""
        rows = self._conn.execute(
            "SELECT verdict_json FROM verdicts WHERE belief_id = ? ORDER BY seq", (id,)
        )
        return [Verdict.model_validate_json(vj) for (vj,) in rows]

    # Internals

    def _remember(self, record: BeliefRecord) -> None:
        self._records[record.claim.id] = record
        for premise in record.claim.premises:
            self._children.setdefault(premise, []).append(record.claim.id)

    def _change(
        self, id: str, new: Status, reason: str, verdict: Verdict | None, step: int
    ) -> list[StatusChange]:
        changes = [self._set(id, new, reason, verdict, step)]
        if new in TERMINAL:
            because = f"depends on {id} ({new})"
            for dep in self.dependents(id):
                if self._records[dep].status not in TERMINAL:
                    changes.append(self._set(dep, Status.INVALIDATED, because, None, step))
        self._conn.commit()
        return changes

    def _set(
        self, id: str, new: Status, reason: str, verdict: Verdict | None, step: int
    ) -> StatusChange:
        old = self._records[id]
        change = StatusChange(
            belief_id=id, old=old.status, new=new, reason=reason, verdict=verdict, step=step
        )
        self._records[id] = old.model_copy(update={"status": new, "status_reason": reason})
        self._conn.execute(
            "UPDATE beliefs SET status = ?, reason = ? WHERE id = ?", (new, reason, id)
        )
        self._conn.execute(
            "INSERT INTO changes (belief_id, old, new, reason, verdict_json, step)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                id,
                old.status,
                new,
                reason,
                verdict.model_dump_json() if verdict is not None else None,
                step,
            ),
        )
        return change
