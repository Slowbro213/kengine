"""A session directory: the belief store plus the event log."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kengine.audit.events import EventLog
from kengine.beliefs.store import BeliefStore


@dataclass
class Session:
    dir: Path
    store: BeliefStore
    log: EventLog

    @classmethod
    def create(cls, root: Path) -> Session:
        name = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
        path = root / name
        path.mkdir(parents=True)
        return cls.open(path)

    @classmethod
    def open(cls, dir: Path) -> Session:
        return cls(
            dir=dir,
            store=BeliefStore.open(dir / "beliefs.sqlite"),
            log=EventLog(dir / "events.jsonl"),
        )

    def close(self) -> None:
        self.store.close()
