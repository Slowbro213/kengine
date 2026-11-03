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

        def acc(k: int) -> str:
            return f"{k / self.n:.3f}" if self.n else "n/a"

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
