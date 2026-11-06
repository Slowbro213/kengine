"""Count M3 verdicts in kengine sessions. Usage:
uv run python eval/scripts/count_verdicts.py <results root>"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

CHECKERS = ("facts", "consistency", "inference-nli")


def main(root: Path) -> None:
    for run in sorted(p for p in root.iterdir() if (p / "sessions").is_dir()):
        counts: Counter[str] = Counter()
        fails: list[str] = []
        for events in run.glob("sessions/*/events.jsonl"):
            for line in events.read_text().splitlines():
                e = json.loads(line)
                v = e.get("verdict")
                if e.get("kind") != "verdict" or v is None or v["checker_id"] not in CHECKERS:
                    continue
                counts[f"{v['checker_id']} {v['outcome']} {v['status']}"] += 1
                if v["checker_id"] == "consistency" and v["outcome"] == "fail":
                    fails.append(f"{events.parent.name} {e['belief_id']}: {v['explanation']}")
        print(f"## {run.name}")
        for k, n in sorted(counts.items()):
            print(f"  {k}: {n}")
        for f in fails:
            print(f"  CONSISTENCY FAIL {f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
