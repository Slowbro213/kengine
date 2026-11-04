import json
import sys
from collections import Counter
from pathlib import Path

# Count `logic` verdicts by outcome and status over every session of the given run dirs.
for run in sys.argv[1:]:
    counts: Counter[tuple[str, str | None]] = Counter()
    for events in Path(run).glob("sessions/*/events.jsonl"):
        for line in events.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e["kind"] == "verdict" and e["verdict"]["checker_id"] == "logic":
                counts[(e["verdict"]["outcome"], e["verdict"]["status"])] += 1
    print(Path(run).name, dict(sorted(counts.items(), key=str)))
