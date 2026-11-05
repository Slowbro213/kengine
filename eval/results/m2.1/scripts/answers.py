import json
import re
import sys
from collections import Counter
from pathlib import Path

# For each run: whether passed and blocked answers were right, and why the last gate blocked.
# Session dirs are found under the run's own sessions/, since some runs were moved after they ran.


def reason(text: str) -> str:
    text = re.sub(r"\bb\d+\b", "bN", text)
    if "is unchecked" in text:
        return "rests on an unchecked claim"
    if "doesn't show the conclusion" in text or "doesn't rule the conclusion out" in text:
        return "what it rests on doesn't prove the answer"
    if "isn't open" in text:
        return "answered Uncertain, but the givens settle it"
    if "has no formal form" in text or "have no formal form" in text:
        return "answered Uncertain, but a given has no formal form"
    if "not true" in text or "isn't false" in text:
        return "the givens prove the opposite"
    return text[:60]


for run in map(Path, sys.argv[1:]):
    answers: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for line in open(run / "items.jsonl"):
        item = json.loads(line)
        events_file = run / "sessions" / Path(item["session_dir"]).name / "events.jsonl"
        events = [json.loads(x) for x in events_file.read_text().splitlines()]
        result = [e for e in events if e["kind"] == "run_finished"][-1]["result"]
        if item["status"] == "answered":
            answers["passed, right" if item["correct"] else "passed, wrong"] += 1
            continue
        blocked = result.get("blocked_answer")
        if blocked is None:
            answers["blocked, no answer"] += 1
        else:
            right = blocked.strip().lower() == str(item["gold"]).lower()
            answers["blocked, right" if right else "blocked, wrong"] += 1
        gates = [e for e in events if e["kind"] == "gate"]
        last = gates[-1]["decision"]["reasons"] if gates else ["no final gate"]
        for r in {reason(x) for x in last}:
            reasons[r] += 1
    print(run.name)
    print("  answers:", dict(answers))
    print("  blocked items by last-gate reason:", reasons.most_common())
