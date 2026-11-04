import json
import sys
from pathlib import Path

# Print a compact audit trail of one item: givens, claims with their verdicts, gates, the result.
run, item_id = Path(sys.argv[1]), sys.argv[2]
it = next(json.loads(x) for x in open(run / "items.jsonl") if json.loads(x)["item_id"] == item_id)
print(item_id, "gold", it["gold"], "status", it["status"], "answer", it["answer"])
for line in (Path(it["session_dir"]) / "events.jsonl").read_text().splitlines():
    e = json.loads(line)
    k = e["kind"]
    if k == "session_started":
        print("TASK", e["task"]["goal"][-300:].replace("\n", " / "))
    elif k == "belief_added":
        c = e["claim"]
        f = c.get("formal") or {}
        form = f"{f.get('mode', '')} {f.get('statements', '-')}"
        print(f"  {c['id']} [{c['type']}] {c['text'][:110]} | prem {c['premises']} | {form}")
    elif k == "verdict" and e["verdict"]["checker_id"] in ("logic", "restatement"):
        v = e["verdict"]
        result = f"{v['checker_id']} {v['outcome']}/{v['status']}"
        print(f"    verdict {e['belief_id']} {result}: {v['explanation'][:150]}")
    elif (
        k == "status_changed"
        and e["change"]["verdict"]
        and e["change"]["verdict"]["checker_id"] == "logic"
    ):
        ch = e["change"]
        print(f"    status {ch['belief_id']} {ch['old']}->{ch['new']}: {ch['reason'][:120]}")
    elif k == "gate":
        d = e["decision"]
        print(f"  GATE step {e['step']} allowed={d['allowed']} blocking={d['blocking']}")
    elif k == "run_finished":
        r = e["result"]
        print(
            "RESULT",
            r["status"],
            r["answer"],
            "blocked_answer",
            r.get("blocked_answer"),
            "steps",
            r["steps"],
        )
