import json
import sys
from collections import Counter
from pathlib import Path

# Why kengine's blocked folio answers were blocked, and whether they'd have been right.
# For each blocking belief at the last gate: the cause, from its type, formal form and
# logic verdicts.
NORM = {"true": "True", "false": "False", "uncertain": "Uncertain"}


def cause(b: dict, verdicts: list[dict], beliefs: dict) -> str:
    if not [v for v in verdicts if v]:
        return "never checked"
    if b.get("formal") is None:
        return "no formal form"
    logic = [v for v in verdicts if v and v["checker_id"] == "logic"]
    if not logic:
        return f"no logic verdict ({b['type']})"
    last = logic[-1]
    if last["outcome"] == "fail":
        return "contradicted" if last["status"] == "contradicted" else "logic: doesn't follow"
    if last["outcome"] == "unknown":
        ex = last["explanation"]
        if "has no formal form" in ex:
            return "logic: a premise isn't formalized"
        if "contradict each other" in ex:
            return "logic: the cited claims contradict each other"
        if "don't settle" in ex:
            return "logic: the givens don't settle it"
        return "logic: unknown (" + ex[:40] + ")"
    return f"logic {last['outcome']}/{last['status']}"


for run in sys.argv[1:]:
    gold = {json.loads(x)["item_id"]: json.loads(x) for x in open(Path(run) / "items.jsonl")}
    causes: Counter[str] = Counter()
    per_item: Counter[str] = Counter()
    would = Counter()
    giv = Counter()
    for item in gold.values():
        ev = Path(item["session_dir"]) / "events.jsonl"
        events = [json.loads(x) for x in ev.read_text().splitlines()]
        beliefs, verdicts, gates = {}, {}, []
        for e in events:
            if e["kind"] == "belief_added":
                beliefs[e["claim"]["id"]] = e["claim"]
            elif e["kind"] == "verdict":
                verdicts.setdefault(e["belief_id"], []).append(e["verdict"])
            elif e["kind"] == "status_changed":
                verdicts.setdefault(e["change"]["belief_id"], []).append(e["change"]["verdict"])
            elif e["kind"] == "gate":
                gates.append(e["decision"])
            elif e["kind"] == "formalize":
                giv["formalize_ok" if not e.get("errors") else "formalize_with_errors"] += 1
        givens = [b for b in beliefs.values() if b["type"] == "given"]
        giv["givens"] += len(givens)
        giv["givens_formal"] += sum(b.get("formal") is not None for b in givens)
        fin = [e for e in events if e["kind"] == "run_finished"][-1]["result"]
        if item["status"] == "answered":
            continue
        ba = fin.get("blocked_answer")
        key = (
            "no answer"
            if ba is None
            else ("right" if NORM.get(str(ba).strip().lower()) == item["gold"] else "wrong")
        )
        would[key] += 1
        if not gates or gates[-1]["allowed"]:
            per_item["no final gate"] += 1
            continue
        cs = set()
        for bid in gates[-1]["blocking"]:
            c = cause(beliefs[bid], verdicts.get(bid, []), beliefs)
            causes[c] += 1
            cs.add(c)
        for c in cs:
            per_item[c] += 1
    print(Path(run).name)
    print("  blocked answers if allowed:", dict(would))
    print("  givens:", dict(giv))
    print("  blocking beliefs by cause:", causes.most_common())
    print("  blocked items with each cause:", per_item.most_common())
