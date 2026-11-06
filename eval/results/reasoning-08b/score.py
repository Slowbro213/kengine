"""Score generated chains with the real checker: score.py GEN_JSONL..."""

import json
import sys
from collections import Counter

from kengine.claims.extraction import Extraction, to_formal
from kengine.eval import suites
from kengine.finetune.chains import Step, check_chain
from kengine.finetune.data import convert_row

rows = {f"folio-{r['example_id']}": r for r in suites.load_rows("tasksource/folio", None, "train")}
for path in sys.argv[1:]:
    c = Counter()
    for line in open(path):
        g = json.loads(line)
        forms = convert_row(rows[g["item_id"]])
        try:
            ext = Extraction.model_validate_json(g["extract"])
        except Exception:
            c["extract not valid JSON"] += 1
            continue
        if ext.action.kind != "final_answer" or ext.action.answer is None:
            c["no final answer"] += 1
            continue
        answer = ext.action.answer.strip()
        refs = {cl.ref: str(i) for i, cl in enumerate(ext.claims, 1)}
        try:
            steps = [
                Step(
                    to_formal(cl.formal).statements,
                    tuple(refs.get(p, p) for p in cl.premises),
                    cl.formal.mode,
                )
                for cl in ext.claims
                if cl.formal is not None
            ]
        except Exception:
            c["bad formal"] += 1
            continue
        if (
            not steps
            or len(steps) != len(ext.claims)
            or ext.action.preconditions != [ext.claims[-1].ref]
        ):
            c["not a plain chain"] += 1
            continue
        try:
            why = check_chain(forms[:-1], forms[-1], answer, steps)
        except Exception as e:
            c[f"check error {type(e).__name__}"] += 1
            continue
        if why is None:
            c["accepted, right" if answer == g["label"] else "accepted, WRONG"] += 1
        else:
            c["refused"] += 1
    print(path, sum(c.values()), dict(c.most_common()))
