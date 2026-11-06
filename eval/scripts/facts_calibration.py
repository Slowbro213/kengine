"""Replay the M3 checkers over the claims of earlier hotpotqa kengine sessions (no LLM) and
count verdicts, to check the thresholds. Usage:
    uv run python eval/scripts/facts_calibration.py <results dir> <out.json>"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from kengine.beliefs.records import BeliefRecord
from kengine.beliefs.status import Status
from kengine.checkers.facts import FactChecker
from kengine.checkers.inference_nli import InferenceEntailmentChecker
from kengine.claims.schema import Claim
from kengine.config import load_config
from kengine.retrieval.index import Corpus
from kengine.retrieval.models import load_models


class Replay:
    def __init__(self, claims: dict[str, Claim]) -> None:
        self._r = {
            i: BeliefRecord(claim=c, status=Status.VERIFIED, status_reason="")
            for i, c in claims.items()
        }

    def get(self, id: str) -> BeliefRecord:
        return self._r[id]

    def exists(self, id: str) -> bool:
        return id in self._r

    def all(self) -> list[BeliefRecord]:
        return list(self._r.values())

    def premise_closure(self, ids: list[str]) -> list[str]:
        return list(ids)


def main(results: Path, out: Path) -> None:
    cfg = load_config([])
    models = load_models(cfg.models_dir)
    counts: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for events_path in sorted(results.glob("sessions/*/events.jsonl")):
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        docs = events[0]["task"]["documents"]
        claims = {
            e["claim"]["id"]: Claim.model_validate(e["claim"])
            for e in events
            if e.get("kind") == "belief_added"
        }
        corpus = Corpus.from_documents(docs, models.embedder)
        facts = FactChecker(corpus, models.entailment, cfg.facts)
        infer = InferenceEntailmentChecker(models.entailment, cfg.facts)
        view = Replay(claims)
        for c in claims.values():
            for checker in (facts, infer):
                if checker.applies_to(c):
                    v = checker.check(c, view)
                    key = f"{checker.id}:{v.outcome}:{v.status}"
                    counts[key] += 1
                    rows.append(
                        {
                            "session": events_path.parent.name,
                            "claim": c.text,
                            "verdict": key,
                            "why": v.explanation,
                            "evidence": v.evidence,
                        }
                    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"counts": dict(counts), "rows": rows}, indent=1))
    print(json.dumps(dict(counts), indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
