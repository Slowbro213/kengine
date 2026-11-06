"""The corpus index (spec §5.5): SQLite FTS5 for keywords plus embeddings, combined by
reciprocal-rank fusion. In memory: a task's corpus lives as long as the task."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence

import numpy as np

from kengine.retrieval.models import Embedder, Vectors
from kengine.retrieval.passages import Passage, split_documents

_WORD_RE = re.compile(r"\w+")
_RRF_K = 60
_POOL = 20


class Corpus:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._passages: list[Passage] = []
        self._vectors: Vectors = np.zeros((0, 0), np.float32)
        self._db = sqlite3.connect(":memory:")
        self._db.execute("CREATE VIRTUAL TABLE p USING fts5(t)")

    @classmethod
    def from_documents(cls, docs: Sequence[str], embedder: Embedder) -> Corpus:
        corpus = cls(embedder)
        corpus.add(split_documents(docs))
        return corpus

    def __len__(self) -> int:
        return len(self._passages)

    def add(self, passages: Sequence[Passage]) -> None:
        if not passages:
            return
        start = len(self._passages)
        self._passages += passages
        self._db.executemany(
            "INSERT INTO p(rowid, t) VALUES (?, ?)",
            [(start + i, p.text) for i, p in enumerate(passages)],
        )
        new = self._embedder.embed([p.text for p in passages])
        self._vectors = new if start == 0 else np.vstack([self._vectors, new])

    def search(self, query: str, k: int = 8) -> list[Passage]:
        if not self._passages:
            return []
        score: dict[int, float] = {}
        for rank, row in enumerate(self._keyword(query)):
            score[row] = score.get(row, 0.0) + 1 / (_RRF_K + rank)
        q = self._embedder.embed([query], query=True)[0]
        for rank, row in enumerate(np.argsort(-(self._vectors @ q))[:_POOL]):
            score[int(row)] = score.get(int(row), 0.0) + 1 / (_RRF_K + rank)
        best = sorted(score, key=lambda r: (-score[r], r))[:k]
        return [self._passages[r] for r in best]

    def _keyword(self, query: str) -> list[int]:
        words = [w for w in _WORD_RE.findall(query.lower()) if len(w) > 2]
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in dict.fromkeys(words))
        rows = self._db.execute(
            "SELECT rowid FROM p WHERE p MATCH ? ORDER BY bm25(p) LIMIT ?", (match, _POOL)
        )
        return [int(r[0]) for r in rows]
