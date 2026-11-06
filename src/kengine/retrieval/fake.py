"""Deterministic stand-ins for the CPU-side models, for tests."""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence

import numpy as np

from kengine.retrieval.models import NLIScores, Vectors

NEUTRAL = NLIScores(0.0, 1.0, 0.0)
_DIM = 64


class FakeEntailment:
    """A rule `(in_premise, hypothesis, scores)` applies when the hypothesis is equal and
    `in_premise` is part of the premise. Anything else is neutral."""

    def __init__(self, rules: Sequence[tuple[str, str, NLIScores]] = ()) -> None:
        self.rules = list(rules)
        self.calls: list[tuple[str, str]] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        out: list[NLIScores] = []
        for premise, hypothesis in pairs:
            self.calls.append((premise, hypothesis))
            match = [s for p, h, s in self.rules if h == hypothesis and p in premise]
            out.append(match[0] if match else NEUTRAL)
        return out


class FakeEmbedder:
    """A hashed bag of lowercase words."""

    def embed(self, texts: Sequence[str], query: bool = False) -> Vectors:
        v = np.zeros((len(texts), _DIM), np.float32)
        for i, text in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", text.lower()):
                v[i, zlib.crc32(w.encode()) % _DIM] += 1.0
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return (v / np.where(norms == 0, 1.0, norms)).astype(np.float32)
