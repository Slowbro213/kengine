"""Web search for the fact checker (spec §5.5): pluggable, SearXNG by default, off unless
configured. Results are untrusted evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from kengine.retrieval.passages import Passage


@dataclass(frozen=True)
class WebResult:
    url: str
    title: str
    content: str
    fetched_at: str  # ISO 8601, UTC


class WebSearch(Protocol):
    def search(self, query: str) -> list[WebResult]: ...


def web_passages(results: Sequence[WebResult]) -> list[Passage]:
    return [
        Passage(0, r.title, f"{r.title}: {r.content}", f"{r.url} (fetched {r.fetched_at})")
        for r in results
        if r.content.strip()
    ]
