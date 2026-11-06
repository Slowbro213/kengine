"""Web search for the fact checker (spec §5.5): pluggable, SearXNG by default, off unless
`facts.searxng_url` is set. Results are untrusted evidence: their text is only ever scored
as a premise, and feedback quoting them marks them untrusted. Only the result snippets are
used; pages aren't fetched."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast

import httpx

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


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SearxngSearch:
    """A SearXNG instance's JSON API (`search.formats` must include `json` in its
    settings.yml)."""

    def __init__(
        self,
        base_url: str,
        client: httpx.Client | None = None,
        max_results: int = 5,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._url = base_url.rstrip("/") + "/search"
        self._client = client or httpx.Client(timeout=15.0)
        self._max = max_results
        self._now = now

    def search(self, query: str) -> list[WebResult]:
        try:
            r = self._client.get(self._url, params={"q": query, "format": "json"})
        except httpx.HTTPError:
            return []
        if r.status_code != 200:
            return []
        try:
            body: object = r.json()
        except ValueError:
            return []
        if not isinstance(body, dict):
            return []
        raw: object = cast(dict[str, object], body).get("results", [])
        if not isinstance(raw, list):
            return []
        when = self._now().strftime("%Y-%m-%dT%H:%M:%SZ")
        out: list[WebResult] = []
        for item in cast(list[object], raw):
            if not isinstance(item, dict):
                continue
            x = cast(dict[str, Any], item)
            content = str(x.get("content", ""))
            if content.strip():
                out.append(WebResult(str(x.get("url", "")), str(x.get("title", "")), content, when))
        return out[: self._max]
