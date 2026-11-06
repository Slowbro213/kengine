from datetime import UTC, datetime

import httpx

from kengine.retrieval.web import SearxngSearch, WebResult, web_passages

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


def client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler)


def test_searxng_results_become_web_results_with_fetch_time():
    seen: list[httpx.Request] = []

    def handle(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": "https://a", "title": "Bo", "content": "Bo was born in Oslo."},
                    {"url": "https://b", "title": "Empty", "content": ""},
                ]
            },
        )

    s = SearxngSearch("http://127.0.0.1:8888", client(httpx.MockTransport(handle)), now=lambda: NOW)
    got = s.search("Bo born")
    assert got == [WebResult("https://a", "Bo", "Bo was born in Oslo.", "2026-11-06T12:00:00Z")]
    assert seen[0].url.params["format"] == "json" and seen[0].url.params["q"] == "Bo born"
    assert seen[0].url.path == "/search"


def test_errors_give_no_results():
    def handle(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    for h in (handle, boom):
        assert SearxngSearch("http://x", client(httpx.MockTransport(h))).search("q") == []


def test_invalid_json_gives_no_results():
    def handle(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>")

    assert SearxngSearch("http://x", client(httpx.MockTransport(handle))).search("q") == []


def test_web_passages_are_marked_with_their_source():
    (p,) = web_passages([WebResult("https://a", "Bo", "Bo was born in Oslo.", "t")])
    assert (p.doc, p.text, p.source) == (0, "Bo: Bo was born in Oslo.", "https://a (fetched t)")
