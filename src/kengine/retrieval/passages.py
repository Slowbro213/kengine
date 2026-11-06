"""Documents as passages: each sentence, and each pair of neighbouring sentences, prefixed
with the document's title so a passage names what it's about."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TITLE_RE = re.compile(r"^([^:\n]{1,120}): (.+)$", re.DOTALL)
_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
_REF_RE = re.compile(
    r"\s*\((?:from\s+)?docs?(?:uments?)?\s+\d+(?:\s*(?:,|and|&)\s*\d+)*\)\s*(\.?)\s*$",
    re.IGNORECASE,
)


def claim_text(text: str) -> str:
    """The claim without a trailing document reference such as "(document 3)": the model
    is asked to cite that way, but the reference isn't part of what the documents say."""
    m = _REF_RE.search(text)
    if m is None:
        return text
    head = text[: m.start()].rstrip()
    return head + (m.group(1) if m.group(1) and not head.endswith(".") else "")


@dataclass(frozen=True)
class Passage:
    doc: int  # 1-based document number; 0 for web results
    title: str
    text: str  # starts with "<title>: "
    source: str = ""  # a web result's "url (fetched <time>)"


def bare_title(title: str) -> str:
    return _PAREN_RE.sub("", title)


def split_document(doc: str, number: int) -> list[Passage]:
    m = _TITLE_RE.match(doc.strip())
    title, body = (m.group(1), m.group(2)) if m else (f"Document {number}", doc.strip())
    sentences = [s.strip() for s in _SENTENCE_RE.split(body) if s.strip()]
    out: list[Passage] = []
    for i, s in enumerate(sentences):
        out.append(Passage(number, title, f"{title}: {s}"))
        if i + 1 < len(sentences):
            out.append(Passage(number, title, f"{title}: {s} {sentences[i + 1]}"))
    return out


def split_documents(docs: Sequence[str]) -> list[Passage]:
    return [p for i, d in enumerate(docs, 1) for p in split_document(d, i)]
