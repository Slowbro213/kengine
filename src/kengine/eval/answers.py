"""Pulling the final answer out of model text, and SQuAD-style text normalization."""

from __future__ import annotations

import re
import string

# The marker may sit mid-line and carry markdown: `**Final answer:** 7`, `**Final answer**: 7`.
_FINAL_RE = re.compile(r"(?im)\bfinal answer[\s*_]*[:：][\s*_]*(.+?)\s*$")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def extract_final_answer(text: str) -> str | None:
    """The text after the last `Final answer:` line, without markdown or a trailing period."""
    matches = _FINAL_RE.findall(text)
    if not matches:
        return None
    answer: str = matches[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    answer = answer.strip("*_ ").strip()
    if answer.endswith("."):
        answer = answer[:-1].rstrip()
    return answer or None


def normalize_text(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())
