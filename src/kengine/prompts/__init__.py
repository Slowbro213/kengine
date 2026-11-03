"""Prompt templates shipped as package data."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib.resources import files
from typing import Literal

from kengine.claims.formal import FormalSyntax, json_statement


def load_prompt(name: str) -> str:
    return (files("kengine.prompts") / f"{name}.md").read_text(encoding="utf-8")


def formal_value(
    statements: Sequence[str], syntax: FormalSyntax, mode: str = "holds"
) -> dict[str, object]:
    """A `formal` object in `syntax`, for prompt examples."""
    if syntax == "text":
        return {"mode": mode, "statements": list(statements)}
    return {
        "mode": mode,
        "statements": [json_statement(s).model_dump(by_alias=True) for s in statements],
    }


_FORMALIZE_EXAMPLE: list[tuple[int, list[str]]] = [
    (1, ["dog(x) -> mammal(x)"]),
    (2, ["dog(rex) | cat(rex)"]),
    (3, ["not cat(rex)"]),
    (4, []),
]


def system_prompt(name: Literal["extract", "formalize"], syntax: FormalSyntax) -> str:
    text = load_prompt(name).replace("{formal_syntax}", load_prompt(f"formal_{syntax}").strip())
    if name == "extract":
        return text.replace("{formal_example}", json.dumps(formal_value(["mammal(rex)"], syntax)))
    example = {
        "givens": [
            {"given": n, "statements": formal_value(s, syntax)["statements"]}
            for n, s in _FORMALIZE_EXAMPLE
        ]
    }
    return text.replace("{example}", json.dumps(example))
