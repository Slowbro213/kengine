"""Effort profiles (spec §5.2). Critic and fact-check settings come in later milestones."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CheckScope(StrEnum):
    LOAD_BEARING = "load_bearing"
    ALL = "all"


class GatePolicy(StrEnum):
    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"


class Profile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    n_samples: int = Field(ge=1)
    check_scope: CheckScope
    max_steps: int = Field(ge=1)
    final_gate: GatePolicy


def _profile(name: str, n_samples: int, scope: CheckScope, max_steps: int) -> Profile:
    return Profile(
        name=name,
        n_samples=n_samples,
        check_scope=scope,
        max_steps=max_steps,
        final_gate=GatePolicy.STANDARD,
    )


PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (
        _profile("quick", 1, CheckScope.LOAD_BEARING, 8),
        _profile("standard", 3, CheckScope.ALL, 16),
        _profile("thorough", 3, CheckScope.ALL, 32),
        _profile("exhaustive", 5, CheckScope.ALL, 64),
    )
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; choose from {', '.join(PROFILES)}") from None
