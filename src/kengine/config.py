"""Configuration: pydantic models plus layered TOML loading."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kengine.backend.types import RoleConfig, SamplingParams
from kengine.claims.formal import FormalSyntax

# Qwen3.5 model card: thinking mode for general tasks.
THINK_SAMPLING = SamplingParams(
    temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5
)
# Non-thinking mode, lowered temperature for schema-constrained extraction.
EXTRACT_SAMPLING = SamplingParams(
    temperature=0.3, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=0.0
)


def _think_role() -> RoleConfig:
    return RoleConfig(sampling=THINK_SAMPLING, thinking=True, max_tokens=8192)


def _extract_role() -> RoleConfig:
    return RoleConfig(sampling=EXTRACT_SAMPLING, thinking=False, max_tokens=4096)


def _critic_role() -> RoleConfig:
    return RoleConfig(
        sampling=EXTRACT_SAMPLING.model_copy(update={"temperature": 0.2}),
        thinking=False,
        max_tokens=4096,
    )


class BackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["llamacpp"] = "llamacpp"
    base_url: str = "http://127.0.0.1:8080"
    model: str = "qwen3.5-9b"
    timeout_s: float = Field(default=600.0, gt=0)


class RolesConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    think: RoleConfig = Field(default_factory=_think_role)
    extract: RoleConfig = Field(default_factory=_extract_role)
    critic: RoleConfig = Field(default_factory=_critic_role)
    # Translates the task's givens into the formal fragment (M2).
    formalize: RoleConfig = Field(default_factory=_extract_role)
    # Filled in by KengineConfig: a copy of `think` with the eval token budget.
    baseline: RoleConfig = Field(default_factory=_think_role)


class ContextConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    step_prompt_tokens: int = Field(default=14000, gt=0)


class EvalConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    token_budget_per_task: int = Field(default=32768, gt=0)


class LogicConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    syntax: FormalSyntax = "text"  # set from the Checkpoint C probe (spec §9)
    formalize_givens: bool = True


class KengineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: BackendConfig = Field(default_factory=BackendConfig)
    roles: RolesConfig = Field(default_factory=RolesConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logic: LogicConfig = Field(default_factory=LogicConfig)
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".local/share/kengine")
    default_profile: str = "standard"

    @field_validator("data_dir")
    @classmethod
    def _expand_home(cls, value: Path) -> Path:
        return value.expanduser()

    @model_validator(mode="before")
    @classmethod
    def _derive_baseline(cls, data: Any) -> Any:
        """Unless set explicitly, the baseline role is `think` with the eval token budget.

        A partial `[roles.baseline]` table is layered over that derived role.
        """
        if not isinstance(data, Mapping):
            return data
        raw = cast(Mapping[str, Any], data)
        roles_raw: dict[str, Any] = dict(cast(Mapping[str, Any], raw.get("roles") or {}))
        if isinstance(roles_raw.get("baseline"), RoleConfig):
            return raw
        think: Any = roles_raw.get("think")
        think_dict: dict[str, Any] = (
            think.model_dump() if isinstance(think, RoleConfig) else _with_defaults(think)
        )
        eval_raw: Any = raw.get("eval")
        budget: Any
        if isinstance(eval_raw, EvalConfig):
            budget = eval_raw.token_budget_per_task
        elif isinstance(eval_raw, Mapping):
            budget = cast(Mapping[str, Any], eval_raw).get(
                "token_budget_per_task", EvalConfig().token_budget_per_task
            )
        else:
            budget = EvalConfig().token_budget_per_task
        derived = {**think_dict, "max_tokens": budget}
        roles_raw["baseline"] = deep_merge(derived, roles_raw.get("baseline") or {})
        return {**raw, "roles": roles_raw}


def _with_defaults(think: Any) -> dict[str, Any]:
    return deep_merge(_think_role().model_dump(), cast(Mapping[str, Any], think or {}))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base`."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = out.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(cast(Mapping[str, Any], current), cast(Mapping[str, Any], value))
        else:
            out[key] = value
    return out


def default_config_paths() -> list[Path]:
    return [Path.home() / ".config/kengine/config.toml", Path.cwd() / "kengine.toml"]


def load_config(paths: Sequence[Path]) -> KengineConfig:
    """Load and deep-merge TOML files in order; missing files are skipped."""
    defaults = KengineConfig().model_dump(mode="json")
    del defaults["roles"]["baseline"]  # re-derived from the merged `think` role
    merged: dict[str, Any] = defaults
    for path in paths:
        if path.is_file():
            with path.open("rb") as f:
                merged = deep_merge(merged, tomllib.load(f))
    return KengineConfig.model_validate(merged)
