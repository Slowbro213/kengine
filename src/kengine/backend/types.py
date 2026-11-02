"""Types shared by every model backend."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SamplingParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int = Field(ge=0)
    min_p: float = Field(ge=0.0, le=1.0)
    presence_penalty: float
    repetition_penalty: float = 1.0


class RoleConfig(BaseModel):
    """Sampling settings, thinking switch and output limit for one model role."""

    model_config = ConfigDict(frozen=True)

    sampling: SamplingParams
    thinking: bool
    max_tokens: int = Field(gt=0)
