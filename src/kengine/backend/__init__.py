"""Model backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kengine.backend.types import ModelBackend

if TYPE_CHECKING:
    from kengine.config import BackendConfig


def make_backend(config: BackendConfig) -> ModelBackend:
    from kengine.backend.llamacpp import LlamaCppBackend

    return LlamaCppBackend(config)


__all__ = ["make_backend"]
