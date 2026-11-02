"""Model backends."""

from kengine.backend.types import ModelBackend
from kengine.config import BackendConfig


def make_backend(config: BackendConfig) -> ModelBackend:
    from kengine.backend.llamacpp import LlamaCppBackend

    return LlamaCppBackend(config)


__all__ = ["make_backend"]
