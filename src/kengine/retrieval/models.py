"""The CPU-side models (spec §5.5): an entailment cross-encoder and a sentence embedder,
both ONNX, pinned by revision and loaded from a local cache only. `kengine fetch-models`
downloads them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download  # pyright: ignore[reportUnknownVariableType]
from huggingface_hub.errors import LocalEntryNotFoundError
from tokenizers import Encoding, Tokenizer

Vectors = npt.NDArray[np.float32]

_FILES = ("onnx/model.onnx", "tokenizer.json")
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BATCH = 32
_MAX_TOKENS = 512


@dataclass(frozen=True)
class NLIScores:
    entail: float
    neutral: float
    contradict: float


class EntailmentModel(Protocol):
    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str], query: bool = False) -> Vectors: ...


@dataclass(frozen=True)
class Models:
    entailment: EntailmentModel
    embedder: Embedder


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str


ENTAILMENT = ModelSpec(
    "Xenova/DeBERTa-v3-base-mnli-fever-anli", "72a1ce83a0144efaf828b3c3844320a61197a53d"
)
EMBEDDING = ModelSpec("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")


class ModelsMissing(RuntimeError):
    pass


class _Session(Protocol):
    def get_inputs(self) -> list[Any]: ...
    def run(
        self, output_names: None, input_feed: dict[str, npt.NDArray[np.int64]]
    ) -> list[npt.NDArray[np.float32]]: ...


def _path(cache_dir: Path, spec: ModelSpec, filename: str, local_only: bool) -> Path:
    return Path(
        hf_hub_download(
            spec.repo,
            filename,
            revision=spec.revision,
            cache_dir=cache_dir,
            local_files_only=local_only,
        )
    )


def _open(cache_dir: Path, spec: ModelSpec) -> tuple[_Session, Tokenizer]:
    import onnxruntime  # pyright: ignore[reportMissingTypeStubs]

    model, tok = (_path(cache_dir, spec, f, local_only=True) for f in _FILES)
    session = cast(
        _Session,
        onnxruntime.InferenceSession(str(model), providers=["CPUExecutionProvider"]),
    )
    tokenizer = Tokenizer.from_file(str(tok))
    tokenizer.enable_truncation(_MAX_TOKENS)  # pyright: ignore[reportUnknownMemberType]
    tokenizer.enable_padding()  # pyright: ignore[reportUnknownMemberType]
    return session, tokenizer


def _feeds(session: _Session, enc: list[Encoding]) -> dict[str, npt.NDArray[np.int64]]:
    feeds = {
        "input_ids": np.array([e.ids for e in enc], np.int64),
        "attention_mask": np.array([e.attention_mask for e in enc], np.int64),
    }
    if "token_type_ids" in {i.name for i in session.get_inputs()}:
        feeds["token_type_ids"] = np.array([e.type_ids for e in enc], np.int64)
    return feeds


class OnnxEntailment:
    """Softmax over (entailment, neutral, contradiction) for each (premise, hypothesis)."""

    def __init__(self, session: _Session, tokenizer: Tokenizer) -> None:
        self._session, self._tok = session, tokenizer

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScores]:
        out: list[NLIScores] = []
        for i in range(0, len(pairs), _BATCH):
            enc = self._tok.encode_batch(list(pairs[i : i + _BATCH]))
            logits = self._session.run(None, _feeds(self._session, enc))[0]
            p = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = p / p.sum(axis=1, keepdims=True)
            out += [NLIScores(float(r[0]), float(r[1]), float(r[2])) for r in p]
        return out


class OnnxEmbedder:
    """CLS-pooled, normalized sentence vectors (bge)."""

    def __init__(self, session: _Session, tokenizer: Tokenizer) -> None:
        self._session, self._tok = session, tokenizer

    def embed(self, texts: Sequence[str], query: bool = False) -> Vectors:
        if query:
            texts = [_QUERY_PREFIX + t for t in texts]
        rows: list[Vectors] = []
        for i in range(0, len(texts), _BATCH):
            enc = self._tok.encode_batch(list(texts[i : i + _BATCH]))
            hidden = self._session.run(None, _feeds(self._session, enc))[0]
            cls = hidden[:, 0].astype(np.float32)
            rows.append(cls / np.linalg.norm(cls, axis=1, keepdims=True))
        return np.vstack(rows) if rows else np.zeros((0, 384), np.float32)


def models_present(
    cache_dir: Path, specs: Sequence[ModelSpec] = (ENTAILMENT, EMBEDDING)
) -> list[str]:
    """The repos in `specs` whose files aren't in `cache_dir`."""
    missing: list[str] = []
    for spec in specs:
        try:
            for f in _FILES:
                _path(cache_dir, spec, f, local_only=True)
        except LocalEntryNotFoundError:
            missing.append(spec.repo)
    return missing


def load_models(
    cache_dir: Path, entailment: ModelSpec = ENTAILMENT, embedding: ModelSpec = EMBEDDING
) -> Models:
    missing = models_present(cache_dir, (entailment, embedding))
    if missing:
        raise ModelsMissing(
            f"{', '.join(missing)} not in {cache_dir}; run `kengine fetch-models` (about 870 MB)"
        )
    return Models(
        OnnxEntailment(*_open(cache_dir, entailment)), OnnxEmbedder(*_open(cache_dir, embedding))
    )


def fetch_models(
    cache_dir: Path, specs: Sequence[ModelSpec] = (ENTAILMENT, EMBEDDING)
) -> list[Path]:
    return [_path(cache_dir, s, f, local_only=False) for s in specs for f in _FILES]
