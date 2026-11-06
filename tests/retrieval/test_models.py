import os
from pathlib import Path

import numpy as np
import pytest

from kengine.retrieval.fake import FakeEmbedder, FakeEntailment
from kengine.retrieval.models import (
    ENTAILMENT,
    ModelsMissing,
    NLIScores,
    load_models,
    models_present,
)

SURE = NLIScores(0.97, 0.02, 0.01)


def test_fake_entailment_matches_a_rule_by_hypothesis_and_premise_substring():
    nli = FakeEntailment([("born in Paris", "Ann was born in Paris.", SURE)])
    got = nli.score([("Ann: Ann was born in Paris in 1990.", "Ann was born in Paris."), ("x", "y")])
    assert got == [SURE, NLIScores(0.0, 1.0, 0.0)]
    assert nli.calls == [
        ("Ann: Ann was born in Paris in 1990.", "Ann was born in Paris."),
        ("x", "y"),
    ]


def test_fake_embedder_rows_are_unit_length_and_similar_for_shared_words():
    v = FakeEmbedder().embed(["Ann was born in Paris", "Ann born Paris", "tax law of Peru"])
    assert v.shape == (3, 64) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_missing_models_are_named_and_loading_says_how_to_fetch(tmp_path: Path):
    assert models_present(tmp_path) == [ENTAILMENT.repo, "BAAI/bge-small-en-v1.5"]
    with pytest.raises(ModelsMissing, match="kengine fetch-models"):
        load_models(tmp_path)


@pytest.mark.models
def test_real_models_score_entailment_and_contradiction():
    cache = Path(os.environ.get("KENGINE_MODELS", Path.home() / ".local/share/kengine/models/hf"))
    m = load_models(cache)
    yes, no = m.entailment.score(
        [
            ("Paris: Paris is the capital of France.", "France's capital is Paris."),
            ("Paris: Paris is the capital of France.", "Paris is the capital of Spain."),
        ]
    )
    assert yes.entail > 0.9 and no.contradict > 0.9
    v = m.embedder.embed(["capital of France"], query=True)
    assert v.shape == (1, 384)
