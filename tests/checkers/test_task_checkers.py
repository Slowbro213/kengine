from kengine.checkers import task_checkers
from kengine.config import FactsConfig
from kengine.retrieval.fake import FakeEmbedder, FakeEntailment
from kengine.retrieval.models import Models

MODELS = Models(FakeEntailment(), FakeEmbedder())


def test_documents_without_givens_get_the_evidence_checkers_and_no_restatement():
    got = task_checkers(["T: a."], has_givens=False, models=MODELS, facts=FactsConfig())
    assert [c.id for c in got] == [
        "structure",
        "arithmetic",
        "logic",
        "consistency",
        "facts",
        "inference-nli",
    ]


def test_a_task_without_documents_keeps_the_m2_set_plus_consistency():
    got = task_checkers([], has_givens=True, models=MODELS, facts=FactsConfig())
    assert [c.id for c in got] == ["structure", "arithmetic", "restatement", "logic", "consistency"]


def test_without_models_it_is_the_m2_set():
    got = task_checkers(["T: a."], has_givens=False, models=None, facts=FactsConfig())
    assert [c.id for c in got] == ["structure", "arithmetic", "logic"]
