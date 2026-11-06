from kengine.checkers import task_checkers
from kengine.config import FactsConfig
from kengine.retrieval.fake import FakeEmbedder, FakeEntailment
from kengine.retrieval.models import Models
from kengine.retrieval.web import WebResult


class FakeWeb:
    def search(self, query: str) -> list[WebResult]:
        return []


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


def test_web_only_gets_the_fact_checker_but_not_the_inference_checker():
    web = FakeWeb()
    got = task_checkers([], has_givens=True, models=MODELS, facts=FactsConfig(), web=web)
    assert [c.id for c in got][-2:] == ["consistency", "facts"]


def test_a_task_without_documents_keeps_the_m2_set_plus_consistency():
    got = task_checkers([], has_givens=True, models=MODELS, facts=FactsConfig())
    assert [c.id for c in got] == ["structure", "arithmetic", "restatement", "logic", "consistency"]


def test_without_models_it_is_the_m2_set():
    got = task_checkers(["T: a."], has_givens=False, models=None, facts=FactsConfig())
    assert [c.id for c in got] == ["structure", "arithmetic", "logic"]
