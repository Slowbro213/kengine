"""Claim checkers (spec §5.5)."""

from __future__ import annotations

from collections.abc import Sequence

from kengine.checkers.arithmetic import ArithmeticChecker
from kengine.checkers.base import Checker
from kengine.checkers.consistency import ConsistencyChecker
from kengine.checkers.facts import FactChecker
from kengine.checkers.inference_nli import InferenceEntailmentChecker
from kengine.checkers.logic import LogicChecker
from kengine.checkers.restatement import RestatementChecker
from kengine.checkers.structure import StructureChecker
from kengine.config import FactsConfig
from kengine.retrieval.index import Corpus
from kengine.retrieval.models import Models
from kengine.retrieval.web import WebSearch


def default_checkers() -> list[Checker]:
    """The M2 checkers, in the order they run: structure first, so a malformed claim is
    caught before anything evaluates it."""
    return [StructureChecker(), ArithmeticChecker(), RestatementChecker(), LogicChecker()]


def task_checkers(
    documents: Sequence[str],
    has_givens: bool,
    models: Models | None,
    facts: FactsConfig,
    web: WebSearch | None = None,
) -> list[Checker]:
    """The checkers for one task. The fact checker's corpus is the task's documents (and the
    web, when configured), so it's built per task. Restatement only checks against givens,
    so a task of documents without givens leaves numeric facts to the fact checker."""
    out: list[Checker] = [StructureChecker(), ArithmeticChecker()]
    if has_givens or not documents:
        out.append(RestatementChecker())
    out.append(LogicChecker())
    if models is None:
        return out
    out.append(ConsistencyChecker(models.entailment, models.embedder, facts))
    if documents or web is not None:
        corpus = Corpus.from_documents(documents, models.embedder)
        out.append(FactChecker(corpus, models.entailment, facts, web))
    if documents:
        out.append(InferenceEntailmentChecker(models.entailment, facts))
    return out
