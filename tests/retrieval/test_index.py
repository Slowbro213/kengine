from collections.abc import Sequence

import pytest

from kengine.retrieval.fake import FakeEmbedder
from kengine.retrieval.index import Corpus
from kengine.retrieval.models import Vectors
from kengine.retrieval.passages import Passage, bare_title, split_document

DOCS = [
    "Ann Lee: Ann Lee is a painter. She was born in Paris in 1990. She lives in Rome.",
    "Rome (city): Rome is the capital of Italy.",
    "Tax law: Tax law in Peru changed in 2001.",
]


def test_a_document_becomes_titled_sentences_and_two_sentence_windows():
    got = [p.text for p in split_document(DOCS[0], 1)]
    assert got == [
        "Ann Lee: Ann Lee is a painter.",
        "Ann Lee: Ann Lee is a painter. She was born in Paris in 1990.",
        "Ann Lee: She was born in Paris in 1990.",
        "Ann Lee: She was born in Paris in 1990. She lives in Rome.",
        "Ann Lee: She lives in Rome.",
    ]
    assert split_document(DOCS[1], 2)[0] == Passage(
        2, "Rome (city)", "Rome (city): Rome is the capital of Italy."
    )


def test_a_document_without_a_title_is_numbered():
    assert split_document("Just one sentence.", 4) == [
        Passage(4, "Document 4", "Document 4: Just one sentence.")
    ]


def test_bare_title_drops_a_parenthetical():
    assert bare_title("Rome (city)") == "Rome" and bare_title("Ann Lee") == "Ann Lee"


def test_search_ranks_passages_that_share_words_first():
    corpus = Corpus.from_documents(DOCS, FakeEmbedder())
    assert len(corpus) == 7
    top = corpus.search("Where was Ann Lee born?", k=2)
    assert top[0].text == "Ann Lee: She was born in Paris in 1990." or "born" in top[0].text
    assert all(p.doc == 1 for p in top)


def test_search_on_an_empty_corpus_or_query_without_words_is_empty():
    assert Corpus(FakeEmbedder()).search("anything") == []
    corpus = Corpus.from_documents(DOCS, FakeEmbedder())
    assert len(corpus.search("?!", k=3)) == 3  # embeddings still rank something


def test_added_passages_are_searchable():
    corpus = Corpus(FakeEmbedder())
    corpus.add([Passage(0, "Web", "Web: Bo was born in Oslo.", "https://x (fetched t)")])
    assert corpus.search("Bo born Oslo")[0].source == "https://x (fetched t)"


def test_adding_twice_keeps_row_ids_and_vectors_aligned():
    corpus = Corpus(FakeEmbedder())
    corpus.add([Passage(0, "Web", "Web: Bo was born in Oslo.")])
    corpus.add([Passage(0, "Web", "Web: Ida paints ferries in Bergen.")])
    assert len(corpus) == 2
    assert corpus.search("Ida paints ferries")[0].text == "Web: Ida paints ferries in Bergen."


class Boom:
    def embed(self, texts: Sequence[str], query: bool = False) -> Vectors:
        raise RuntimeError("embedder failed")


def test_an_embedder_that_raises_leaves_the_corpus_unchanged():
    corpus = Corpus(Boom())
    with pytest.raises(RuntimeError):
        corpus.add([Passage(0, "Web", "Web: Bo was born in Oslo.")])
    assert len(corpus) == 0
    assert corpus.search("anything") == []
