import pytest

from kengine.retrieval.passages import claim_text


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("Heat was directed by Michael Mann (document 3)", "Heat was directed by Michael Mann"),
        ("Heat was directed by Michael Mann (document 3).", "Heat was directed by Michael Mann."),
        ("Heat was directed by Michael Mann. (Document 3)", "Heat was directed by Michael Mann."),
        ("It is X (doc 3)", "It is X"),
        ("It is X (from document 4).", "It is X."),
        ("It is X (From Document 4)", "It is X"),
        ("It is X (documents 2 and 5)", "It is X"),
        ("It is X (docs 2, 5)", "It is X"),
        ("It is X (see below)", "It is X (see below)"),
        ("Document 3 says X", "Document 3 says X"),
    ],
)
def test_claim_text_strips_a_trailing_document_reference(raw: str, want: str) -> None:
    assert claim_text(raw) == want
