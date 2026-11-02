from pathlib import Path

import pytest
from pydantic import ValidationError

from kengine.eval.taxonomy import ErrorLabel, summarize_taxonomy


def test_summarize_taxonomy(tmp_path: Path) -> None:
    p = tmp_path / "taxonomy.jsonl"
    p.write_text(
        '{"item_id": "a", "labels": ["arithmetic_error"]}\n'
        "\n"
        '{"item_id": "b", "labels": ["arithmetic_error", "truncated"], "note": "x"}\n'
    )
    assert summarize_taxonomy(p) == {ErrorLabel.ARITHMETIC_ERROR: 2, ErrorLabel.TRUNCATED: 1}


def test_unknown_label_names_line(tmp_path: Path) -> None:
    p = tmp_path / "taxonomy.jsonl"
    p.write_text('{"item_id": "a", "labels": ["other"]}\n{"item_id": "b", "labels": ["oops"]}\n')
    with pytest.raises(ValidationError, match="line 2"):
        summarize_taxonomy(p)
