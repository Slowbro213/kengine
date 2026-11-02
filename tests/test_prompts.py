import json

from kengine.claims.extraction import Extraction
from kengine.prompts import load_prompt


def test_prompts_load_and_contain_required_markers():
    t, e = load_prompt("think"), load_prompt("extract")
    assert all(
        m in t for m in ["Working:", "ASSUMPTION:", "Final answer:", "Rests on:", "Next: continue"]
    )
    assert all(m in e for m in ["computation", "premises", "preconditions", "final_answer"])


def test_extract_example_is_valid_extraction():
    example = load_prompt("extract").strip().splitlines()[-1]
    parsed = Extraction.model_validate(json.loads(example))
    assert parsed.action.kind == "final_answer" and len(parsed.claims) == 3
