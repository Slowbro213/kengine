import json
from pathlib import Path

from kengine.eval.compare import compare_runs


def write(d: Path, rows: list[tuple[str, str, bool]]) -> Path:
    d.mkdir()
    with (d / "items.jsonl").open("w") as f:
        for item_id, status, correct in rows:
            f.write(json.dumps({"item_id": item_id, "status": status, "correct": correct}) + "\n")
    return d


def test_compare_categories(tmp_path: Path):
    base = write(
        tmp_path / "base",
        [("a", "answered", True), ("b", "answered", True), ("c", "answered", True),
         ("d", "answered", False), ("e", "partial", False), ("f", "answered", False),
         ("only-base", "answered", True)],
    )  # fmt: skip
    other = write(
        tmp_path / "other",
        [("a", "answered", True), ("b", "answered", False), ("c", "blocked", False),
         ("d", "answered", True), ("e", "blocked", False), ("f", "answered", False)],
    )  # fmt: skip
    c = compare_runs(base, other)
    assert (c.n, c.base_correct, c.other_correct) == (6, 3, 2)
    assert (c.both_right, c.broke, c.false_alarm) == (1, 1, 1)
    assert (c.fixed, c.caught, c.both_wrong) == (1, 1, 1)
    assert c.ids["false_alarm"] == ["c"] and c.ids["caught"] == ["e"]
