from kengine.eval.suites import SUITES, select_indices


def test_select_indices_seeded_sorted_and_sized() -> None:
    a = select_indices(1000, "smoke", 0)
    assert len(a) == 10 and a == sorted(a) and a == select_indices(1000, "smoke", 0)
    assert a != select_indices(1000, "smoke", 1)
    assert len(select_indices(1000, "small", 0)) == 100
    assert select_indices(50, "small", 0) == list(range(50))  # fewer rows than size -> all
    assert select_indices(30, "full", 0) == list(range(30))


def test_suites_pinned() -> None:
    assert {n: (s.hf_id, s.hf_config, s.split) for n, s in SUITES.items()} == {
        "gsm8k": ("openai/gsm8k", "main", "test"),
        "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "validation"),
        "folio": ("yale-nlp/FOLIO", None, "validation"),
    }
