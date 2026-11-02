import pytest

from kengine.loop.profiles import PROFILES, CheckScope, GatePolicy, get_profile


def test_profiles_pinned():
    assert {n: (p.n_samples, p.check_scope, p.max_steps) for n, p in PROFILES.items()} == {
        "quick": (1, CheckScope.LOAD_BEARING, 8),
        "standard": (3, CheckScope.ALL, 16),
        "thorough": (3, CheckScope.ALL, 32),
        "exhaustive": (5, CheckScope.ALL, 64),
    }
    assert {p.final_gate for p in PROFILES.values()} == {GatePolicy.STANDARD}


def test_get_profile():
    assert get_profile("quick").name == "quick"


def test_get_profile_unknown_lists_names():
    with pytest.raises(KeyError, match="quick"):
        get_profile("turbo")
