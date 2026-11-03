from pathlib import Path

import pydantic
import pytest

from kengine.config import KengineConfig, load_config


def test_defaults_match_model_card() -> None:
    cfg = load_config([])
    assert cfg.backend.base_url == "http://127.0.0.1:8080"
    t = cfg.roles.think
    assert (
        t.thinking,
        t.sampling.temperature,
        t.sampling.top_p,
        t.sampling.top_k,
        t.sampling.min_p,
        t.sampling.presence_penalty,
    ) == (True, 1.0, 0.95, 20, 0.0, 1.5)
    e = cfg.roles.extract
    assert e.thinking is False and e.sampling.temperature == 0.3
    assert e.sampling.presence_penalty == 0.0
    assert cfg.roles.baseline == cfg.roles.think.model_copy(
        update={"max_tokens": cfg.eval.token_budget_per_task}
    )
    assert cfg.context.step_prompt_tokens == 14000
    assert cfg.eval.token_budget_per_task == 32768
    assert cfg.default_profile == "standard"


def test_later_files_override_earlier(tmp_path: Path) -> None:
    a = tmp_path / "a.toml"
    a.write_text('[backend]\nbase_url = "http://a:1"\nmodel = "m-a"\n')
    b = tmp_path / "b.toml"
    b.write_text('[backend]\nmodel = "m-b"\n[roles.think]\nmax_tokens = 999\n')
    cfg = load_config([a, b])
    assert cfg.backend.base_url == "http://a:1"  # deep merge keeps a's key
    assert cfg.backend.model == "m-b"
    assert cfg.roles.think.max_tokens == 999
    assert cfg.roles.think.sampling.top_k == 20  # untouched nested default survives


def test_missing_files_are_skipped(tmp_path: Path) -> None:
    assert load_config([tmp_path / "nope.toml"]) == load_config([])


def test_invalid_value_rejected(tmp_path: Path) -> None:
    p = tmp_path / "c.toml"
    p.write_text('[context]\nstep_prompt_tokens = "lots"\n')
    with pytest.raises(pydantic.ValidationError):
        load_config([p])


def test_explicit_baseline_is_kept(tmp_path: Path) -> None:
    p = tmp_path / "d.toml"
    p.write_text("[roles.baseline]\nthinking = false\nmax_tokens = 100\n")
    cfg = load_config([p])
    assert cfg.roles.baseline.max_tokens == 100 and cfg.roles.baseline.thinking is False


def test_data_dir_expands_home(tmp_path: Path) -> None:
    p = tmp_path / "e.toml"
    p.write_text('data_dir = "~/kdata"\n')
    assert load_config([p]).data_dir == Path.home() / "kdata"


def test_logic_defaults():
    cfg = KengineConfig()
    assert cfg.logic.syntax == "text" and cfg.logic.formalize_givens
    assert cfg.roles.formalize == cfg.roles.extract
