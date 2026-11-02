"""Runs the agent against a real llama-server. Set KENGINE_TEST_LLM_URL to enable."""

import os
from pathlib import Path
from typing import get_args

import pytest

from kengine.audit.session import Session
from kengine.backend import make_backend
from kengine.checkers import default_checkers
from kengine.config import BackendConfig, KengineConfig
from kengine.loop.agent import Agent, RunStatus
from kengine.loop.profiles import get_profile
from kengine.loop.task import Task

URL = os.environ.get("KENGINE_TEST_LLM_URL")

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(URL is None, reason="KENGINE_TEST_LLM_URL is not set"),
]


def test_agent_runs_end_to_end(tmp_path: Path):
    assert URL is not None
    cfg = KengineConfig(backend=BackendConfig(base_url=URL))
    session = Session.create(tmp_path)
    agent = Agent(
        backend=make_backend(cfg.backend),
        roles=cfg.roles,
        profile=get_profile("quick"),
        context_budget=cfg.context.step_prompt_tokens,
        checkers=default_checkers(),
        session=session,
    )
    result = agent.run(Task(goal="What is 17 * 23?"))
    assert result.status in get_args(RunStatus)
    events = session.log.read()
    assert events[0].kind == "session_started" and events[-1].kind == "run_finished"
