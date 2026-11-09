"""A summary's rebuilt conversation shows ``world_model.py`` in full, once it
is the model's own: never the seeded example, which is a different game,
and never without a world model in use or with the switch off."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest

from seal import SealAgent, seeds

SOURCE = "def do_move(state, action):\n    return state\n"


def _agent(use_world_model: Literal["none", "optional", "mandatory"] = "optional", **kwargs: Any) -> SealAgent:
    agent = SealAgent(use_world_model=use_world_model, game_action_names=["RESET", "UP"], **kwargs)
    agent._seed_world_model(agent._ensure_repl())  # pyright: ignore[reportPrivateUsage]
    return agent


def _world_model(agent: SealAgent) -> Path:
    directory = agent._ensure_repl()._tmpdir  # pyright: ignore[reportPrivateUsage]
    assert directory is not None
    return Path(directory) / "world_model.py"


@pytest.fixture
def agents() -> Any:
    made: list[SealAgent] = []

    def make(*args: Any, **kwargs: Any) -> SealAgent:
        made.append(_agent(*args, **kwargs))
        return made[-1]

    yield make
    for agent in made:
        agent._ensure_repl()._reset_impl(keep_tmpdir_contents=False)  # pyright: ignore[reportPrivateUsage]


def test_the_models_own_world_model_is_shown(agents: Any) -> None:
    agent = agents()
    _world_model(agent).write_text(SOURCE)
    assert agent._summary_world_model_source() == SOURCE  # pyright: ignore[reportPrivateUsage]


def test_the_seeded_example_is_not(agents: Any) -> None:
    agent = agents()
    assert _world_model(agent).read_text() == seeds.seed_source(mouse=False)
    assert agent._summary_world_model_source() is None  # pyright: ignore[reportPrivateUsage]


def test_nothing_is_shown_without_a_world_model_or_with_the_switch_off(agents: Any) -> None:
    for agent in (agents("none"), agents(add_world_model_to_summary=False)):
        _world_model(agent).write_text(SOURCE)
        assert agent._summary_world_model_source() is None  # pyright: ignore[reportPrivateUsage]


def test_a_fence_inside_the_file_cannot_close_the_one_around_it() -> None:
    shown = SealAgent().prompts.compaction_world_model("DOC = '```'\n")
    assert shown.splitlines()[1] == "````python" and shown.splitlines()[-1] == "````"
