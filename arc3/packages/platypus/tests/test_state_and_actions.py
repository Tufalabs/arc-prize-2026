"""A1 (instances pickle), A3 (templates deepcopy cheaply, stay pristine),
the taaf picklable-until-setup contract, and the Action bijection (§2)."""

from __future__ import annotations

import pickle

import pytest
from infra_fixtures import ProbeAgent
from taaf.benchmark import Benchmark
from taaf.game_examples import ExampleGame

from platypus import Action, AgentSolver, ContractViolationError, RunConfig


async def test_template_survives_run_untouched() -> None:
    ProbeAgent.reset_log()
    template = ProbeAgent(script=("ACTION1", "ACTION2"))
    benchmark = Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=template), n_passes=2)
    await benchmark.run()
    # Copies played (state advanced inside the run), the template did not.
    assert template.step == 0
    assert len(ProbeAgent.finishes) == 2


def test_agent_pickles_with_config_and_state() -> None:
    agent = ProbeAgent(script=("ACTION1",), run_config=RunConfig(rng_seed=7, on_agent_error="raise"))
    agent.step = 1
    restored = pickle.loads(pickle.dumps(agent))
    assert restored == agent


def test_solver_picklable_before_setup() -> None:
    solver = AgentSolver(agent=ProbeAgent(script=("ACTION1",)))
    restored = pickle.loads(pickle.dumps(solver))
    assert isinstance(restored.agent, ProbeAgent)


def test_action_bijection_with_action_input() -> None:
    click = Action("ACTION6", row=3, col=60)
    action_input = click.to_action_input()
    assert action_input.data == {"x": 60, "y": 3}  # the ONE defined bijection
    assert Action.from_action_input(action_input) == click

    simple = Action("ACTION2")
    round_tripped = Action.from_action_input(simple.to_action_input())
    assert round_tripped == simple


def test_action_rejects_malformed() -> None:
    with pytest.raises(ContractViolationError):
        Action("ACTION8")
    with pytest.raises(ContractViolationError):
        Action("ACTION6", row=3)  # coordinate action needs both
    with pytest.raises(ContractViolationError):
        Action("ACTION1", row=1, col=1)  # simple action carries no data
