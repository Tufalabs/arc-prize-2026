"""I9 observational equivalence: identical seeded outcomes across
subprocess and in-process hosting, judged by the engine-side record."""

from __future__ import annotations

import arcengine
from taaf.benchmark import Benchmark
from taaf.game import Game
from taaf.game_examples import ExampleGame

from platypus import AgentSolver, RunConfig
from reference_agents import RandomAgent


def _games() -> list[Game]:
    action1, action2 = arcengine.GameAction.ACTION1, arcengine.GameAction.ACTION2
    return [
        ExampleGame(label="two_level"),  # default: 2 levels, actions {1, 2}
        ExampleGame(target_sequence_per_level=[[action2], [action1, action2]], label="short"),
    ]


async def _run(mode: str) -> list[tuple[str, str, int, list[str]]]:
    agent = RandomAgent(
        max_moves=40,
        run_config=RunConfig(rng_seed=17, process_mode=mode),  # type: ignore[arg-type]
    )
    benchmark = Benchmark(games=_games(), solver=AgentSolver(agent=agent), n_passes=2)
    await benchmark.run()
    return [
        (
            run.game_id or "",
            run.state,
            run.levels_completed,
            [record.action.id.name for record in run.history],
        )
        for run in benchmark.game_runs
    ]


async def test_process_modes_observationally_equivalent() -> None:
    in_process = await _run("inprocess")
    subprocess_mode = await _run("subprocess")
    assert in_process == subprocess_mode
    # Sanity: the runs did real work (some game pressed buttons).
    assert any(actions for _, _, _, actions in in_process)
