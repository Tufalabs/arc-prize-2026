"""Per-game seed resolution (I11/I13): the seed a game's RNG is built
from is explicit, overridable, and recorded faithfully enough to replay
the run it came from."""

from __future__ import annotations

import arcengine
import pytest
from taaf.benchmark import Benchmark
from taaf.game import Game
from taaf.game_examples import ExampleGame

from platypus import AgentSolver, RunConfig
from reference_agents import RandomAgent

Outcome = list[tuple[str, str, int, list[str]]]


def _games() -> list[Game]:
    action1, action2 = arcengine.GameAction.ACTION1, arcengine.GameAction.ACTION2
    return [
        ExampleGame(label="two_level"),  # default: 2 levels, actions {1, 2}
        ExampleGame(target_sequence_per_level=[[action2], [action1, action2]], label="short"),
    ]


def _outcome(benchmark: Benchmark) -> Outcome:
    return [
        (run.game_id or "", run.state, run.levels_completed, [record.action.id.name for record in run.history])
        for run in benchmark.game_runs
    ]


async def _run(config: RunConfig) -> Benchmark:
    # n_passes=1 throughout: rng_seed_per_game is keyed by game_id, so with
    # several passes of one game every pass copy would take the same seed
    # (the same limitation llm_config.replay_dir documents).
    benchmark = Benchmark(games=_games(), solver=AgentSolver(agent=RandomAgent(max_moves=40, run_config=config)))
    await benchmark.run()
    return benchmark


@pytest.mark.parametrize("mode", ["inprocess", "subprocess"])
async def test_per_game_override_replaces_the_derived_seed(mode: str) -> None:
    """An override is exactly the derived seed it replaces, and the
    run-level ``rng_seed`` no longer has a say."""
    derived = await _run(RunConfig(rng_seed=777, process_mode=mode))  # type: ignore[arg-type]
    game_ids = [run.game_id or "" for run in derived.game_runs]
    # Derived seeds are rng_seed + game index, so these overrides name the
    # same two integers by a different route.
    overridden = await _run(
        RunConfig(
            rng_seed=999,
            rng_seed_per_game={game_id: 777 + index for index, game_id in enumerate(game_ids)},
            process_mode=mode,  # type: ignore[arg-type]
        )
    )
    assert _outcome(overridden) == _outcome(derived)


@pytest.mark.parametrize("mode", ["inprocess", "subprocess"])
async def test_override_covers_only_the_games_it_names(mode: str) -> None:
    """A partial override leaves the other games on their derived seeds,
    so one recorded game can be pinned without disturbing the rest."""
    derived = await _run(RunConfig(rng_seed=777, process_mode=mode))  # type: ignore[arg-type]
    pinned = (derived.game_runs[0].game_id or "",)
    partial = await _run(
        RunConfig(
            rng_seed=777,
            rng_seed_per_game={pinned[0]: 777},  # game index 0: the same seed it would derive
            process_mode=mode,  # type: ignore[arg-type]
        )
    )
    assert _outcome(partial) == _outcome(derived)
