# type: ignore
"""Tests for the localhost competition Arcade simulator (R11.13)."""

import warnings

import pytest

import taaf
from taaf.competition_arcade import CompetitionArcadeServer
from taaf.game_api import GameAPI

ENV_NAME = "ls20"


async def test_competition_arcade_runs_cloned_games_with_shared_scorecard():
    with CompetitionArcadeServer(game_ids=(ENV_NAME,), total_runs=2) as server:
        bm = taaf.benchmark.Benchmark(
            label="competition_arcade_smoke",
            games=[GameAPI(env_name=game_id, arcade_spec=server.arcade_spec) for game_id in server.exposed_game_ids],
            solver=taaf.solver_examples.SolverRandom(seed=7, max_actions_per_game=1),
            n_passes=1,
            job_dir=None,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="arcengine returned no baseline_actions")
            await bm.run()

    assert [run.game_id for run in bm.game_runs] == ["k000", "k001"]
    assert all(run.base_actions_per_level is None for run in bm.game_runs)
    assert all(run.final_score == 0.0 for run in bm.game_runs)


def test_competition_arcade_rejects_reusing_same_game_id_in_one_scorecard():
    with CompetitionArcadeServer(game_ids=(ENV_NAME,)) as server:
        session = taaf.game.RunSession()
        game_id = server.exposed_game_ids[0]
        first = GameAPI(env_name=game_id, arcade_spec=server.arcade_spec)
        second = GameAPI(env_name=game_id, arcade_spec=server.arcade_spec)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="arcengine returned no baseline_actions")
            first.start_game(session)
        try:
            with pytest.raises(RuntimeError, match="observation_space is None|returned None"):
                second.start_game(session)
        finally:
            first.finish_game()


def test_official_110_competition_server_exposes_unique_clone_ids():
    with CompetitionArcadeServer.official_110() as server:
        game_ids = server.exposed_game_ids

    assert len(game_ids) == 110
    assert len(set(game_ids)) == 110
    assert game_ids[:3] == ["k000", "k001", "k002"]
    assert game_ids[-1] == "k109"


async def test_competition_sim_arcade_spec_autostarts_and_tears_down_server():
    """``ArcadeSpec(competition_sim=True)`` starts the sim lazily on the worker,
    shares it across games, and ``Benchmark.run`` stops it via ``RunSession.close``."""
    bm = taaf.standard_benchmarks.make_benchmark_kaggle_official_110(
        solver=taaf.solver_examples.SolverRandom(seed=3, max_actions_per_game=1),
        competition_sim=True,
    )
    # Subset for speed; the lazily-started server still exposes all 110 clones.
    bm.games = bm.games[:2]
    bm.job_dir = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="arcengine returned no baseline_actions")
        await bm.run()

    assert [run.game_id for run in bm.game_runs] == ["k000", "k001"]
    assert all(run.state != "crashed" for run in bm.game_runs)
    assert all(run.base_actions_per_level is None for run in bm.game_runs)
