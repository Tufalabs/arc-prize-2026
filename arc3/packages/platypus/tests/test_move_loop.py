"""Move loop semantics: I1 (deliver complete, execute what do_move
returns), §2 move splitting/partition rule, I6 history, give-up flow."""

from __future__ import annotations

import arcengine
from infra_fixtures import LosableExampleGame, ProbeAgent
from taaf.benchmark import Benchmark
from taaf.game import Game
from taaf.game_examples import ExampleGame

from platypus import AgentSolver

WIN_SCRIPT = ("ACTION1",) * 3 + ("ACTION2",) * 3  # default ExampleGame levels


async def test_win_records_full_history_and_result() -> None:
    ProbeAgent.reset_log()
    benchmark = Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=ProbeAgent(script=WIN_SCRIPT)))
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "won"
    ((game_id, result, history),) = ProbeAgent.finishes
    assert game_id == "example_game"
    assert result.state == "won"
    assert result.levels_completed == 2
    assert result.score == run.final_score is not None
    assert [move.action.name for move in history] == list(WIN_SCRIPT)
    # Engine-side record agrees move for move (I1: execute exactly what
    # do_move returned).
    assert [record.action.id.name for record in run.history] == list(WIN_SCRIPT)


async def test_level_transition_splits_observations() -> None:
    ProbeAgent.reset_log()
    benchmark = Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=ProbeAgent(script=WIN_SCRIPT)))
    await benchmark.run()
    (_, _, history) = ProbeAgent.finishes[0]

    transition_move = history[2]  # third ACTION1 completes level 0
    assert [obs.outcome for obs in transition_move.observations] == ["level_won", "playing"]
    level_won, new_initial = transition_move.observations
    # Partition rule: preceding frames (here the auto-inserted green win
    # screen) belong to the level-win observation; the final frame is the
    # new level's initial observation.
    assert level_won.frame.data[0, 0] == 14
    assert new_initial.frame.data[0, 0] == 4  # level 1, progress 0
    # Level = the level the observation BELONGS to.
    assert level_won.level == 0
    assert new_initial.level == 1
    assert level_won.caused_by is not None and level_won.caused_by.name == "ACTION1"
    assert new_initial.caused_by is None

    # Every observation is delivered through its own do_move call: the
    # level-win observation first (delivery-only), then the new initial.
    # The history holds older moves compressed, so these are equal copies
    # of what was delivered rather than the same objects.
    for seen, recorded in ((ProbeAgent.seen[3], level_won), (ProbeAgent.seen[4], new_initial)):
        assert (seen.index, seen.outcome, seen.level) == (recorded.index, recorded.outcome, recorded.level)
        assert seen.frame == recorded.frame
    # The delivery-only call's returned action (ProbeAgent acks with
    # RESET) was discarded — the engine saw only the script.
    assert [obs.outcome for obs in ProbeAgent.seen] == [
        "playing",
        "playing",
        "playing",
        "level_won",
        "playing",
        "playing",
        "playing",
    ]

    # Observation indices are a single global counter (initial obs = 0).
    all_observations = [obs for move in history for obs in move.observations]
    assert ProbeAgent.seen[0].index == 0
    assert [obs.index for obs in all_observations] == list(range(1, len(all_observations) + 1))

    # Final move ends the game: single game_won observation, no split, no
    # do_move call (only 7 calls happened).
    assert [obs.outcome for obs in history[-1].observations] == ["game_won"]
    assert history[-1].observations[-1].level == 1
    assert len(ProbeAgent.seen) == 7


async def test_give_up_finishes_game_with_reason() -> None:
    ProbeAgent.reset_log()
    benchmark = Benchmark(
        games=[ExampleGame()], solver=AgentSolver(agent=ProbeAgent(script=("ACTION1",), give_up_reason="bored"))
    )
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "gave_up"
    assert run.solver_note is not None and "bored" in run.solver_note
    (_, result, history) = ProbeAgent.finishes[0]
    assert result.state == "gave_up"
    assert len(history) == 1  # the give-up move itself executes nothing


async def test_lost_is_ordinary_observation_and_reset_recovers() -> None:
    ProbeAgent.reset_log()
    agent = ProbeAgent(script=("ACTION5", "RESET", "ACTION1"))
    benchmark = Benchmark(games=[LosableExampleGame()], solver=AgentSolver(agent=agent))
    await benchmark.run()
    (_, result, history) = ProbeAgent.finishes[0]
    assert result.state == "gave_up"  # script exhausted after recovery
    assert [move.observations[-1].outcome for move in history] == ["lost", "playing", "playing"]
    # The agent saw the lost state and decided RESET itself (ledger item 8:
    # infra never injects actions).
    assert [obs.outcome for obs in ProbeAgent.seen] == ["playing", "lost", "playing", "playing"]
    assert history[1].action.name == "RESET"


async def test_multi_pass_runs_are_independent() -> None:
    ProbeAgent.reset_log()
    benchmark = Benchmark(
        games=[ExampleGame()],
        solver=AgentSolver(agent=ProbeAgent(script=WIN_SCRIPT)),
        n_passes=2,
    )
    await benchmark.run()
    assert [run.state for run in benchmark.game_runs] == ["won", "won"]
    assert len(ProbeAgent.finishes) == 2
    # Each pass got a fresh agent copy off the template (I11): both played
    # the full script from step 0.
    for _, result, history in ProbeAgent.finishes:
        assert result.state == "won"
        assert len(history) == len(WIN_SCRIPT)


async def test_give_up_on_delivery_only_call_is_honored() -> None:
    ProbeAgent.reset_log()
    agent = ProbeAgent(script=WIN_SCRIPT, give_up_on_level_won=True)
    benchmark = Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=agent))
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "gave_up"
    assert run.solver_note is not None and "saw level_won" in run.solver_note
    # Only the first level's script executed; nothing after the give-up.
    assert [record.action.id.name for record in run.history] == ["ACTION1"] * 3


async def test_multiple_games_play_concurrently_with_isolated_agents() -> None:
    ProbeAgent.reset_log()
    # One template, two games with the same action set but opposite target
    # sequences: the script wins game_a, and on game_b runs out after two
    # legal-but-wrong moves -> gave_up. Distinct outcomes prove each game
    # got its own agent copy and history (a shared copy would have burned
    # the script on one game).
    action1, action2 = arcengine.GameAction.ACTION1, arcengine.GameAction.ACTION2
    games: list[Game] = [
        ExampleGame(target_sequence_per_level=[[action1, action2]], label="game_a"),
        ExampleGame(target_sequence_per_level=[[action2, action1]], label="game_b"),
    ]
    agent = ProbeAgent(script=("ACTION1", "ACTION2"))
    benchmark = Benchmark(games=games, solver=AgentSolver(agent=agent))
    await benchmark.run()
    assert {run.game_id: run.state for run in benchmark.game_runs} == {"game_a": "won", "game_b": "gave_up"}
    results = {game_id: (result, history) for game_id, result, history in ProbeAgent.finishes}
    assert set(results) == {"game_a", "game_b"}
    # Both copies played the full script from step 0, each on its own game.
    assert [move.action.name for move in results["game_a"][1]] == ["ACTION1", "ACTION2"]
    assert [move.action.name for move in results["game_b"][1]] == ["ACTION1", "ACTION2"]
    assert results["game_a"][0].state == "won" and results["game_a"][0].levels_completed == 1
    assert results["game_b"][0].state == "gave_up" and results["game_b"][0].levels_completed == 0
