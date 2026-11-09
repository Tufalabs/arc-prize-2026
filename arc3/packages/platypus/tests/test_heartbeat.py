"""The sacrificed liveness game: infra plays it, the agent never sees it.

Kaggle's arena ends a run that goes too long without a move, and a real
agent's first turn can exceed that on its own. One game is given up to
keep the run alive.
"""

from __future__ import annotations

import asyncio

import arcengine
import pytest
import taaf.game
from infra_fixtures import ProbeAgent, SlowAgent
from taaf.benchmark import Benchmark
from taaf.game_examples import ExampleGame

from platypus import AgentSolver, RunConfig

WIN_SCRIPT = ("ACTION1",) * 3 + ("ACTION2",) * 3


def _games(count: int) -> list[taaf.game.Game]:
    return [ExampleGame(label=f"g{index}") for index in range(count)]


def _unwinnable_first(count: int) -> list[taaf.game.Game]:
    """Games where the sacrificed one (``g0``) cannot be won by luck.

    ``ExampleGame``'s default target is three of one action then three of
    another, which random play beats often enough to make any assertion
    about the heartbeat's end state a coin flip. A 16-long target over
    three available actions puts that at ~4e-8 per attempt. The agent's
    own games keep the default so ``WIN_SCRIPT`` still wins them."""
    hard = [[arcengine.GameAction.ACTION1, arcengine.GameAction.ACTION2] * 8]
    games: list[taaf.game.Game] = [ExampleGame(label="g0", target_sequence_per_level=hard)]
    games.extend(ExampleGame(label=f"g{index}") for index in range(1, count))
    return games


def _config(**overrides: object) -> RunConfig:
    return RunConfig(process_mode="inprocess", heartbeat_game=True, heartbeat_interval_s=0.05, **overrides)  # type: ignore[arg-type]


async def test_the_agent_is_handed_every_game_but_the_first() -> None:
    """The contract for any agent: a shorter list, and no way to tell why.
    Nothing about the heartbeat reaches agent code, which is what makes it
    work for agents that do not exist yet."""
    ProbeAgent.reset_log()
    agent = ProbeAgent(script=WIN_SCRIPT, run_config=_config())
    benchmark = Benchmark(games=_games(3), solver=AgentSolver(agent=agent))
    await benchmark.run()

    played = {game_id for game_id, _, _ in ProbeAgent.finishes}
    assert played == {"g1", "g2"}, played


async def test_the_sacrificed_game_is_played_and_closed_out() -> None:
    """It has to actually move — that is the entire point — and it has to
    end in a terminal state of its own. A game left ``playing`` would be
    stamped ``crashed`` by the benchmark's teardown, which would report a
    failure where there was none.

    ``move_delay_s`` keeps the agent's games in flight long enough for the
    heartbeat to tick; a scripted win otherwise finishes the whole run
    before the first interval elapses."""
    agent = ProbeAgent(script=WIN_SCRIPT, move_delay_s=0.1, run_config=_config())
    benchmark = Benchmark(games=_unwinnable_first(3), solver=AgentSolver(agent=agent))
    await benchmark.run()

    heartbeat_run = next(run for run in benchmark.game_runs if run.game_id == "g0")
    assert heartbeat_run.history, "the heartbeat game never moved"
    assert heartbeat_run.state == "gave_up", heartbeat_run.state


async def test_a_lucky_random_win_stops_the_heartbeat_cleanly() -> None:
    """Random play beats the default toy game fairly often, and the loop
    must notice: continuing to act on a finished game would raise on every
    tick for the rest of the run."""
    agent = ProbeAgent(script=WIN_SCRIPT, move_delay_s=0.05, run_config=_config())
    benchmark = Benchmark(games=_games(3), solver=AgentSolver(agent=agent))
    await benchmark.run()

    heartbeat_run = next(run for run in benchmark.game_runs if run.game_id == "g0")
    assert heartbeat_run.state in ("won", "gave_up"), heartbeat_run.state
    assert heartbeat_run.solver_note is None, heartbeat_run.solver_note


async def test_it_keeps_moving_while_the_agent_is_stuck_on_its_first_turn() -> None:
    """The situation the feature exists for: the agent is silent for
    longer than the watchdog window, and the arena has to see moves anyway.
    Cancellation stands in for the watchdog that would otherwise fire."""
    agent = SlowAgent(run_config=_config())
    benchmark = Benchmark(games=_games(3), solver=AgentSolver(agent=agent))
    task = asyncio.create_task(benchmark.run())
    await asyncio.sleep(1.0)
    moves_while_stuck = len(next(r for r in benchmark.game_runs if r.game_id == "g0").history)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert moves_while_stuck >= 5, f"only {moves_while_stuck} heartbeat moves in 1s at a 0.05s interval"
    # The agent's own games made no move at all in that window — so the
    # activity the arena saw came from the heartbeat and nothing else.
    assert all(not run.history for run in benchmark.game_runs if run.game_id != "g0")


async def test_off_by_default_every_game_goes_to_the_agent() -> None:
    ProbeAgent.reset_log()
    agent = ProbeAgent(script=WIN_SCRIPT, run_config=RunConfig(process_mode="inprocess"))
    benchmark = Benchmark(games=_games(3), solver=AgentSolver(agent=agent))
    await benchmark.run()
    assert {game_id for game_id, _, _ in ProbeAgent.finishes} == {"g0", "g1", "g2"}


async def test_sacrificing_the_only_game_is_refused() -> None:
    """Silently playing nothing would look like a working run that scored
    zero — the most expensive kind of misconfiguration to debug."""
    agent = ProbeAgent(script=WIN_SCRIPT, run_config=_config(on_agent_error="raise"))
    benchmark = Benchmark(games=_games(1), solver=AgentSolver(agent=agent))
    await benchmark.run()
    # Benchmark contains solver exceptions rather than re-raising them
    # (R2.15), so the refusal surfaces as a run that played nothing and was
    # stamped crashed at teardown — loud, and impossible to mistake for a
    # legitimate zero.
    assert not benchmark.game_runs[0].history
    assert benchmark.game_runs[0].state == "crashed"


async def test_other_passes_of_the_sacrificed_game_still_reach_the_agent() -> None:
    """Only one (game, pass) entry is given up. Dropping every pass of that
    game would cost n_passes games instead of one."""
    ProbeAgent.reset_log()
    agent = ProbeAgent(script=WIN_SCRIPT, run_config=_config())
    benchmark = Benchmark(games=_games(2), solver=AgentSolver(agent=agent), n_passes=2)
    await benchmark.run()

    played = [game_id for game_id, _, _ in ProbeAgent.finishes]
    assert sorted(played) == ["g0", "g1", "g1"], played


async def test_the_heartbeat_game_is_reported_like_any_other(capsys: pytest.CaptureFixture[str]) -> None:
    """A run of N games that hosts N-1 would otherwise read as having lost
    one: the status line's count is the first thing anyone checks, and a
    silent discrepancy there costs more than the line is worth."""
    agent = ProbeAgent(script=WIN_SCRIPT, move_delay_s=0.15, run_config=_config(status_interval_s=0.2))
    benchmark = Benchmark(games=_unwinnable_first(3), solver=AgentSolver(agent=agent))
    await benchmark.run()

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[games]")]
    assert any("3 games" in line for line in lines), lines
    assert any("1 heartbeat" in line for line in lines), lines


if __name__ == "__main__":
    pytest.main([__file__])
