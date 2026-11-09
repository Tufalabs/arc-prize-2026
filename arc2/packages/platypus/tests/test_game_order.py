"""Playing the set in a random order, and naming the order afterwards.

A run that cannot finish its set sees only a prefix of it, and the
benchmark's order decides which games that is — the same prefix every
time. Shuffling makes the sample vary; recording the seed makes it
reproducible.
"""

from __future__ import annotations

import datetime

from infra_fixtures import ProbeAgent

import taaf.benchmark
import taaf.game
from platypus import AgentSolver, RunConfig
from taaf.game_examples import flip_game


async def _play(**config: object) -> list[str]:
    """The order the agent actually saw, game by game."""
    ProbeAgent.reset_log()
    # In-process: the log is class state, which a spawned child could not
    # write where this process reads it. max_concurrent=1 so the finishing
    # order IS the play order.
    agent = ProbeAgent(
        run_config=RunConfig(process_mode="inprocess", status_interval_s=None, max_concurrent=1, **config),  # type: ignore[arg-type]
    )
    games: list[taaf.game.Game] = [flip_game(f"g{index}") for index in range(6)]
    await taaf.benchmark.Benchmark(label="order", games=games, solver=AgentSolver(agent=agent)).run()
    return [game_id for game_id, _, _ in ProbeAgent.finishes]


async def test_off_by_default_keeps_the_benchmarks_order() -> None:
    """A reference run's order is part of what it pins, so the default has
    to be the benchmark's own."""
    assert await _play() == [f"g{index}" for index in range(6)]


async def test_a_seed_fixes_the_order() -> None:
    """Same seed, same order — what lets a shuffled run be rerun, and two
    arms be compared on the same sample."""
    first = await _play(shuffle_games=True, shuffle_seed=20260701)
    second = await _play(shuffle_games=True, shuffle_seed=20260701)
    assert first == second, (first, second)
    assert sorted(first) == [f"g{index}" for index in range(6)], first


async def test_different_seeds_give_different_orders() -> None:
    """Otherwise the knob is decoration. Six games is 720 orders, so a
    collision between two fixed seeds would be a real failure, not luck."""
    assert await _play(shuffle_games=True, shuffle_seed=1) != await _play(shuffle_games=True, shuffle_seed=2)


async def test_no_seed_uses_todays_date() -> None:
    """The default an unattended submission gets, and still nameable
    afterwards: resubmitting on the same day replays the same order."""
    today = datetime.date.today()
    expected = today.year * 10000 + today.month * 100 + today.day
    assert await _play(shuffle_games=True) == await _play(shuffle_games=True, shuffle_seed=expected)


async def test_play_priority_starts_the_highest_first() -> None:
    """A run that knows its long tasks starts them first."""
    order = await _play(play_priority={"g4": 9.0, "g1": 5.0})
    assert order == ["g4", "g1", "g0", "g2", "g3", "g5"]


async def test_play_priority_keeps_the_shuffle_within_a_priority() -> None:
    """Applied after the shuffle and stably: games of one priority keep the
    shuffled order among themselves, so arms of one task stay mixed."""
    shuffled = await _play(shuffle_games=True, shuffle_seed=7)
    order = await _play(shuffle_games=True, shuffle_seed=7, play_priority={"g4": 9.0, "g1": 5.0})
    assert order[:2] == ["g4", "g1"]
    assert order[2:] == [game for game in shuffled if game not in ("g4", "g1")]
