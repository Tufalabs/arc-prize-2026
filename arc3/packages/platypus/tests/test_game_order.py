"""Playing the set in a random order, and naming the order afterwards.

A run that cannot finish its set sees only a prefix of it, and the
benchmark's order decides which games that is — the same prefix every
time. Shuffling makes the sample vary; recording the seed makes it
reproducible. A priority starts the games a run names first, so a run
that knows its long games does not end on a tail of them.
"""

from __future__ import annotations

import datetime

import taaf.game
from infra_fixtures import ProbeAgent
from taaf.benchmark import Benchmark
from taaf.game_api import GameAPI
from taaf.game_examples import ExampleGame

from platypus import AgentSolver, RunConfig
from platypus.solver import _planned_game_id  # pyright: ignore[reportPrivateUsage]

WIN_SCRIPT = ("ACTION1",) * 3 + ("ACTION2",) * 3


def _games(count: int) -> list[taaf.game.Game]:
    return [ExampleGame(label=f"g{index}") for index in range(count)]


async def _play(**config: object) -> list[str]:
    """The order the agent actually saw, game by game."""
    ProbeAgent.reset_log()
    # inprocess: ProbeAgent's log is class state, so a spawned child would
    # record where this process cannot read it. max_concurrent=1 so the
    # finish order IS the play order.
    agent = ProbeAgent(
        script=WIN_SCRIPT,
        run_config=RunConfig(process_mode="inprocess", max_concurrent=1, **config),  # type: ignore[arg-type]
    )
    await Benchmark(games=_games(6), solver=AgentSolver(agent=agent)).run()
    return [game_id for game_id, _, _ in ProbeAgent.finishes]


async def test_off_by_default_keeps_the_benchmarks_order() -> None:
    """The default has to be the old behaviour: a reference run's order is
    part of what it pins."""
    assert await _play() == [f"g{index}" for index in range(6)]


async def test_a_seed_fixes_the_order() -> None:
    """Same seed, same order — the property that lets a shuffled run be
    rerun, and two arms be compared on the same sample."""
    first = await _play(shuffle_games=True, shuffle_seed=20260701)
    second = await _play(shuffle_games=True, shuffle_seed=20260701)
    assert first == second, (first, second)
    assert sorted(first) == [f"g{index}" for index in range(6)], first


async def test_different_seeds_give_different_orders() -> None:
    """Otherwise the knob is decoration. Six games is 720 orders, so a
    collision between two fixed seeds would be a real failure, not luck."""
    assert await _play(shuffle_games=True, shuffle_seed=1) != await _play(shuffle_games=True, shuffle_seed=2)


async def test_no_seed_uses_todays_date() -> None:
    """The default a Kaggle submission gets: unattended, and still nameable
    afterwards — resubmitting on the same day replays the same order."""
    today = datetime.date.today()
    expected = today.year * 10000 + today.month * 100 + today.day
    assert await _play(shuffle_games=True) == await _play(shuffle_games=True, shuffle_seed=expected)


async def test_the_heartbeat_is_drawn_from_the_shuffle() -> None:
    """The sacrifice is whichever entry lands first, not the benchmark's
    first game — otherwise one game is always the one nobody plays."""
    played = await _play(shuffle_games=True, shuffle_seed=20260701, heartbeat_game=True)
    assert len(played) == 5, played
    unplayed = set(f"g{index}" for index in range(6)) - set(played)
    assert unplayed != {"g0"}, "the shuffle did not reach the heartbeat's pick"


async def test_play_priority_starts_the_highest_first() -> None:
    """A run that knows its long games starts them first."""
    order = await _play(play_priority={"g4": 9.0, "g1": 5.0})
    assert order == ["g4", "g1", "g0", "g2", "g3", "g5"]


async def test_play_priority_keeps_the_shuffle_within_a_priority() -> None:
    """Applied after the shuffle and stably: games of one priority keep the
    shuffled order among themselves, so arms of one game stay mixed."""
    shuffled = await _play(shuffle_games=True, shuffle_seed=7)
    order = await _play(shuffle_games=True, shuffle_seed=7, play_priority={"g4": 9.0, "g1": 5.0})
    assert order[:2] == ["g4", "g1"]
    assert order[2:] == [game for game in shuffled if game not in ("g4", "g1")]


def test_an_unstarted_engine_game_is_named_by_the_id_it_starts_under() -> None:
    """The priority is read before any game starts, when TAAF's games have
    no ``game_id`` yet: an engine game goes by its env's id, or by the
    external id it is given."""
    assert _planned_game_id(GameAPI(env_name="bp35-0a0ad940")) == "bp35-0a0ad940"
    assert _planned_game_id(GameAPI(env_name="ls20", external_game_id="ls20-pass2")) == "ls20-pass2"
    assert _planned_game_id(ExampleGame(label="g3")) == "g3"
