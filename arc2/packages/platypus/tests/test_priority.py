"""``PrioritySlots``: the run's concurrency, granted by rank.

The scheduler this backs decides which games spend the machine's tokens
under a deadline, so the properties here are the ones a wrong grant would
break silently: a freed slot goes to the best-ranked waiter rather than
the longest-waiting one, rank is read at the moment of the grant rather
than from a snapshot, a better rank never takes a slot off a running
game, a yield by the top-ranked game costs it nothing, and a run with no
ranking policy grants in plain FIFO order.

The solver's side of it is here too, where it is about the slots: what a
parked game reports while it waits, when a non-free offer is taken up,
and the two process modes swapping games end to end.
"""

from __future__ import annotations

import asyncio
import itertools
import time
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest
from infra_fixtures import ProbeAgent, mirror

import platypus.solver
import platypus.status
import taaf.benchmark
import taaf.game
import taaf.game_examples
from platypus import GiveUp, MoveContext, RunConfig
from platypus.contract import HANDICAP_HINT, TAKE_INDEX_HINT
from platypus.priority import PrioritySlots
from platypus.solver import AgentSolver, _rank_key


async def _settle() -> None:
    """Let every task that could run, run."""
    for _ in range(4):
        await asyncio.sleep(0)


def _by_spend(key: str, hints: Mapping[str, Any]) -> tuple[Any, ...]:
    """Cheapest reported spend first — the shape of the shipped policy."""
    del key
    return (hints.get("spent", 0),)


async def test_a_freed_slot_goes_to_the_best_ranked_waiter_not_the_oldest() -> None:
    """The whole point: rank beats arrival order.

    ``b`` waits first and ``c`` waits last, and ``c`` ranks ahead — so a
    FIFO semaphore and this must disagree, which is what makes the
    assertion meaningful.
    """
    ranks = {"a": 0, "c": 1, "b": 2}
    slots = PrioritySlots(active=1, rank_key=lambda key, hints: (ranks[key],))
    await slots.acquire("a")
    waiters = [asyncio.create_task(slots.acquire(key)) for key in ("b", "c")]
    await _settle()
    assert not any(task.done() for task in waiters), "one slot, and it is held"

    slots.release("a")
    await _settle()

    assert waiters[1].done() and not waiters[0].done()
    for task in waiters:
        task.cancel()


async def test_rank_is_read_at_the_grant_not_from_a_snapshot() -> None:
    """The property the policy rests on: a game that has just become cheap
    is cheap at its very next comparison.

    A game reports a large spend, is outranked, then reports zero. Were the
    ranking a periodically-recomputed order, the fresh number would sit
    unused until the next tick and the game would hand its slot to a rival
    it now beats — exactly the case the whole scheduler exists to get
    right.
    """
    slots = PrioritySlots(active=1, rank_key=_by_spend)
    await slots.acquire("holder", {"spent": 90_000})
    rival = asyncio.create_task(slots.acquire("rival", {"spent": 50_000}))
    await _settle()

    parks: list[int] = []
    parked = await slots.yield_slot("holder", {"spent": 0}, on_park=lambda: parks.append(1))

    assert not parked and not parks, "the yielder is cheapest now, so it keeps its slot"
    assert not rival.done()
    rival.cancel()


async def test_an_unranked_run_grants_in_arrival_order() -> None:
    """Parity with a plain concurrency cap: no policy, arrival order.

    This is what makes the mechanism safe to land before any policy uses
    it — a run that passes no ``rank_key`` grants in arrival order, which
    is a plain concurrency cap and schedules nothing.
    """
    slots = PrioritySlots(active=1)
    await slots.acquire("first")

    waiters = [asyncio.create_task(slots.acquire(key)) for key in ("second", "third")]
    await _settle()

    slots.release("first")
    await _settle()
    assert waiters[0].done() and not waiters[1].done()

    slots.release("second")
    await _settle()
    assert waiters[1].done()


async def test_an_outranked_yield_hands_the_slot_over_and_gets_it_back() -> None:
    """The swap the whole mechanism exists for, end to end."""
    slots = PrioritySlots(active=1, rank_key=_by_spend)
    await slots.acquire("expensive", {"spent": 90_000})
    waiter = asyncio.create_task(slots.acquire("cheap", {"spent": 1_000}))
    await _settle()

    parks: list[int] = []
    yielding = asyncio.create_task(slots.yield_slot("expensive", {"spent": 95_000}, on_park=lambda: parks.append(1)))
    await _settle()

    assert parks, "on_park fires as it parks, not once the wait is over"
    assert waiter.done() and not yielding.done()

    slots.release("cheap")
    await _settle()
    assert await yielding is True


async def test_a_better_rank_never_takes_a_slot_off_a_running_game() -> None:
    """Rank decides the next grant and nothing else.

    Preemption would cut a turn in half and throw away the tokens already
    spent on it, so a waiter overtaking the holder must leave it running —
    the ranking only says who *should* be running, which is what
    ``admitted`` reports.
    """
    slots = PrioritySlots(active=1, rank_key=_by_spend)
    await slots.acquire("holder", {"spent": 10})
    waiter = asyncio.create_task(slots.acquire("challenger", {"spent": 20}))
    await _settle()

    slots.report("holder", {"spent": 9_000})
    await _settle()

    assert not waiter.done()
    assert slots.admitted() == ["challenger"], "the ranking has changed its mind; the slot has not moved"
    assert slots.running == 1
    waiter.cancel()


async def test_a_retired_game_leaves_the_population_that_is_ranked() -> None:
    """``live_keys`` is the population the ranking ranks, so a finished
    game has to be gone from it — otherwise every comparison keeps giving
    standing to games that will never ask for a slot again."""
    slots = PrioritySlots(active=2, rank_key=_by_spend)
    await slots.acquire("a", {"spent": 5})
    await slots.acquire("b")

    slots.retire("a")

    assert sorted(slots.live_keys()) == ["b"]
    assert slots.hints("a") == {}
    assert slots.running == 1


async def test_a_cancelled_wait_leaves_nothing_queued() -> None:
    """A game cancelled while waiting must not keep a claim on the run.

    Left in the queue it would take the next freed slot and drop it: the
    grant goes to a task that no longer exists, and the slot is lost for
    the rest of the run.
    """
    slots = PrioritySlots(active=1)
    await slots.acquire("holder")
    abandoned = asyncio.create_task(slots.acquire("gone"))
    survivor = asyncio.create_task(slots.acquire("stays"))
    await _settle()

    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned

    slots.release("holder")
    await _settle()

    assert survivor.done()
    assert slots.waiting == 0


async def test_reports_do_not_move_slots() -> None:
    """Hints arrive far more often than slots change hands; recording one
    must not be a scheduling event of its own."""
    slots = PrioritySlots(active=1, rank_key=_by_spend)
    await slots.acquire("running", {"spent": 1})
    waiter = asyncio.create_task(slots.acquire("queued"))
    await _settle()

    slots.report("running", {"spent": 50_000})

    assert slots.hints("running") == {"spent": 50_000}
    assert not waiter.done()
    waiter.cancel()


def test_least_tokens_ranks_by_spend_then_by_play_order() -> None:
    """The shipped policy: fewest tokens spent first, the run's own order
    breaking ties — so a game that has never reported is at zero and ahead
    of any game that has spent, and two unplayed games keep their order.
    ``queue_order`` is that order alone, whatever anyone reports."""
    games: list[taaf.game.Game] = [taaf.game_examples.flip_game(f"g{i}") for i in range(3)]
    ranked = _rank_key(RunConfig(slot_policy="least_tokens"), games)
    assert ranked("2", {}) < ranked("0", {"tokens": 1, "compacted": False})
    assert ranked("1", {"tokens": 500}) < ranked("0", {"tokens": 501})
    assert ranked("0", {"tokens": 7}) < ranked("1", {"tokens": 7})

    fifo = _rank_key(RunConfig(slot_policy="queue_order"), games)
    assert fifo("0", {"tokens": 10**9}) < fifo("1", {})


def test_a_game_that_asks_to_rank_further_back_does() -> None:
    """``HANDICAP_HINT`` adds to the spend on top of the take handicap, so a
    game that asks for 10M ranks behind every game that has not, whatever they
    have spent, and among others that asked by spend."""
    ranked = _rank_key(RunConfig(slot_policy="least_tokens", take_handicap_tokens=150_000), [])
    deferred = {"tokens": 10, TAKE_INDEX_HINT: 1, HANDICAP_HINT: 10_000_000}
    assert ranked("1", {"tokens": 900_000, TAKE_INDEX_HINT: 1}) < ranked("0", deferred)
    assert ranked("1", {"tokens": 5, TAKE_INDEX_HINT: 1, HANDICAP_HINT: 10_000_000}) < ranked("0", deferred)
    assert ranked("0", {"tokens": 10, HANDICAP_HINT: 0}) == ranked("0", {"tokens": 10})


async def test_retiring_a_game_that_is_waiting_gives_up_its_place_too() -> None:
    """A game can be retired while it sits in the queue rather than on a
    slot, and the queue entry has to go with it.

    Left behind, the next freed slot is granted to a future nobody is
    waiting on: the slot is consumed for the rest of the run, and enough
    of them stall it outright.
    """
    slots = PrioritySlots(active=1)
    await slots.acquire("holder")
    queued = asyncio.create_task(slots.acquire("gone"))
    await _settle()
    assert slots.waiting == 1

    slots.retire("gone")
    slots.release("holder")
    await _settle()

    assert slots.running == 0, "a retired waiter must not take the freed slot"
    assert slots.waiting == 0
    queued.cancel()


async def test_a_pause_callback_that_raises_does_not_cost_the_run_a_slot() -> None:
    """``on_park`` prints, so a closed stdout raises straight through it.

    The yield then unwinds into the game's own hosting, which retires it —
    from the queue it is parked in, not from the slot it has already given
    away. This is the path that makes retiring a waiter reachable in
    production rather than only in theory.
    """
    slots = PrioritySlots(active=1, rank_key=_by_spend)
    await slots.acquire("holder", {"spent": 0})
    rival = asyncio.create_task(slots.acquire("rival", {"spent": 0}))
    await _settle()

    def closed_stdout() -> None:
        raise RuntimeError("stdout is closed")

    with pytest.raises(RuntimeError):
        await slots.yield_slot("holder", {"spent": 10}, on_park=closed_stdout)
    slots.retire("holder")
    await _settle()

    assert rival.done(), "the yielder handed its slot over before the callback ran"
    slots.release("rival")
    await _settle()
    assert (slots.running, slots.waiting) == (0, 0), "the run got its slot back"


def _solver_with_two_keys(config: RunConfig) -> AgentSolver:
    """A solver holding a live scheduler with ``0`` (holder) and ``1``
    (rival) registered, as ``_fan_out`` would leave it."""
    solver = AgentSolver(agent=ProbeAgent())
    solver._priority = PrioritySlots(config.max_concurrent)
    solver._slot_labels = {"0": "holder", "1": "rival"}
    for key, game_id in solver._slot_labels.items():
        solver._status.register(key, game_id)
    return solver


async def test_a_resumed_game_still_reports_what_it_was_doing() -> None:
    """A pause must not cost the agent its published label.

    Moving between coarse states clears the refinement, and the agent's
    handle publishes only on change — it cannot know the registry dropped
    a label underneath it, so a game that resumes inside the same status
    scope would report the bare coarse state until its next label change,
    which for a duck mid-turn is minutes of a run reading as unattributed
    ``running``. The registry is parent-side in both process modes, so
    this one path covers both.
    """
    config = RunConfig(max_concurrent=1)
    solver = _solver_with_two_keys(config)
    assert solver._priority is not None
    await solver._priority.acquire("0")
    solver._status.set_coarse("0", platypus.status.RUNNING)
    solver._status.set_detail("0", "waiting for LLM response")

    rival = asyncio.create_task(solver._priority.acquire("1"))
    await _settle()

    yield_slot = solver._yield_handle(taaf.game_examples.flip_game("g"), "0", config)

    async def offer() -> None:
        """The game offering its slot back, from inside its own move."""
        await yield_slot({"tokens": 10, "compacted": True})

    parking = asyncio.create_task(offer())
    await _settle()
    assert solver._status.snapshot()[0].label == "paused", "not running, and not its old label"

    solver._priority.release("1")
    await parking

    assert solver._status.snapshot()[0].label == "waiting for LLM response"
    rival.cancel()


async def test_a_non_free_offer_is_taken_only_after_the_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """An offer that is not free (no compaction behind it) is declined until
    the game has held its slot for ``slot_hold_s``, and the clock restarts
    at the grant — so the hold bounds one game's tenure without letting an
    offer made right after a grant cost it the slot again.
    """
    now = 1000.0
    # The solver's own clock, not the process's: asyncio keeps time too.
    monkeypatch.setattr(platypus.solver, "time", types.SimpleNamespace(monotonic=lambda: now))
    config = RunConfig(max_concurrent=1, slot_hold_s=60.0)
    solver = _solver_with_two_keys(config)
    assert solver._priority is not None
    await solver._priority.acquire("0")
    rival = asyncio.create_task(solver._priority.acquire("1"))
    await _settle()
    yield_slot = solver._yield_handle(taaf.game_examples.flip_game("g"), "0", config)

    now = 1059.0
    await yield_slot({"tokens": 5, "compacted": False})
    assert not rival.done(), "declined: the hold has not elapsed and the swap is not free"

    now = 1060.0

    async def offer() -> None:
        """A non-free offer, once the hold has elapsed."""
        await yield_slot({"tokens": 5, "compacted": False})

    parking = asyncio.create_task(offer())
    await _settle()
    assert rival.done() and not parking.done(), "taken: the hold elapsed"

    solver._priority.release("1")
    await parking
    # The grant reset the clock: an immediate non-free offer is declined again.
    now = 1061.0
    other = asyncio.create_task(solver._priority.acquire("1"))
    await _settle()
    await yield_slot({"tokens": 6, "compacted": False})
    assert not other.done()
    other.cancel()


async def test_no_hold_declines_a_non_free_offer_however_long_the_game_has_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``slot_hold_s=None`` leaves the free points as the only swap points.

    A submission's deadline does not move, so a non-free swap buys the run
    nothing and costs the parked game the context it had built. The default
    therefore retires the fallback rather than setting it past the length
    of a submission, and a game that never compacts keeps its slot until it
    finishes.
    """
    now = 1000.0
    monkeypatch.setattr(platypus.solver, "time", types.SimpleNamespace(monotonic=lambda: now))
    config = RunConfig(max_concurrent=1, slot_hold_s=None)
    solver = _solver_with_two_keys(config)
    assert solver._priority is not None
    await solver._priority.acquire("0")
    rival = asyncio.create_task(solver._priority.acquire("1"))
    await _settle()
    yield_slot = solver._yield_handle(taaf.game_examples.flip_game("g"), "0", config)

    now = 1000.0 + 100 * 3600
    await yield_slot({"tokens": 5, "compacted": False})
    assert not rival.done(), "declined: no hold can elapse, and the swap is not free"

    async def offer() -> None:
        """The same game at a free point: a compaction."""
        await yield_slot({"tokens": 5, "compacted": True})

    parking = asyncio.create_task(offer())
    await _settle()
    assert rival.done(), "taken: a free offer is still taken"
    solver._priority.release("1")
    await parking


def test_a_misspelt_slot_policy_is_refused_at_setup() -> None:
    """A typo must not silently schedule a run that asked not to be.

    ``RunConfig`` is mutable and its ``Literal`` is not a runtime check,
    so ``"queue-order"`` reaches the solver looking like a request for
    the neutral policy and would fall through to the ranking branch —
    changing which games get the GPU under a deadline, in a run whose
    author believed they had opted out.
    """
    agent = ProbeAgent()
    agent.run_config = RunConfig(slot_policy="queue-order")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="not a policy"):
        AgentSolver(agent=agent)._setup()


@dataclass
class YieldingAgent(ProbeAgent):
    """Offers its slot back at every round, each round costing 100 more
    tokens, and records the order in which its rounds actually ran.

    In-process the record is the class-level list; a child cannot reach
    that, so with ``log_dir`` set each round also writes one file whose
    name sorts by wall clock — rounds are strictly sequential under one
    slot, so the names order the same way the class list would.
    """

    rounds: int = 3
    log_dir: str = ""
    order: ClassVar[list[str]] = []

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Play ``rounds`` rounds, offering the slot back after each."""
        for i in range(self.rounds):
            # A real round suspends on its LLM call; without a suspension
            # here an in-process game reaches its first offer before its
            # siblings have so much as queued, and has nobody to yield to.
            await asyncio.sleep(0)
            event = f"{ctx.game_id}:{i}"
            YieldingAgent.order.append(event)
            if self.log_dir:
                (Path(self.log_dir) / f"{time.time_ns():020d}_{event}").write_text("")
            await ctx.yield_slot({"tokens": (i + 1) * 100, "compacted": True})
        ctx.submit(mirror(view))
        return None


def _benchmark(agent: ProbeAgent, n_games: int = 3) -> taaf.benchmark.Benchmark:
    """``n_games`` flip tasks hosted by ``agent``."""
    return taaf.benchmark.Benchmark(
        label="t",
        games=[taaf.game_examples.flip_game(f"g{i}") for i in range(n_games)],
        solver=AgentSolver(agent=agent),
    )


def _round_robin(n_games: int, rounds: int) -> list[str]:
    """Every game's round ``i`` before any game's round ``i + 1``."""
    return [f"g{g}:{i}" for i in range(rounds) for g in range(n_games)]


def _one_after_another(n_games: int, rounds: int) -> list[str]:
    """Each game plays every round before the next game starts."""
    return [f"g{g}:{i}" for g in range(n_games) for i in range(rounds)]


async def test_least_tokens_round_robins_games_that_yield_at_compaction(capsys: pytest.CaptureFixture[str]) -> None:
    """Three games, one slot, every offer free: under ``least_tokens`` the
    slot goes round the games a round at a time, because after each round
    the holder has spent more than the two behind it; under
    ``queue_order`` the same offers are all handed straight back and each
    game plays out before the next starts. Every swap is announced.
    """
    YieldingAgent.order.clear()
    ranked = YieldingAgent(
        run_config=RunConfig(
            process_mode="inprocess", status_interval_s=None, max_concurrent=1, slot_policy="least_tokens"
        )
    )
    bm = _benchmark(ranked)
    await bm.run()
    assert YieldingAgent.order == _round_robin(3, 3)
    assert all(run.state == "won" for run in bm.game_runs)
    out = capsys.readouterr().out
    # Every offer parks: after each round the yielder has spent more than a
    # waiter, or ties one that queued earlier.
    assert out.count("has been paused") == out.count("has been resumed") == 9, out
    assert "[slots] (g0 finished)" in out

    YieldingAgent.order.clear()
    fifo = YieldingAgent(run_config=RunConfig(process_mode="inprocess", status_interval_s=None, max_concurrent=1))
    await _benchmark(fifo).run()
    assert YieldingAgent.order == _one_after_another(3, 3)
    assert "has been paused" not in capsys.readouterr().out


async def test_a_child_parks_and_resumes_across_the_pipe(tmp_path: Path) -> None:
    """The same round-robin with the agents in child processes: a yield
    crosses the pipe as a request, the child blocks in it while parked,
    and the parent's re-grant is what lets it run again."""
    agent = YieldingAgent(
        log_dir=str(tmp_path),
        run_config=RunConfig(
            process_mode="subprocess", status_interval_s=None, max_concurrent=1, slot_policy="least_tokens"
        ),
    )
    bm = _benchmark(agent)
    await bm.run()
    recorded = [path.name.split("_", 1)[1] for path in sorted(tmp_path.iterdir())]
    assert recorded == _round_robin(3, 3)
    assert all(run.state == "won" and run.final_score == 100.0 for run in bm.game_runs)


async def test_cancelling_a_run_unwinds_a_parked_child(tmp_path: Path) -> None:
    """A parked child is blocked in a synchronous wait on the parent; the
    cancellation has to be raised into that wait, or the child sits out
    the whole grace window and the run cannot end."""
    agent = YieldingAgent(
        rounds=1000,
        log_dir=str(tmp_path),
        run_config=RunConfig(
            process_mode="subprocess", status_interval_s=None, max_concurrent=1, slot_policy="least_tokens"
        ),
    )
    bm = _benchmark(agent, n_games=2)
    task = asyncio.create_task(bm.run())
    while len(list(tmp_path.iterdir())) < 4:  # both games have started and swapped at least once
        await asyncio.sleep(0.05)
    task.cancel()
    started = time.monotonic()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - started < 30, "the parked child did not unwind promptly"
    assert all(run.state == "cancelled" for run in bm.game_runs)


async def test_start_pacer_spaces_starts_by_the_gap() -> None:
    from platypus.priority import StartPacer

    loop = asyncio.get_running_loop()

    async def run(gap_s: float) -> list[float]:
        pacer = StartPacer(gap_s)
        stamps: list[float] = []

        async def start() -> None:
            await pacer.wait()
            stamps.append(loop.time())

        await asyncio.gather(*(start() for _ in range(4)))
        return stamps

    spaced = await run(0.02)
    gaps = [b - a for a, b in itertools.pairwise(spaced)]
    assert len(gaps) == 3 and all(gap >= 0.02 * 0.9 for gap in gaps)
    # A zero gap is a pass-through, not a serialized queue.
    burst = await run(0.0)
    assert max(burst) - min(burst) < 0.02
