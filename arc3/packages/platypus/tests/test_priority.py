"""``PrioritySlots``: the run's concurrency, granted by rank (I1 hosting).

The scheduler this backs decides which games spend the machine's tokens
under a deadline, so the properties here are the ones a wrong grant would
break silently: a freed slot goes to the best-ranked waiter rather than
the longest-waiting one, rank is read at the moment of the grant rather
than from a snapshot, a better rank never takes a slot off a running
game, a yield by the top-ranked game costs it nothing, and a run with no
ranking policy grants in plain FIFO order.

The solver's side of it is here too, where it is about the slots: what a
parked game reports while it waits.
"""

from __future__ import annotations

import asyncio
import types
from typing import Any, Mapping

import pytest
import taaf.game
from infra_fixtures import ProbeAgent
from taaf.game_examples import ExampleGame

import platypus.solver
import platypus.status
from platypus.contract import RunConfig
from platypus.priority import PrioritySlots
from platypus.solver import AgentSolver, _rank_key, _total_level_weight, charged_tokens


async def _settle() -> None:
    """Let every task that could run, run."""
    for _ in range(4):
        await asyncio.sleep(0)


def _example_game(levels: int) -> taaf.game.Game:
    """A stand-in carrying only the level count the weighting reads."""
    game = ExampleGame(label="g")
    game.number_of_levels = levels
    return game


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

    A game reports a large spend, is outranked, then wins its level and
    reports zero. Were the ranking a periodically-recomputed order, the
    fresh number would sit unused until the next tick and the game would
    hand its slot to a rival it now beats — exactly the case the whole
    scheduler exists to get right.
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


async def test_every_grant_names_the_slot_it_took() -> None:
    """Slot provenance (what the traces chain into lanes): an opening grant
    takes a slot nobody held, a handover names the yielder or the finisher,
    and a yield nothing outranks is the yielder's own slot back."""
    grants: list[tuple[str, str | None]] = []
    slots = PrioritySlots(active=1, rank_key=_by_spend, on_grant=lambda key, freed: grants.append((key, freed)))
    await slots.acquire("expensive", {"spent": 90_000})
    waiter = asyncio.create_task(slots.acquire("cheap", {"spent": 1_000}))
    await _settle()
    yielding = asyncio.create_task(slots.yield_slot("expensive", {"spent": 95_000}))
    await _settle()
    assert waiter.done()
    await slots.yield_slot("cheap", {"spent": 2_000})  # still the cheapest: kept
    slots.retire("cheap")
    await _settle()
    assert await yielding is True
    assert grants == [
        ("expensive", None),
        ("cheap", "expensive"),
        ("cheap", "cheap"),
        ("expensive", "cheap"),
    ]


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


def test_level_weighting_of_zero_is_the_raw_spend() -> None:
    """The default has to leave the charge equal to the tokens.

    ``(weight / depth) ** 0`` is 1 whatever the game — asserted on a deep
    level of a long game, where every factor it could get wrong is
    non-trivial rather than incidentally 1.
    """
    assert charged_tokens({"level": 3, "level_tokens": 40_000}, weight=21, weighting=0.0) == 40_000.0


def test_level_weighting_of_one_prices_a_token_by_what_its_level_is_worth() -> None:
    """The worked example the weighting is defined by, in code.

    ARC gives level ``i`` the share ``i / total_weight`` of a game's
    score, so a token there is charged ``total_weight / i``: on a 6-level
    game (total weight 21) 21 on level 1 and 10.5 on level 2. Both halves
    of what that buys are asserted — the same tokens are charged less
    deeper into a game, and less in a shallower game — because a sign
    error in either would still produce a plausible-looking ordering.
    """
    assert _total_level_weight(_example_game(6)) == 21, "n(n+1)/2"
    assert _total_level_weight(_example_game(3)) == 6

    # level counts levels WON, so 0 is playing level 1 and 1 is level 2.
    shallow = charged_tokens({"level": 0, "level_tokens": 1_000}, weight=21, weighting=1.0)
    deep = charged_tokens({"level": 1, "level_tokens": 1_000}, weight=21, weighting=1.0)
    assert (shallow, deep) == (21_000.0, 10_500.0), "21 and 10.5 a token"

    short_game = charged_tokens({"level": 0, "level_tokens": 1_000}, weight=6, weighting=1.0)
    assert short_game < shallow, "the same level is cheaper to sit on in a shallower game"


def test_a_weighting_above_one_sharpens_the_same_ordering() -> None:
    """Values past 1 are allowed and must keep meaning more of the same
    thing — the discount for depth compounds rather than reversing."""
    spend = {"level": 2, "level_tokens": 1_000}
    at_one = charged_tokens(spend, weight=21, weighting=1.0)
    at_two = charged_tokens(spend, weight=21, weighting=2.0)

    assert at_two > at_one > 1_000.0


def test_a_game_that_has_not_started_yet_still_ranks() -> None:
    """TAAF leaves the level count at 0 until a game starts, and the
    ranking runs at every grant — including before that.

    Weight 1 prices such a game's tokens like anyone's; the one thing it
    must not do is divide by zero. It costs nothing in practice because a
    game that has not started has spent nothing to price.
    """
    rank = _rank_key(RunConfig(slot_policy="least_level_tokens", scheduler_level_weighting=1.0), [_example_game(0)])

    assert rank("0", {"level": 0, "level_tokens": 500})[0] == 500.0


def test_least_game_tokens_ranks_on_the_whole_spend_alone() -> None:
    """The least-served game first, whatever level it stands on: a game
    that has won its levels cheaply is not favoured for it, and the level
    weighting, which prices a level's tokens, does not enter."""
    rank = _rank_key(
        RunConfig(slot_policy="least_game_tokens", scheduler_level_weighting=1.0),
        [_example_game(6), _example_game(6)],
    )
    fresh_level_rich_game = {"level": 4, "level_tokens": 100, "game_tokens": 90_000}
    stuck_poor_game = {"level": 0, "level_tokens": 40_000, "game_tokens": 40_000}

    assert rank("1", stuck_poor_game) < rank("0", fresh_level_rich_game)
    assert rank("0", {"level": 0, "level_tokens": 0})[0] == 0, "a game that never reported ranks at zero"


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

    yield_slot = solver._yield_handle(_example_game(2), "0", config)

    async def offer() -> None:
        """The game offering its slot back, from inside its own move."""
        await yield_slot({"level": 0, "level_tokens": 10, "compacted": True})

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
    yield_slot = solver._yield_handle(_example_game(2), "0", config)

    now = 1059.0
    await yield_slot({"level": 0, "level_tokens": 5, "compacted": False})
    assert not rival.done(), "declined: the hold has not elapsed and the swap is not free"

    now = 1060.0

    async def offer() -> None:
        """A non-free offer, once the hold has elapsed."""
        await yield_slot({"level": 0, "level_tokens": 5, "compacted": False})

    parking = asyncio.create_task(offer())
    await _settle()
    assert rival.done() and not parking.done(), "taken: the hold elapsed"

    solver._priority.release("1")
    await parking
    # The grant reset the clock: an immediate non-free offer is declined again.
    now = 1061.0
    other = asyncio.create_task(solver._priority.acquire("1"))
    await _settle()
    await yield_slot({"level": 0, "level_tokens": 6, "compacted": False})
    assert not other.done()
    other.cancel()


async def test_no_hold_declines_a_non_free_offer_however_long_the_game_has_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``slot_hold_s=None``, the default, leaves the free points as the only
    swap points: a game that never compacts keeps its slot however long it
    has held it, and a free offer is still taken.
    """
    now = 1000.0
    monkeypatch.setattr(platypus.solver, "time", types.SimpleNamespace(monotonic=lambda: now))
    config = RunConfig(max_concurrent=1)
    solver = _solver_with_two_keys(config)
    assert solver._priority is not None
    await solver._priority.acquire("0")
    rival = asyncio.create_task(solver._priority.acquire("1"))
    await _settle()
    yield_slot = solver._yield_handle(_example_game(2), "0", config)

    now = 1000.0 + 100 * 3600
    await yield_slot({"level": 0, "level_tokens": 5, "compacted": False})
    assert not rival.done(), "declined: no hold can elapse, and the swap is not free"

    async def offer() -> None:
        """The same game at a free point: a compaction."""
        await yield_slot({"level": 0, "level_tokens": 5, "compacted": True})

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
    gaps = [b - a for a, b in zip(spaced, spaced[1:])]
    assert len(gaps) == 3 and all(gap >= 0.02 * 0.9 for gap in gaps)
    # A zero gap is a pass-through, not a serialized queue.
    burst = await run(0.0)
    assert max(burst) - min(burst) < 0.02
