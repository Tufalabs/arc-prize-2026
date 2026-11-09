"""The per-level token budget: which allowance a level gets, when it is
enforced, and what a spent allowance does.

This is what bounds a game, in place of a wall clock
(``packages/duck/design.md``, *Run budgets*). The level is read from the settled
observation in ``ctx.history`` on every consultation rather than cached: the
budget is consulted from inside a turn, where the move's own observation is not
in scope, and a level can turn over mid-move under an in-move execute.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import arcengine
import numpy as np
import pytest
import taaf.game
import ullm.backend

from duck import DuckAgent
from platypus import GiveUpError
from platypus.contract import Action, GameInfo, Move, Observation
from platypus.llm import LLMMoveContext
from platypus.slots import SlotView, write_slots

ACTION = "ACTION1"


def observation_on(level: int, *, index: int = 1) -> Observation:
    """The settled observation of a game standing on ``level``."""
    raw = arcengine.FrameDataRaw(
        game_id="duck-level-budget",
        state=arcengine.GameState.NOT_FINISHED,
        levels_completed=level,
        win_levels=8,
        available_actions=[1],
    )
    raw.frame = [np.zeros((8, 8), dtype=np.int8)]
    return Observation.from_state(
        taaf.game.GameState(raw=raw), outcome="playing", level=level, index=index, caused_by=Action(ACTION)
    )


def context_on(level: int) -> LLMMoveContext:
    """A context whose history leaves the game on ``level``."""
    return LLMMoveContext(
        history=[Move(action=Action(ACTION), observations=(observation_on(level),))],
        game=GameInfo(game_id="g", hint=None, action_space=(ACTION,), number_of_levels=8),
        rng=random.Random(0),
        shared={},
        backend=ullm.backend.DummyBackend(),
    )


def slots_with(waiting: int) -> SlotView:
    """A real SlotView over a real tally file, so the read path is exercised."""
    directory = Path(tempfile.mkdtemp())
    write_slots(directory, waiting=waiting, running=1, limit=1)
    return SlotView(job_dir=directory)


def charge(
    agent: DuckAgent,
    ctx: LLMMoveContext,
    completion: int,
    *,
    started_on: int | None = None,
    started_used: int | None = None,
) -> None:
    """Advance the backend meter and charge that exact delta to a level."""
    before = ctx.backend.total_usage.completion_tokens
    ctx.backend.total_usage.completion_tokens += completion
    agent._charge_budget(
        ctx,
        agent._current_level(ctx) if started_on is None else started_on,
        agent.level_tokens_used if started_used is None else started_used,
        before,
    )


# --- which allowance a level gets ------------------------------------------


def test_the_last_entry_applies_to_every_level_past_the_end() -> None:
    # What lets a two-entry tuple mean "squeeze the first level, leave the rest
    # alone" without naming every level a game might reach.
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", max_tokens_per_level=(40_000, 90_000))
    assert agent._level_allowance(context_on(0)) == 40_000
    assert agent._level_allowance(context_on(1)) == 90_000
    assert agent._level_allowance(context_on(7)) == 90_000


def test_a_single_entry_is_one_allowance_for_every_level() -> None:
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", max_tokens_per_level=(50_000,))
    assert agent._level_allowance(context_on(0)) == 50_000
    assert agent._level_allowance(context_on(4)) == 50_000


def test_none_disables_the_budget() -> None:
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", max_tokens_per_level=None)
    assert agent._level_allowance(context_on(2)) is None
    assert agent._budget_cap(context_on(2)) is None


# --- when it is enforced ---------------------------------------------------


def test_the_budget_stops_applying_once_nothing_is_queued() -> None:
    # Giving up frees a slot; with an empty queue there is nothing to put in it,
    # so the cap would end this game early and buy nobody anything.
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
        max_tokens_per_level=(40_000,),
        uncap_when_queue_drained=True,
    )
    ctx = context_on(0)

    assert agent._budget_in_force(ctx), "a queue nobody published is not a drained one"

    ctx.slots = slots_with(waiting=3)
    assert agent._budget_in_force(ctx), "still games queued behind this one"

    ctx.slots = slots_with(waiting=0)
    assert not agent._budget_in_force(ctx)
    # The allowance is unchanged by the lift: what the level was allowed and
    # whether that was applied are separate facts, so a lifted cap can be
    # reported as lifted rather than as zero.
    assert agent._level_allowance(ctx) == 40_000
    assert agent._budget_cap(ctx) is None


def test_the_default_keeps_the_cap_in_force_on_a_drained_queue() -> None:
    # The default is off, because a lifted cap makes a game's spend depend on
    # where in the run it was scheduled — noise in any comparison between runs.
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", max_tokens_per_level=(40_000,))
    ctx = context_on(0)
    ctx.slots = slots_with(waiting=0)
    assert agent._budget_in_force(ctx)


def test_only_the_kaggle_preset_lifts_the_cap() -> None:
    """A submission runs uncapped and ranks its slots instead; everywhere
    else the cap is what bounds a game.

    Both halves matter. On a deadline a cap can only end a game early,
    forfeiting what it would have scored later, and the scheduler frees
    the same slot for nothing. Off a deadline there is no queue to hand
    the slot to, so lifting it would make a game's spend depend on where
    in the run it happened to be scheduled.
    """
    pytest.importorskip("duck.presets", reason="needs the vllm extra (linux)")
    ctx = context_on(0)

    local = DuckAgent.build(engine="qwen3.8-27b")
    kaggle = DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-rtx")

    assert local.max_tokens_per_level == (90_000,)
    assert local._budget_in_force(ctx)
    assert kaggle.max_tokens_per_level is None
    assert kaggle._budget_cap(ctx) is None, "nothing bounds a Kaggle game's spend on a level"
    assert kaggle.run_config.slot_policy == "least_game_tokens", "the scheduler is what replaced it"


# --- charging, and what a spent allowance does ------------------------------


def test_a_turn_is_charged_to_the_level_it_ran_on() -> None:
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", max_tokens_per_level=(40_000,))
    ctx = context_on(0)

    charge(agent, ctx, 1_500)
    charge(agent, ctx, 2_500)
    assert agent.level_tokens_used == 4_000


def test_a_spent_allowance_gives_up_rather_than_running_another_entry() -> None:
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", max_tokens_per_level=(3_000,))
    ctx = context_on(0)

    assert agent._per_turn_cap(ctx) == 3_000  # nothing spent yet: the whole allowance

    charge(agent, ctx, 3_000)
    with pytest.raises(GiveUpError, match="per-level token budget exhausted"):
        agent._per_turn_cap(ctx)


def test_a_lifted_cap_never_gives_up_however_much_was_spent() -> None:
    agent = DuckAgent(
        reply_cap_tokens=None, compaction_strategy="prune", max_tokens_per_level=(3_000,), uncap_when_queue_drained=True
    )
    ctx = context_on(0)
    ctx.slots = slots_with(waiting=0)

    charge(agent, ctx, 30_000)
    assert agent._per_turn_cap(ctx) is None  # ten times the allowance, and no refusal


def test_a_game_budget_counts_across_level_changes() -> None:
    """The comparison cap follows the backend meter, not the level counter."""
    agent = DuckAgent(
        reply_cap_tokens=None, compaction_strategy="prune", max_tokens_per_level=None, max_tokens_per_game=70_000
    )
    ctx = context_on(0)
    ctx.backend.total_usage.completion_tokens = 40_000

    assert agent._per_turn_cap(ctx) == 30_000
    agent._record_observation(ctx, observation_on(1, index=2))
    assert agent.level_tokens_used == 0
    assert agent._per_turn_cap(ctx) == 30_000

    ctx.backend.total_usage.completion_tokens = 70_000
    with pytest.raises(GiveUpError, match="per-game token budget exhausted"):
        agent._per_turn_cap(ctx)


def test_the_tighter_of_game_and_level_budgets_binds() -> None:
    """Enabling the experiment control does not weaken the normal policy."""
    agent = DuckAgent(
        reply_cap_tokens=None, compaction_strategy="prune", max_tokens_per_level=(3_000,), max_tokens_per_game=70_000
    )
    ctx = context_on(0)
    ctx.backend.total_usage.completion_tokens = 69_000

    assert agent._per_turn_cap(ctx) == 1_000


def test_a_game_budget_never_lifts_when_the_queue_drains() -> None:
    """Equal-work comparisons must not depend on a game's schedule position."""
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
        max_tokens_per_level=(3_000,),
        max_tokens_per_game=70_000,
        uncap_when_queue_drained=True,
    )
    ctx = context_on(0)
    ctx.slots = slots_with(waiting=0)

    assert not agent._budget_in_force(ctx), "the Kaggle per-level rule still retires"
    assert not agent._per_turn_cap_lifted(ctx), "the whole-game comparison cap does not"
    assert agent._per_turn_cap(ctx) == 70_000


def test_a_new_level_gets_a_fresh_allowance() -> None:
    # The reset rides the observation path rather than the level-won branch,
    # because a level can also turn over mid-move under an in-move execute.
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", max_tokens_per_level=(3_000,))
    ctx = context_on(0)
    charge(agent, ctx, 3_000)

    agent._record_observation(ctx, observation_on(1, index=2))
    assert agent.level_tokens_used == 0
    assert agent._per_turn_cap(context_on(1)) == 3_000  # the new level pays nothing for the old one's spend


def test_a_turn_is_never_handed_a_ceiling_of_zero() -> None:
    """A turn capped at zero generates nothing, so the manager appends a
    synthetic placeholder; the placeholder carries no tool call and no usage,
    so the entry nudges into another identical turn and the level's spend never
    advances to end it. The ceiling must be positive, or the turn must not
    happen — there is no third option that terminates.
    """
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", max_tokens_per_level=(1_000,))
    ctx = context_on(0)

    charge(agent, ctx, 999)
    assert agent._per_turn_cap(ctx) == 1, "a sliver of allowance is still a runnable turn"

    charge(agent, ctx, 1)  # exactly spent
    with pytest.raises(GiveUpError, match="per-level token budget exhausted"):
        agent._per_turn_cap(ctx)

    # And overspent, which a truncated turn can leave behind.
    charge(agent, ctx, 500)
    with pytest.raises(GiveUpError):
        agent._per_turn_cap(ctx)


def test_a_turn_whose_level_was_won_mid_flight_leaves_the_new_level_at_zero() -> None:
    # The live counter belongs to the level the game moved on to, which must not
    # pay for work it did not do; what the crossing turn spent is reported
    # against the level it ran under instead (end-to-end pair in
    # tests/test_duck_budget_exhaustion.py).
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", max_tokens_per_level=(40_000,))

    crossed_ctx = context_on(1)
    charge(agent, crossed_ctx, 5_000, started_on=0, started_used=12_000)
    assert agent.level_tokens_used == 0

    charge(agent, crossed_ctx, 5_000)
    assert agent.level_tokens_used == 5_000, "a turn that stayed on its level is charged normally"


def test_a_level_window_bills_its_first_turn_to_the_level_it_starts_on() -> None:
    # A game started on a later level (TAAF's ``start_at_level`` window) has an
    # empty history until its first action lands, yet stands on that level:
    # the duck folded it in from the game's first observation. Its first turn
    # must be charged like any other, or the window's first move plays free of
    # the level's allowance.
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", max_tokens_per_level=(40_000,))
    agent.budget_level = 3
    fresh = LLMMoveContext(
        history=[],
        game=GameInfo(game_id="g", hint=None, action_space=(ACTION,), number_of_levels=8),
        rng=random.Random(0),
        shared={},
        backend=ullm.backend.DummyBackend(),
    )
    started_on = agent._current_level(fresh)
    assert started_on == 3

    acted = context_on(3)  # the first action has landed, still on level 3
    charge(agent, acted, 5_000, started_on=started_on, started_used=0)
    assert agent.level_tokens_used == 5_000


def test_game_and_level_budgets_charge_the_same_backend_meter_delta() -> None:
    """A non-retained billed round cannot make the two allowances diverge."""
    agent = DuckAgent(
        reply_cap_tokens=None, compaction_strategy="prune", max_tokens_per_level=(3_000,), max_tokens_per_game=5_000
    )
    ctx = context_on(0)
    ctx.backend.total_usage.completion_tokens = 500

    charge(agent, ctx, 750)

    assert agent.level_tokens_used == 750
    assert agent._game_tokens_used(ctx) == 1_250
    assert agent._per_turn_cap(ctx) == 2_250
