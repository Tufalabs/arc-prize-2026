"""Direct tests of DuckAgent's observation bookkeeping: step numbering,
level-win deferral, auto-RESET invisibility, board_changed, reward.

These drive internals — ``_record_observation``, ``history_entries``,
``pending_batch`` — against the suite's usual rule of testing outcomes
rather than internals. Deliberate: these fields are one rendering step from
what the model reads, so a wrong step number or a level-win counted twice
reaches the prompt directly, and reaching them through a played game would
take a scripted run per case to pin one integer. The tests around them
cover the behaviour; these pin the arithmetic underneath it."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import taaf.game

from duck import DuckAgent
from platypus import Action
from platypus.contract import GameInfo, Observation

_OBS_INDEX = iter(range(1_000_000))


def _obs(
    *,
    outcome: str,
    fill: int,
    caused_by: Action | None,
    levels_completed: int,
    raw_state: str = "NOT_FINISHED",
    index: int | None = None,
    animation_fills: tuple[int, ...] | None = None,
) -> Observation:
    # Strictly increasing per delivery, like the solver's global counter —
    # _record_observation's index guard drops repeats.
    if index is None:
        index = next(_OBS_INDEX)
    frame = taaf.game.Frame(data=np.full((64, 64), fill, dtype=np.int8))
    animation_frames = tuple(
        taaf.game.Frame(data=np.full((64, 64), value, dtype=np.int8))
        for value in (animation_fills if animation_fills is not None else (fill,))
    )
    state: Any = SimpleNamespace(
        levels_completed=levels_completed,
        available_actions=[0, 1, 2],
        raw=SimpleNamespace(state=SimpleNamespace(name=raw_state)),
        frame=frame,
        animation_frames=list(animation_frames),
    )
    return Observation(
        state=state,
        outcome=outcome,  # type: ignore[arg-type]
        level=levels_completed + 1,
        index=index,
        caused_by=caused_by,
        frame=frame,
        animation_frames=animation_frames,
    )


def _ctx(n_levels: int = 4) -> Any:
    return SimpleNamespace(
        game=GameInfo(game_id="fake", hint=None, action_space=("ACTION1", "ACTION2"), number_of_levels=n_levels)
    )


def test_step_numbering_matches_original_semantics() -> None:
    # Original: entry for action N reads step == N; seed entry reads 0.
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    ctx = _ctx()
    agent._record_observation(ctx, _obs(outcome="playing", fill=0, caused_by=None, levels_completed=0))
    agent.awaiting_display = "UP"
    agent._record_observation(ctx, _obs(outcome="playing", fill=1, caused_by=Action("ACTION1"), levels_completed=0))
    agent.awaiting_display = "UP"
    agent._record_observation(ctx, _obs(outcome="playing", fill=2, caused_by=Action("ACTION1"), levels_completed=0))
    steps = [entry.frame.step for entry in agent.history_entries]
    assert steps == [0, 1, 2]
    assert agent.last_action_result["action_num"] == 2  # consistent with the entry's step


def test_level_win_defers_entry_with_winning_action_step() -> None:
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    ctx = _ctx()
    agent._record_observation(ctx, _obs(outcome="playing", fill=0, caused_by=None, levels_completed=0))
    agent.awaiting_display = "UP"
    agent._record_observation(ctx, _obs(outcome="level_won", fill=14, caused_by=Action("ACTION1"), levels_completed=1))
    # Entry deferred: the win-screen frame must NOT be appended.
    assert len(agent.history_entries) == 1
    # New-level initial state resolves it, with the WINNING action's step.
    agent._record_observation(ctx, _obs(outcome="playing", fill=5, caused_by=None, levels_completed=1))
    assert [e.frame.step for e in agent.history_entries] == [0, 1]
    assert agent.history_entries[-1].action == "UP"
    # No duplicate steps when the next action lands.
    agent.awaiting_display = "DOWN"
    agent._record_observation(ctx, _obs(outcome="playing", fill=6, caused_by=Action("ACTION2"), levels_completed=1))
    steps = [e.frame.step for e in agent.history_entries]
    assert steps == sorted(set(steps)) == [0, 1, 2]


def test_timeline_preserves_animation_frames_and_level_split() -> None:
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    ctx = _ctx()
    agent._record_observation(
        ctx,
        _obs(outcome="playing", fill=0, caused_by=None, levels_completed=0, animation_fills=(15, 0)),
    )
    agent.awaiting_display = "UP"
    agent._record_observation(
        ctx,
        _obs(
            outcome="level_won",
            fill=3,
            caused_by=Action("ACTION1"),
            levels_completed=1,
            animation_fills=(1, 2),
        ),
    )
    agent._record_observation(
        ctx,
        _obs(outcome="playing", fill=4, caused_by=None, levels_completed=1),
    )

    timeline = agent._state_payload()["timeline"]
    assert [entry["action"] for entry in timeline] == ["", "UP"]
    assert [frame["grid_b64"] for frame in timeline[0]["animation"]] == [
        agent.history_entries[0].animation[0].grid_b64,
        agent.history_entries[0].frame.grid_b64,
    ]
    # Winning action: intermediate old-level frames, old settled win frame,
    # then the new-level settled frame.
    assert [frame["level"] for frame in timeline[1]["animation"]] == [1, 1, 1, 2]
    assert timeline[1]["animation"][-1] == agent.history_entries[-1].frame.payload()


def test_level_win_board_changed_resolves_against_new_level_frame() -> None:
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    ctx = _ctx()
    agent._record_observation(ctx, _obs(outcome="playing", fill=3, caused_by=None, levels_completed=0))
    agent.awaiting_display = "UP"
    agent._record_observation(ctx, _obs(outcome="level_won", fill=14, caused_by=Action("ACTION1"), levels_completed=1))
    assert agent.pending_batch is not None and agent.pending_batch.board_changed is False  # unresolved yet
    agent._record_observation(ctx, _obs(outcome="playing", fill=7, caused_by=None, levels_completed=1))
    assert agent.pending_batch.board_changed is True  # pre-action fill 3 vs new-level fill 7
    assert agent.last_action_result["board_changed"] is True


def test_reward_is_levels_delta_over_level_count() -> None:
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    ctx = _ctx(n_levels=4)
    agent._record_observation(ctx, _obs(outcome="playing", fill=0, caused_by=None, levels_completed=0))
    agent.awaiting_display = "UP"
    agent._record_observation(ctx, _obs(outcome="playing", fill=1, caused_by=Action("ACTION1"), levels_completed=0))
    assert agent.last_action_result["reward"] == 0.0
    agent.awaiting_display = "UP"
    agent._record_observation(ctx, _obs(outcome="level_won", fill=14, caused_by=Action("ACTION1"), levels_completed=1))
    assert agent.last_action_result["reward"] == 0.25


async def test_summary_keeps_explicit_memory_across_transition_and_reentry() -> None:
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    ctx = _ctx()
    agent._record_observation(ctx, _obs(outcome="playing", fill=0, caused_by=None, levels_completed=0))
    agent.response_step = 3
    await agent._append_memory("walls are lava; levels share controls")
    agent.awaiting_display = "UP"
    agent._record_observation(ctx, _obs(outcome="level_won", fill=14, caused_by=Action("ACTION1"), levels_completed=1))
    agent._take_step_summary()
    assert agent.last_step_summary is not None and agent.last_step_summary.level_transition is True
    assert "walls are lava; levels share controls" in agent._render_memory()
    # A no-action re-entry keeps the previous summary (original behavior).
    previous = agent.last_step_summary
    agent._take_step_summary()
    assert agent.last_step_summary == previous
