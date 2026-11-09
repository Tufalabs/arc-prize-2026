"""Loss handling: a loss is shown to the model instead of being auto-RESET
past. Covers the move-loop branch, the RESET-only action guard (arcengine
refuses non-RESET actions after GAME_OVER and TAAF raises on the empty frame
it returns, so the refusal is what keeps the game alive), and the prompt."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import taaf.game

from duck import DuckAgent
from platypus import Action
from platypus.contract import GameInfo, Observation

_OBS_INDEX = iter(range(1_000_000))


def _obs(*, outcome: str, fill: int, caused_by: Action | None, levels_completed: int = 0) -> Observation:
    frame = taaf.game.Frame(data=np.full((64, 64), fill, dtype=np.int8))
    raw_state = {"playing": "NOT_FINISHED", "lost": "GAME_OVER"}[outcome]
    state: Any = SimpleNamespace(
        levels_completed=levels_completed,
        # Engine-true: the action list stays WIDE after a loss.
        available_actions=[1, 2, 0],
        raw=SimpleNamespace(state=SimpleNamespace(name=raw_state)),
        frame=frame,
        animation_frames=[frame],
    )
    return Observation(
        state=state,
        outcome=outcome,  # type: ignore[arg-type]
        level=levels_completed + 1,
        index=next(_OBS_INDEX),
        caused_by=caused_by,
        frame=frame,
        animation_frames=(frame,),
    )


def _ctx(replies: list[Any] | None = None) -> Any:
    """Context with a recording ``execute``; ``replies`` are thunks building
    observation tuples (evaluated at execute time so indices stay ordered)."""
    queue = list(replies or [])
    executed: list[Action] = []

    async def execute(action: Action) -> tuple[Observation, ...]:
        executed.append(action)
        return queue.pop(0)()

    ctx = SimpleNamespace(
        game=GameInfo(game_id="fake", hint=None, action_space=("ACTION1", "ACTION2"), number_of_levels=3),
        execute=execute,
    )
    ctx.executed = executed
    return ctx


def _lost_agent(ctx: Any, **kwargs: Any) -> DuckAgent:
    """Agent that has just lost: seed observation then a losing ACTION1."""
    agent = DuckAgent(**kwargs)
    agent._record_observation(ctx, _obs(outcome="playing", fill=0, caused_by=None))
    agent.awaiting_display = "UP"
    agent._record_observation(ctx, _obs(outcome="lost", fill=9, caused_by=Action("ACTION1")))
    agent.live_available_actions = ["ACTION1", "ACTION2", "RESET"]
    agent.executed_this_move = 0
    agent.terminal_this_move = None
    return agent


async def test_loss_reaches_the_llm_instead_of_an_auto_reset() -> None:
    """The whole point: no RESET is returned without asking the model, and the
    turn the model gets is built on the LOST frame (fill 9), not a fresh board."""
    ctx = _ctx()
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    agent._record_observation(ctx, _obs(outcome="playing", fill=0, caused_by=None))
    seen_prompt: list[str] = []

    async def fake_turns(_ctx: Any, _obs: Observation) -> None:
        seen_prompt.append(agent._build_user_prompt_text())
        agent.executed_this_move = 1  # the model acted

    agent._analyzer_turns_until_acted = fake_turns  # type: ignore[method-assign]
    agent.awaiting_display = "UP"
    decision = await agent._do_move(ctx, _obs(outcome="lost", fill=9, caused_by=Action("ACTION1")))

    assert decision is None  # live mode: acted in-snippet, no auto-RESET action
    assert seen_prompt, "the model must get a turn on the lost state"
    prompt = seen_prompt[0]
    assert "Valid actions right now: RESET." in prompt
    assert 'submit_action(["RESET"])' in prompt
    # The frame the turn is built on is the losing one.
    assert agent.history_entries[-1].action == "UP"
    assert agent.game_over_active is True


async def test_non_reset_action_is_refused_while_game_over() -> None:
    """arcengine would return an empty frame here and TAAF would raise; the
    model must get a snippet-level ValueError instead."""
    ctx = _ctx()
    agent = _lost_agent(ctx)
    reply = await agent._execute_actions_live(ctx, [{"action": "DOWN"}])
    assert "error" in reply and "RESET is the only valid action" in reply["error"]
    assert ctx.executed == []  # nothing reached the engine


async def test_reset_clears_game_over_and_the_rest_of_the_batch_runs() -> None:
    ctx = _ctx(
        [
            lambda: (_obs(outcome="playing", fill=0, caused_by=Action("RESET")),),
            lambda: (_obs(outcome="playing", fill=1, caused_by=Action("ACTION1")),),
        ]
    )
    agent = _lost_agent(ctx)
    reply = await agent._execute_actions_live(ctx, [{"action": "RESET"}, {"action": "UP"}])
    assert [a.name for a in ctx.executed] == ["RESET", "ACTION1"]
    assert agent.game_over_active is False
    assert reply["result"]["executed_actions"] == ["RESET", "UP"]
    # Back to normal: the next prompt carries the real action list again.
    assert agent.current_valid_actions == ["UP", "DOWN", "RESET"]
    assert "The game is over" not in agent._build_user_prompt_text()


async def test_stop_detail_does_not_promise_an_auto_reset() -> None:
    ctx = _ctx([lambda: (_obs(outcome="lost", fill=9, caused_by=Action("ACTION1")),)])
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    agent._record_observation(ctx, _obs(outcome="playing", fill=0, caused_by=None))
    agent.live_available_actions = ["ACTION1", "ACTION2", "RESET"]
    reply = await agent._execute_actions_live(ctx, [{"action": "UP"}, {"action": "DOWN"}])
    detail = reply["result"]["stop_detail"]
    assert "auto-reset" not in detail
    assert "RESET on your next turn" in detail


async def test_shown_loss_keeps_the_present_tense() -> None:
    """The lost frame IS the current one, so the prompt says so plainly."""
    ctx = _ctx()
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    agent._record_observation(ctx, _obs(outcome="playing", fill=0, caused_by=None))
    agent.awaiting_display = "UP"
    agent._record_observation(ctx, _obs(outcome="lost", fill=9, caused_by=Action("ACTION1")))
    agent._take_step_summary()

    prompt = agent._build_user_prompt_text()

    assert "The game is over." in prompt
    assert agent.game_over_active is True
