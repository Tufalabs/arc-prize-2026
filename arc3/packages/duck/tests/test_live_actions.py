"""Live ``submit_action()``: each action executes via ctx.execute and the
reply carries the original's result + refreshed state; batch staging is
all-or-nothing while validity stops a batch mid-way; batches stop early on
terminal events; post-terminal calls are stop_detail no-ops; the kernel
re-installs the refreshed state so snippet globals are live."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import taaf.game

from duck import DuckAgent
from duck.python_tool import DuckPythonTool
from platypus import Action
from platypus.contract import GameInfo, Observation

_OBS_INDEX = iter(range(1_000_000))


def _obs(*, outcome: str, fill: int, caused_by: Action | None, levels_completed: int = 0) -> Observation:
    frame = taaf.game.Frame(data=np.full((64, 64), fill, dtype=np.int8))
    raw_state = {"playing": "NOT_FINISHED", "lost": "GAME_OVER", "level_won": "NOT_FINISHED", "game_won": "WIN"}[
        outcome
    ]
    state: Any = SimpleNamespace(
        levels_completed=levels_completed,
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


def _live_ctx(replies: list[Any]) -> Any:
    """``replies`` are thunks building observation tuples — evaluated at
    execute time, so their indices come AFTER the agent's seed observation
    (the index guard drops anything at-or-below the last folded index)."""
    queue = list(replies)
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


def _fresh_live_agent(ctx: Any) -> DuckAgent:
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    agent._record_observation(ctx, _obs(outcome="playing", fill=0, caused_by=None))
    agent.live_available_actions = ["ACTION1", "ACTION2"]
    agent.executed_this_move = 0
    agent.terminal_this_move = None
    return agent


def _empty_action(actions: list[dict[str, Any]]) -> dict[str, Any]:
    del actions
    return {}


async def test_live_batch_executes_and_reports_results() -> None:
    ctx = _live_ctx(
        [
            lambda: (_obs(outcome="playing", fill=1, caused_by=Action("ACTION1")),),
            lambda: (_obs(outcome="playing", fill=2, caused_by=Action("ACTION2")),),
        ]
    )
    agent = _fresh_live_agent(ctx)
    reply = await agent._execute_actions_live(ctx, [{"action": "UP"}, {"action": "DOWN"}])
    result = reply["result"]
    assert [a.name for a in ctx.executed] == ["ACTION1", "ACTION2"]
    assert result["executed"] is True
    assert result["executed_actions"] == ["UP", "DOWN"]
    assert result["requested_count"] == 2 and result["executed_count"] == 2
    assert result["stopped_early"] is False
    assert agent.executed_this_move == 2
    # Refreshed state rides the reply: current frame is the last executed one.
    assert reply["state"]["timeline"][-1]["action"] == "DOWN"
    assert len(agent.history_entries) == 3  # seed + 2 actions


async def test_live_batch_stops_early_on_game_over_and_guards_afterwards() -> None:
    ctx = _live_ctx([lambda: (_obs(outcome="lost", fill=9, caused_by=Action("ACTION1")),)])
    agent = _fresh_live_agent(ctx)
    reply = await agent._execute_actions_live(ctx, [{"action": "UP"}, {"action": "DOWN"}])
    result = reply["result"]
    assert [a.name for a in ctx.executed] == ["ACTION1"]  # DOWN never executed
    assert result["game_over"] is True
    assert result["stopped_early"] is True and result["executed_count"] == 1
    assert result["stop_reason"] == "game_over"
    assert "RESET on your next turn" in result["stop_detail"]
    # A second call this move is a no-op carrying the original's stop_detail.
    reply2 = await agent._execute_actions_live(ctx, [{"action": "UP"}])
    result2 = reply2["result"]
    assert result2["executed"] is False
    assert result2["executed_count"] == 0
    assert result2["stop_reason"] == "previous_game_over"
    assert [a.name for a in ctx.executed] == ["ACTION1"]


async def test_live_batch_stops_at_the_max_moves_cap() -> None:
    """``max_moves`` is a hard action cap even inside one batch: this loop is
    the only place actions execute, and ``_do_move``'s own check runs before
    the analyzer, which a single multi-action call never revisits."""
    ctx = _live_ctx([lambda: (_obs(outcome="playing", fill=1, caused_by=Action("ACTION1")),)])
    agent = _fresh_live_agent(ctx)
    agent.max_moves = 1
    reply = await agent._execute_actions_live(ctx, [{"action": "UP"}, {"action": "DOWN"}])
    result = reply["result"]
    assert [a.name for a in ctx.executed] == ["ACTION1"]  # DOWN never executed
    assert result["stopped_early"] is True and result["executed_count"] == 1
    assert result["stop_reason"] == "max_moves"
    assert "action limit" in result["stop_detail"]
    # A follow-up call this move executes nothing either.
    reply2 = await agent._execute_actions_live(ctx, [{"action": "UP"}])
    assert reply2["result"]["executed_count"] == 0
    assert reply2["result"]["stop_reason"] == "max_moves"
    assert [a.name for a in ctx.executed] == ["ACTION1"]


async def test_live_level_win_folds_split_observations() -> None:
    ctx = _live_ctx(
        [
            lambda: (
                _obs(outcome="level_won", fill=3, caused_by=Action("ACTION1"), levels_completed=1),
                _obs(outcome="playing", fill=4, caused_by=None, levels_completed=1),
            )
        ]
    )
    agent = _fresh_live_agent(ctx)
    reply = await agent._execute_actions_live(ctx, [{"action": "UP"}])
    result = reply["result"]
    assert result["level_completed"] is True
    assert agent.terminal_this_move is not None
    # Deferred level-win entry resolved by the split's initial observation:
    # the entry's frame is the NEW level's board.
    assert agent.history_entries[-1].action == "UP"
    assert agent.history_entries[-1].frame.level == 2


async def test_level_win_in_a_snippet_still_owes_a_kernel_reset() -> None:
    """persistence="level" must clear the kernel across a level boundary in
    LIVE mode too. The win never arrives as its own do_move there — it comes
    back through ctx.execute mid-snippet, and the next delivered observation
    is the split's "playing" initial — so the reset is owed and paid at the
    next decision move, where the kernel is idle."""
    ctx = _live_ctx(
        [
            lambda: (
                _obs(outcome="level_won", fill=3, caused_by=Action("ACTION1"), levels_completed=1),
                _obs(outcome="playing", fill=4, caused_by=None, levels_completed=1),
            )
        ]
    )
    agent = _fresh_live_agent(ctx)
    agent.persistence = "level"
    # A kernel the model has actually built something in — the state whose
    # survival across the boundary is the whole question.
    tool = agent._ensure_repl()

    def _no_state() -> dict[str, Any]:
        return {
            "frame_api": "ergonomic",
            "timeline": [],
            "valid_actions": [],
            "last_action_result": {},
        }

    def _no_action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        return {}

    tool.host_functions = {"get_state": _no_state, "action": _no_action}
    await tool._execute({"code": "carried = 1234"})

    await agent._execute_actions_live(ctx, [{"action": "UP"}])
    assert agent.pending_level_reset, "the win was folded without recording that a reset is owed"

    # The next decision move pays it: the kernel goes, and the model is told.
    agent._apply_pending_level_reset()
    assert not agent.pending_level_reset
    assert tool.is_fresh, "scratch state survived the level boundary"
    assert "a new level started" in agent.repl_reset_pending
    await tool.clear_state()


async def test_turn_stops_after_first_acting_tool_call_in_live_mode() -> None:
    """The original's ``if step_executed: break``: once a tool call has
    executed actions, the stop hook ends the turn and skips the
    response's remaining tool calls."""
    ctx = _live_ctx([lambda: (_obs(outcome="playing", fill=1, caused_by=Action("ACTION1")),)])
    agent = _fresh_live_agent(ctx)
    assert agent._stop_when_acted(None, None) is None  # type: ignore[arg-type]  # not acted yet
    await agent._execute_actions_live(ctx, [{"action": "UP"}])
    skip = agent._stop_when_acted(None, None)  # type: ignore[arg-type]
    assert skip is not None and "executed actions" in skip


async def test_live_invalid_first_action_is_an_error_and_executes_nothing() -> None:
    ctx = _live_ctx([])
    agent = _fresh_live_agent(ctx)
    agent.live_available_actions = ["ACTION2"]
    reply = await agent._execute_actions_live(ctx, [{"action": "UP"}])
    assert "error" in reply and "not currently valid" in reply["error"]
    assert ctx.executed == []


async def test_staging_rejects_all_or_nothing_and_clamps_mouse() -> None:
    """A batch that fails staging (unknown action, malformed coordinates)
    rejects whole — nothing reaches the engine — while MOUSE coordinates
    clamp to the board instead of rejecting."""
    ctx = _live_ctx([lambda: (_obs(outcome="playing", fill=1, caused_by=Action("ACTION6", row=63, col=0)),)])
    agent = _fresh_live_agent(ctx)
    agent.live_available_actions = ["ACTION1", "ACTION6"]

    reply = await agent._execute_actions_live(ctx, [{"action": "UP"}, {"action": "TELEPORT"}])
    assert "unknown action 'TELEPORT'" in reply["error"]
    reply = await agent._execute_actions_live(ctx, [{"action": "UP"}, {"action": "MOUSE", "row": 3}])
    assert "integer row and col" in reply["error"]
    assert ctx.executed == []  # staging failures execute nothing, UP included

    reply = await agent._execute_actions_live(ctx, [{"action": "MOUSE", "row": 99, "col": -5}])
    assert reply["result"]["executed_actions"] == ["MOUSE(row=63, col=0)"]
    (mouse,) = ctx.executed
    assert (mouse.name, mouse.row, mouse.col) == ("ACTION6", 63, 0)


async def test_model_reachable_host_action_rejects_malformed_arguments() -> None:
    """Raw callback traversal cannot turn malformed model input into a fatal error."""
    ctx = _live_ctx([])
    agent = _fresh_live_agent(ctx)
    action = agent._kernel_host_functions(ctx, _obs(outcome="playing", fill=0, caused_by=None))["action"]
    reply = await action([], {})
    assert reply == {"error": "submit_action() requires exactly one positional actions argument."}
    assert ctx.executed == []


async def test_action_spin_loop_hits_the_wall_clock() -> None:
    """ullm's kernel timeout is a wall-clock deadline over the whole call,
    host-call time included: a model loop calling submit_action() every iteration
    (e.g. spinning on the terminal guard's no-op reply after a level win)
    must die at ~timeout, not run until the per-game budget — the measured
    17.5 min spin. The next call must get a working kernel."""

    def _host_get_state() -> dict[str, Any]:
        return {
            "frame_api": "ergonomic",
            "timeline": [],
            "valid_actions": ["UP"],
            "last_action_result": {},
        }

    def _host_action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        # The terminal guard's shape: a no-op result, NOT an error — the
        # model-visible reply that fed the real spin.
        return {"result": {"executed": False, "stop_reason": "previous_level_completed"}}

    tool = DuckPythonTool(timeout=2.0)
    tool.host_functions = {
        "get_state": _host_get_state,
        "action": _host_action,
    }
    content = await tool._execute({"code": "while True:\n    submit_action(['UP'])"})
    payload = json.loads(content.to_string())
    assert payload.get("error") == "Tool timed out after 2s", payload
    assert tool.is_error
    # A follow-up call works (fresh kernel per call regardless).
    content2 = await tool._execute({"code": "print('alive')"})
    payload2 = json.loads(content2.to_string())
    tool.reset(keep_tmpdir_contents=False)
    assert payload2.get("returncode") == 0 and "alive" in payload2.get("stdout", ""), payload2


async def test_kernel_record_uses_model_visible_python_error() -> None:
    """A caught snippet exception is still an error in kernel diagnostics."""

    def _host_get_state() -> dict[str, Any]:
        return {
            "frame_api": "ergonomic",
            "timeline": [],
            "valid_actions": [],
            "last_action_result": {},
        }

    tool = DuckPythonTool(timeout=2.0)
    tool.host_functions = {"get_state": _host_get_state, "action": _empty_action}
    try:
        content = await tool._execute({"code": "raise ValueError('bad probe')"})
        payload = json.loads(content.to_string())
        (record,) = tool.pop_execution_records()
    finally:
        await tool.clear_state()

    assert "ValueError: bad probe" in payload["error"]
    assert record["is_error"] is True


async def test_kernel_installs_refreshed_state_after_live_action() -> None:
    """End to end through the real kernel: a snippet calls submit_action() twice;
    after each call current_frame/valid_actions reflect the host's
    refreshed state — the act-inspect-act loop the queue design forbade."""
    frames = [np.full((2, 2), fill, dtype=np.uint8) for fill in (5, 6)]
    import base64 as b64

    def _frame_payload(index: int) -> dict[str, Any]:
        return {
            "grid_b64": b64.b64encode(frames[index].tobytes()).decode("ascii"),
            "shape": [2, 2],
            "step": index + 1,
            "level": 1,
        }

    calls: list[list[dict[str, Any]]] = []

    def _host_get_state() -> dict[str, Any]:
        return {
            "frame_api": "ergonomic",
            "timeline": [{"action": "", "animation": [_frame_payload(0)]}],
            "valid_actions": ["UP"],
            "last_action_result": {},
        }

    def _host_action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        calls.append(normalized)
        index = len(calls) - 1
        return {
            "result": {"executed": True, "action_display": normalized[0]["action"], "board_changed": True},
            "state": {
                "frame_api": "ergonomic",
                "timeline": [
                    {"action": "", "animation": [_frame_payload(0)]},
                    {"action": normalized[0]["action"], "animation": [_frame_payload(min(index + 1, 1))]},
                ],
                "valid_actions": ["DOWN"] if index == 0 else ["UP"],
                "last_action_result": {"executed": True},
            },
        }

    tool = DuckPythonTool()
    tool.host_functions = {
        "get_state": _host_get_state,
        "action": _host_action,
    }

    code = (
        "first = valid_actions[0]\n"
        "r1 = submit_action([first])\n"
        "second = valid_actions[0]\n"
        "r2 = submit_action([second])\n"
        "print(first, second, r1['executed'], r2['executed'], "
        "current_frame is frames[-1], last_action == actions[-1], len(animations) == len(frames))\n"
    )
    content = await tool._execute({"code": code})
    payload = json.loads(content.to_string())
    tool.reset(keep_tmpdir_contents=False)
    assert payload.get("returncode") == 0, payload
    # The snippet saw UP first, then the refreshed DOWN — state was live.
    assert "UP DOWN True True True True True" in payload.get("stdout", ""), payload


async def test_accidental_submit_action_import_resolves_to_preloaded_function() -> None:
    """Qwen's common ``import submit_action`` mistake must not discard the action."""
    calls: list[list[dict[str, Any]]] = []

    def get_state() -> dict[str, Any]:
        return {
            "frame_api": "ergonomic",
            "timeline": [],
            "valid_actions": ["UP"],
            "last_action_result": {},
        }

    def execute(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        calls.append(normalized)
        return {"result": {"executed": True}}

    tool = DuckPythonTool()
    tool.host_functions = {"get_state": get_state, "action": execute}
    content = await tool._execute({"code": "import submit_action\nsubmit_action(['UP'])"})
    payload = json.loads(content.to_string())
    await tool.clear_state()
    assert payload.get("returncode") == 0, payload
    assert calls == [[{"action": "UP"}]]


async def test_numpy_coordinates_reach_the_host_as_python_ints() -> None:
    """A MOUSE coordinate read off an array is a numpy scalar. The action host
    call rebuilds plain data only, so without conversion the call is refused
    on its way to the host (sweep 12's first full run: 8 games in 30 minutes);
    it goes as the int it stands for."""
    calls: list[list[dict[str, Any]]] = []

    def get_state() -> dict[str, Any]:
        return {
            "frame_api": "ergonomic",
            "timeline": [],
            "valid_actions": ["MOUSE"],
            "last_action_result": {},
        }

    def execute(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        calls.append(normalized)
        return {"result": {"executed": True}}

    tool = DuckPythonTool()
    tool.host_functions = {"get_state": get_state, "action": execute}
    code = (
        "row, col = np.argwhere(np.eye(4, dtype=bool))[2]\n"
        "submit_action([{'action': 'MOUSE', 'row': row, 'col': np.int8(col)}])\n"
    )
    content = await tool._execute({"code": code})
    payload = json.loads(content.to_string())
    await tool.clear_state()
    assert payload.get("returncode") == 0 and "error" not in payload, payload
    assert calls == [[{"action": "MOUSE", "row": 2, "col": 2}]]
    assert {type(calls[0][0]["row"]), type(calls[0][0]["col"])} == {int}


async def test_the_wide_sandbox_is_granted_only_when_asked() -> None:
    """Off, a snippet has the first allowlist, preloaded, and nothing of the
    wide sandbox: its modules refused, no `open`, no `KeyError`. On, it has
    both, the wide modules preloaded too."""
    probe = (
        "print('preloaded', collections.Counter('aa')['a'], 'time' in dir())\n"
        "try:\n"
        "    import time\n"
        "    print('time', time.time() > 0)\n"
        "except Exception as exc:\n"
        "    print('refused', exc)\n"
        "print('open' in __builtins__, 'KeyError' in __builtins__)\n"
    )

    async def run(wide: bool | None) -> str:
        def get_state() -> dict[str, Any]:
            state: dict[str, Any] = {
                "frame_api": "ergonomic",
                "timeline": [],
                "valid_actions": [],
                "last_action_result": {},
            }
            if wide is not None:
                state["wide_sandbox"] = wide
            return state

        tool = DuckPythonTool()
        tool.host_functions = {"get_state": get_state, "action": _empty_action}
        try:
            payload = json.loads((await tool._execute({"code": probe})).to_string())
        finally:
            await tool.clear_state()
        assert "error" not in payload, payload
        return payload["stdout"]

    narrow = await run(None)
    assert narrow == await run(False)
    assert narrow == "preloaded 2 False\nrefused Module 'time' is not allowed in the sandbox.\nFalse False\n"
    assert await run(True) == "preloaded 2 True\ntime True\nTrue True\n"


async def test_kernel_frame_surface_is_lazy_segmented_and_legacy_names_are_absent() -> None:
    grid = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    import base64 as b64

    frame = {
        "grid_b64": b64.b64encode(grid.tobytes()).decode("ascii"),
        "shape": [2, 2],
        "step": 0,
        "level": 1,
    }
    prior = {
        **frame,
        "grid_b64": b64.b64encode(np.zeros((2, 2), dtype=np.uint8).tobytes()).decode("ascii"),
    }

    def get_state() -> dict[str, Any]:
        return {
            "frame_api": "ergonomic",
            "timeline": [{"action": "", "animation": [prior, frame]}],
            "valid_actions": ["UP"],
            "last_action_result": {},
        }

    tool = DuckPythonTool()
    tool.host_functions = {
        "get_state": get_state,
        "action": _empty_action,
    }
    code = (
        "first = current_frame.segment()\n"
        "second = current_frame.segment()\n"
        "node = first['nodes'][0]\n"
        "try:\n"
        "    action\n"
        "except Exception:\n"
        "    has_action = False\n"
        "else:\n"
        "    has_action = True\n"
        "print(first is second, bool(first['nodes']), len(node['pixels']) == node['pixel_count'], "
        "isinstance(node['pixels'][0], list), current_frame.array.shape == current_frame.shape, "
        "np.array_equal(current_frame.array, np.array([list(row) for row in current_frame.ascii.splitlines()])), "
        "animations[-1][-1] is current_frame, "
        "len(animations[-1]), hasattr(current_frame, 'segmentation'), has_action)\n"
    )
    content = await tool._execute({"code": code})
    payload = json.loads(content.to_string())
    await tool.clear_state()
    assert payload.get("returncode") == 0, payload
    assert "True True True True True True True 2 False False" in payload.get("stdout", ""), payload
