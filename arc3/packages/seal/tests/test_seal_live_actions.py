"""Live ``submit_action()``: each action executes via ctx.execute and the
reply carries the original's result + refreshed state; batch staging is
all-or-nothing while validity stops a batch mid-way; batches stop early on
terminal events; post-terminal calls are stop_detail no-ops; the kernel
re-installs the refreshed state so snippet globals are live."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import taaf.game

from platypus import Action
from platypus.contract import GameInfo, Observation
from platypus.diagnostics import NULL_SPAN
from seal import SealAgent
from seal.harness import game as harness_game
from seal.repl import GameREPL

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
        diag=NULL_SPAN,
    )
    ctx.executed = executed
    return ctx


def _fresh_live_agent(ctx: Any) -> SealAgent:
    # Acts without a world model, which the default gate refuses.
    agent = SealAgent(use_world_model="none")
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
    assert result["executed_actions"] == [{"action": "UP"}, {"action": "DOWN"}]
    assert result["requested_count"] == 2 and result["executed_count"] == 2
    assert result["stopped_early"] is False
    assert agent.executed_this_move == 2
    # Refreshed state rides the reply: current frame is the last executed one.
    assert reply["state"]["turns"][-1]["action"] == "DOWN"
    assert len(agent.history_entries) == 3  # seed + 2 actions


@pytest.mark.parametrize("move_changes", [True, False])
async def test_a_batch_records_what_each_move_changed_only_when_asked(move_changes: bool) -> None:
    """The first move repaints the board, the second changes nothing; off, the summary carries no changes."""
    ctx = _live_ctx(
        [
            lambda: (_obs(outcome="playing", fill=1, caused_by=Action("ACTION1")),),
            lambda: (_obs(outcome="playing", fill=1, caused_by=Action("ACTION2")),),
        ]
    )
    agent = _fresh_live_agent(ctx)
    agent.move_changes = move_changes
    await agent._execute_actions_live(ctx, [{"action": "UP"}, {"action": "DOWN"}])
    agent._take_step_summary()
    summary = agent.last_step_summary
    assert summary is not None
    assert summary.changes == ([(64 * 64, (0,), (1,)), (0, (), ())] if move_changes else [])


@pytest.mark.parametrize("replay_reminder", [True, False])
async def test_a_batch_records_a_repeated_attempt_only_when_asked(replay_reminder: bool) -> None:
    """UP and RESET twice from the level's start: the second RESET closes a repeat."""
    ctx = _live_ctx(
        [
            lambda: (_obs(outcome="playing", fill=1, caused_by=Action("ACTION1")),),
            lambda: (_obs(outcome="playing", fill=0, caused_by=Action("RESET")),),
            lambda: (_obs(outcome="playing", fill=1, caused_by=Action("ACTION1")),),
            lambda: (_obs(outcome="playing", fill=0, caused_by=Action("RESET")),),
        ]
    )
    agent = _fresh_live_agent(ctx)
    agent.replay_reminder = replay_reminder
    await agent._execute_actions_live(
        ctx, [{"action": "UP"}, {"action": "RESET"}, {"action": "UP"}, {"action": "RESET"}]
    )
    agent._take_step_summary()
    summary = agent.last_step_summary
    assert summary is not None
    assert summary.repeated == (["UP", "RESET"] if replay_reminder else None)


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
    # the winning turn's frame is the NEW level's board, and the new level
    # then opens with a seed turn sharing it.
    won, opening = agent.history_entries[-2:]
    assert (won.action, won.outcome, won.frame.level) == (Action("ACTION1"), "won", 2)
    assert (opening.action, opening.outcome, opening.level) == (None, "playing", 2)


async def test_a_level_won_while_exploring_leaves_the_next_levels_budget_whole() -> None:
    """The move that wins a level while exploring belongs to that level: the
    next one opens with its whole allowance, as the prompt promises."""
    ctx = _live_ctx(
        [
            lambda: (
                _obs(outcome="level_won", fill=3, caused_by=Action("ACTION1"), levels_completed=1),
                _obs(outcome="playing", fill=4, caused_by=None, levels_completed=1),
            )
        ]
    )
    agent = _fresh_live_agent(ctx)
    explore = agent._kernel_host_functions(ctx, _obs(outcome="playing", fill=0, caused_by=None))["action_exploring"]

    reply = await explore([{"action": "UP"}])

    assert reply["result"]["level_completed"] is True
    assert agent.exploration_moves_left == agent.exploration_moves_new_level


class _SpanRecorder:
    """A ``ctx.diag`` that keeps each span's kind and kv."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, dict[str, Any]]] = []

    @contextlib.contextmanager
    def span(self, kind: str, name: str = "") -> Generator[Any, None, None]:
        """Open a span whose kv lands in :attr:`spans`."""
        del name
        record: dict[str, Any] = {}
        self.spans.append((kind, record))
        yield SimpleNamespace(kv=record.update)

    def moves(self) -> list[dict[str, Any]]:
        """The ``seal/moves`` spans' kv, oldest first."""
        return [kv for kind, kv in self.spans if kind == "seal/moves"]


async def test_every_acting_call_leaves_one_moves_span_naming_its_route() -> None:
    """``seal/moves`` says how each batch was sent and how much of it ran: an
    exploratory move, a refused second one, and a checked batch the world
    model stopped after its first move."""
    ctx = _live_ctx(
        [
            lambda: (_obs(outcome="playing", fill=1, caused_by=Action("ACTION1")),),
            lambda: (_obs(outcome="playing", fill=2, caused_by=Action("ACTION2")),),
        ]
    )
    ctx.diag = _SpanRecorder()
    agent = _fresh_live_agent(ctx)
    agent.use_world_model = "mandatory"
    agent.exploration_moves_left = 5
    host = agent._kernel_host_functions(ctx, _obs(outcome="playing", fill=0, caused_by=None))

    await host["action_exploring"]([{"action": "UP"}])
    await host["action_exploring"]([{"action": "UP"}, {"action": "DOWN"}])
    paused = await host["action_stepped"]([{"action": "DOWN"}, {"action": "UP"}])
    assert paused["paused"] is True
    host["action_stop"]("world_model_mismatch", "render does not match")

    exploring, refused, checked = ctx.diag.moves()
    assert exploring == {
        "mode": "mandatory",
        "route": "exploring",
        "requested": 1,
        "executed": 1,
        "stop_reason": None,
        "world_model_stop": None,
        "refused": None,
    }
    assert refused["route"] == "exploring" and refused["executed"] == 0 and refused["refused"]
    assert (checked["route"], checked["requested"], checked["executed"], checked["world_model_stop"]) == (
        "checked",
        2,
        1,
        "mismatch",
    )


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
            "level": 1,
            "turns": [],
            "available_actions": [],
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
    agent.current_valid_actions = ["RESET", "UP", "DOWN"]
    reply = await agent._execute_actions_live(ctx, [{"action": "UP"}])
    assert "error" in reply and "not currently valid" in reply["error"]
    assert reply["error"].endswith(" This game's actions: RESET, UP, DOWN."), "the refusal names the game's actions"
    assert ctx.executed == []


async def test_staging_rejects_all_or_nothing_and_clamps_mouse() -> None:
    """A batch that fails staging (unknown action, malformed coordinates)
    rejects whole — nothing reaches the engine — while MOUSE coordinates
    clamp to the board instead of rejecting."""
    ctx = _live_ctx([lambda: (_obs(outcome="playing", fill=1, caused_by=Action("ACTION6", row=63, col=0)),)])
    agent = _fresh_live_agent(ctx)
    agent.live_available_actions = ["ACTION1", "ACTION6"]
    agent.current_valid_actions = ["RESET", "UP", "MOUSE"]

    reply = await agent._execute_actions_live(ctx, [{"action": "UP"}, {"action": "TELEPORT"}])
    assert reply["error"] == "Action 2: unknown action 'TELEPORT'. This game's actions: RESET, UP, MOUSE."
    reply = await agent._execute_actions_live(ctx, [{"action": "UP"}, {"action": "MOUSE", "row": 3}])
    assert "integer row and col" in reply["error"] and reply["error"].endswith("This game's actions: RESET, UP, MOUSE.")
    assert ctx.executed == []  # staging failures execute nothing, UP included

    reply = await agent._execute_actions_live(ctx, [{"action": "MOUSE", "row": 99, "col": -5}])
    assert reply["result"]["executed_actions"] == [{"action": "MOUSE", "row": 63, "col": 0}]
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
            "level": 1,
            "turns": [],
            "available_actions": ["UP"],
        }

    def _host_action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        # The terminal guard's shape: a no-op result, NOT an error — the
        # model-visible reply that fed the real spin.
        return {"result": {"executed": False, "stop_reason": "previous_level_completed"}}

    tool = GameREPL(timeout=2.0, use_world_model="none")
    tool.host_functions = {
        "get_state": _host_get_state,
        "action": _host_action,
    }
    content = await tool._execute({"code": "while True:\n    submit_action(['UP'])"})
    text = content.to_string()
    assert "timed out after 2" in text, text
    assert tool.is_error
    # A follow-up call works (fresh kernel per call regardless).
    content2 = await tool._execute({"code": "print('alive')"})
    text2 = content2.to_string()
    tool.reset(keep_tmpdir_contents=False)
    assert not tool.is_error and "alive" in text2, text2


async def test_kernel_record_uses_model_visible_python_error() -> None:
    """A caught snippet exception is still an error in kernel diagnostics."""

    def _host_get_state() -> dict[str, Any]:
        return {
            "level": 1,
            "turns": [],
            "available_actions": [],
        }

    tool = GameREPL(timeout=2.0, use_world_model="none")
    tool.host_functions = {"get_state": _host_get_state, "action": _empty_action}
    try:
        content = await tool._execute({"code": "raise ValueError('bad probe')"})
        text = content.to_string()
        (record,) = tool.pop_execution_records()
    finally:
        await tool.clear_state()

    assert "ValueError: bad probe" in text
    assert record["is_error"] is True


async def test_kernel_installs_refreshed_state_after_live_action() -> None:
    """End to end through the real kernel: a snippet calls submit_action() twice;
    after each call history/valid_actions reflect the host's
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
            "level": 1,
            "turns": [{"action": "", "outcome": "playing", "animation": [_frame_payload(0)]}],
            "available_actions": ["UP", "DOWN"],
        }

    def _host_action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        calls.append(normalized)
        return {
            "result": {"executed": True, "action_display": normalized[0]["action"], "board_changed": True},
            "state": {
                "level": 1,
                "turns": [
                    {"action": "", "outcome": "playing", "animation": [_frame_payload(0)]},
                    *[
                        {
                            "action": call[0]["action"],
                            "outcome": "playing",
                            "animation": [_frame_payload(min(n, 1))],
                        }
                        for n, call in enumerate(calls)
                    ],
                ],
                "available_actions": ["UP", "DOWN"],
            },
        }

    tool = GameREPL(use_world_model="none")
    tool.host_functions = {
        "get_state": _host_get_state,
        "action": _host_action,
    }

    code = (
        "before = len(history)\n"
        "held = history\n"
        "r1 = submit_action(['UP'])\n"
        "after_one = len(history)\n"
        "r2 = submit_action(['DOWN'])\n"
        "print(before, after_one, len(history), r1['executed'], r2['executed'],\n"
        "      history is held,\n"
        "      history[-1].action.name, history[-1].outcome)\n"
    )
    content = await tool._execute({"code": code})
    text = content.to_string()
    tool.reset(keep_tmpdir_contents=False)
    assert not tool.is_error, text
    # history grew as the snippet acted, and it is the SAME object throughout:
    # the append rides the action reply rather than replacing the graph, so a
    # name the model bound before acting still points at the live history.
    assert "1 2 3 True True True DOWN playing" in text, text


async def test_accidental_submit_action_import_is_a_plain_module_error() -> None:
    """``import submit_action`` is a mistake real Python answers itself: a
    ``ModuleNotFoundError`` naming the global, with nothing executed. The
    duck's kernel resolved the import to the callable; with real imports
    there is no allowlist to hang that shim on, and the error is enough."""
    calls: list[list[dict[str, Any]]] = []

    def get_state() -> dict[str, Any]:
        return {
            "level": 1,
            "turns": [],
            "available_actions": ["UP"],
        }

    def execute(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        calls.append(normalized)
        return {"result": {"executed": True}}

    tool = GameREPL(use_world_model="none")
    tool.host_functions = {"get_state": get_state, "action": execute}
    content = await tool._execute({"code": "import submit_action\nsubmit_action(['UP'])"})
    text = content.to_string()
    await tool.clear_state()
    assert tool.is_error and "ModuleNotFoundError: No module named 'submit_action'" in text, text
    assert calls == []


async def test_numpy_coordinates_reach_the_host_as_plain_ints() -> None:
    """A MOUSE target computed from the grid carries numpy integers, and the
    host call's restricted decoder refuses numpy types, so the kernel hands
    the host plain ints; the host validates them as it would any others."""
    calls: list[list[dict[str, Any]]] = []

    def get_state() -> dict[str, Any]:
        return {
            "level": 1,
            "turns": [],
            "available_actions": ["MOUSE"],
        }

    def execute(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        calls.append(normalized)
        return {"result": {"executed": True}}

    tool = GameREPL(use_world_model="none")
    tool.host_functions = {"get_state": get_state, "action": execute}
    code = "r, c = np.argwhere(np.eye(8, dtype=np.int8))[3]\nprint(submit_action([{'action': 'MOUSE', 'row': r, 'col': c}]))"
    content = await tool._execute({"code": code})
    text = content.to_string()
    await tool.clear_state()
    assert not tool.is_error, text
    assert calls == [[{"action": "MOUSE", "row": 3, "col": 3}]]
    assert all(type(calls[0][0][key]) is int for key in ("row", "col"))


async def test_kernel_frame_surface_is_the_whales_with_segmentation() -> None:
    """The frame is the whale's ``Frame`` (``.grid``, ``.diff_sparse``), the
    animation holds only what came BEFORE the settled board, nodes are
    attribute-carrying ``Node``s with array pixels, and the duck's legacy
    names are absent."""
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
            "level": 1,
            "turns": [{"action": "", "outcome": "playing", "animation": [prior, frame]}],
            "available_actions": ["UP"],
        }

    tool = GameREPL(use_world_model="none")
    tool.host_functions = {
        "get_state": get_state,
        "action": _empty_action,
    }
    code = (
        "seg = segment(history[-1].frame.grid)\n"
        "node = seg[0]\n"
        "try:\n"
        "    action\n"
        "except Exception:\n"
        "    has_action = False\n"
        "else:\n"
        "    has_action = True\n"
        "try:\n"
        "    history[-1].frame.step = 99\n"
        "except AttributeError:\n"
        "    read_only = True\n"
        "else:\n"
        "    read_only = False\n"
        "print(len(seg) == len(seg.nodes), isinstance(node.color, int), "
        "len(node.pixels) == node.pixel_count, "
        "node.pixels.shape == (node.pixel_count, 2) and node.pixels.dtype.kind == 'i', "
        "history[-1].frame.grid.dtype == np.int8, "
        # ascii_grid renders the grid one hex digit per cell after the row's
        # index and a tab -- the only place a colour stops being the index it is.
        "np.array_equal(history[-1].frame.grid, "
        "np.array([[int(c, 16) for c in row.split('\\t')[1]] for row in ascii_grid(history[-1].frame.grid).splitlines()])), "
        "len(history[-1].animation_frames), read_only, has_action, "
        "history[-1].frame.diff_sparse is None, sorted(history[-1].frame.color_counts) == [0, 1])\n"
    )
    content = await tool._execute({"code": code})
    text = content.to_string()
    await tool.clear_state()
    assert not tool.is_error, text
    # The settled board is the turn's own frame, and the animation holds only
    # what came BEFORE it -- so a model diffing the two sees real motion.
    # The level's first frame has nothing before it, so no diff.
    assert "True True True True True True 1 True False True True" in text, text


async def test_without_a_world_model_a_turn_names_nothing_a_world_model_fills() -> None:
    """Under ``use_world_model='none'`` a ``Turn`` has no ``state`` or
    ``predicted_frame``: ``help(Turn)`` does not describe them and the
    missing-attribute hint does not offer them."""
    grid = np.zeros((2, 2), dtype=np.uint8)
    import base64 as b64

    frame = {"grid_b64": b64.b64encode(grid.tobytes()).decode("ascii"), "shape": [2, 2], "step": 0, "level": 1}

    def get_state() -> dict[str, Any]:
        return {
            "level": 1,
            "turns": [{"action": "", "outcome": "playing", "animation": [frame]}],
            "available_actions": ["UP"],
        }

    tool = GameREPL(use_world_model="none")
    tool.host_functions = {"get_state": get_state, "action": _empty_action}
    content = await tool._execute({"code": "help(Turn)\nhistory[-1].state\n"})
    text = content.to_string()
    await tool.clear_state()
    assert "Available: action, animation_frames, frame, level, outcome, step." in text, text
    assert "world model" not in text and "predicted_frame" not in text, text


async def test_when_a_turn_was_played_is_on_the_turn_and_its_board_on_the_frame() -> None:
    """``step`` and ``level`` are the turn's; a frame still answers them but
    does not offer them. A board attribute asked of a turn names the frame,
    and a slice of the history is a history."""
    grid = np.zeros((2, 2), dtype=np.uint8)
    import base64 as b64

    frame = {"grid_b64": b64.b64encode(grid.tobytes()).decode("ascii"), "shape": [2, 2], "step": 3, "level": 2}

    def get_state() -> dict[str, Any]:
        return {
            "level": 2,
            "turns": [{"action": "", "outcome": "playing", "animation": [frame]}],
            "available_actions": ["UP"],
        }

    tool = GameREPL(use_world_model="none")
    tool.host_functions = {"get_state": get_state, "action": _empty_action}
    code = (
        "t = history[-1]\n"
        "print(t.step, t.level, t.frame.step, t.frame.level, 'step' in dir(t.frame))\n"
        "print(type(history[:1]).__name__, len(history[:1]), type(history[-1]).__name__)\n"
        "t.grid\n"
    )
    content = await tool._execute({"code": code})
    text = content.to_string()
    await tool.clear_state()
    assert "3 2 3 2 False" in text, text
    assert "History 1 Turn" in text, text
    assert "'Turn' object has no attribute 'grid'; the board is the turn's frame: .frame.grid" in text, text


async def test_moves_cross_the_kernel_boundary_as_actions() -> None:
    """End to end through the host's own bindings and a real kernel: a move
    is an Action going into ``submit_action``, in its result, and in
    ``history`` — a click with its coordinates — so the list a world model
    was stepped through is the list that plays, and each played turn equals
    the move that was planned."""
    ctx = _live_ctx(
        [
            lambda: (_obs(outcome="playing", fill=1, caused_by=Action("ACTION6", row=4, col=7)),),
            lambda: (_obs(outcome="playing", fill=2, caused_by=Action("ACTION1")),),
        ]
    )
    agent = _fresh_live_agent(ctx)
    agent.live_available_actions = ["ACTION1", "ACTION6"]
    tool = GameREPL(use_world_model="none")
    tool.host_functions = agent._kernel_host_functions(ctx, _obs(outcome="playing", fill=0, caused_by=None))
    code = (
        "plan = [Action('MOUSE', row=4, col=7), Action('UP')]\n"
        "r = submit_action(plan)\n"
        "print(r['executed_actions'] == plan)\n"
        "print([turn.action for turn in history[1:]] == plan, history[1].action.row, history[1].action.col)\n"
        "print(history[1].action)\n"
    )
    try:
        text = (await tool._execute({"code": code})).to_string()
    finally:
        await tool.clear_state()

    assert "True\nTrue 4 7\nAction('MOUSE', row=4, col=7)" in text, text
    assert [(action.name, action.row, action.col) for action in ctx.executed] == [
        ("ACTION6", 4, 7),
        ("ACTION1", None, None),
    ]


def test_a_planned_move_is_spelled_as_it_will_be_played() -> None:
    """A move built for a lookahead has to equal the turn it becomes: the name
    is read the way ``submit_action`` reads it, and a click without its
    coordinates is refused where it is written rather than inside do_move."""
    assert harness_game.Action(" up ") == harness_game.Action("UP")
    with pytest.raises(ValueError, match=r"Action\('MOUSE', row=r, col=c\)"):
        harness_game.Action("MOUSE")
    with pytest.raises(ValueError, match="Only MOUSE takes a row and a col"):
        harness_game.Action("UP", row=1, col=2)


async def test_the_executed_actions_line_names_a_click_by_where_and_what_it_hit() -> None:
    """The next prompt names each move bare, and a click by its coordinates
    and the colour of the cell it landed on — read from the board it was
    made on, not the one it produced."""
    ctx = _live_ctx([lambda: (_obs(outcome="playing", fill=2, caused_by=Action("ACTION6", row=4, col=7)),)])
    agent = SealAgent(use_world_model="none")
    agent._record_observation(ctx, _obs(outcome="playing", fill=9, caused_by=None))
    agent.live_available_actions = ["ACTION1", "ACTION6"]
    agent.executed_this_move = 0
    agent.terminal_this_move = None

    await agent._execute_actions_live(ctx, [{"action": "MOUSE", "row": 4, "col": 7}])
    agent._take_step_summary()

    assert agent.last_step_summary is not None
    assert agent.last_step_summary.executed_actions == ["MOUSE(row=4, col=7) on color 9"]
