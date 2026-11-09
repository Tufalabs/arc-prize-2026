"""A new frame that repeats an earlier frame of its level is named as that repeat.

A RESET or an UNDO puts back a board the level has shown, and the game
restoring a board is no move's effect: its frame gets one line naming the
turn it repeats, in place of a block. Any other move that brings a board
back keeps its block, which says what the move did, and the line follows.
A move that changed nothing is not a repeat: its block is the comparison
alone, as before. Played in a live kernel without a world model, one call
per move, since only a call's newest frame is described.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

from seal import frame_delta as fd
from seal.repl import GameREPL

_HOOK = "import runtime._repl_state as _s\nprint(_s.frame_delta_after_call())\ndel _s\n"

#: Where each move puts the block, from where it stands; UNDO and RESET are the level's own bookkeeping.
_STEPS = {"RIGHT": 4, "LEFT": -4, "SPACE": 0}


def _grid(col: int) -> np.ndarray:
    """A 64x64 board with a 3x3 blue block at (10, col)."""
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[10:13, col : col + 3] = 9
    return grid


def _frame(step: int, col: int) -> dict[str, Any]:
    """The board as the host sends a frame."""
    return {"grid_b64": base64.b64encode(_grid(col).tobytes()).decode(), "shape": [64, 64], "step": step, "level": 1}


class _Level:
    """A kernel over a level where the block moves sideways, RESET puts it back at the start and UNDO takes
    back the last move."""

    def __init__(self) -> None:
        self.cols = [10]
        self.turns: list[dict[str, Any]] = [{"action": "", "outcome": "playing", "animation": [_frame(0, 10)]}]
        self.tool = GameREPL(timeout=30.0, reset_before_each_call=False, use_world_model="none", settled_delta=True)
        self.tool.host_functions = {"get_state": self._state, "action": self._action}

    def _state(self) -> dict[str, Any]:
        return {"level": 1, "turns": list(self.turns), "available_actions": ["RIGHT", "LEFT", "SPACE", "UNDO"]}

    def _action(self, normalized: list[dict[str, Any]]) -> dict[str, Any]:
        name = normalized[0]["action"]
        if name == "RESET":
            col = 10
        elif name == "UNDO":
            col = self.cols[-2]
        else:
            col = self.cols[-1] + _STEPS[name]
        self.cols.append(col)
        self.turns.append({"action": name, "outcome": "playing", "animation": [_frame(len(self.turns), col)]})
        return {"result": {"executed": True}, "state": self._state()}

    async def move(self, name: str) -> dict[str, Any]:
        """One call playing ``name``, and what the host collects after it."""
        await self.tool._execute({"code": f"submit_action(['{name}'])\n"})
        content = await self.tool._execute({"code": _HOOK})
        return json.loads(content.to_string().strip().splitlines()[-1])

    async def close(self) -> None:
        """Drop the kernel."""
        await self.tool.clear_state()


async def test_a_restored_board_is_named_and_a_revisited_one_keeps_its_block() -> None:
    level = _Level()
    try:
        first = await level.move("RIGHT")
        assert "repeats" not in first["records"][0], "a new board"
        await level.move("RIGHT")

        back = await level.move("LEFT")  # history[3] is history[1]'s board again
        (record,) = back["records"]
        assert record["shown"] and record["repeats"] == 1
        assert "# moved left 4" in back["text"], "the block still says what the move did"
        assert back["text"].endswith("\n" + fd.repeat_text(1))

        reset = await level.move("RESET")
        (record,) = reset["records"]
        assert reset["text"] == "The current frame is the same as `history[0].frame`, the level's start."
        assert (record["shown"], record["repeats"], record["reason"]) == (False, 0, "unpredicted")
        assert "chosen" not in record, "no decomposition ran"

        again = await level.move("RIGHT")  # history[5] repeats history[1] and history[3]: the most recent
        assert again["records"][0]["repeats"] == 3

        await level.move("RIGHT")
        undo = await level.move("UNDO")  # history[7] puts back history[6]'s predecessor, history[5]
        assert undo["text"] == fd.repeat_text(5)
        assert undo["records"][0]["shown"] is False

        still = await level.move("SPACE")  # nothing changed: the comparison alone, as before
        (record,) = still["records"]
        assert "repeats" not in record and record["chosen"] == "previous" and record["cost"] == 0
        assert "assert (history[-2].frame.grid == history[-1].frame.grid).all()" in still["text"]
    finally:
        await level.close()


def test_the_level_start_is_named_first_and_the_frame_before_is_no_repeat() -> None:
    a, b, c = _grid(10), _grid(14), _grid(18)
    assert fd.repeated_turn([a, b, a], 2) == 0
    assert fd.repeated_turn([a, b, c, b, a, b], 5) == 3, "the most recent below the start"
    assert fd.repeated_turn([a, b, a, a], 3) is None, "a move that changed nothing, though the start matches"
    assert fd.repeated_turn([a, a], 1) is None
    assert fd.repeated_turn([a, b, c], 2) is None
