"""The settled delta on and off, in a live kernel, without a world model.

Without a world model nothing is ever predicted, so a new frame is explained
from the previous frame with reason ``unpredicted`` and the level never
counts as understood; with the settled delta off (and animations not
described) the hook answers with nothing at all. And what the agent sets
for framediff reaches the kernel, and a description the model asks for is
never cut by the output cap.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np

from framediff import BlockCopy, Decomposer
from seal.frame_delta import UNCAPPED_END, UNCAPPED_START
from seal.harness._repl_state import CALL_MARGIN_S
from seal.repl import GameREPL

_HOOK = "import runtime._repl_state as _s\nprint(_s.frame_delta_after_call())\ndel _s\n"

#: The kernel's per-call limit in these tests.
_TIMEOUT_S = 30.0

#: What the block's move costs: a copy, priced by the four columns it travels.
_MOVE = BlockCopy.COST + 4 * Decomposer().move_distance_cost

#: What the hook answers when there is nothing to say.
_NOTHING: dict[str, Any] = {"records": [], "text": None, "animation_text": None, "described": None, "concise": False}


def _frame(step: int, col: int) -> dict[str, Any]:
    """A 64x64 board with a 3x3 blue block at (10, col), as the host sends a frame."""
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[10:13, col : col + 3] = 9
    return {"grid_b64": base64.b64encode(grid.tobytes()).decode(), "shape": [64, 64], "step": step, "level": 1}


class _Level:
    """A kernel over a one-move level: the seed turn, and a move that shifts the block four columns, passing
    through two columns on the way when ``animated``."""

    def __init__(
        self,
        settled_delta: bool,
        crash_on_framediff_timeout: bool = True,
        animated: bool = False,
        use_world_model: Literal["none", "optional", "mandatory"] = "none",
        **repl: Any,
    ) -> None:
        self.turns: list[dict[str, Any]] = [{"action": "", "outcome": "playing", "animation": [_frame(0, 10)]}]
        self.animated = animated
        self.tool = GameREPL(
            timeout=_TIMEOUT_S,
            reset_before_each_call=False,
            use_world_model=use_world_model,
            settled_delta=settled_delta,
            crash_on_framediff_timeout=crash_on_framediff_timeout,
            **repl,
        )
        self.tool.host_functions = {"get_state": self._state, "action": self._action}

    def _state(self) -> dict[str, Any]:
        return {"level": 1, "turns": list(self.turns), "available_actions": ["UP"]}

    def _action(self, normalized: list[dict[str, Any]]) -> dict[str, Any]:
        animation = [_frame(1, 12), _frame(1, 14)] if self.animated else [_frame(1, 14)]
        self.turns.append({"action": normalized[0]["action"], "outcome": "playing", "animation": animation})
        return {"result": {"executed": True}, "state": self._state()}

    async def run(self, code: str) -> str:
        """One snippet, as the model's tool call would run it."""
        content = await self.tool._execute({"code": code})
        return content.to_string()

    async def hook(self) -> dict[str, Any]:
        """What the host collects after a call."""
        return json.loads((await self.run(_HOOK)).strip().splitlines()[-1])

    async def close(self) -> None:
        """Drop the kernel."""
        await self.tool.clear_state()


async def test_without_a_world_model_every_new_frame_is_unpredicted() -> None:
    """No render ever exists, so the delta comes from the previous frame at the higher limit; a call that
    plays nothing adds nothing."""
    level = _Level(settled_delta=True)
    try:
        assert await level.hook() == _NOTHING, "the seed turn alone is a level's first frame"
        out = await level.run("submit_action(['UP'])\nprint(len(history), hasattr(history[-1], 'predicted_frame'))\n")
        assert "2 False" in out, out
        payload = await level.hook()
        (record,) = payload["records"]
        assert (record["event"], record["reason"], record["understood"], record["limit"]) == (
            "frame",
            "unpredicted",
            False,
            150,
        )
        assert record["chosen"] == "previous" and record["shown"] and record["cost"] == _MOVE
        assert payload["text"].startswith(
            "The current frame can be most easily understood as a delta from the previous frame"
        )
        assert "g[10:13, 14:17] = history[-2].frame.grid[10:13, 10:13]  # moved right 4" in payload["text"]
        assert await level.hook() == _NOTHING, "no new frame since"
    finally:
        await level.close()


async def test_the_settled_delta_off_answers_with_nothing() -> None:
    """The switch off: the hook has nothing to say after a move either."""
    level = _Level(settled_delta=False)
    try:
        await level.run("submit_action(['UP'])\n")
        assert await level.hook() == _NOTHING
    finally:
        await level.close()


#: Makes every framediff search in the kernel start past its deadline.
_OUT_OF_TIME = "import framediff.decompose as _d\n_d.Decomposer.deadline = lambda self, deadline=None: 0.0\ndel _d\n"


async def test_out_of_time_after_a_call_ends_the_game_or_reads_as_undescribed() -> None:
    """The settled delta the harness makes after a call: out of time, it carries the error that ends the
    game, or with the flag off it is a frame with no concise description."""
    for crash in (True, False):
        level = _Level(settled_delta=True, crash_on_framediff_timeout=crash)
        try:
            await level.run(_OUT_OF_TIME)
            await level.run("submit_action(['UP'])\n")
            (record,) = (await level.hook())["records"]
            if crash:
                assert "framediff ran out of time on history[-1].frame.grid" in record["error"], record
            else:
                assert record["timed_out"] and not record["shown"] and "no concise description" in record["text"]
        finally:
            await level.close()


async def test_a_search_inside_a_call_stops_before_the_call_does() -> None:
    """Every call sets the deadline a decomposition inside it gives up at: the call's limit, less a margin."""
    level = _Level(settled_delta=True)
    try:
        out = await level.run(
            "import time\nimport runtime._repl_state as _s\nprint(_s._call_deadline - time.monotonic())\n"
        )
        left = float(out.strip().splitlines()[-1])
        assert _TIMEOUT_S - CALL_MARGIN_S - 5 < left <= _TIMEOUT_S - CALL_MARGIN_S
    finally:
        await level.close()


async def test_the_agents_framediff_settings_reach_the_kernel() -> None:
    """A budget under the move's cost skips the block the default budget shows; the understood budget is the
    one given."""
    level = _Level(settled_delta=True, decomposer=Decomposer(max_bits=5.0), max_bits_understood=70.0)
    try:
        await level.run("submit_action(['UP'])\n")
        (record,) = (await level.hook())["records"]
        assert (record["cost"], record["limit"], record["shown"]) == (_MOVE, 5.0, False)
        out = await level.run("import runtime._repl_state as _s\nprint(_s.max_bits_understood)\ndel _s\n")
        assert out.strip().splitlines()[-1] == "70.0"
    finally:
        await level.close()


#: The cut the output cap makes, as a pattern.
_CUT = r"\n\.\.\. \[truncated \d+ chars of tool output\] \.\.\.\n"


async def test_a_requested_description_is_never_cut_by_the_output_cap() -> None:
    """The cap cuts the model's own printing and leaves a description whole: where it was printed when that
    survives the cut, right after the cut when it was printed in what the cap removed. A stream the model
    redirected gets the text without the marks."""
    level = _Level(settled_delta=False, animated=True, tool_output_tokens=64)
    cut = _CUT
    try:
        await level.run("submit_action(['UP'])\n")
        whole = await level.run("describe_animation(-1)\n")
        assert whole.startswith("The animation of history[1] (1 frames) can be rebuilt as follows"), whole
        assert len(whole) > level.tool.max_output_chars and "truncated" not in whole
        head = await level.run("describe_animation(-1)\nprint('z' * 300)\n")
        assert head.startswith(whole + "\nzzz") and re.search(cut + "z{128}$", head), head
        middle = await level.run("print('a' * 300)\ndescribe_animation(-1)\nprint('z' * 300)\n")
        assert re.fullmatch("a{128}" + cut + re.escape(whole) + "\nz{128}", middle), middle
        tail = await level.run("print('a' * 300)\ndescribe_animation(-1)\n")
        assert re.fullmatch("a{128}" + cut + "a{127}\n" + re.escape(whole), tail), tail
        assert not any(mark in text for text in (whole, head, middle, tail) for mark in (UNCAPPED_START, UNCAPPED_END))
        redirected = await level.run(
            "import contextlib, io\nbuf = io.StringIO()\nwith contextlib.redirect_stdout(buf):\n"
            "    describe_animation(-1)\nprint(buf.getvalue().strip() == " + repr(whole) + ")\n"
        )
        assert redirected.strip() == "True", redirected
    finally:
        await level.close()


async def test_an_animated_opening_is_built_from_its_own_first_frame() -> None:
    """The game's first observation can carry an animation, and its turn has no turn before it: the block
    starts from the animation's first frame, a name that holds however far the game has gone, never from
    the newest board."""
    level = _Level(settled_delta=False)
    level.turns[0]["animation"] = [_frame(0, 2), _frame(0, 6), _frame(0, 10)]
    try:
        await level.run("submit_action(['UP'])\n")
        out = await level.run("describe_animation(0)\n")
        assert out.startswith("The animation of history[0] (2 frames) can be rebuilt as follows"), out
        assert "# Frame 0: the same as history[0].animation_frames[0].grid" in out, out
        assert "history[-1]" not in out
    finally:
        await level.close()


async def test_a_world_model_report_is_never_cut_by_the_output_cap() -> None:
    """``test_world_model()``'s report is the harness's text too: printed or as the call's last expression, the
    cap cuts the model's own printing around it and leaves the report whole."""
    level = _Level(settled_delta=False, use_world_model="mandatory", tool_output_tokens=64)
    try:
        assert level.tool._tmpdir is not None  # pyright: ignore[reportPrivateUsage] — the test writes where it runs
        Path(level.tool._tmpdir, "world_model.py").write_text(  # pyright: ignore[reportPrivateUsage] — same
            "def initial_state():\n    raise ValueError('not written yet ' * 20)\n\n\n"
            "def do_move(state, move):\n    return state\n\n\ndef render(state):\n    return None\n"
        )
        whole = await level.run("print(test_world_model())\n")
        assert whole.startswith("TestResult: FAIL"), whole
        assert len(whole) > level.tool.max_output_chars and "truncated" not in whole
        printed = await level.run("print('a' * 300)\nprint(test_world_model())\nprint('z' * 300)\n")
        assert re.fullmatch("a{128}" + _CUT + re.escape(whole) + "\nz{128}", printed), printed
        last = await level.run("print('a' * 300)\ntest_world_model()\n")
        assert re.fullmatch("a{128}" + _CUT + "a{127}\n" + re.escape(whole), last), last
    finally:
        await level.close()


async def test_a_world_model_that_does_not_load_is_reported_whole() -> None:
    """The notice that ``world_model.py`` did not load quotes the model's own
    traceback, which is what it fixes the file from: the cap cuts the cell's
    printing and leaves the notice whole."""
    level = _Level(settled_delta=False, use_world_model="mandatory", tool_output_tokens=64)
    try:
        assert level.tool._tmpdir is not None  # pyright: ignore[reportPrivateUsage] — the test writes where it runs
        Path(level.tool._tmpdir, "world_model.py").write_text("raise ValueError('broken ' * 60)\n")  # pyright: ignore[reportPrivateUsage] — same
        printed = await level.run("print('z' * 300)\n")
        assert printed.startswith("!!! world_model could not be (re)loaded"), printed
        assert "ValueError: " + "broken " * 59 + "broken" in printed, printed
        assert re.search(_CUT + "z{128}$", printed), printed
        assert not any(mark in printed for mark in (UNCAPPED_START, UNCAPPED_END))
    finally:
        await level.close()
