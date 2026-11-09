"""A python call that runs into its time limit, and a harness cell the kernel cannot run.

A call that runs into its limit takes the kernel's child processes with it and
earns no search; the search's retest runs the model's code under the call's own
limit; and a harness cell that fails for the state the model left the kernel in
-- it hung, or a library could not get memory to load -- restarts the kernel and
tells the model in the call's output, where a failure of ours still ends the
game. On a real kernel, the hangs produced by the model's own code.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest

import seal.agent
from platypus.diagnostics import SpanLogger
from seal import SealAgent
from seal.frame_delta import FrameDeltaError
from seal.repl import GameREPL
from seal.state import FramePayload

#: A one-colour board that never changes: passes on any history of such boards.
STILL_MODEL = """\
import numpy as np


def initial_state():
    return {'outcome': 'playing', 'energy': 2000}


def do_move(state, action):
    state['energy'] -= 1


def render(state):
    return np.full((8, 8), 3, dtype=np.int8)
"""

#: The same model, edited so that replaying a move never returns.
LOOPING_MODEL = STILL_MODEL.replace("    state['energy'] -= 1\n", "    while True:\n        pass\n")

LIMIT_S = 3


def _frame() -> dict[str, Any]:
    """An 8x8 board of colour 3, as the kernel receives it."""
    grid = np.full((8, 8), 3, dtype=np.uint8)
    return FramePayload(
        grid_b64=base64.b64encode(grid.tobytes()).decode("ascii"), shape=[8, 8], step=0, level=1
    ).payload()


def _tool(*, use_world_model: Literal["none", "optional"] = "optional", timeout: float = 30.0) -> GameREPL:
    """A level-long kernel on a level of two turns: the seed, and one UP."""
    turns = [
        {"action": "", "outcome": "playing", "animation": [_frame()]},
        {"action": "UP", "outcome": "playing", "animation": [_frame()]},
    ]
    state: dict[str, Any] = {"level": 1, "turns": turns, "available_actions": ["UP"]}

    def get_state() -> dict[str, Any]:
        return state

    def action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        del normalized
        return {}

    tool = GameREPL(timeout=timeout, reset_before_each_call=False, use_world_model=use_world_model)
    tool.host_functions = {"get_state": get_state, "action": action}
    return tool


def _write(tool: GameREPL, name: str, text: str) -> None:
    assert tool._tmpdir is not None  # pyright: ignore[reportPrivateUsage] — the test writes where it runs
    Path(tool._tmpdir, name).write_text(text)  # pyright: ignore[reportPrivateUsage]


def _ctx() -> Any:
    """What the after-call steps read off a move's context: its spans."""
    return SimpleNamespace(diag=SpanLogger())


def _alive(pid: int) -> bool:
    """Whether ``pid`` is a process that has not exited (a zombie has)."""
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


async def test_a_call_past_its_limit_takes_the_kernels_children_with_it() -> None:
    tool = _tool(use_world_model="none", timeout=LIMIT_S)
    code = "import subprocess, time\np = subprocess.Popen(['sleep', '600'])\nprint(p.pid, flush=True)\ntime.sleep(60)"
    try:
        content = await tool._execute({"code": code})  # pyright: ignore[reportPrivateUsage]
        assert tool.timed_out and not tool.is_fresh, content.to_string()
        assert tool.strays_killed == 1
        pid = int(re.findall(r"^(\d+)$", content.to_string(), flags=re.MULTILINE)[0])
        assert not _alive(pid), "the shell-out outlived the call"
        await tool._execute({"code": "print('next')"})  # pyright: ignore[reportPrivateUsage]
        assert not tool.timed_out and tool.strays_killed == 0
    finally:
        await tool.clear_state()


async def test_a_call_past_its_limit_earns_no_search() -> None:
    """The call passed its test, then ran on past its limit: whatever it left
    of the model's files is what timed out, so nothing is searched."""
    agent = SealAgent(use_world_model="optional")
    tool = _tool(timeout=LIMIT_S)
    try:
        _write(tool, "world_model.py", STILL_MODEL)
        content = await tool._execute({"code": "print(test_world_model())\nwhile True:\n    pass"})  # pyright: ignore[reportPrivateUsage]
        assert tool.timed_out and "PASS" in content.to_string(), content.to_string()
        after_call = agent._after_call_hook(_ctx(), tool)  # pyright: ignore[reportPrivateUsage]
        assert await after_call(agent.action_count, content.to_string()) is None
    finally:
        await tool.clear_state()


async def test_a_retest_past_the_calls_limit_earns_no_search() -> None:
    """The model's last test passed, then its file was edited into one whose
    replay never returns. The retest stops at the call's own limit, and the
    chain reports nothing; the kernel lives on."""
    agent = SealAgent(use_world_model="optional", python_timeout_s=LIMIT_S)
    tool = _tool()
    try:
        _write(tool, "world_model.py", STILL_MODEL)
        content = await tool._execute({"code": "print(test_world_model())"})  # pyright: ignore[reportPrivateUsage]
        assert "PASS" in content.to_string(), content.to_string()
        _write(tool, "world_model.py", LOOPING_MODEL)
        assert await agent._search_after_call(_ctx(), tool, content.to_string()) is None  # pyright: ignore[reportPrivateUsage]
        assert tool.timed_out and not tool.is_fresh
    finally:
        await tool.clear_state()


#: Rebinds a json function to one that never returns. The kernel's worker
#: parses each request with ``json.loads`` and writes each reply with
#: ``json.dumps``, so either wedges it.
_WEDGE = "import json\ndef _forever(*args, **kwargs):\n    while True:\n        pass\njson.{name} = _forever"


async def test_a_call_that_loses_its_kernel_says_the_session_is_gone() -> None:
    """The call's own output says it was killed; the notice says what that
    means for the next call."""
    agent = SealAgent(use_world_model="none")
    tool = _tool(use_world_model="none", timeout=LIMIT_S)
    try:
        content = await tool._execute({"code": _WEDGE.format(name="dumps")})  # pyright: ignore[reportPrivateUsage]
        assert tool.kernel_lost, content.to_string()
        after_call = agent._after_call_hook(_ctx(), tool)  # pyright: ignore[reportPrivateUsage]
        assert await after_call(
            agent.action_count, content.to_string()
        ) == agent.prompts.repl_reset_notice_with_files.format(reason=agent.prompts.kernel_stopped_responding_reason)
    finally:
        await tool.clear_state()


async def test_a_harness_cell_the_kernel_cannot_run_restarts_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model wedged the kernel for the next request, which is the
    after-call cell's. That cell runs into its limit, the kernel is
    restarted, the frame goes undescribed and the call's output says so."""
    monkeypatch.setattr(seal.agent, "_HARNESS_CELL_TIMEOUT_S", LIMIT_S)
    agent = SealAgent(use_world_model="none")
    tool = _tool(use_world_model="none")
    try:
        content = await tool._execute({"code": _WEDGE.format(name="loads")})  # pyright: ignore[reportPrivateUsage]
        assert not tool.kernel_lost, content.to_string()
        note = await agent._collect_after_call(_ctx(), tool)  # pyright: ignore[reportPrivateUsage]
        assert note == agent.prompts.repl_reset_notice_with_files.format(
            reason=agent.prompts.kernel_stopped_responding_reason
        )
        assert tool.is_fresh, "the kernel was not restarted"
        assert agent.frame_delta_text is None
    finally:
        await tool.clear_state()


@pytest.mark.parametrize(
    ("last_line", "out_of_memory"),
    [
        ("MemoryError", True),
        # numpy's own subclass, as the kernel's address-space cap refuses an allocation.
        (
            "numpy._core._exceptions._ArrayMemoryError: Unable to allocate 130. KiB for an array with shape "
            "(128, 65) and data type complex128",
            True,
        ),
        (
            "ImportError: /venv/numpy/fft/_pocketfft_umath.so: failed to map segment from shared object",
            True,
        ),
        ("KeyError: 'records'", False),
        # Named like one, and no MemoryError: the class decides, not the spelling.
        ("framediff.decompose.NotAMemoryError: no", False),
        ("ValueError: MemoryError in the message", False),
    ],
)
async def test_only_the_kernels_own_failures_restart_it(
    monkeypatch: pytest.MonkeyPatch, last_line: str, out_of_memory: bool
) -> None:
    """A library that could not get memory to load is the kernel's state; any
    other failure of the after-call cell is ours, and still ends the game."""

    async def failing(self: GameREPL, code: str, *, timeout: float) -> tuple[str, bool]:
        del self, code, timeout
        return f'Traceback (most recent call last):\n  File "<harness>", line 1\n{last_line}\n', True

    monkeypatch.setattr(GameREPL, "run_harness", failing)
    agent = SealAgent(use_world_model="none")
    tool = _tool(use_world_model="none")
    try:
        if out_of_memory:
            note = await agent._collect_after_call(_ctx(), tool)  # pyright: ignore[reportPrivateUsage]
            assert note is not None and agent.prompts.kernel_out_of_memory_reason in note
        else:
            with pytest.raises(FrameDeltaError, match="could not describe the frame"):
                await agent._collect_after_call(_ctx(), tool)  # pyright: ignore[reportPrivateUsage]
    finally:
        await tool.clear_state()
