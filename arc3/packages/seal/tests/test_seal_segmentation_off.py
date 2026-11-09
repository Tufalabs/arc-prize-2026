"""``SealAgent.segmentation`` off: no ``segment()`` in the kernel, and no first view in the prompt.

What is pinned: off, ``segment``, ``Segmentation`` and ``Node`` are not names in a snippet or in a module the
model writes, ``runtime.game`` does not hold them and ``help()`` does not list them, and a call to ``segment``
is a plain ``NameError``; the prompt, the tool description and the level's checklist name none of them and
point at no view in their place. On, all of it is there, so the checks tell the two apart. Every prompt name
being a kernel name, in both modes, is ``test_seal_prompt_names_the_runtime``'s.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import pytest

from seal import SealAgent
from seal.repl import GameREPL

_PROBE = (
    "import runtime.game as _g\n"
    "print('globals', [n for n in ('segment', 'Segmentation', 'Node') if n in dir()])\n"
    "print('module', [n for n in ('segment', 'Segmentation', 'Node') if hasattr(_g, n) or n in dir(_g)])\n"
    "open('probe.py', 'w').write(\"FOUND = [n for n in ('segment', 'Segmentation', 'Node') if n in globals()]\\n\")\n"
    "import probe\n"
    "print('written module', probe.FOUND)\n"
    "help()\n"
    "try:\n"
    "    segment(history[-1].frame.grid)\n"
    "except NameError as error:\n"
    "    print('call', error)\n"
)


async def _probe(*, segmentation: bool) -> str:
    """What a snippet finds of segmentation, asked of a real kernel."""
    grid = np.zeros((2, 2), dtype=np.uint8)
    frame = {"grid_b64": base64.b64encode(grid.tobytes()).decode("ascii"), "shape": [2, 2], "step": 0, "level": 1}
    state: dict[str, Any] = {
        "level": 1,
        "turns": [{"action": "", "outcome": "playing", "animation": [frame]}],
        "available_actions": ["UP"],
    }
    tool = GameREPL(segmentation=segmentation)

    def _action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        del normalized
        return {"ack": {}}

    tool.host_functions = {"get_state": lambda: state, "action": _action}
    try:
        return (await tool._execute({"code": _PROBE})).to_string()  # pyright: ignore[reportPrivateUsage]
    finally:
        await tool.clear_state()


async def test_off_the_kernel_has_no_segmentation() -> None:
    out = await _probe(segmentation=False)
    assert "globals []" in out and "module []" in out and "written module []" in out, out
    assert "segment" not in out.split("Available objects:")[1].split("Call help")[0], out
    assert "call name 'segment' is not defined" in out, out


async def test_on_the_kernel_has_segmentation() -> None:
    out = await _probe(segmentation=True)
    names = "['segment', 'Segmentation', 'Node']"
    assert f"globals {names}" in out and f"module {names}" in out and f"written module {names}" in out, out
    assert "from_ascii_grid, segment, color_names" in out and "call" not in out.split("Call help")[1], out


@pytest.mark.parametrize("segmentation", [True, False])
def test_the_prompt_points_at_segmentation_only_where_the_kernel_has_it(segmentation: bool) -> None:
    agent = SealAgent(segmentation=segmentation)
    agent.game_action_names = ["UP", "DOWN", "LEFT", "RIGHT", "RESET"]
    text = agent._system_prompt_text() + agent._ensure_repl().description + agent._build_user_prompt_text()  # pyright: ignore[reportPrivateUsage]
    named = [name for name in ("segment(", "`segment`", "Segmentation", "`Node`", "primary") if name in text]
    if segmentation:
        assert named == ["segment(", "`segment`", "Segmentation", "`Node`", "primary"]
    else:
        assert named == []
        assert "1. Orient on the newest frame.\n" in text
