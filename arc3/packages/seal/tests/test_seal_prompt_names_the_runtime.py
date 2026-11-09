"""Every global the prompt names is one the kernel actually binds.

Ported in spirit from ``duck-prompt-rewrite``, which found its legacy
surface naming ``frames`` — a global that runtime does not install. Nothing
fails when a prompt lies about the runtime: the model calls the name, gets
an error it cannot attribute to the prompt, and spends a turn on it.

Checked against the KERNEL rather than against a list kept here. A list
would have to be maintained alongside the prompt, which is the same
maintenance the bug came from; asking the kernel means the test cannot
drift from what is installed.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

import numpy as np
import pytest

from seal import SealAgent
from seal.repl import GameREPL

#: Names the prompt may use that are not kernel globals: attributes reached
#: through something else, or modules the model imports for itself.
NOT_GLOBALS = frozenset(
    {
        # attributes and methods, named as `x.y` or bare in prose
        "action",
        "outcome",
        "animation_frames",
        "nodes",
        "adjacency_list",
        "by_color",
        "name",
        "row",
        "col",
        "step",
        "level",
        "grid",
        "diff_sparse",
        "color_counts",
        "id",
        "color",
        "hash",
        "pixels",
        "pixel_count",
        "bbox",
        "row0",
        "row1",
        "col0",
        "col1",
        "boundary",
        "children",
        "splitlines",
        "deepcopy",
        # importable, not preloaded
        "numpy",
        "importlib",
        "pickle",
        # native tools, which are not Python names at all
        "memory",
        "python",
        "view_file",
        "search_files",
        "str_replace",
        "create_file",
        "undo_edit",
        # types and literals used in the type notation
        "int",
        "str",
        "dict",
        "bool",
        "list",
        "tuple",
        "None",
        "old_color",
        "new_color",
        "playing",
        "won",
        "lost",
        "obj",
        "c",
        "seg",
        "actions",
        "x",
        "y",
        "py",
        # the memory tool's argument, not a Python name
        "text",
        # bound by the prelude once the model's own file exists, which this
        # kernel has none of, plus what that file defines and the Action
        # its do_move is handed
        "world_model",
        "initial_state",
        "do_move",
        "render",
        "distance_to_win",
        "move",
        "state",
        # bound by the search after a call that earned one, which this kernel
        # never runs
        "suggested_actions",
        "suggested_history",
        # the pixel tolerance setter's argument
        "n",
    }
)

#: Python keywords the prompt quotes as syntax rather than as names.
KEYWORDS = frozenset({"assert", "class", "import", "for", "in", "if", "else", "def", "return"})


async def _kernel_globals(*, segmentation: bool, tolerance_offered: bool) -> set[str]:
    """Every name a snippet can see — globals and the builtins it is given —
    asked of a real kernel, offered the setters the agent offers."""
    grid = np.zeros((2, 2), dtype=np.uint8)
    frame = {
        "grid_b64": base64.b64encode(grid.tobytes()).decode("ascii"),
        "shape": [2, 2],
        "step": 0,
        "level": 1,
    }
    state: dict[str, Any] = {
        "level": 1,
        "turns": [{"action": "", "outcome": "playing", "animation": [frame]}],
        "available_actions": ["UP"],
    }
    tool = GameREPL(segmentation=segmentation, pixel_tolerance_offered=tolerance_offered)

    def _action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        del normalized
        return {"ack": {}}

    tool.host_functions = {"get_state": lambda: state, "action": _action}
    try:
        content = await tool._execute({"code": "import builtins\nprint(json.dumps(sorted(dir()) + dir(builtins)))"})
    finally:
        # In the finally: a kernel left running leaks its socket into whatever
        # test the garbage collector happens to reach it in, which then fails
        # there instead of here.
        await tool.clear_state()
    names: list[str] = [str(name) for name in json.loads(content.to_string())]
    return set(names)


@pytest.mark.parametrize("segmentation", [True, False])
async def test_the_prompt_names_no_global_the_kernel_does_not_bind(segmentation: bool) -> None:
    agent = SealAgent(segmentation=segmentation)
    agent.game_action_names = ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE", "RESET"]
    # The first turn's user prompt too: its checklist names the structural view.
    text = agent._system_prompt_text() + agent._ensure_repl().description + agent._build_user_prompt_text()
    installed = await _kernel_globals(segmentation=segmentation, tolerance_offered=agent.expose_pixel_tolerance)

    # Backticked bare identifiers, and the receiver of a backticked call.
    named: set[str] = set()
    for token in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)(?:\(|\.|`)", text):
        named.add(str(token))

    # Action names are the game's vocabulary, not the runtime's.
    known = installed | NOT_GLOBALS | KEYWORDS | set(agent.game_action_names)
    unknown = sorted(name for name in named if name not in known)
    assert not unknown, f"the prompt names {unknown}, which the kernel does not install"
