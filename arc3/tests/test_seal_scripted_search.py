"""The world model, phase 2: the search a passing test earns.

One scripted reference over a room game, read through the viewer as the
review of the search. Every python cell opens with comments saying what it
exercises. After a python call whose last ``test_world_model()`` passed and
which played no move, the harness tests the model again in the REPL, tests
the same files in the agent process, searches the model in a child process
for a win or a move it does not predict, replays a found path in the REPL,
and appends a report to the call's output.

- A win found, through a model that calibrates in a helper module using the
  preloaded names unimported (the host's loads must give them, as the
  kernel's do): the fewest moves the model allows, ``suggested_actions`` and
  ``suggested_history`` bound, the path played as it stands, and the names
  gone after the move. The MOUSE line: clicks were tried per region only.
- The limits: a slow model stopped on time, a greedy one on the search's own
  memory check, and an allocation refused by the address-space cap.
- In a MOUSE game, a board the model cannot draw: a target of its own, with
  what ``render`` raised, and an all-wildcard last turn.
- Each kind of nothing: every reachable state searched (exhausted). A win
  that repeats one board for many moves is still suggested: the repetition
  check is off. A boundary where the model's
  code raises, with the traceback, and where it raises ``Unknown``, with the
  path ending before that move.
- The model's own ``suggested_actions`` overwritten by a search.
- No search: a call whose passing test is followed by a move, and a call
  whose model passed and then broke its file (a short note instead).
- A ``distance_to_win`` that raises on a state the history never reached:
  the search stops ordering by it and says so, and still finds the win.

Regenerate (rewrites the committed file; free, stub-driven):

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_seal_scripted_search'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import arcengine
import numpy as np
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv
from taaf.game import Game, GameState, RunSession

from platypus import LLMAgentSolver, RunConfig
from reference_support import ReferenceCase, ReferenceRun, regenerate_case, replay_case, require_regeneration, tool_call
from seal import SealAgent
from seal.search import ASKED_EARLY_NOTE, MAX_CLICKS, RETEST_FAILED_NOTE

RNG_SEED = 20260917

#: Where the room sits on the board.
_TOP, _LEFT = 10, 10


@dataclass
class RoomGame(Game):
    """A player walking a small room to its goal, over two levels.

    Level 1 is an open 4x4 room. Level 2 is 4x5 with a wall down its middle
    column; stepping on the switch beside the wall's foot removes the wall
    and the switch. UP/DOWN/LEFT/RIGHT move the player, a wall or the room's edge
    blocks, and SPACE and MOUSE do nothing — offered so the search has moves
    that lead nowhere, and MOUSE so its click line renders. Reaching the goal
    wins the level.
    """

    label: str = field(default="room_game", init=True)
    _level: int = field(default=0, init=False, repr=False)
    _player: tuple[int, int] = field(default=(0, 0), init=False, repr=False)
    _switched: bool = field(default=False, init=False, repr=False)

    _LEVELS: tuple[dict[str, Any], ...] = (
        {"size": (4, 4), "goal": (3, 3), "switch": None, "wall": ()},
        {"size": (4, 5), "goal": (0, 4), "switch": (3, 1), "wall": ((0, 2), (1, 2), (2, 2), (3, 2))},
    )
    _MOVES = {
        arcengine.GameAction.ACTION1: (-1, 0),
        arcengine.GameAction.ACTION2: (1, 0),
        arcengine.GameAction.ACTION3: (0, -1),
        arcengine.GameAction.ACTION4: (0, 1),
    }

    def _start_game(self, session: RunSession) -> GameState:
        del session
        self.game_id = self.label
        self.number_of_levels = len(self._LEVELS)
        self.base_actions_per_level = [5, 6]
        self.hint = "scripted search exercise"
        self.grid_size = (64, 64)
        self._level = 0
        self._restart_level()
        return self._build(arcengine.GameState.NOT_FINISHED)

    def _restart_level(self) -> None:
        self._player = (0, 0)
        self._switched = False

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        if action.id == arcengine.GameAction.RESET:
            self._restart_level()
            return self._build(arcengine.GameState.NOT_FINISHED)
        if action.id in self._MOVES:
            spec = self._LEVELS[self._level]
            dr, dc = self._MOVES[action.id]
            r, c = self._player[0] + dr, self._player[1] + dc
            rows, cols = spec["size"]
            walled = (r, c) in spec["wall"] and not self._switched
            if 0 <= r < rows and 0 <= c < cols and not walled:
                self._player = (r, c)
                if (r, c) == spec["switch"]:
                    self._switched = True
                if (r, c) == spec["goal"]:
                    self._level += 1
                    if self._level >= len(self._LEVELS):
                        return self._build(arcengine.GameState.WIN)
                    self._restart_level()
        return self._build(arcengine.GameState.NOT_FINISHED)

    def _build(self, state: arcengine.GameState) -> GameState:
        grid = np.zeros((64, 64), dtype=np.int8)
        if state == arcengine.GameState.WIN:
            grid[:, :] = 14
        else:
            spec = self._LEVELS[self._level]
            rows, cols = spec["size"]
            room = grid[_TOP : _TOP + rows, _LEFT : _LEFT + cols]
            room[:, :] = 5
            if not self._switched:
                for cell in spec["wall"]:
                    room[cell] = 3
                if spec["switch"] is not None:
                    room[spec["switch"]] = 9
            room[spec["goal"]] = 14
            room[self._player] = 12
        raw = arcengine.FrameDataRaw(
            game_id=self.game_id or self.label,
            state=state,
            levels_completed=len(self._LEVELS) if state == arcengine.GameState.WIN else self._level,
            win_levels=len(self._LEVELS),
            available_actions=[1, 2, 3, 4, 5, 6],
        )
        raw.frame = [grid]
        return GameState(raw=raw)


# ---------------------------------------------------------------------------
# The world models the script writes.
# ---------------------------------------------------------------------------

#: Walls block, and everything else is floor — the switch included, since
#: nobody has stepped on it.
ROOM_MODEL = """\
import numpy as np

# Calibrated from this level's first frame: the room is at rows 10-13,
# columns 10-14; 5 is floor, 3 a wall, 0 outside, 12 the player, 14 the goal.
FIRST = history[0].frame.grid
ROOM = FIRST[10:14, 10:15]
START = tuple(int(v) for v in np.argwhere(ROOM == 12)[0])
GOAL = tuple(int(v) for v in np.argwhere(ROOM == 14)[0])
MOVES = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}


def initial_state():
    return {'outcome': 'playing', 'energy': 2000, 'player': START}


def blocked(state, r, c):
    return not (0 <= r < 4 and 0 <= c < 5) or int(ROOM[r, c]) in (0, 3)


def do_move(state, action):
    state['energy'] -= 1
    if action.name not in MOVES:
        return
    dr, dc = MOVES[action.name]
    r, c = state['player'][0] + dr, state['player'][1] + dc
    if blocked(state, r, c):
        return
    state['player'] = (r, c)
    if (r, c) == GOAL:
        state['outcome'] = 'won'


def render(state):
    grid = FIRST.copy()
    grid[10 + START[0], 10 + START[1]] = 5
    grid[10 + state['player'][0], 10 + state['player'][1]] = 12
    return grid
"""

#: A model that takes its time: SPACE sleeps past the reference's whole time
#: limit, so the first expansion outlasts it however fast the machine is.
SLOW_MODEL = ROOM_MODEL.replace("import numpy as np\n", "import time\n\nimport numpy as np\n", 1).replace(
    "    if action.name not in MOVES:\n",
    "    if action.name == 'SPACE':\n        time.sleep(12)\n    if action.name not in MOVES:\n",
)

#: A model whose SPACE puts 300 MB into the state: after the first expansion the
#: search holds more than the reference's memory budget leaves its own check,
#: and one large block rather than many small ones keeps copying and hashing
#: states quick, so the time limit cannot fire first.
GREEDY_MODEL = ROOM_MODEL.replace(
    "    if action.name not in MOVES:\n",
    "    if action.name == 'SPACE':\n        state['scratch'] = np.ones(300 * 2**20, dtype=np.uint8)\n"
    "    if action.name not in MOVES:\n",
)

#: A model whose SPACE asks for more memory than any search may have: the
#: allocation itself is refused.
HUGE_MODEL = ROOM_MODEL.replace(
    "    if action.name not in MOVES:\n",
    "    if action.name == 'SPACE':\n        state['scratch'] = np.ones(64 * 2**30, dtype=np.uint8)\n"
    "    if action.name not in MOVES:\n",
)

#: A helper module the level-1 model calibrates through. It uses names every
#: piece of the model's code starts with — ``Counter`` and ``np`` — without
#: importing them, as the model's helpers do in a real run (tr87).
BOARD_HELPER = """\
def find(grid, colour):
    # Counter and np are preloaded here, as in a cell.
    assert Counter(grid.ravel().tolist())[colour] == 1, f"colour {colour} is not a single cell"
    return tuple(int(v) for v in np.argwhere(grid == colour)[0])
"""

#: The level-1 room model, calibrated through the helper.
HELPER_ROOM_MODEL = ROOM_MODEL.replace(
    "START = tuple(int(v) for v in np.argwhere(ROOM == 12)[0])\nGOAL = tuple(int(v) for v in np.argwhere(ROOM == 14)[0])\n",
    "START = find(ROOM, 12)\nGOAL = find(ROOM, 14)\n",
).replace("import numpy as np\n", "import numpy as np\nfrom board import find\n", 1)

#: A board the model has not worked out how to draw: the bottom row, where the
#: player has never been.
UNDRAWN_ROW_MODEL = ROOM_MODEL.replace(
    "    grid = FIRST.copy()\n",
    "    if state['player'][0] == 3:\n        raise ValueError('no drawing for the bottom row yet')\n    grid = FIRST.copy()\n",
)

#: A guess nothing has contradicted: eight presses of SPACE open the wall. The
#: counter never shows on the board, so a win repeats one board press after
#: press — which the whale's repetition check would reject, and the seal's
#: search, with the check off, suggests.
CHARGE_MODEL = (
    ROOM_MODEL.replace("'player': START}", "'player': START, 'charge': 0}")
    .replace(
        "    if action.name not in MOVES:\n",
        "    if action.name == 'SPACE':\n        state['charge'] += 1\n        return\n    if action.name not in MOVES:\n",
    )
    .replace("int(ROOM[r, c]) in (0, 3)", "int(ROOM[r, c]) == 0 or (int(ROOM[r, c]) == 3 and state['charge'] < 8)")
)

#: The same guess with a bug: the counter is never initialised.
BROKEN_CHARGE_MODEL = CHARGE_MODEL.replace(", 'charge': 0}", "}")

#: Honest about the switch: nobody has stepped on it.
UNKNOWN_SWITCH_MODEL = ROOM_MODEL.replace(
    "MOVES = {",
    "SWITCH = tuple(int(v) for v in np.argwhere(ROOM == 9)[0]) if (ROOM == 9).any() else None\nMOVES = {",
).replace(
    "    state['player'] = (r, c)\n",
    "    if (r, c) == SWITCH:\n        raise Unknown('nobody has stepped on the switch')\n    state['player'] = (r, c)\n",
)

#: What the step onto the switch showed: the switch and the wall are gone. And
#: a heuristic that divides by zero one column short of the goal's — a column
#: the history never reached.
SWITCH_MODEL = (
    ROOM_MODEL.replace(
        "MOVES = {",
        "SWITCH = tuple(int(v) for v in np.argwhere(ROOM == 9)[0]) if (ROOM == 9).any() else None\nMOVES = {",
    )
    .replace("'player': START}", "'player': START, 'switched': False}")
    .replace("int(ROOM[r, c]) in (0, 3)", "int(ROOM[r, c]) == 0 or (int(ROOM[r, c]) == 3 and not state['switched'])")
    .replace(
        "    state['player'] = (r, c)\n",
        "    state['player'] = (r, c)\n    if (r, c) == SWITCH:\n        state['switched'] = True\n",
    )
    .replace(
        "    grid[10 + START[0], 10 + START[1]] = 5\n",
        "    grid[10 + START[0], 10 + START[1]] = 5\n"
        "    if state['switched']:\n"
        "        room = grid[10:14, 10:15]\n"
        "        room[(ROOM == 3) | (ROOM == 9)] = 5\n",
    )
    + """

def distance_to_win(state):
    r, c = state['player']
    return abs(r - GOAL[0]) + abs(c - GOAL[1]) + 6 // (3 - c)
"""
)


def _python(code: str) -> ullm.conv.LLMResponse:
    """The scripted model running a snippet."""
    return tool_call("python", {"code": code})


def _create(path: str, content: str) -> ullm.conv.LLMResponse:
    """The scripted model writing a whole file."""
    return tool_call("create_file", {"path": path, "content": content})


def _script() -> list[ullm.conv.LLMResponse]:
    """One entry per backend call. A decision ends at the cell that acts."""
    return [
        # === decision 1 (level 1; history: the seed turn) ====================
        _python(
            "# Exploring, to see a move: one action per call.\n"
            "print(submit_action([Action('RIGHT')])['executed_actions'])\n"
        ),
        # === decision 2 (level 1; history: seed, RIGHT) =====================
        _create("board.py", BOARD_HELPER),
        _create("world_model.py", HELPER_ROOM_MODEL),
        _python(
            "# A passing test in a call that plays no move earns a search. The model\n"
            "# calibrates through a helper module that uses preloaded names, which\n"
            "# the search's own loads have to give it too. The search runs\n"
            "# after the call, and its report follows this cell's own output: a win\n"
            "# in the fewest moves the model allows, and — this game has MOUSE —\n"
            "# the clicks it tried.\n"
            "print(test_world_model())\n"
        ),
        _python(
            "# The path is bound in the kernel: suggested_actions, and history as the\n"
            "# model predicts it after those moves. Played as it stands, it wins the\n"
            "# level; the move clears both names.\n"
            "print(suggested_actions)\n"
            "print(len(history), len(suggested_history), suggested_history[-1].outcome)\n"
            "r = submit_action(suggested_actions)\n"
            "print(r['executed_actions'], 'suggested_actions' in globals())\n"
        ),
        # === decision 3 (level 2; a fresh kernel, the files kept) ============
        _python(
            "# The level-1 model passes on level 2's first frame. The wall blocks and\n"
            "# the switch is plain floor to it, so no win is reachable, and nothing\n"
            "# raises: the search reports every reachable state searched.\n"
            "print(test_world_model())\n"
        ),
        _create("world_model.py", SLOW_MODEL),
        _python(
            "# SPACE sleeps 12 seconds, and this reference gives a search 10: the\n"
            "# search stops on its time limit with nothing found.\n"
            "print(test_world_model())\n"
        ),
        _create("world_model.py", GREEDY_MODEL),
        _python(
            "# SPACE puts 300 MB into the state, and this reference gives a search\n"
            "# 768 MB: the search's own memory check stops it with nothing found.\n"
            "print(test_world_model())\n"
        ),
        _create("world_model.py", HUGE_MODEL),
        _python(
            "# SPACE asks for 64 GB. The allocation is refused outright under the\n"
            "# search's address-space cap, and the search fails with MemoryError.\n"
            "print(test_world_model())\n"
        ),
        _create("world_model.py", UNDRAWN_ROW_MODEL),
        _python(
            "# render raises on the bottom row, which the history never reached. In\n"
            "# a game with MOUSE the search draws every state to list its clicks, so\n"
            "# a board that does not draw is a target of its own, with what render\n"
            "# raised.\n"
            "print(test_world_model())\n"
        ),
        _python(
            "# The path ends on the undrawn board: its last turn is all wildcards.\n"
            "print(len(suggested_actions), len(suggested_history), set(suggested_history[-1].frame.grid.flatten()))\n"
        ),
        _create("world_model.py", CHARGE_MODEL),
        _python(
            "# The file changed since the last search, so its path is unbound as the\n"
            "# call starts: it predicts a model that is gone.\n"
            "print('bound after the edit:', 'suggested_actions' in globals())\n"
            "# A guess nothing contradicts: eight SPACEs open the wall. The win it\n"
            "# allows repeats the same board press after press; with the\n"
            "# repetition check off, it is suggested all the same.\n"
            "print(test_world_model())\n"
        ),
        _create("world_model.py", BROKEN_CHARGE_MODEL),
        _python(
            "# The same guess with its counter never set: SPACE raises in do_move, a\n"
            "# move the history never played. That is a boundary, reported with the\n"
            "# traceback.\n"
            "print(test_world_model())\n"
        ),
        _create("charge_rules.py", BROKEN_CHARGE_MODEL),
        _create("world_model.py", "from charge_rules import initial_state, do_move, render\n"),
        _python(
            "# The same rule, in a helper world_model.py imports do_move from: the\n"
            "# search calls it with no frame of world_model.py in between, and the\n"
            "# traceback is still the helper's alone.\n"
            "print(test_world_model())\n"
        ),
        _create("world_model.py", UNKNOWN_SWITCH_MODEL),
        _python(
            "# Honest about the switch: Unknown. The search's path ends on that move,\n"
            "# and this model's own suggested_actions is overwritten by it.\n"
            "suggested_actions = 'my own plan'\n"
            "print(test_world_model())\n"
        ),
        _python(
            "# suggested_history stops before the unknown move: one turn short of the\n"
            "# path.\n"
            "print(suggested_actions)\n"
            "print(len(history), len(suggested_actions), len(suggested_history))\n"
        ),
        _python(
            "# A file written in the call: test_world_model() finds the files changed\n"
            "# since the search and unbinds its path. Looking it up after the test is\n"
            "# too early — this call's search runs when the call ends — and its\n"
            "# report says the names are there now.\n"
            "with open('notes.txt', 'w') as f:\n"
            "    f.write('the switch opens the wall\\n')\n"
            "print('bound before the test:', 'suggested_actions' in globals())\n"
            "print(test_world_model())\n"
            "print('bound after the test:', 'suggested_actions' in globals())\n"
            "print(suggested_actions)\n"
        ),
        _python(
            "# A pass, and then an edit that breaks the file in the same call: the\n"
            "# harness tests again after the call, so there is no search — a short\n"
            "# note instead.\n"
            "print(test_world_model())\n"
            "with open('world_model.py', 'a') as f:\n"
            "    f.write(\"\\nraise RuntimeError('half-way through an edit')\\n\")\n"
        ),
        _create("world_model.py", UNKNOWN_SWITCH_MODEL),
        _python(
            "# A pass followed by a move: no search. The path runs stepped, and the\n"
            "# model declines its last move, which is played and stands.\n"
            "print(test_world_model())\n"
            "r = submit_action(suggested_actions)\n"
            "print(r['executed_actions'])\n"
        ),
        # === decision 4 (level 2; the player on the switch, the wall gone) ===
        _create("world_model.py", SWITCH_MODEL),
        _python(
            "# The switch modelled. distance_to_win divides by zero in a column the\n"
            "# history never reached, so the search stops ordering by it, says so,\n"
            "# and still finds the win.\n"
            "print(test_world_model())\n"
        ),
        _python("r = submit_action(suggested_actions)\nprint(r['executed_actions'])\n"),
    ]


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The seal over the room game, with the default world-model gate."""
    agent = SealAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=2000,
        # No side turns: a reflection or an audit would take the script's
        # next replies, and none of the corners here is theirs.
        level_reflection="none",
        strategy_audit="none",
        llm_config=run.llm_config,
        run_config=RunConfig(rng_seed=RNG_SEED, rng_seed_per_game=run.seeds),
        # Level 2's third decision writes and searches eight models before it
        # acts, the longest run of calls in one decision.
        max_llm_calls_per_move=30,
        # Low, so a slow model and a greedy one hit them in one expansion;
        # every other search here finishes well inside both. The time is
        # generous enough that copying the greedy model's state under a
        # loaded gate still reaches the memory check first.
        search_seconds=10.0,
        search_memory_bytes=768 * 2**20,
    )
    return taaf.benchmark.Benchmark(
        games=[RoomGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir
    )


CASE = ReferenceCase(
    name="seal_scripted_search",
    build=_build,
    profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)},
)


async def test_search_reference_replays_and_every_corner_fired(tmp_path: Path) -> None:
    (ref,) = await replay_case(CASE, tmp_path)
    # The last decision's calls reach no later request (the game ends on its
    # move), so they are read from the file.
    text = ref.path.read_text(encoding="utf-8")

    assert ref.expected["state"] == "won"
    # A win, bound and played; the names gone after the move.
    assert "Found: a win in 5 moves \\u2014 the fewest your model allows." in text
    # ...through a model calibrated by a helper module that uses preloaded names
    # unimported, which the search's own loads give it as the kernel does: the
    # win above needs it. The report names no files.
    assert "Searched: level 1, 2 turns of history." in text
    # A game with MOUSE: complete for the clicks tried, every region of every
    # board clicked, and where a search stopped early only the cap.
    assert "Search space: complete for the clicks tried \\u2014 every shorter path using them was ruled out." in text
    assert "on every region of its board" in text
    assert f"on its {MAX_CLICKS} largest regions" in text
    assert "2 7 won" in text
    assert "Action('RIGHT')] False" in text
    # Each kind of nothing.
    assert "Found: no win, and no move your world model does not predict." in text
    assert "Search space: complete for the clicks tried \\u2014 every state they reach was searched." in text
    assert "Found: a win in 12 moves" in text
    # The limits: time, the search's memory check, and a refused allocation.
    assert "Found: nothing before the search stopped." in text
    assert text.count("The state space may or may not have a win or unknown state.") >= 3
    assert "The states those clicks reach were fully searched, and no win or unknown state is among them." in text
    assert "Search space: not complete \\u2014 the search ran out of time." in text
    assert "Search space: not complete \\u2014 the search ran out of memory." in text
    assert "the search failed: numpy._core._exceptions._ArrayMemoryError: Unable to allocate 64.0 GiB" in text
    # A board the model cannot draw: its own target, with render's traceback.
    assert "Found: a board your world model cannot draw, after the last of 3 moves." in text
    assert "ValueError: no drawing for the bottom row yet" in text
    assert "3 4 {-1}" in text
    # Boundaries: the traceback, and Unknown with the history one turn short
    # of the path.
    assert "Found: a move your world model does not predict, the last of 1 move." in text
    assert "KeyError: 'charge'" in text
    # ...and from a helper: the model's own file, and none of ours.
    assert 'File \\"charge_rules.py\\", line' in text
    assert (
        "No win is reachable with the clicks tried without passing a move your model does not predict or a board it cannot draw"
        in text
    )
    assert (
        "To reproduce: world_model.do_move(copy.deepcopy(suggested_history[-1].state), suggested_actions[-1])" in text
    )
    assert "To reproduce: world_model.render(copy.deepcopy(suggested_history[-1].state))" in text
    assert "Found: a move your world model does not predict, the last of 4 moves." in text
    assert "do_move raised Unknown('nobody has stepped on the switch')." in text
    assert "Action('DOWN')]\\n1 4 4" in text
    # No search after an edit that broke the model; none after a move.
    assert RETEST_FAILED_NOTE.replace("\u2014", "\\u2014") in text
    # A path lasts only while the files are the ones it was searched with:
    # unbound as the next call starts, or at the model's test within a call.
    # Looked up in the call that tested, it is not there yet, and the report
    # says it is now.
    assert "bound after the edit: False" in text
    assert "bound before the test: True" in text and "bound after the test: False" in text
    assert "NameError: name 'suggested_actions' is not defined" in text
    assert ASKED_EARLY_NOTE.replace("\u2014", "\\u2014") in text
    # The heuristic dropped, the win still found.
    assert "Found: a win in 6 moves" in text
    assert "Order: distance_to_win raised: ZeroDivisionError" in text

    requests = ref.all_request_text()
    assert "/scratch" not in requests and "/tmp/" not in requests and " at 0x" not in requests
    assert "seal.harness" not in requests and "_repl_state" not in requests and "seal-model-" not in requests
    assert "py_model" not in requests and "pyright" not in requests


async def test_regenerate_seal_scripted_search(tmp_path: Path) -> None:
    """Regenerates the committed reference (stub-driven, no API spend); gated
    because it rewrites a committed file. Validated by immediate replay (I13)."""
    require_regeneration()
    backend = ullm.backends.stub.StubBackend(responses=_script(), on_exhausted="raise")
    await regenerate_case(CASE, backend=backend, tmp_path=tmp_path)
